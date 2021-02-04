from frankapy import FrankaArm
import os
import numpy as np
import math
import rospy
import pickle
import argparse
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
from sampling_methods import pick_random_skill_from_top_n, pick_most_uncertain, pick_top_skill
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import load_model

import time
import cv2


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

AZURE_KINECT_INTRINSICS = 'calib/azure_kinect_intrinsics.intr'
AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead_to_world.tf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
    parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS) 
    parser.add_argument('--num_trials', '-n', type=int, default=5)
    parser.add_argument('--trial_num', '-t', type=int, default=-1)
    parser.add_argument('--baseline_type', '-bt', type=str, default='baseline')
    parser.add_argument('--inputs', '-i', type=str, default='data/franka_fingers_inputs.npy')
    parser.add_argument('--sampling_method', '-s', type=int, default=2)
    args = parser.parse_args()
   
    path = '/home/klz/Humanoids/Baseline/Sampling_'+str(args.sampling_method)+'/trial_'+str(args.trial_num) + '/'

    if args.trial_num == -1:
        print('Trial Number not provided!')
    else:
        createFolder(path)

    x_y_thetas = np.load(args.inputs)

    if args.baseline_type == 'baseline_same_friction':
        baseline_data = np.load('baseline_same_friction/pick_up/franka_fingers_data.npy')
    elif args.baseline_type == 'baseline_same_mass':
        baseline_data = np.load('baseline_same_mass/pick_up/franka_fingers_data.npy')
    else:
        baseline_data = np.load('baseline/pick_up/good_franka_fingers_data.npy')

    #rospy.init_node('collect_data')
    print('Starting robot')
    fa = FrankaArm()    

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()

    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)    

    x_y_thetas = np.load(args.inputs)

    num_skills = x_y_thetas.shape[0]
    num_blocks_baseline = baseline_data.shape[1]
    block_affinities = np.ones(num_blocks_baseline)
    block_probabilities = np.ones(num_blocks_baseline)
    skill_probabilities = np.ones(num_skills) * 0.5

    num_successes = 0

    for i in range(args.num_trials):

        if args.sampling_method == 0:
            skill_num = pick_random_skill_from_top_n(skill_probabilities, int(num_skills * 0.03))
        elif args.sampling_method == 1:
            skill_num = pick_top_skill(skill_probabilities)
        elif args.sampling_method == 2:
            if i < 3:
                skill_num = pick_most_uncertain(skill_probabilities)
            else:
                skill_num = pick_random_skill_from_top_n(skill_probabilities, int(num_skills * 0.03))

        current_input = x_y_thetas[skill_num]

        print(current_input)

        azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
        azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
        #print(azure_kinect_depth_image)

        cutting_board_x_min = 750
        cutting_board_x_max = 1170
        cutting_board_y_min = 290
        cutting_board_y_max = 620

        cropped_rgb_image = azure_kinect_rgb_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]
        cropped_depth_image = azure_kinect_depth_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]

        object_image_position = np.array([220, 175])

        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
               print('x = %d, y = %d'%(x, y))
               param[0] = x
               param[1] = y
        
        cv2.namedWindow('image')
        cv2.imshow('image', cropped_rgb_image)
        #time.sleep(2)
        cv2.setMouseCallback('image', onMouse, object_image_position)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cutting_board_z_height = 0.03
        intermediate_pose_z_height = 0.19

        object_center_point_in_world = get_object_center_point_in_world(object_image_position[0] + cutting_board_x_min,
                                                                        object_image_position[1] + cutting_board_y_min,
                                                                        azure_kinect_depth_image, azure_kinect_intrinsics,
                                                                        azure_kinect_to_world_transform)
        object_center_pose = fa.get_pose()
        object_center_pose.translation = [object_center_point_in_world[0] + current_input[0], object_center_point_in_world[1] + current_input[1], cutting_board_z_height]
        theta = current_input[2]
        if theta < -1.5:
            theta += np.pi
        elif theta > 1.5:
            theta -= np.pi
        
        new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [-np.sin(theta), -np.cos(theta), 0],
                              [0, 0, -1]])
        object_center_pose.rotation = new_rotation


        intermediate_robot_pose = object_center_pose.copy()
        intermediate_robot_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], intermediate_pose_z_height]

        #Move to intermediate robot pose
        fa.goto_pose(intermediate_robot_pose)

        fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

        #Close Gripper
        fa.goto_gripper(0.045, grasp=True, force=10.0)
        
        #Move to intermediate robot pose
        fa.goto_pose(intermediate_robot_pose)

        fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])

        print('Opening Grippers')
        #Open Gripper
        fa.open_gripper()

        fa.goto_pose(intermediate_robot_pose)

        #Reset Pose
        fa.reset_pose() 
        #Reset Joints
        fa.reset_joints()

        val = input("Enter 0 if the skill failed or 1 if the skill succeeded: ")
        if int(val) == 1:
            block_affinities += baseline_data[skill_num]
            num_successes += 1
        else: 
            block_affinities += (1-baseline_data[skill_num])

        block_probabilities = block_affinities / (i+1)

        skill_probabilities = np.sum(np.transpose(baseline_data) * block_probabilities.reshape(-1,1), axis=0)
        skill_probabilities /= np.sum(block_probabilities)

        fig = plt.figure()
        ax = plt.axes()

        p = ax.scatter(x_y_thetas[:,0], x_y_thetas[:,1], c=skill_probabilities);
        ax.scatter(current_input[0], current_input[1], c='red');
        if i == 0:
            ax.set_title('Skill Parameter Probabilities after ' + str(i+1) + ' skill')
        else:
            ax.set_title('Skill Parameter Probabilities after ' + str(i+1) + ' skills')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        fig.colorbar(p)
        plt.savefig(path + str(i+1)+'.png')
        plt.show()
        