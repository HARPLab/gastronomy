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

import time
import cv2

AZURE_KINECT_INTRINSICS = 'calib/azure_kinect_intrinsics.intr'
AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead_to_world.tf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
    parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS) 
    parser.add_argument('--data_dir', '-d', type=str, default='playing_data/')
    parser.add_argument('--food_type', '-f', type=str)   
    parser.add_argument('--slice_num', '-s', type=int)    
    parser.add_argument('--trial_num', '-t', type=int)
    args = parser.parse_args()

    dir_path = args.data_dir
    createFolder(dir_path)
    dir_path += args.food_type + '/'
    createFolder(dir_path)
    dir_path += str(args.slice_num) + '/'
    createFolder(dir_path)
    dir_path += str(args.trial_num) + '/'
    createFolder(dir_path)
   
    #rospy.init_node('collect_data')
    print('Starting robot')
    fa = FrankaArm()    

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)    

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()
    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()

    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    realsense_rgb_image = get_realsense_rgb_image(cv_bridge)
    realsense_depth_image = get_realsense_depth_image(cv_bridge)

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

    cutting_board_z_height = 0.04
    intermediate_pose_z_height = 0.19

    object_center_point_in_world = get_object_center_point_in_world(object_image_position[0] + cutting_board_x_min,
                                                                    object_image_position[1] + cutting_board_y_min,
                                                                    azure_kinect_depth_image, azure_kinect_intrinsics,
                                                                    azure_kinect_to_world_transform)
    object_center_pose = fa.get_pose()
    object_center_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], cutting_board_z_height]
    
    intermediate_robot_pose = object_center_pose.copy()
    intermediate_robot_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], intermediate_pose_z_height]

    # Saving images
    cv2.imwrite(dir_path + 'starting_azure_kinect_rgb_image.png', cropped_rgb_image)
    np.save(dir_path + 'starting_azure_kinect_depth_image.npy', cropped_depth_image)
    cv2.imwrite(dir_path + 'starting_realsense_rgb_image.png', realsense_rgb_image)
    np.save(dir_path + 'starting_realsense_depth_image.npy', realsense_depth_image)

    # Saving object positions
    object_world_position = np.array(object_center_point_in_world)
    intermediate_robot_position = intermediate_robot_pose.translation

    print('Closing Grippers')
    #Close Gripper
    fa.close_gripper()

    #Move to intermediate robot pose
    fa.goto_pose(intermediate_robot_pose)

    #Pushing down on the object
    fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10], block=False)

    #Save Audio, Realsense Camera, Robot Positions, and Forces
    audio_p = save_audio(dir_path, 'push_down_audio.wav')
    realsense_p = save_realsense(dir_path, 'push_down_realsense')

    (push_down_robot_positions, push_down_robot_forces) = get_robot_positions_and_forces(fa, 5)
    audio_p.terminate()
    realsense_p.terminate()

    #Move to intermediate robot pose
    fa.goto_pose(intermediate_robot_pose)

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()

    fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

    # Save Audio, Realsense Camera, Finger Vision
    audio_p = save_audio(dir_path, 'grasp_audio.wav')
    realsense_p = save_realsense(dir_path, 'grasp_realsense')
    finger_vision_p = save_finger_vision(dir_path, 'grasp_finger_vision')

    print('Closing Grippers')
    #Close Gripper
    fa.close_gripper()
    rospy.sleep(1)
    grasp_gripper_width = fa.get_gripper_width()
    audio_p.terminate()
    realsense_p.terminate()
    finger_vision_p.terminate()

    #Move to intermediate robot pose
    fa.goto_pose(intermediate_robot_pose)

    # Save Audio, Realsense Camera, Finger Vision
    audio_p = save_audio(dir_path, 'release_audio.wav')
    realsense_p = save_realsense(dir_path, 'release_realsense')
    finger_vision_p = save_finger_vision(dir_path, 'release_finger_vision')

    rospy.sleep(1)
    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()
    rospy.sleep(1)
    audio_p.terminate()
    realsense_p.terminate()
    finger_vision_p.terminate()


    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()

    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    realsense_rgb_image = get_realsense_rgb_image(cv_bridge)
    realsense_depth_image = get_realsense_depth_image(cv_bridge)

    cropped_rgb_image = azure_kinect_rgb_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]
    cropped_depth_image = azure_kinect_depth_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]

    # Saving images
    cv2.imwrite(dir_path + 'ending_azure_kinect_rgb_image.png', cropped_rgb_image)
    np.save(dir_path + 'ending_azure_kinect_depth_image.npy', cropped_depth_image)
    cv2.imwrite(dir_path + 'ending_realsense_rgb_image.png', realsense_rgb_image)
    np.save(dir_path + 'ending_realsense_depth_image.npy', realsense_depth_image)

    np.savez(dir_path+'trial_info.npz', object_image_position=object_image_position, 
                                        object_world_position=object_world_position, 
                                        intermediate_robot_position=intermediate_robot_position,
                                        push_down_robot_positions=push_down_robot_positions,
                                        push_down_robot_forces=push_down_robot_forces,
                                        grasp_gripper_width=np.array([grasp_gripper_width]))

