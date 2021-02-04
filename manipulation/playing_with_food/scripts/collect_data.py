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


    # joints_home=[-1.45437655e-04, -7.39825729e-01,  1.97288847e-04, -2.84437687e+00,-1.33295213e-03,  2.08647642e+00,  7.86089686e-01]    
    # num_reps=10
    # x_transl_stiffness=500    

    # for i in range(0,num_reps):

    #     # check start time
    #     timestamp_str = str(time.time()).split('.')

    #     # create a directory to save frames
    #     directory = timestamp_str[0] + "_" + timestamp_str[1]
    #     save_loc = os.path.join(folder_loc, "data_" + directory) 
    #     os.mkdir(save_loc)

    #     object_center_point_in_world = get_object_center_point_in_world()
    #     vertical_stop_on_contact_forces=[20.0, 20.0, 5.0, 20.0, 20.0, 20.0]        

    #      ##########Take Kinect image (at beginning)

    #     rgb_cv_image = get_snapshot_rgb()
    #     depth_cv_image = get_snapshot_depth()
    #     cv2.imwrite(save_loc + "/rgb_pre_drag.png", rgb_cv_image)
    #     cv2.imwrite(save_loc + "/depth_pre_drag.png", depth_cv_image)

    #     ######ACTION 1 - Drag object
    #     #Close gripper partially
    #     fa.goto_gripper(0.02, force=5)     

    #     #Go close to cutting board in front of object
    #     desired_object_pose = RigidTransform(rotation=np.array([[1, 0,  0],[0, -1,  0],[0, 0,  -1]]),translation= \
    #         np.array([object_center_point_in_world.x + 0.05, object_center_point_in_world.y, object_center_point_in_world.z+0.3]),from_frame='franka_tool')
    #     fa.goto_pose_with_cartesian_control(desired_object_pose, 4.0)    

    #     #Go down to cutting board
    #     neg_ztranslation = RigidTransform(rotation=np.eye(3), translation=np.array([0, 0, -0.1]),from_frame='franka_tool', to_frame='franka_tool')
    #     fa.goto_pose_delta_with_cartesian_control(neg_ztranslation,4.0,vertical_stop_on_contact_forces)
        
    #     #Drag backward 
    #     neg_xtranslation_10cm = RigidTransform(rotation=np.eye(3), translation=np.array([-0.12, 0, 0]),from_frame='franka_tool', to_frame='franka_tool')
    #     fa.goto_pose_delta_with_cartesian_control(neg_xtranslation_10cm,cartesian_impedances=[x_transl_stiffness,3000,3000,100,100,100])
                
    #     #Need to move arm out of way for Kinect to see slice after this - take post-drag image
    #     up = RigidTransform(rotation=np.eye(3), translation=np.array([0, 0, 0.15]),from_frame='franka_tool', to_frame='franka_tool')
    #     fa.goto_pose_delta_with_cartesian_control(up,cartesian_impedances=[x_transl_stiffness,3000,3000,100,100,100])
    #     back = RigidTransform(rotation=np.eye(3), translation=np.array([-0.2, 0, 0]),from_frame='franka_tool', to_frame='franka_tool')
    #     fa.goto_pose_delta_with_cartesian_control(back,duration=2.0,cartesian_impedances=[x_transl_stiffness,3000,3000,100,100,100],skill_desc='post_drag_image')
        
    #     ##############Take Kinect image now (after drag)   
        
    #     rgb_cv_image = get_snapshot_rgb()
    #     depth_cv_image = get_snapshot_depth()    
    #     cv2.imwrite(save_loc + "/rgb_post_drag.png", rgb_cv_image)
    #     cv2.imwrite(save_loc + "/depth_post_drag.png", depth_cv_image)

    #     fa.open_gripper()

    #     ########ACTION 2 - Grasp object 
    #     #Go down to to object center
    #     object_center_point_in_world = get_object_center_point_in_world()
    #     desired_object_pose = RigidTransform(rotation=np.array([[1, 0,  0],[0, -1,  0],[0, 0,  -1]]),translation= \
    #         np.array([object_center_point_in_world.x, object_center_point_in_world.y, object_center_point_in_world.z+0.3]),from_frame='franka_tool')
    #     fa.goto_pose_with_cartesian_control(desired_object_pose, 3.0, vertical_stop_on_contact_forces)    

    #     #Go down to cutting board
    #     neg_ztranslation = RigidTransform(rotation=np.eye(3), translation=np.array([0, 0, -0.1]),from_frame='franka_tool', to_frame='franka_tool')
    #     fa.goto_pose_delta_with_cartesian_control(neg_ztranslation,3.0,vertical_stop_on_contact_forces)

    #     #Close gripper to grasp 
    #     fa.goto_gripper(0.0, force=0.1, skill_desc='grasp_object')

    #     #Lift up and move forward
    #     forward_and_up = RigidTransform(rotation=np.eye(3), translation=np.array([0.08, 0, 0.05]),from_frame='franka_tool', to_frame='franka_tool')
    #     fa.goto_pose_delta_with_cartesian_control(forward_and_up,duration=2.0)

    #     ######Action 3 - Drop object
    #     #Drop object 
    #     fa.open_gripper()
    #     fa.goto_joints(joints_home)


    #     rgb_cv_image = get_snapshot_rgb()
    #     depth_cv_image = get_snapshot_depth()    
    #     cv2.imwrite(save_loc + "/rgb_post_drop.png", rgb_cv_image)
    #     cv2.imwrite(save_loc + "/depth_post_drop.png", depth_cv_image)

    #     time.sleep(5)

    # 