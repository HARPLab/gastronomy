import os
import subprocess
import numpy as np
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm
from cv_bridge import CvBridge

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--position_dmp_weights_file_path', '-w', type=str)
    parser.add_argument('--data_dir', '-d', type=str, default='cutting_data/')
    parser.add_argument('--food_type', '-f', type=str)
    parser.add_argument('--slice_num', '-s', type=int)    
    parser.add_argument('--trial_num', '-t', type=int)
    parser.add_argument('--veg_food_height', '-ht', type=float, default=0.05)
    args = parser.parse_args()

    dir_path = args.data_dir
    createFolder(dir_path)
    dir_path += args.food_type + '/'
    createFolder(dir_path)
    dir_path += str(args.slice_num) + '/'
    createFolder(dir_path)
    dir_path += str(args.trial_num) + '/'
    createFolder(dir_path, delete_previous=True)

    position_dmp_file = open(args.position_dmp_weights_file_path,"rb")
    position_dmp_info = pickle.load(position_dmp_file)

    print('Starting robot')
    fa = FrankaArm()

    tool_delta_pose = RigidTransform(translation=np.array([0.04, 0.16, 0.0]), from_frame='franka_tool', to_frame='franka_tool_base')

    reset_joint_positions = [ 0.02846037, -0.51649966, -0.12048514, -2.86642333, -0.05060268,  2.30209197, 0.7744833 ]
        
    fa.set_tool_delta_pose(tool_delta_pose)

    fa.goto_joints(reset_joint_positions)

    cv_bridge = CvBridge()

    # Capture Overhead Azure Kinect Depth and RGB Images
    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    
    cutting_board_x_min = 685
    cutting_board_x_max = 1245
    cutting_board_y_min = 449
    cutting_board_y_max = 829

    cropped_azure_kinect_rgb_image = azure_kinect_rgb_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]
    cropped_azure_kinect_depth_image = azure_kinect_depth_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]


    # Capture Realsense Depth and RGB Images from Side and Tong Realsense
    side_realsense_rgb_image = get_realsense_rgb_image(cv_bridge, topic='/side_camera/color/image_raw')
    side_realsense_depth_image = get_realsense_depth_image(cv_bridge, topic='/side_camera/depth/image_rect_raw')
    tong_realsense_rgb_image = get_realsense_rgb_image(cv_bridge, topic='/tong_camera/color/image_raw')
    tong_realsense_depth_image = get_realsense_depth_image(cv_bridge, topic='/tong_camera/depth/image_rect_raw')
    
    # Saving images
    cv2.imwrite(dir_path + 'starting_azure_kinect_rgb_image.png', cropped_azure_kinect_rgb_image)
    np.save(dir_path + 'starting_azure_kinect_depth_image.npy', cropped_azure_kinect_depth_image)
    cv2.imwrite(dir_path + 'starting_side_realsense_rgb_image.png', side_realsense_rgb_image)
    np.save(dir_path + 'starting_side_realsense_depth_image.npy', side_realsense_depth_image)
    cv2.imwrite(dir_path + 'starting_tong_realsense_rgb_image.png', tong_realsense_rgb_image)
    np.save(dir_path + 'starting_tong_realsense_depth_image.npy', tong_realsense_depth_image)

    # knife_orientation = np.array([[-0.0126934,   0.9804153,  -0.19648464],
    #                               [ 0.9998221,   0.01504747,  0.01049279],
    #                               [ 0.01324389, -0.1963165,  -0.98045076]])

    knife_orientation = np.array([[0.0,   0.9805069,  -0.19648464],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.19648464,  -0.9805069]])

    flat_knife_orientation = np.array([[0.0,   1,  0.0],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, 0.0,  -1]])

    in_hand_knife_orientation = np.array([[-0.99893883, -0.04193431,  0.0185333 ],
                                          [-0.04429461,  0.98704811, -0.1541273 ],
                                          [-0.01183003, -0.15478467, -0.98787716]])

    if args.slice_num > 0 and args.slice_num < 9:

        starting_position = RigidTransform(rotation=knife_orientation, translation=np.array([0.6, -0.19, 0.19]),
        from_frame='franka_tool', to_frame='world')    

        fa.goto_pose(starting_position, duration=5, use_impedance=False)

        starting_cutting_board_position = RigidTransform(rotation=knife_orientation, translation=np.array([0.6, -0.19, 0.0]),
        from_frame='franka_tool', to_frame='world')   

        fa.goto_pose(starting_cutting_board_position, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 5.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)

        lift_up_slightly_delta_z_position = RigidTransform(translation=np.array([0.0, 0.0, 0.01]),
        from_frame='world', to_frame='world')   

        fa.goto_pose_delta(lift_up_slightly_delta_z_position, duration=2, use_impedance=False)

        positive_delta_y_position = RigidTransform(rotation=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]), translation=np.array([0.0, 0.4, 0.0]),
        from_frame='world', to_frame='world')   

        fa.goto_pose_delta(positive_delta_y_position, duration=10, use_impedance=False, force_thresholds=[10.0, 4.5, 10.0, 10.0, 10.0, 10.0])

        move_backwards_y_slightly = RigidTransform(translation=np.array([0.0, -0.012, 0.0]),
        from_frame='world', to_frame='world')   

        fa.goto_pose_delta(move_backwards_y_slightly, duration=2, use_impedance=False)

        lift_up_veg_food_height = RigidTransform(translation=np.array([0.0, 0.0, args.veg_food_height]),
        from_frame='world', to_frame='world')   

        fa.goto_pose_delta(lift_up_veg_food_height, duration=3, use_impedance=False)

        if args.slice_num == 6 or args.slice_num == 7:
        
            flat_knife_current_pose = fa.get_pose()

            flat_knife_current_pose.rotation = flat_knife_orientation

            fa.goto_pose(flat_knife_current_pose, duration=3, use_impedance=False)

            # 30 deg
            # angled_left_cut_position = RigidTransform(rotation=np.array([
            # [ 0.866,  0.5,   0], 
            # [ -0.5, 0.866,  0],
            # [ 0,   0, 1]
            #     ]), translation=np.array([0.0, 0, 0.0]),
            # from_frame='franka_tool', to_frame='franka_tool')

            # 15 deg
            angled_left_cut_position = RigidTransform(rotation=np.array([
            [ 0.9659,  0.2588,   0], 
            [ -0.2588, 0.9659,  0],
            [ 0,   0, 1]
                ]), translation=np.array([0.0, 0, 0.0]),
            from_frame='franka_tool', to_frame='franka_tool')

            fa.goto_pose_delta(angled_left_cut_position, duration=3, use_impedance=False)

            angled_down_cut_position = RigidTransform(rotation=np.array([
            [ 1.0, 0.0,  0.0],
            [ 0.0, 0.9805069,  -0.19648464],
            [ 0.0, 0.19648464,  0.9805069]
                ]), translation=np.array([0.0, 0, 0.0]),
            from_frame='franka_tool', to_frame='franka_tool')

            fa.goto_pose_delta(angled_down_cut_position, duration=3, use_impedance=False)

        elif args.slice_num == 8:
            
            flat_knife_current_pose = fa.get_pose()

            flat_knife_current_pose.rotation = flat_knife_orientation

            fa.goto_pose(flat_knife_current_pose, duration=3, use_impedance=False)

            # 30 deg
            # angled_right_cut_position = RigidTransform(rotation=np.array([
            # [ 0.866,  -0.5,   0], 
            # [ 0.5, 0.866,  0],
            # [ 0,   0, 1]
            #     ]), translation=np.array([0.0, 0, 0.0]),
            # from_frame='franka_tool', to_frame='franka_tool')

            # 10 deg
            angled_right_cut_position = RigidTransform(rotation=np.array([
            [ 0.9659,  -0.2588,   0], 
            [ 0.2588, 0.9659,  0],
            [ 0,   0, 1]
                ]), translation=np.array([0.0, 0, 0.0]),
            from_frame='franka_tool', to_frame='franka_tool')

            fa.goto_pose_delta(angled_right_cut_position, duration=3, use_impedance=False)

            angled_down_cut_position = RigidTransform(rotation=np.array([
            [ 1.0, 0.0,  0.0],
            [ 0.0, 0.9805069,  -0.19648464],
            [ 0.0, 0.19648464,  0.9805069]
                ]), translation=np.array([0.0, 0, 0.0]),
            from_frame='franka_tool', to_frame='franka_tool')

            fa.goto_pose_delta(angled_down_cut_position, duration=3, use_impedance=False)

        if args.slice_num == 1:
            slice_thickness = 0.03 #0.04
        elif args.slice_num == 2:
            slice_thickness = 0.005
        elif args.slice_num == 3:
            slice_thickness = 0.01
        elif args.slice_num == 4:
            slice_thickness = 0.003
        elif args.slice_num == 5:
            slice_thickness = 0.02
        elif args.slice_num == 6:
            slice_thickness = 0.01 #0.008 #0.0 for 30 deg
        elif args.slice_num == 7:
            slice_thickness = 0.013 #0.02
        elif args.slice_num == 8:
            slice_thickness = 0.03 #0.02 #0.03 #0.01 for tomato
 
        move_over_slice_thickness = RigidTransform(translation=np.array([0.0, slice_thickness, 0.0]),
        from_frame='world', to_frame='world')   

        fa.goto_pose_delta(move_over_slice_thickness, duration=5, use_impedance=False)

    elif args.slice_num == 9 or args.slice_num == 10:

        starting_position = RigidTransform(rotation=knife_orientation, translation=np.array([0.6, 0.0, 0.19]),
        from_frame='franka_tool', to_frame='world')    

        fa.goto_pose(starting_position, duration=5, use_impedance=False)

        in_hand_cut_position = RigidTransform(rotation=np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ]), translation=np.array([0.0, 0.0, 0.0]),
        from_frame='franka_tool', to_frame='franka_tool') 

        #fa.goto_pose_delta(in_hand_cut_position, duration=5, use_impedance=False)

        in_hand_pose = RigidTransform(rotation=in_hand_knife_orientation, translation=np.array([0.64, 0.05, args.veg_food_height]),
        from_frame='franka_tool', to_frame='world') 

        fa.goto_pose(in_hand_pose, duration=5, use_impedance=False)

        in_hand_cut_move_forward = RigidTransform(rotation=in_hand_knife_orientation, translation=np.array([0.64, 0.1, args.veg_food_height]),
        from_frame='franka_tool', to_frame='world') 
        fa.goto_pose(in_hand_cut_move_forward, duration=3, use_impedance=False)
        
    move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -args.veg_food_height]),
    from_frame='world', to_frame='world')   

    fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 4.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
    
    current_ht = 0.03 #fa.get_pose().translation[2]
    dmp_num = 0
    while current_ht > 0.023: #0.028: # 0.025 is cutting board ht    
        #Save Audio, Realsense Camera, Robot Positions, and Forces
        audio_p = save_audio(dir_path, 'cutting_audio_'+str(dmp_num)+'.wav')
        knife_realsense_p = save_realsense(dir_path, 'knife_hand_realsense_'+str(dmp_num), '/knife_camera', use_depth=False)
        tong_realsense_p = save_realsense(dir_path, 'tong_hand_realsense_'+str(dmp_num), '/tong_camera')
        side_realsense_p = save_realsense(dir_path, 'side_hand_realsense_'+str(dmp_num), '/side_camera')

        fa.execute_pose_dmp(position_dmp_info, position_only=True, ee_frame=True, duration=6, initial_sensor_values=[1.0,1.0,1.0,1.0,1.0,0.0], use_impedance=False, block=False)

        (robot_positions, robot_forces) = get_robot_positions_and_forces(fa, 8)
        audio_p.terminate()
        knife_realsense_p.terminate()
        tong_realsense_p.terminate()
        side_realsense_p.terminate()

        np.savez(dir_path+'trial_info_'+str(dmp_num)+'.npz', robot_positions=robot_positions, robot_forces=robot_forces)

        current_ht = fa.get_pose().translation[2]
        dmp_num += 1

    if args.slice_num > 0 and args.slice_num < 9:

        fa.goto_pose_delta(lift_up_slightly_delta_z_position, duration=2, use_impedance=False)

        move_slice = RigidTransform(translation=np.array([0.0, -0.05, 0.0]),
        from_frame='world', to_frame='world')   

        fa.goto_pose_delta(move_slice, duration=5, use_impedance=False)

        fa.goto_pose_delta(lift_up_veg_food_height, duration=3, use_impedance=False)

    elif args.slice_num == 9 or args.slice_num == 10:

        lift_up_veg_food_height = RigidTransform(translation=np.array([0.0, 0.0, args.veg_food_height]),
        from_frame='world', to_frame='world')  
        fa.goto_pose_delta(lift_up_veg_food_height, duration=3, use_impedance=False)

    fa.goto_joints(reset_joint_positions)

    # Capture Overhead Azure Kinect Depth and RGB Images
    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)

    cropped_azure_kinect_rgb_image = azure_kinect_rgb_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]
    cropped_azure_kinect_depth_image = azure_kinect_depth_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]


    # Capture Realsense Depth and RGB Images from Side and Tong Realsense
    side_realsense_rgb_image = get_realsense_rgb_image(cv_bridge, topic='/side_camera/color/image_raw')
    side_realsense_depth_image = get_realsense_depth_image(cv_bridge, topic='/side_camera/depth/image_rect_raw')
    tong_realsense_rgb_image = get_realsense_rgb_image(cv_bridge, topic='/tong_camera/color/image_raw')
    tong_realsense_depth_image = get_realsense_depth_image(cv_bridge, topic='/tong_camera/depth/image_rect_raw')
    
    # Saving images
    cv2.imwrite(dir_path + 'ending_azure_kinect_rgb_image.png', cropped_azure_kinect_rgb_image)
    np.save(dir_path + 'ending_azure_kinect_depth_image.npy', cropped_azure_kinect_depth_image)
    cv2.imwrite(dir_path + 'ending_side_realsense_rgb_image.png', side_realsense_rgb_image)
    np.save(dir_path + 'ending_side_realsense_depth_image.npy', side_realsense_depth_image)
    cv2.imwrite(dir_path + 'ending_tong_realsense_rgb_image.png', tong_realsense_rgb_image)
    np.save(dir_path + 'ending_tong_realsense_depth_image.npy', tong_realsense_depth_image)

    
    '''
    Order of cuts:
    -cut 1: straight back and forth, thickness = 40mm 
    -cut 2: straight back and forth, thickness = 5mm 
    -cut 3: straight back and forth, thickness = 10mm 
    -cut 4: straight back and forth, thickness = 3mm
    -cut 5: straight back and forth, thickness = 20mm
    -cut 6: 30 deg L angled slice, thickness = 0mm *NOTE: first start with a flat side and cut 30 from there
    -cut 7: 30 deg L angled slice, thickness = 10mm  *NOTE: This didnt work. too hard to grasp angled cut - not enough space. take previous piece and cut 30 again in same direction   
    -cut 8: 30 deg R angled slice, thickness = 10/30mm (cut one sideways L and then cut same slice on the right)
    -cut 9: in hand cut (90 deg), mid-cut 5
    -cut 10: in hand cut (90 deg), mid-cut 9
    '''

       

    # angled_right_cut_position = RigidTransform(rotation=np.array([
    # [ 0.866,  0.0,   -0.5], 
    # [ 0.0, 1.0,  0.0],
    # [ 0.5,   0.0, 0.866]
    #     ]), translation=np.array([0.0, 0, 0.0]),
    # from_frame='franka_tool', to_frame='franka_tool')

    # angled_left_cut_position = RigidTransform(rotation=np.array([
    # [ 0.866,  0.0,   0.5], move_back
    # [ 0.0, 1.0,  0.0],
    # [ -0.5,   0.0, 0.866]
    #     ]), translation=np.array([0.0, 0, 0.0]),
    # from_frame='franka_tool', to_frame='franka_tool')

    # fa.goto_pose_delta(angled_cut_position, duration=5, use_impedance=False)

    # print(fa.get_pose())

    # import pdb; pdb.set_trace()
    
    # fa.execute_pose_dmp(position_dmp_info, position_only=True, ee_frame=True, duration=6)