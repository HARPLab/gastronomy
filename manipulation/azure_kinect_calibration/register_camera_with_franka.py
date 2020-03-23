"""
Adapted from autolab/perception package's register camera script:
Script to register sensors to a chessboard for the YuMi setup
Authors: Jeff Mahler and Brenton Chu
""" 
import argparse
import cv2
import IPython
import logging
import numpy as np
import os
import sys
import time
import traceback
import rospy

from mpl_toolkits.mplot3d import Axes3D

from autolab_core import Point, PointCloud, RigidTransform, YamlConfig
from perception import CameraChessboardRegistration, RgbdSensorFactory, CameraIntrinsics

# from visualization import Visualizer2D as vis2d
# from visualization import Visualizer3D as vis3d

from frankapy import FrankaArm

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Hello world")	

    # parse args
    parser = argparse.ArgumentParser(description='Register a camera to a robot')
    parser.add_argument('--config_filename', type=str, default='cfg/register_camera_azure_kinect_with_franka.yaml', help='filename of a YAML configuration for registration')
    parser.add_argument('--intrinsics_dir', type=str, default='calib/kinect2_overhead/ir_intrinsics_azure_kinect.intr')
    args = parser.parse_args()
    config_filename = args.config_filename
    config = YamlConfig(config_filename)
    rospy.init_node('register_camera', anonymous=True)
    
    # get known tf from chessboard to world
    T_cb_world = RigidTransform.load(config['chessboard_tf'])

    robot = FrankaArm()
    T_ee_world = robot.get_pose()

    # Get T_cb_world by using T_ee_world*T_cb_ee
    T_cb_ee = RigidTransform(rotation=np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]]),translation=np.array([0.02275, 0, -0.0732]), from_frame='cb', to_frame='franka_tool')
    # T_cb_ee = RigidTransform(rotation=np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]),
    #                          translation=np.array([0.02275, 0, -0.0732]), 
    #                          from_frame='cb', to_frame='franka_tool')

    print(T_cb_ee)
    print(T_ee_world)

    T_cb_world = T_ee_world * T_cb_ee

    # get camera sensor object
    for sensor_frame, sensor_data in config['sensors'].items():
        rospy.loginfo('Registering %s' %(sensor_frame))
        sensor_config = sensor_data['sensor_config']
        if 'registration_config' in sensor_data:
            registration_config = sensor_data['registration_config'].copy()
        else:
            registration_config = {}
        registration_config.update(config['chessboard_registration'])

	logging.info("here")
        
        # open sensor
        try:
            sensor_type = sensor_config['type']
            sensor_config['frame'] = sensor_frame
            rospy.loginfo('Creating sensor')
            sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)
            rospy.loginfo('Starting sensor')
            sensor.start()
            if args.intrinsics_dir:
                ir_intrinsics = CameraIntrinsics.load(args.intrinsics_dir)
            else:
                ir_intrinsics = sensor.ir_intrinsics
            rospy.loginfo('Sensor initialized')

            # ==== DEBUG: Visualize what the camera sees ====
            # rgb_im, depth_im, _ = sensor.frames()
            # img = rgb_im.raw_data.astype(np.uint8)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('hello_world', img_rgb)
            # cv2.waitKey(1000)

            # register
            reg_result = CameraChessboardRegistration.register(sensor, registration_config)
            T_camera_world = T_cb_world * reg_result.T_camera_cb

            rospy.loginfo('Final Result for sensor %s' %(sensor_frame))
            rospy.loginfo('Rotation: ')
            rospy.loginfo(T_camera_world.rotation)
            rospy.loginfo('Quaternion: ')
            rospy.loginfo(T_camera_world.quaternion)
            rospy.loginfo('Translation: ')
            rospy.loginfo(T_camera_world.translation)

        except Exception as e:
            rospy.logerr('Failed to register sensor {}'.format(sensor_frame))
            traceback.print_exc()
            continue

        # save tranformation arrays based on setup
        output_dir = os.path.join(config['calib_dir'], sensor_frame)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        rospy.loginfo('Saving to {}'.format(output_dir))
        pose_filename = os.path.join(output_dir, '%s_to_world.tf' %(sensor_frame))
        T_camera_world.save(pose_filename)
        if not args.intrinsics_dir:
            intr_filename = os.path.join(output_dir, '%s.intr' %(sensor_frame))
            ir_intrinsics.save(intr_filename)
        f = os.path.join(output_dir, 'corners_cb_%s.npy' %(sensor_frame))
        np.save(f, reg_result.cb_points_cam.data)
                
        # move the robot to the chessboard center for verification
        if config['use_robot']:  

            # find the rightmost and further cb point in world frame
            cb_points_world = T_camera_world * reg_result.cb_points_cam
            cb_point_data_world = cb_points_world.data
            dir_world = np.array([1.0, -1.0, 0])
            dir_world = dir_world / np.linalg.norm(dir_world)
            ip = dir_world.dot(cb_point_data_world)

            # open interface to robot
            logging.info('Setting up robot')
            robot = FrankaArm() # TODO(jacky): use Franka Arm
            robot.close_gripper()
            time.sleep(1)

            # choose target point #1
            target_ind = np.where(ip == np.max(ip))[0]
            target_pt_world = cb_points_world[int(target_ind[0])]
                
            # create robot pose relative to target point
            R_gripper_world = np.array([[1.0, 0, 0],
                                        [0, -1.0, 0],
                                        [0, 0, -1.0]])
            t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                        target_pt_world.y + config['gripper_offset_y'],
                                        target_pt_world.z + config['gripper_offset_z']])
            T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                             translation=t_gripper_world,
                                             from_frame='franka_tool',
                                             to_frame='world')
            logging.info('Moving robot to point x=%f, y=%f, z=%f' %(t_gripper_world[0], t_gripper_world[1], t_gripper_world[2]))

            T_lift = RigidTransform(translation=(0,0,0.05), from_frame='world', to_frame='world')
            T_gripper_world_lift = T_lift * T_gripper_world
            T_orig_gripper_world_lift = T_gripper_world_lift.copy()

            if config['vis_cb_corners']:
                _, depth_im, _ = sensor.frames()
                points_world = T_camera_world * ir_intrinsics.deproject(depth_im)
                vis3d.figure()
                vis3d.points(cb_points_world, color=(0,0,1), scale=0.005)
                vis3d.points(points_world, color=(0,1,0), subsample=10, random=True, scale=0.001)
                vis3d.pose(T_camera_world)
                vis3d.pose(T_gripper_world_lift)
                vis3d.pose(T_gripper_world)
                vis3d.pose(T_cb_world)
                vis3d.pose(RigidTransform())
                vis3d.table(dim=0.5, T_table_world=T_cb_world)
                vis3d.show()

            robot.goto_pose(T_gripper_world_lift, duration=5)
            robot.goto_pose(T_gripper_world, duration=3)
            
            # wait for human measurement
            yesno = input('Take measurement. Hit [ENTER] when done')
            robot.goto_pose(T_gripper_world_lift, duration=3)

            # choose target point 2
            target_ind = np.where(ip == np.min(ip))[0]
            target_pt_world = cb_points_world[int(target_ind[0])]
                
            # create robot pose relative to target point
            t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                        target_pt_world.y + config['gripper_offset_y'],
                                        target_pt_world.z + config['gripper_offset_z']])
            T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                             translation=t_gripper_world,
                                             from_frame='franka_tool',
                                             to_frame='world')
            logging.info('Moving robot to point x=%f, y=%f, z=%f' %(t_gripper_world[0], t_gripper_world[1], t_gripper_world[2]))
            
            T_lift = RigidTransform(translation=(0,0,0.05), from_frame='world', to_frame='world')
            T_gripper_world_lift = T_lift * T_gripper_world
            robot.goto_pose(T_gripper_world_lift, duration=3)
            robot.goto_pose(T_gripper_world, duration=3)
            
            # wait for human measurement
            yesno = input('Take measurement. Hit [ENTER] when done')
            robot.goto_pose(T_gripper_world_lift, duration=3)
            robot.goto_pose(T_orig_gripper_world_lift, duration=3)

            # choose target point 3
            dir_world = np.array([1.0, 1.0, 0])
            dir_world = dir_world / np.linalg.norm(dir_world)
            ip = dir_world.dot(cb_point_data_world)
            target_ind = np.where(ip == np.max(ip))[0]
            target_pt_world = cb_points_world[int(target_ind[0])]
                
            # create robot pose relative to target point
            t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                        target_pt_world.y + config['gripper_offset_y'],
                                        target_pt_world.z + config['gripper_offset_z']])
            T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                             translation=t_gripper_world,
                                             from_frame='franka_tool',
                                             to_frame='world')
            logging.info('Moving robot to point x=%f, y=%f, z=%f' %(t_gripper_world[0], t_gripper_world[1], t_gripper_world[2]))
            
            T_lift = RigidTransform(translation=(0,0,0.05), from_frame='world', to_frame='world')
            T_gripper_world_lift = T_lift * T_gripper_world
            robot.goto_pose(T_gripper_world_lift, duration=3)
            robot.goto_pose(T_gripper_world, duration=3)
            
            # # wait for human measurement
            yesno = input('Take measurement. Hit [ENTER] when done')
            robot.goto_pose(T_gripper_world_lift, duration=3)
            robot.goto_pose(T_orig_gripper_world_lift, duration=3)
            
            # stop robot
            robot.reset_joints()
                
        sensor.stop()
