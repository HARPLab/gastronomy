#!/usr/bin/env python

import rospy
from frankapy import FrankaArm
import numpy as np
import math
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics

from darknet_ros_msgs.msg import BoundingBoxes

from magnetic_stickers.msg import MagneticData
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from franka_action_lib.msg import RobotState

AZURE_KINECT_CALIB_DATA = \
    '/path/to/azure_kinect_calibration/calib/azure_kinect_overhead/ir_intrinsics_azure_kinect.intr'
AZURE_KINECT_EXTRINSICS = \
    '/path/to/azure_kinect_calibration/calib/azure_kinect_overhead/kinect2_overhead_to_world.tf'

class MagneticCalibration:
    def __init__(self):
        
        self.magnetic_data_sub = rospy.Subscriber('/magnetic_stickers/magnetic_data', MagneticData, self.magnetic_data_callback)
        self.robot_state_sub = rospy.Subscriber('/robot_state_publisher_node_1/robot_state', RobotState, self.robot_state_callback)

        self.magnetic_data_index = 0
        self.robot_state_index = 0
        self.magnetic_data_frequency = 40 # 40hz
        self.robot_state_frequency = 100 # 100hz
        self.max_data_time = 10 # 10 seconds
        self.num_magnetic_data_samples = self.magnetic_data_frequency * self.max_data_time
        self.num_robot_state_samples = self.robot_state_frequency * self.max_data_time
        self.magnetic_data = np.zeros((self.num_magnetic_data_samples,9))
        self.robot_state_data = np.zeros((self.num_robot_state_samples,4))

        self.recording_data = False

        self.affine = np.zeros((4,4))
        self.offsets = np.zeros((1,8))
        self.scales = np.zeros((1,8))
        
    def magnetic_data_callback(self,data):

        if(self.recording_data):
            self.magnetic_data[self.magnetic_data_index,0] = data.ref_x
            self.magnetic_data[self.magnetic_data_index,1] = data.ref_y
            self.magnetic_data[self.magnetic_data_index,2] = data.ref_z
            self.magnetic_data[self.magnetic_data_index,3] = data.ref_t
            self.magnetic_data[self.magnetic_data_index,4] = data.x
            self.magnetic_data[self.magnetic_data_index,5] = data.y
            self.magnetic_data[self.magnetic_data_index,6] = data.z
            self.magnetic_data[self.magnetic_data_index,7] = data.t
            self.magnetic_data[self.magnetic_data_index,8] = data.header.stamp.to_sec()

            self.magnetic_data_index += 1
            self.magnetic_data_index = self.magnetic_data_index % self.num_magnetic_data_samples

    def robot_state_callback(self,data):

        if(self.recording_data):
            self.robot_state_data[self.robot_state_index,0] = data.O_T_EE[12]
            self.robot_state_data[self.robot_state_index,1] = data.O_T_EE[13]
            self.robot_state_data[self.robot_state_index,2] = data.O_T_EE[14]
            self.robot_state_data[self.robot_state_index,3] = data.header.stamp.to_sec()

            self.robot_state_index += 1
            self.robot_state_index = self.robot_state_index % self.num_robot_state_samples

    def start_recording_data(self):
        self.magnetic_data = np.zeros((self.num_magnetic_data_samples,9))
        self.robot_state_data = np.zeros((self.num_robot_state_samples,4))
        self.magnetic_data_index = 0
        self.robot_state_index = 0
        self.recording_data = True

    def stop_recording_data(self):
        self.recording_data = False


    def get_max_location(self):
        data_mask = self.magnetic_data[:,8] == 0
        my_data = self.magnetic_data[~data_mask,:]

        processed_data = self.process_data(my_data) # shape of processed_data is m x 4

        print(processed_data)

        max_magnetic_data_idx = np.argmax(processed_data[:,2])
        max_magnetic_data = processed_data[max_magnetic_data_idx,:]
        print("max magnetic data: " + str(max_magnetic_data))
        max_magnetic_data_stamp = max_magnetic_data[3]

        closest_robot_state_idx = np.argmin(abs(self.robot_state_data[:,3] - max_magnetic_data_stamp))
        closest_robot_state = self.robot_state_data[closest_robot_state_idx,:]
        max_location = closest_robot_state[:3]
        print(closest_robot_state)

        return max_location


    def process_data(self, raw_data):

        processed_data = np.zeros((raw_data.shape[0],4))

        for i in range(raw_data.shape[0]):
            calibrated = np.multiply(self.scales, (raw_data[i,:8]-self.offsets))
            calibrated[3] = 1
            calibrated[7] = 1
            transform_ref = calibrated[0:4].dot(self.affine)

            processed_data[i,:3] = calibrated[4:7] - transform_ref[0:3] # raw_data[i,4:7] 
            processed_data[i, 3] = raw_data[i,8]

        return processed_data

    def get_magnetic_data(self):
        data1 = self.magnetic_data[self.magnetic_data_index:,:]
        data2 = self.magnetic_data[:self.magnetic_data_index,:]
        data = np.concatenate((data1, data2), axis=0)
        return data

    def get_last_magnetic_data(self):
        last_data = self.magnetic_data[self.magnetic_data_index-1,:]
        
        return last_data

    def load_calibration_file(self, filename):

        calibration = np.load(filename)
        self.affine=calibration['affine']
        self.offsets=calibration['offsets']
        self.scales=calibration['scales']

    def get_processed_data(self):
        current_magnetic_data = self.get_last_magnetic_data()

        calibrated = np.multiply(self.scales, (current_magnetic_data[:8]-self.offsets))
        calibrated[3] = 1
        calibrated[7] = 1
        transform_ref = calibrated[0:4].dot(self.affine)

        magnetic_data = calibrated[4:7]-transform_ref[0:3]

        return magnetic_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_CALIB_DATA)
    parser.add_argument('--transform_file_path', type=str, default=AZURE_KINECT_EXTRINSICS)
    parser.add_argument('--object', type=str, default=AZURE_KINECT_EXTRINSICS)
    args = parser.parse_args()

    # rospy.init_node("Test azure kinect calibration")
    print('Starting robot')
    fa = FrankaArm()

    cv_bridge = CvBridge()
    ir_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)

    kinect2_overhead_to_world_transform = RigidTransform.load(args.transform_file_path)


    print('Opening Grippers')
    fa.open_gripper()

    print('Reset with pose')
    fa.reset_pose()

    print('Reset with joints')
    fa.reset_joints()

    depth_image_msg = rospy.wait_for_message('/depth_to_rgb/image_raw', Image)
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)
    

    object_max_probability = 0.0

    while object_max_probability < 0.3:
        bounding_boxes_msg = rospy.wait_for_message('/darknet_ros/bounding_boxes', BoundingBoxes)
        
        for bounding_box in bounding_boxes_msg.bounding_boxes:
            if bounding_box.Class == args.object:
                if(bounding_box.probability > object_max_probability):
                    object_max_probability = bounding_box.probability
                    object_xmin = bounding_box.xmin
                    object_ymin = bounding_box.ymin
                    object_xmax = bounding_box.xmax
                    object_ymax = bounding_box.ymax

                    if ((object_ymax - object_ymin) < (object_xmax - object_xmin)):
                        width_direction = "y"
                        object_width = object_ymax - object_ymin
                        object_height = object_xmax - object_xmin
                    else:
                        width_direction = "x"
                        object_width = object_xmax - object_xmin
                        object_height = object_ymax - object_ymin


    
    object_xcenter = int((object_xmin + object_xmax) / 2)
    object_ycenter = int((object_ymin + object_ymax) / 2)

    print(object_xcenter)
    print(object_ycenter)
    print(cv_image.shape)

    object_center = Point(np.array([object_xcenter, object_ycenter]), 'kinect2_overhead')
    object_depth = cv_image[object_ycenter, object_xcenter]
    print("x, y, z: ({:.4f}, {:.4f}, {:.4f})".format(
        object_xcenter, object_ycenter, object_depth))
    object_center_point_in_world = kinect2_overhead_to_world_transform * ir_intrinsics.deproject_pixel(object_depth, object_center)
    
    print(object_center_point_in_world)

    constant_z = 0.11

    desired_pose = RigidTransform(
        rotation=np.array([[0, -1,  0],
                           [0,  0,  1],
                           [-1, 0,  0]]), 
        translation=np.array([object_center_point_in_world.x, object_center_point_in_world.y, constant_z]), 
        from_frame='franka_tool')
    fa.goto_pose_with_cartesian_control(desired_pose, 10.0)


    magnetic_calibration = MagneticCalibration()

    filename = '../calibration/2020-01-14 11-08 E1 2X Board Calibration.npz'
    magnetic_calibration.load_calibration_file(filename)

    magnetic_calibration.start_recording_data()
    desired_object_pose = RigidTransform(
        rotation=np.array([[0, -1,  0],
                           [0,  0,  1],
                           [-1, 0,  0]]), 
        translation=np.array([object_center_point_in_world.x, object_center_point_in_world.y - 0.01, constant_z]),
         from_frame='franka_tool')
    fa.goto_pose_with_cartesian_control(desired_object_pose, 10.0)
    magnetic_calibration.stop_recording_data()
    max_location = magnetic_calibration.get_max_location()


    current_position = fa.get_pose().position
    current_data = magnetic_calibration.get_processed_data()
    # print("My current data is: " + str(current_data[0]) + ", " + str(current_data[1]) + ", " + str(current_data[2]))
    # print("My current position is: " + str(current_position[0]) + ", " + str(current_position[1]) + ", " + str(current_position[2]))
    # print("My next position is: " + str(max_location[0]) + ", " + str(max_location[1]) + ", " + str(max_location[2]))
    # input("Press enter to continue to command")

    # franka_arm.move -relative dist m in x and y
    next_position = RigidTransform(rotation=np.array([
                                            [0, -1,  0],
                                            [0,  0,  1],
                                            [-1, 0,  0]
                                        ]), translation=np.array(max_location), # add x and z later
                                    from_frame='franka_tool', to_frame='world')

    fa.goto_pose_with_cartesian_control(next_position, 3)

    #fa.close_gripper()

    fa.goto_gripper(0.0, 0.04, 1)

    relative_z_dist = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0, 0.0, 0.1]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    fa.goto_pose_delta_with_cartesian_control(relative_z_dist, 5)
