#!/usr/bin/env python

import rospy
from magnetic_stickers.msg import MagneticData
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from frankapy import FrankaArm
import math
import time
import argparse
from autolab_core import RigidTransform
from franka_action_lib.msg import RobotState

class MagneticCalibration:
    def __init__(self):
        
        self.magnetic_data_sub = rospy.Subscriber('/magnetic_stickers/magnetic_data', MagneticData, self.magnetic_data_callback)
        self.robot_state_sub = rospy.Subscriber('/robot_state_publisher_node_1/robot_state', RobotState, self.robot_state_callback)

        self.magnetic_data_index = 0
        self.robot_state_index = 0
        self.force_state_index = 0
        
        self.magnetic_data_frequency = 100 # 40hz
        self.robot_state_frequency = 100 # 100hz
        self.force_state_frequency = 100
        
        self.max_data_time = 60 # 10 seconds
        self.num_magnetic_data_samples = self.magnetic_data_frequency * self.max_data_time
        self.num_robot_state_samples = self.robot_state_frequency * self.max_data_time
        self.num_force_state_samples = self.force_state_frequency * self.max_data_time
        
        self.magnetic_data = np.zeros((self.num_magnetic_data_samples,9))
        self.robot_state_data = np.zeros((self.num_robot_state_samples,4))
        self.force_state_data = np.zeros((self.num_force_state_samples,4))
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

            self.force_state_data[self.force_state_index,0] = data.O_F_ext_hat_K[0]
            self.force_state_data[self.force_state_index,1] = data.O_F_ext_hat_K[1]
            self.force_state_data[self.force_state_index,2] = data.O_F_ext_hat_K[2]
            self.force_state_data[self.force_state_index,3] = data.header.stamp.to_sec()

            # if there is a sign change greater than noise, set contact flag
            self.force_state_index += 1
            self.force_state_index = self.force_state_index % self.num_force_state_samples

    def start_recording_data(self):
        self.magnetic_data = np.zeros((self.num_magnetic_data_samples,9))
        self.robot_state_data = np.zeros((self.num_robot_state_samples,4))
        self.force_state_data = np.zeros((self.num_force_state_samples,4))
        
        self.magnetic_data_index = 0
        self.robot_state_index = 0
        self.force_state_index = 0
        self.recording_data = True

    def stop_recording_data(self):
        self.recording_data = False

    def get_max_location(self):
        data_mask = self.magnetic_data[:,8] == 0
        my_data = self.magnetic_data[~data_mask,:]

        processed_data = self.process_data(my_data) # shape of processed_data is m x 4

        print(processed_data)

        #max_magnetic_data_idx = np.argmax(processed_data[:,2])
        max_magnetic_data_idx = np.argmax(processed_data[:,1])
        max_magnetic_data = processed_data[max_magnetic_data_idx,:]
        print("max magnetic data: " + str(max_magnetic_data))
        max_magnetic_data_stamp = max_magnetic_data[3]

        closest_robot_state_idx = np.argmin(abs(self.robot_state_data[:,3] - max_magnetic_data_stamp))
        closest_robot_state = self.robot_state_data[closest_robot_state_idx,:]
        max_location = closest_robot_state[:3]

        return max_location

    def process_data(self, raw_data):

        processed_data = np.zeros((raw_data.shape[0],4))

        for i in range(raw_data.shape[0]):
            calibrated = np.multiply(self.scales, (raw_data[i,:8]-self.offsets))
            calibrated[3] = 1
            calibrated[7] = 1
            transform_ref = calibrated[0:4].dot(self.affine)

            processed_data[i,:3] = calibrated[4:7]-transform_ref[0:3]
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

    def get_previous_magnetic_samples(self, numSamples):

        data = self.magnetic_data[self.magnetic_data_index-numSamples:self.magnetic_data_index-1,:]

        return data

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

    def get_processed_samples(self, numSamples):

        all_data = np.zeros((numSamples,3))
        for i in range(numSamples):
            all_data[i] = self.get_processed_data()
            time.sleep(0.03)
        return all_data

    def plotData(self, magnetic_data, figure_title):
        
        magnetic_data_mask = magnetic_data[:,8] == 0
        data = magnetic_data[~magnetic_data_mask,:]

        plt.figure(figure_title)
        plt.scatter(data[:,8], data[:,4], c="red")
        plt.scatter(data[:,8], data[:,5], c="green")
        plt.scatter(data[:,8], data[:,6], c="blue")
        plt.title('Bfield Over Time')
        plt.show()

    def plotDataVsRobotState(self, magnetic_data, robot_data, figure_title):
        
        magnetic_data_mask = magnetic_data[:,8] == 0
        good_magnet_data = magnetic_data[~magnetic_data_mask,:]

        robot_data_mask = robot_data[:,3] == 0
        good_robot_data = robot_data[~robot_data_mask,:]

        num_data = np.min([good_magnet_data.shape[0], good_robot_data.shape[0]])

        plt.figure(figure_title)
        plt.scatter(good_robot_data[:num_data,1], good_magnet_data[:num_data,5], c="red")
        # plt.scatter(data[:,8], data[:,5], c="green")
        # plt.scatter(data[:,8], data[:,6], c="blue")
        plt.title('Bfield Over Location')
        plt.show()

    def plotForceData(self, force_data, figure_title):
        
        force_data_mask = force_data[:,3] == 0
        data = force_data[~force_data_mask,:]

        plt.figure(figure_title)
        plt.scatter(data[:,3], data[:,0], c="red")
        plt.scatter(data[:,3], data[:,1], c="green")
        plt.scatter(data[:,3], data[:,2], c="blue")
        plt.show()

    def plotRobotStateData(self, robot_data, figure_title):
        
        robot_data_mask = robot_data[:,3] == 0
        data = robot_data[~robot_data_mask,:]

        plt.figure(figure_title)
        plt.scatter(data[:,3], data[:,0], c="red")
        plt.scatter(data[:,3], data[:,1], c="green")
        plt.scatter(data[:,3], data[:,2], c="blue")
        plt.show()



    def collectAndSaveSamples(self, num_samples, filename):

        self.magnetic_data = np.zeros((num_samples,8))
        self.num_samples = num_samples

        # blocking loop waiting to collect all the samples
        while(self.num_data < num_samples):
            time.sleep(0.01)

        np.savez(filename, data=self.magnetic_data)

    def saveData(self, filename):
        magnetic_data_mask = self.magnetic_data[:,8] == 0
        my_magnetic_data = self.magnetic_data[~magnetic_data_mask,:]

        robot_state_data_mask = self.robot_state_data[:,3] == 0
        my_robot_state_data = self.robot_state_data[~robot_state_data_mask,:]

        force_state_data_mask = self.force_state_data[:,3] == 0
        my_force_state_data = self.force_state_data[~force_state_data_mask,:]

        #np.savez(filename, magnetic_data=my_magnetic_data, robot_state_data=my_robot_state_data)
        np.savez(filename, magnetic_data=my_magnetic_data, robot_state_data=my_robot_state_data, force_state_data=my_force_state_data)
        

    def getdBzdx(self, xdata):
        eps = 0.0001
        dBz = xdata[-1,5]-xdata[-1,11]
        dx = xdata[-1,0]-xdata[-1,6]
        if(abs(dx)<eps):
            return dBz/eps
        else:
            return dBz/(dx * 1000)

    def getdBzdy(self, ydata):
        eps = 0.0001
        dBz = ydata[-1,5]-ydata[-1,11]
        dy = ydata[-1,1]-ydata[-1,7]
        if(abs(dy)<eps):
            return dBz/eps
        else:
            return dBz/(dy * 1000) 

    def getdBzdz(self, zdata):
        eps = 0.0001
        dBz = zdata[-1,5]-zdata[-1,11]
        dz = zdata[-1,2]-zdata[-1,8]
        if(abs(dz)<eps):
            return dBz/eps
        else:
            return dBz/(dz * 1000) 

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()


    fa = FrankaArm(async_cmds=True)

    fa.close_gripper()

    while fa.is_skill_done() == False:
        time.sleep(1)

    # fa.reset_pose()
    # fa.reset_joints()

    drawer_2_z_height = 0.12919677
    drawer_3_z_height = 0.08264024

    drawer_height = drawer_3_z_height
    
    initial_magnet_position1 = RigidTransform(rotation=np.array([
                                            [0, -1,  0],
                                            [0,  0,  1],
                                            [-1, 0,  0]
                                        ]), translation=np.array([0.40926815, 0.20738414, drawer_height]),
                                    from_frame='franka_tool', to_frame='world')

    initial_magnet_position2 = RigidTransform(rotation=np.array([
                                            [0, -1,  0],
                                            [0,  0,  1],
                                            [-1, 0,  0]
                                        ]), translation=np.array([0.40926815, 0.23738414, drawer_height]),
                                    from_frame='franka_tool', to_frame='world')

    relative_pos_dist_z = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.0, 0.08]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_pos_dist_y = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.08, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_rotation_z = RigidTransform(rotation=np.array([
                                                [0,  1,  0],
                                                [-1, 0,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.0, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_neg_rotation_z = RigidTransform(rotation=np.array([
                                                [0,  -1,  0],
                                                [1, 0,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.0, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_neg_dist_y = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, -0.13, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_neg_dist_z = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.0, -0.08]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_pos_dist_x = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.03, 0.0, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_neg_dist_x = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([-0.03, 0.0, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')


    fa.goto_pose_with_cartesian_control(initial_magnet_position1)

    while fa.is_skill_done() == False:
        time.sleep(1)

    fa.goto_pose_with_cartesian_control(initial_magnet_position2)

    while fa.is_skill_done() == False:
        time.sleep(1)

    magnetic_calibration = MagneticCalibration()

    filename = '../calibration/2020-01-14 11-08 E1 2X Board Calibration.npz'
    magnetic_calibration.load_calibration_file(filename)
    
    num_samples = 10
    noise_level = 1 # uT

    #fa.goto_gripper(0.0, force=160)

    # while fa.is_skill_done() == False:
    #     time.sleep(1)

    magnetic_calibration.start_recording_data()
    time.sleep(0.5)

    greater_than_noise = False

    fa.goto_pose_delta_with_cartesian_control(relative_pos_dist_y,5,stop_on_contact_forces=[20,2,20,20,20,20])
    
    while fa.is_skill_done() == False:
        mag_data = magnetic_calibration.get_previous_magnetic_samples(num_samples)
        dif_mag_data = np.diff(mag_data[:,5])
        mean_change = np.mean(dif_mag_data)
        print(mean_change)
        
        if(abs(mean_change) > noise_level) and not greater_than_noise:
            greater_than_noise = True

        if greater_than_noise and abs(mean_change) < 1:
            print("stopped skill")
            fa.stop_skill()
        time.sleep(0.01)

    # while fa.is_skill_done() == False:
    #     time.sleep(1)

    fa.goto_pose_delta_with_cartesian_control(relative_neg_dist_y, 5)

    while fa.is_skill_done() == False:
        time.sleep(1)

    magnetic_calibration.stop_recording_data()

    #fa.open_gripper()

    while fa.is_skill_done() == False:
        time.sleep(1)

    print(magnetic_calibration.magnetic_data[0,:])

    if args.filename is not None:
        magnetic_calibration.saveData(args.filename)

    magnetic_calibration.plotDataVsRobotState(magnetic_calibration.magnetic_data, magnetic_calibration.robot_state_data, 'Raw Data Over Distance')
    magnetic_calibration.plotData(magnetic_calibration.magnetic_data, 'Raw Data Over Time')
    magnetic_calibration.plotForceData(magnetic_calibration.force_state_data, 'Raw Data Over Time')
    magnetic_calibration.plotRobotStateData(magnetic_calibration.robot_state_data, 'Raw Data Over Time')


if __name__ == '__main__':
    run_main()