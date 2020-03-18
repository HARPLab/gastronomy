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
        #self.robot_state_sub = rospy.Subscriber('/robot_state_publisher_node_1/gripper_state', GripperState, self.gripper_state_callback)

        self.magnetic_data_index = 0
        self.robot_state_index = 0
        self.gripper_state_index = 0
        
        self.magnetic_data_frequency = 100 # 40hz
        self.robot_state_frequency = 100 # 100hz
        self.gripper_state_frequency = 100
        
        self.max_data_time = 100 # 10 seconds
        self.num_magnetic_data_samples = self.magnetic_data_frequency * self.max_data_time
        self.num_robot_state_samples = self.robot_state_frequency * self.max_data_time
        self.num_gripper_state_samples = self.gripper_state_frequency * self.max_data_time
        
        self.magnetic_data = np.zeros((self.num_magnetic_data_samples,9))
        self.robot_state_data = np.zeros((self.num_robot_state_samples,4))
        self.gripper_state_data = np.zeros((self.num_gripper_state_samples,4))
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

            self.gripper_state_data[self.gripper_state_index,0] = data.gripper_width
            self.gripper_state_data[self.gripper_state_index,1] = data.gripper_max_width
            self.gripper_state_data[self.gripper_state_index,2] = np.float64(data.gripper_is_grasped)
            self.gripper_state_data[self.gripper_state_index,3] = data.header.stamp.to_sec()

            # if there is a sign change greater than noise, set contact flag
            self.gripper_state_index += 1
            self.gripper_state_index = self.gripper_state_index % self.num_gripper_state_samples

    # def gripper_state_callback(self,data):

    #     if(self.recording_data):
    #         self.gripper_state_data[self.gripper_state_index,0] = data.width[12]
    #         self.gripper_state_data[self.gripper_state_index,1] = data.max_width[13]
    #         self.gripper_state_data[self.gripper_state_index,2] = data.is_grasped[14] #/.temperature, .time
    #         self.gripper_state_data[self.gripper_state_index,3] = data.header.stamp.to_sec()

    #         self.gripper_state_data += 1
    #         self.gripper_state_data = self.gripper_state_index % self.num_gripper_state_samples

    def start_recording_data(self):
        self.magnetic_data = np.zeros((self.num_magnetic_data_samples,9))
        self.robot_state_data = np.zeros((self.num_robot_state_samples,4))
        self.gripper_state_data = np.zeros((self.num_gripper_state_samples,4))
        
        self.magnetic_data_index = 0
        self.robot_state_index = 0
        self.gripper_state_index = 0
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

    def get_current_gripper_width(self):

        return self.gripper_state_data[self.gripper_state_index-1,0]

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

    def plotGripperData(self, gripper_data, figure_title):
        
        gripper_data_mask = gripper_data[:,3] == 0
        data = gripper_data[~gripper_data_mask,:]

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

    def saveData(self, filename, min_contact, min_gripper_width):
        magnetic_data_mask = self.magnetic_data[:,8] == 0
        my_magnetic_data = self.magnetic_data[~magnetic_data_mask,:]

        robot_state_data_mask = self.robot_state_data[:,3] == 0
        my_robot_state_data = self.robot_state_data[~robot_state_data_mask,:]


        gripper_state_data_mask = self.gripper_state_data[:,3] == 0
        my_gripper_state_data = self.gripper_state_data[~gripper_state_data_mask,:]

        #np.savez(filename, magnetic_data=my_magnetic_data, robot_state_data=my_robot_state_data)
        np.savez(filename, magnetic_data=my_magnetic_data, robot_state_data=my_robot_state_data, gripper_state_data=my_gripper_state_data, min_contact=min_contact, min_gripper_width=min_gripper_width)
        

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

    # def close_gripper_magnetic_feedback(self, franka, step_size):

    #     # Manually close gripper by a small step until magnetic signal changes direction
    #     initial_signal = self.get_last_magnetic_data() #1x9 array
    #     initial_filtered_signal = self.get_processed_data() #1x3 array [Bx By Bz]
    #     current_width = self.gripper_state_data[1] # get current position of gripper
    #     next_width = current_width - step_size # units in m?
    #     franka.goto_gripper(next_width , speed=0.04, force=1)
    #     end_filtered_signal = self.get_processed_data()
    #     if((end_filtered_signal-initial_filtered_signal)<0):
    #         print('Stopped by Magnetic Feedback')
    #     else:
    #         #continue looping
    #     # motion of gripper might be too fast to get enough data. check magnetic-Data_index before and after

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()


    fa = FrankaArm()

    fa.open_gripper()

    # fa.reset_pose()
    fa.reset_joints(10)
    
    initial_magnet_position1 = RigidTransform(rotation=np.array([
                                            [0, -1,  0],
                                            [0,  0,  1],
                                            [-1, 0,  0]
                                        ]), translation=np.array([0.38592997, 0.10820438, 0.08264024]),
                                    from_frame='franka_tool', to_frame='world')

    initial_magnet_position2 = RigidTransform(rotation=np.array([
                                            [0, -1,  0],
                                            [0,  0,  1],
                                            [-1, 0,  0]
                                        ]), translation=np.array([0.38592997, 0.20820438, 0.08264024]),
                                    from_frame='franka_tool', to_frame='world')

    squeeze_position1 = RigidTransform(rotation=np.array([
                                            [0, 1,  0],
                                            [0,  0,  1],
                                            [1, 0,  0]
                                        ]), translation=np.array([0.54900978, 0.20820438, 0.20654183]),
                                    from_frame='franka_tool', to_frame='world')

    squeeze_position2 = RigidTransform(rotation=np.array([
                                            [0, 1,  0],
                                            [0,  0,  1],
                                            [1, 0,  0]
                                        ]), translation=np.array([0.54900978, 0.20820438, 0.15654183]),
                                    from_frame='franka_tool', to_frame='world')

    relative_pos_dist_z = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.0, 0.1]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_pos_dist_y = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.1, 0.0]),
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
                                            ]), translation=np.array([0.0, -0.1, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_neg_dist_z = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.0, 0.0, -0.1]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_pos_dist_x = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0.075, 0.0, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    relative_neg_dist_x = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([-0.03, 0.0, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')


    fa.goto_pose_with_cartesian_control(initial_magnet_position1)

    fa.goto_pose_with_cartesian_control(initial_magnet_position2)

    magnetic_calibration = MagneticCalibration()

    filename = '../../scripts/2020-01-14 11-08 E1 2X Board Calibration.npz'
    magnetic_calibration.load_calibration_file(filename)
   
    # Close the gripper and save corresponding robot state, gripper state, and magnetic state data
    magnetic_calibration.start_recording_data()
    time.sleep(1)
    # fa.goto_pose_delta_with_cartesian_control(relative_x_dist, 10)
    # max width is about 0.080 m
    
    gripper_step_size = 0.002 # in meters
    num_samples = 30
    noise_level = 13 # uT
    force_threshold = 1.05

    GRIPPER_CONTACT = False

    min_contact = None
    min_gripper_width = None
    global_min = None
    current_force_estimate = 1

    while(GRIPPER_CONTACT == False):

        current_width = magnetic_calibration.get_current_gripper_width()
        print(current_width)
        fa.goto_gripper(current_width-gripper_step_size)

        #magnetic_calibration.get_last_magnetic_data()
        mag_data = magnetic_calibration.get_previous_magnetic_samples(num_samples) # grab last ten samples 

        xyz_mag_data = mag_data[:,6]
        print(xyz_mag_data)
        #slope = np.diff(xyz_mag_data, axis=0)
        #print(slope)
        # asign = np.sign(slope)
        # signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        # print(signchange)

        # first make sure the signal changes are large enough to not just be noise jitter
        # then see if there is a signal change in slope - this signifies a contact. 
        #if(np.any(abs(np.diff(mag_data)>noise_level))):
        #if(abs(np.sum(np.sign(np.diff(mag_data)))<numSamples) & abs(np.diff(mag_data))>noise_level):
        
        if global_min is None:
            global_min = np.min(xyz_mag_data, axis=0)

        elif global_min > np.min(xyz_mag_data, axis=0):
            global_min = np.min(xyz_mag_data, axis=0)

        if xyz_mag_data[-1] - global_min > noise_level:
            GRIPPER_CONTACT = True
            min_gripper_width = current_width-2*gripper_step_size
            print(min_gripper_width)
            print("Min Magnetic Values: ")
            print(min_contact)

        gripper_step_size -= 0.00003

        # if (np.any(np.logical_and((abs(slope)>noise_level), (slope > 0)))):
        #     # if last value is negative, contact. if last value is positive, release.
        #     # save max value so we can compare current value and strength of grip
        #     GRIPPER_CONTACT = True
        #     min_contact = np.min(xyz_mag_data, axis=0)
        #     min_gripper_width = current_width-gripper_step_size
        #     print(min_gripper_width)
        #     print("Min Magnetic Values: ")
        #     print(min_contact)
        #input("Press enter to continue to next step")

        #magnetic_calibration.close_gripper_magnetic_feedback()

    # current_width = min_gripper_width
    # num_samples = 100

    # while(current_force_estimate < force_threshold):
    #     current_width -= gripper_step_size
    #     print(current_width)
    #     fa.goto_gripper(current_width)

    #     mag_data = magnetic_calibration.get_previous_magnetic_samples(num_samples) # grab last ten samples 

    #     z_mag_data = np.max(mag_data[:,6])
    #     print(z_mag_data)

    #     current_force_estimate = global_min / z_mag_data 
    #     print(current_force_estimate)

    fa.goto_pose_delta_with_cartesian_control(relative_pos_dist_z)

    current_joints = fa.get_joints()
    current_joints[6] -= math.pi

    fa.goto_joints(list(current_joints))

    current_position = fa.get_pose()

    fa.goto_pose_with_cartesian_control(squeeze_position1)

    fa.goto_pose_with_cartesian_control(squeeze_position2)

    for i in range(3):
        for j in range(i+1):
            fa.goto_gripper(0.0, force = 20)

            fa.goto_gripper(min_gripper_width)
        if i < 2:
            fa.goto_pose_delta_with_cartesian_control(relative_pos_dist_x)

    fa.goto_pose_with_cartesian_control(current_position,5)

    current_joints = fa.get_joints()
    current_joints[6] += math.pi

    fa.goto_joints(list(current_joints))

    fa.goto_pose_delta_with_cartesian_control(relative_neg_dist_z)

    fa.open_gripper()

    fa.goto_pose_with_cartesian_control(initial_magnet_position1)

    #fa.goto_pose_delta_with_cartesian_control(relative_neg_dist_z)

    
    magnetic_calibration.stop_recording_data()

    #fa.open_gripper()

    print(magnetic_calibration.magnetic_data[0,:])

    if args.filename is not None:
        magnetic_calibration.saveData(args.filename, min_contact, min_gripper_width)

    magnetic_calibration.plotData(magnetic_calibration.magnetic_data, 'Raw Data Over Time')
    magnetic_calibration.plotGripperData(magnetic_calibration.gripper_state_data, 'Raw Data Over Time')
    magnetic_calibration.plotGripperData(magnetic_calibration.robot_state_data, 'Raw Data Over Time')


if __name__ == '__main__':
    run_main()