#!/usr/bin/env python

from frankapy import FrankaArm
import numpy as np
import math
import rospy
import argparse
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from magnetic_stickers.msg import MagneticData
import matplotlib.pyplot as plt
from datetime import datetime
import time
from franka_action_lib.msg import RobotState
from scipy.optimize import curve_fit, fsolve
from mpl_toolkits.mplot3d import Axes3D

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
        
        self.max_data_time = 60 # 10 seconds
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

    def restart_recording_data(self):
        self.recording_data = True


    def get_max_location(self):
        data_mask = self.magnetic_data[:,8] == 0
        my_data = self.magnetic_data[~data_mask,:]

        processed_data = self.process_data(my_data) # shape of processed_data is m x 4

        print(processed_data)

        max_magnetic_data_idx = np.argmax(my_data[:,6])

        # max_magnetic_data_idx = np.argmax(processed_data[:,2])
        # max_magnetic_data_idx = np.argmax(processed_data[:,1])
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

    def saveData(self, filename, starting_location, vision_object_location, magnet_object_location, robot_final_location):
        magnetic_data_mask = self.magnetic_data[:,8] == 0
        my_magnetic_data = self.magnetic_data[~magnetic_data_mask,:]

        robot_state_data_mask = self.robot_state_data[:,3] == 0
        my_robot_state_data = self.robot_state_data[~robot_state_data_mask,:]


        gripper_state_data_mask = self.gripper_state_data[:,3] == 0
        my_gripper_state_data = self.gripper_state_data[~gripper_state_data_mask,:]

        #np.savez(filename, magnetic_data=my_magnetic_data, robot_state_data=my_robot_state_data)
        np.savez(filename, magnetic_data=my_magnetic_data, robot_state_data=my_robot_state_data, gripper_state_data=my_gripper_state_data, 
                 starting_location=starting_location, vision_object_location=vision_object_location, magnet_object_location=magnet_object_location, robot_final_location=robot_final_location)
      
    def getData(self):
        magnetic_data_mask = self.magnetic_data[:,8] == 0
        my_magnetic_data = self.magnetic_data[~magnetic_data_mask,:]

        robot_state_data_mask = self.robot_state_data[:,3] == 0
        my_robot_state_data = self.robot_state_data[~robot_state_data_mask,:]

        return (my_magnetic_data, my_robot_state_data)

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

# function for magnetic strength on centerline, solve for Br and offset
def centerline(x, offset, Br, br_offset):
    # distance in mm, B in uT
    R = 7.5 # 7.5mm radius
    T = 2 # 2 mm thickness
    x = x + offset
    return Br/2*(((x+T)/np.sqrt(R**2+(x+T)**2))-(x/np.sqrt(R**2+x**2))) + br_offset

# function for magnetic strength on centerline, solve for Br, R, T
# def centerline(x, offset, Br, R, T):
#     x = x + offset
#     return Br/2*((x+T/np.sqrt(R**2+(x+T)**2))-(x/np.sqrt(R**2+x**2)))

def func(x, a, b, c, d):
    return a * np.exp(-np.square((x-b)/(2*c))) + d

AZURE_KINECT_CALIB_DATA = \
    '/path/to/azure_kinect_calibration/calib/azure_kinect_left/ir_intrinsics.intr'
AZURE_KINECT_EXTRINSICS = \
    '/path/to/azure_kinect_calibration/calib/azure_kinect_left/kinect2_left_to_world.tf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_CALIB_DATA)
    parser.add_argument('--transform_file_path', type=str, default=AZURE_KINECT_EXTRINSICS)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    # rospy.init_node("Test azure kinect calibration")
    print('Starting robot')
    fa = FrankaArm()

    cv_bridge = CvBridge()
    ir_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)

    kinect2_left_to_world_transform = RigidTransform.load(args.transform_file_path)


    # print('Opening Grippers')
    # fa.open_gripper()

    relative_y_dist = RigidTransform(rotation=np.array([
                                                [1,  0,  0],
                                                [0,  1,  0],
                                                [0,  0,  1]
                                            ]), translation=np.array([0, 0.01, 0.0]),
                                        from_frame='franka_tool', to_frame='franka_tool')

    fa.goto_pose_delta_with_cartesian_control(relative_y_dist, 3)

    print('Reset with pose')
    fa.reset_pose()

    print('Reset with joints')
    fa.reset_joints()

    fa.close_gripper()

    rgb_image_msg = rospy.wait_for_message('/rgb/image_raw', Image)
    try:
        rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    except CvBridgeError as e:
        print(e)

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
           # draw circle here (etc...)
           print('x = %d, y = %d'%(x, y))

    object_xcenter = 1073
    object_ycenter = 753
    radius=30

    cropped_image = rgb_cv_image[(object_ycenter-radius):(object_ycenter+radius+1), (object_xcenter-radius):(object_xcenter+radius+1)]

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
 
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,128,255,1)

    M = cv2.moments(thresh)

    # print(int(M["m01"] / M["m00"]))
    # print(int(M["m10"] / M["m00"]))

    object_xcenter = int(M["m10"] / M["m00"]) - radius + object_xcenter
    object_ycenter = int(M["m01"] / M["m00"]) - radius + object_ycenter

    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.circle(rgb_cv_image,(object_xcenter,object_ycenter),10,(255,0,0),-1)
    cv2.imshow('image', rgb_cv_image)

    #cv2.resizeWindow('image', 600,600)
    cv2.setMouseCallback('image',onMouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    depth_image_msg = rospy.wait_for_message('/depth_to_rgb/image_raw', Image)
    try:
        depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)
    
    object_center = Point(np.array([object_xcenter, object_ycenter]), 'kinect2_left')
    object_depth = depth_cv_image[object_ycenter, object_xcenter]
    print("x, y, z: ({:.4f}, {:.4f}, {:.4f})".format(
        object_xcenter, object_ycenter, object_depth))
    object_center_point_in_world = kinect2_left_to_world_transform * ir_intrinsics.deproject_pixel(object_depth, object_center)
    
    vision_object_location = np.array([object_center_point_in_world.x, object_center_point_in_world.y, object_center_point_in_world.z])
    print(object_center_point_in_world)

    constant_x = np.random.rand(1)[0] * 0.06 + 0.47
    constant_y = np.random.rand(1)[0] * 0.02 - 0.03
    constant_z = np.random.rand(1)[0] * 0.03 + 0.07
    x_scan_dist = 0.03
    y_scan_dist = 0.01
    z_scan_dist = 0.03

    desired_pose = RigidTransform(
        rotation=np.array([[1, 0,  0],
                           [0,  -1,  0],
                           [0, 0,  -1]]), 
        translation=np.array([constant_x, constant_y, constant_z]), 
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(desired_pose, 10.0)

    correct_x = constant_x
    correct_y = constant_y
    correct_z = constant_z

    starting_location = [correct_x, correct_y, correct_z]
    print("Starting location:")
    print(starting_location)

    for i in range(2):

        ### X scan ###
        desired_object_pose = RigidTransform(
            rotation=np.array([[1, 0,  0],
                               [0,  -1,  0],
                               [0, 0,  -1]]), 
            translation=np.array([correct_x+x_scan_dist, correct_y, correct_z]),
             from_frame='franka_tool', to_frame='world')


        magnetic_calibration = MagneticCalibration()

        filename = '../calibration/2020-01-14 11-08 E1 2X Board Calibration.npz'
        magnetic_calibration.load_calibration_file(filename)

        magnetic_calibration.start_recording_data()  
        fa.goto_pose_with_cartesian_control(desired_object_pose, 3.0)
        magnetic_calibration.stop_recording_data()

        max_location = magnetic_calibration.get_max_location()
        print(max_location)

        (magnet_data, robot_data) = magnetic_calibration.getData()

        num_data = np.min((robot_data.shape[0], magnet_data.shape[0]))

        xdata = np.transpose(robot_data[:num_data,0])
        zdata = np.transpose(magnet_data[:num_data,6])

        guess_prm = [400, max_location[0], 0.01, np.min(zdata)]
        bounds = ([0, max_location[0]-0.05, 0.0, 0],[600, max_location[0]+0.05, 0.02, 200])

        # Do the fit, using our custom _gaussian function which understands our
        # flattened (ravelled) ordering of the data points.
        popt, pcov = curve_fit(func, xdata, zdata, p0=guess_prm, bounds=bounds, maxfev=1000000)

        fit = np.zeros(zdata.shape)

        print('Fitted parameters:')
        print(popt)

        rms = np.sqrt(np.mean((zdata - fit)**2))
        print('RMS residual =', rms)

        correct_x = popt[1]
        current_location = [correct_x, correct_y, correct_z]

        print("Next Location:")
        print(current_location)

        input("Press enter to continue to command")

        next_position = RigidTransform(rotation=np.array([
                                                    [1, 0,  0],
                                                   [0,  -1,  0],
                                                   [0, 0,  -1]
                                                ]), translation=np.array(current_location), # add x and z later
                                            from_frame='franka_tool', to_frame='world')

        fa.goto_pose_with_cartesian_control(next_position, 3)

        print(fa.get_pose().translation)

        if args.filename is not None:
            magnetic_calibration.saveData(args.filename+'_x'+str(i), starting_location, vision_object_location, current_location, fa.get_pose().translation)


        starting_location = current_location

        ### Z scan ###
        desired_object_pose = RigidTransform(
        rotation=np.array([[1, 0,  0],
                           [0,  -1,  0],
                           [0, 0,  -1]]), 
        translation=np.array([correct_x, correct_y, correct_z+z_scan_dist]),
         from_frame='franka_tool', to_frame='world')

        magnetic_calibration.start_recording_data()  
        fa.goto_pose_with_cartesian_control(desired_object_pose, 3.0)
        magnetic_calibration.stop_recording_data()

        max_location = magnetic_calibration.get_max_location()
        print(max_location)

        (magnet_data, robot_data) = magnetic_calibration.getData()


        num_data = np.min((robot_data.shape[0], magnet_data.shape[0]))

        xdata = np.transpose(robot_data[:num_data,2])
        zdata = np.transpose(magnet_data[:num_data,6])

        guess_prm = [400, max_location[2], 0.01, np.min(zdata)]
        bounds = ([0, max_location[2]-0.05, 0.0, 0],[600, max_location[2]+0.05, 0.02, 200])
        popt, pcov = curve_fit(func, xdata, zdata, p0=guess_prm, bounds=bounds, maxfev=1000000)

        print('Fitted parameters:')
        print(popt)

        correct_z = popt[1]
        current_location = [correct_x, correct_y, correct_z]

        print("Next Location:")
        print(current_location)

        input("Press enter to continue to command")
        
        

        next_position = RigidTransform(rotation=np.array([
                                                    [1, 0,  0],
                                                   [0,  -1,  0],
                                                   [0, 0,  -1]
                                                ]), translation=np.array(current_location), # add x and z later
                                            from_frame='franka_tool', to_frame='world')

        fa.goto_pose_with_cartesian_control(next_position, 3)

        print(fa.get_pose().translation)

        if args.filename is not None:
            magnetic_calibration.saveData(args.filename+'_z'+str(i), starting_location, vision_object_location, current_location, fa.get_pose().translation)

        x_scan_dist = 0.01
        z_scan_dist = 0.01

        starting_location = current_location

    desired_object_pose = RigidTransform(rotation=np.array([[1, 0,  0],
                                                           [0,  -1,  0],
                                                           [0, 0,  -1]]), 
                                        translation=np.array([correct_x, correct_y+y_scan_dist, correct_z]),
                                         from_frame='franka_tool', to_frame='world')

    magnetic_calibration.start_recording_data()  
    fa.goto_pose_with_cartesian_control(desired_object_pose, 3.0)
    magnetic_calibration.stop_recording_data()

    (magnet_data, robot_data) = magnetic_calibration.getData()

    num_data = np.min((robot_data.shape[0], magnet_data.shape[0]))

    xdata = np.transpose(robot_data[:num_data,1])*1000
    zdata = np.transpose(magnet_data[:num_data,6])

     #[offset(mm) Br(uT)]
    guess_prm = [-np.min(xdata), 4500, 80]
    bounds = ([-np.min(xdata)-30,4000, 60],[-np.min(xdata)+30,5000, 100])
    popt, pcov = curve_fit(centerline, xdata, zdata, p0=guess_prm, bounds=bounds, maxfev=10000)

    popt /= [1000.0, 1, 1]

    print('Fitted parameters:')
    print(popt)

    step = (popt[0]+(np.min(xdata)/1000))
    print('Step Size (m):')
    print(step)

    while abs(step) > 0.001:
        next_y_position = (np.min(xdata)/1000)-step
        print("Next Y Position:")
        print(next_y_position)
        input("Press enter to continue")

        desired_object_pose = RigidTransform(rotation=np.array([[1, 0,  0],
                                                               [0,  -1,  0],
                                                               [0, 0,  -1]]), 
                                            translation=np.array([correct_x, next_y_position, correct_z]),
                                             from_frame='franka_tool', to_frame='world')
        #NOTE: make sure time is long enough
        magnetic_calibration.restart_recording_data()  
        fa.goto_pose_with_cartesian_control(desired_object_pose, 3.0)
        magnetic_calibration.stop_recording_data()
        
        (magnet_data, robot_data) = magnetic_calibration.getData()

        num_data = np.min((robot_data.shape[0], magnet_data.shape[0]))

        xdata = np.transpose(robot_data[:num_data,1])*1000
        zdata = np.transpose(magnet_data[:num_data,6])

         #[offset(mm) Br(uT)]
        guess_prm = [-np.min(xdata), 4500, 80]
        bounds = ([-np.min(xdata)-30,4000, 60],[-np.min(xdata)+30,5000,100])
        popt, pcov = curve_fit(centerline, xdata, zdata, p0=guess_prm, bounds=bounds, maxfev=10000)

        popt /= [1000.0, 1, 1]

        print('Fitted parameters:')
        print(popt)

        step = (popt[0]+(np.min(xdata)/1000))
        print('Step Size (m):')
        print(step)


    # when step is sufficiently small
    next_y_position = (np.min(xdata)/1000)-step
    print("Final Y Position:")
    print(next_y_position)

    input("Press enter to continue")

    desired_object_pose = RigidTransform(rotation=np.array([[1, 0,  0],
                                                           [0,  -1,  0],
                                                           [0, 0,  -1]]), 
                                        translation=np.array([correct_x, next_y_position, correct_z]),
                                         from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(desired_object_pose, 3.0)

    current_location = np.array([correct_x, next_y_position, correct_z])

    if args.filename is not None:
        magnetic_calibration.saveData(args.filename+'_y', starting_location, vision_object_location, current_location, fa.get_pose().translation)

    print("Final Error: ")
    error = vision_object_location-fa.get_pose().translation
    print(error)

    rgb_image_msg = rospy.wait_for_message('/rgb/image_raw', Image)
    try:
        rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    except CvBridgeError as e:
        print(e)

    radius = 50
    cropped_image = rgb_cv_image[(object_ycenter-radius):(object_ycenter+radius+1), (object_xcenter-radius):(object_xcenter+radius+1)]

    if args.filename is not None:
        cv2.imwrite(args.filename+'.png', cropped_image)

    magnetic_calibration.plotData(magnetic_calibration.magnetic_data, 'Raw Data Over Time')
    magnetic_calibration.plotGripperData(magnetic_calibration.gripper_state_data, 'Raw Data Over Time')
    magnetic_calibration.plotGripperData(magnetic_calibration.robot_state_data, 'Raw Data Over Time')

    cv2.imshow('image', cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()