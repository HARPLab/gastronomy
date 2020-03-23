#!/usr/bin/env python

import rospy
from magnetic_stickers.msg import MagneticData
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from frankapy import FrankaArm
import math
import time

class MagneticCalibration:
    def __init__(self):
        
        sub = rospy.Subscriber('/magnetic_stickers/magnetic_data', MagneticData, self.callback)

        self.sample_index = 0
        self.num_samples = 1000
        self.magnetic_data = np.zeros((self.num_samples,8))
        
    def callback(self,data):
        
        #rospy.loginfo("[" + str(data.ref_x) + "," + str(data.ref_y) + "," + str(data.ref_z) + "," + str(data.ref_t) + 
        #              str(data.x) + "," + str(data.y) + "," + str(data.z) + "," + str(data.t) + "]")

        self.magnetic_data[self.sample_index,0] = data.ref_x
        self.magnetic_data[self.sample_index,1] = data.ref_y
        self.magnetic_data[self.sample_index,2] = data.ref_z
        self.magnetic_data[self.sample_index,3] = data.ref_t
        self.magnetic_data[self.sample_index,4] = data.x
        self.magnetic_data[self.sample_index,5] = data.y
        self.magnetic_data[self.sample_index,6] = data.z
        self.magnetic_data[self.sample_index,7] = data.t

        self.sample_index += 1
        self.sample_index = self.sample_index % self.num_samples

    def get_magnetic_data(self):
        data1 = self.magnetic_data[self.sample_index:,:]
        data2 = self.magnetic_data[:self.sample_index,:]
        data = np.concatenate((data1, data2), axis=0)
        return data

    def get_last_magnetic_data(self):
        last_data = self.magnetic_data[self.sample_index-1,:]
        
        return last_data

    def calibrate(self):
        
        # first calibrate mag individually, then transform reference
        date_time = datetime.now().strftime("%Y-%m-%d %H-%M")
        filename = date_time + " E1 2X Board Calibration.npz"

        data = self.get_magnetic_data()

        self.plotData(data, "Raw Data")
        # calculate offsets and scales for calibration
        offsets = np.zeros(8)
        scales = np.zeros(8)

        minValues = np.amin(data, axis=0)
        maxValues = np.amax(data, axis=0)

        offsets = (maxValues+minValues)/2
        offsets[3] = 0
        offsets[7] = 0

        scales = (maxValues-minValues)/2
        scales[0:3] = scales[0:3]/(np.sum(scales[0:3])/3)
        scales[3] = 1
        scales[4:7] = scales[4:7]/(np.sum(scales[4:7])/3)
        scales[7] = 1

        print("Offsets: " + str(offsets))
        print("Scales: " + str(scales))
        calibrated = np.multiply(scales, (data-offsets))
        self.plotData(calibrated, "Calibrated Data")

        # calculate affine transform from reference signal
        transform_ref = np.zeros(shape=(np.size(calibrated,0),4))
        calibrated[:,3] = 1
        calibrated[:,7] = 1
        A, residuals, rank, s = np.linalg.lstsq(calibrated[:,0:4], calibrated[:,4:])
        print("Affine Matrix: " + str(A))
        print("Residuals: " + str(residuals))
        transform_ref = calibrated[:,0:4].dot(A)
        
        self.plotData(calibrated[:,4:7]-transform_ref[:,0:3], "Transformed Data")
        avg_error = np.mean(calibrated[:,4:7]-transform_ref[:,0:3])
        print("Average error: " + str(avg_error))
        np.savez(filename, data=data, affine=A, offsets=offsets, scales=scales)


    def plotData(self, data, figure_title):
        
        plt.figure(figure_title)
        plt.subplot(131)
        plt.scatter(data[:,0], data[:,1], c="red")
        plt.title('XY Plane')
        plt.axis('equal')

        plt.subplot(132)
        plt.scatter(data[:,1], data[:,2], c="green")
        plt.title('YZ Plane')
        plt.axis('equal')

        plt.subplot(133)
        plt.scatter(data[:,2], data[:,0], c="blue")
        plt.title('ZX Plane')
        plt.axis('equal')

        #plt.draw()
        #plt.pause(0.001)
        plt.show()

    def collectAndSaveSamples(self, num_samples, filename):

        self.magnetic_data = np.zeros((num_samples,8))
        self.num_samples = num_samples

        # blocking loop waiting to collect all the samples
        while(self.num_data < num_samples):
            time.sleep(0.01)

        np.savez(filename, data=self.magnetic_data)


def run_main():
    fa = FrankaArm()

    fa.reset_pose()
    fa.reset_joints()
    
    magnetic_calibration = MagneticCalibration()

    fa.apply_effector_forces_torques(45, 0, 0, 0)
    
    magnetic_calibration.calibrate()

if __name__ == '__main__':
    run_main()