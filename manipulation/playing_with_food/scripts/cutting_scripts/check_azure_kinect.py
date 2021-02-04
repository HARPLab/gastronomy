import numpy as np
import math
import cv2
import rospy
from cv_bridge import CvBridge

from utils import *

if __name__ == '__main__':

    rospy.init_node('AzureKinectSubscriber')

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

    cv2.imwrite('starting_azure_kinect_rgb_image.png', cropped_azure_kinect_rgb_image)
    np.save('starting_azure_kinect_depth_image.npy', cropped_azure_kinect_depth_image)