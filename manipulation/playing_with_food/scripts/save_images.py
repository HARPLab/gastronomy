import numpy as np
import argparse
import math
import cv2
import rospy
from cv_bridge import CvBridge

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='playing_data/')
    parser.add_argument('--food_type', '-f', type=str)   
    parser.add_argument('--end', '-e', action='store_true')
    args = parser.parse_args()

    dir_path = args.data_dir
    createFolder(dir_path)
    dir_path += args.food_type + '/'
    createFolder(dir_path)

    rospy.init_node('SaveImages')

    cv_bridge = CvBridge()

    # Capture Overhead Azure Kinect Depth and RGB Images
    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    realsense_rgb_image = get_realsense_rgb_image(cv_bridge)
    realsense_depth_image = get_realsense_depth_image(cv_bridge)
    
    cutting_board_x_min = 750
    cutting_board_x_max = 1170
    cutting_board_y_min = 290
    cutting_board_y_max = 620

    cropped_azure_kinect_rgb_image = azure_kinect_rgb_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]
    cropped_azure_kinect_depth_image = azure_kinect_depth_image[cutting_board_y_min:cutting_board_y_max, cutting_board_x_min:cutting_board_x_max]
    
    if args.end:
        cv2.imwrite(dir_path + 'end_azure_kinect_rgb_image.png', cropped_azure_kinect_rgb_image)
        np.save(dir_path + 'end_azure_kinect_depth_image.npy', cropped_azure_kinect_depth_image)
        cv2.imwrite(dir_path + 'end_realsense_rgb_image.png', realsense_rgb_image)
        np.save(dir_path + 'end_realsense_depth_image.npy', realsense_depth_image)
    else:
        cv2.imwrite(dir_path + 'azure_kinect_rgb_image.png', cropped_azure_kinect_rgb_image)
        np.save(dir_path + 'azure_kinect_depth_image.npy', cropped_azure_kinect_depth_image)
        cv2.imwrite(dir_path + 'realsense_rgb_image.png', realsense_rgb_image)
        np.save(dir_path + 'realsense_depth_image.npy', realsense_depth_image)