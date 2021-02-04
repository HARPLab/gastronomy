#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2 

import signal
import time

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'filename', nargs='?', metavar='FILENAME',
        help='video file to store recording to')
    parser.add_argument('--topic', '-t', type=str, default='/camera')
    args = parser.parse_args()

    rospy.init_node('realsense_subscriber')

    cv_bridge = CvBridge()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    realsense_rgb = cv2.VideoWriter(args.filename+'_rgb.avi', fourcc, 15.0, (640,480))
    realsense_depth = cv2.VideoWriter(args.filename+'_depth.avi', fourcc, 15.0, (640,480))

    def callback1(msg):
        try:
            cv_image = cv_bridge.imgmsg_to_cv2(msg)
            rgb_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            realsense_rgb.write(rgb_cv_image)
        except CvBridgeError as e:
            print(e)

    
    boundaries = np.zeros((0,3,2))
    #depth_images = np.zeros((0,480,640), dtype=np.uint16)

    def callback2(msg):
        global boundaries
        #global depth_images
        try:
            depth_image = cv_bridge.imgmsg_to_cv2(msg)
            depth_image_max_value = np.max(depth_image)
            depth_image_min_value = np.min(depth_image[np.nonzero(depth_image)])
            section_length = (depth_image_max_value - depth_image_min_value + 1) / 3.0
            boundary = np.zeros((3,2))

            boundary[0,0] = depth_image_min_value - 1
            boundary[0,1] = boundary[0,0] + section_length
            boundary[1,0] = boundary[0,0] + section_length
            boundary[1,1] = boundary[0,0] + 2 * section_length
            boundary[2,0] = boundary[0,0] + 2 * section_length
            boundary[2,1] = depth_image_max_value

            scaled_image = depth_image.copy().astype('float64')

            new_scaled_rgb_cv_image = np.zeros((480,640,3), dtype=float)
            for i in range(3):
                nonzero_element_indices = np.nonzero(np.logical_and(scaled_image > boundary[i,0],scaled_image <= boundary[i,1]))
                if nonzero_element_indices[0].shape[0] > 0:
                    boundary[i,0] = np.min(scaled_image[nonzero_element_indices]) - 1
                    boundary[i,1] = np.max(scaled_image[nonzero_element_indices])
                    alpha = (boundary[i,1] - boundary[i,0]) / 255.0
                    new_scaled_cv_image = np.zeros((480,640))
                    new_scaled_cv_image[nonzero_element_indices] = (scaled_image[nonzero_element_indices] - boundary[i,0]) / alpha 
                    new_scaled_rgb_cv_image[:,:,i] = new_scaled_cv_image

            rgb_cv_image = new_scaled_rgb_cv_image.astype(np.uint8)

            boundaries = np.vstack((boundaries, boundary.reshape(1,3,2)))

            realsense_depth.write(rgb_cv_image)
            #depth_images = np.vstack((depth_images, depth_image.reshape((1,480,640))))
        except CvBridgeError as e:
            print(e)

    realsense_rgb_sub = rospy.Subscriber(args.topic + '/color/image_raw', Image, callback1)
    realsense_depth_sub = rospy.Subscriber(args.topic + '/aligned_depth_to_color/image_raw', Image, callback2)
    
    killer = GracefulKiller()
    while not killer.kill_now:
        time.sleep(0.01)
    
    realsense_rgb.release()
    realsense_depth.release()
    np.save(args.filename+'_depth_image_boundaries.npy', boundaries)
    #np.save(args.filename+'_depth_images.npy', depth_images)
