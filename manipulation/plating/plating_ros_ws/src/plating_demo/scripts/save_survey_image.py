#!/usr/bin/env python
import numpy as np
import rospkg
import os

#https://answers.ros.org/question/210294/ros-python-save-snapshot-from-camera/
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

import wait_logic as wl

import feedbot_trajectory_logic.tracker_interface as tracker
from learn_trajectory.srv import PlayTrajectory
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Quaternion, PointStamped
import rospy

#https://answers.ros.org/question/210294/ros-python-save-snapshot-from-camera/
def save_image(msg):
  # Instantiate CvBridge
  bridge = CvBridge()
  rospy.logwarn("Received an image!")
  try:
      # Convert your ROS Image message to OpenCV2
      cv2_img = bridge.imgmsg_to_cv2(msg, 'bgr8') 
  except CvBridgeError, e:
      print(e)
  else:
      # Save your OpenCV2 image as a jpeg 
      rospack= rospkg.RosPack()
      filename = rospack.get_path('food_perception') + '/test/input_data/survey_img.jpg'
      cv2.imshow('image', cv2_img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      cv2.imwrite(filename, cv2_img)

class ServingDemo:
  def __init__(self):
    # quaternion is defined in order x,y,z,w
    #self.defaultQuat = Quaternion(0.5, 0.5, 0.5, 0.5)
    self.defaultQuat = Quaternion(0, 1, 0, 0)
    self.tracker = tracker.TrackerInterface(self.defaultQuat)
    self.play_trajectory_topic = "/spoon/example_poses"
    self._play_trajectory = rospy.ServiceProxy("play_trajectory", PlayTrajectory)

    self.locations = { 
                       "survey" : [-0.15, 0.55, 0.3],
                     }

    self.quats = {
                   "vert" : Quaternion(0.707, 0.707, 0, 0),
                 }

    # go to location, pause, take photo, die
    self.go_to_pose(self.locations["survey"], self.quats["vert"])
    rospy.sleep(1)
    image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
    save_image(image_msg)
    rospy.logwarn("saved image")


  def go_to_pose(self, position, quat):
    # clear out any numpy-arrayness of position
    position = np.array(position)
    position = position.tolist()
    self.tracker.start_tracking_fixed_pose(position, quat)
    wl.wait(wl.State.MOVING_ARM) 

if __name__=="__main__":
  rospy.init_node('proper_plating_learner', anonymous=True)
  sd = ServingDemo()
