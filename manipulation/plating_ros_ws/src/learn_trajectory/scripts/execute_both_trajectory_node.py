#!/usr/bin/env python
import rospy
import time

import feedbot_trajectory_logic.tracker_interface as tracker
import numpy as np
from learn_trajectory.srv import PlayTrajectory
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Quaternion


class SpoonFeeder:
  def __init__(self):
    rospy.logwarn("sleeping for 5 seconds before starting recorded motion")
    rospy.sleep(5)
    # quaternion is defined in order x,y,z,w
    self.defaultQuat = Quaternion(0.5, 0.5, 0.5, 0.5)
    self.tracker = tracker.TrackerInterface(self.defaultQuat)
    self.trackertoo = tracker.TrackerInterface(self.defaultQuat, '/domusromus/update_pose_target')
    self.play_trajectory_topic = "trained_poses"
    self._play_trajectory = rospy.ServiceProxy("play_trajectory", PlayTrajectory)

  def follow_trajectory(self, recording_file_name):
    self.tracker.start_updating_target_to_pose(self.play_trajectory_topic, [0,0,0])
    self.trackertoo.start_updating_target_to_pose("/domusromus/"+self.play_trajectory_topic,[0,0,0])
    rospy.logwarn("Playing trajectory at " + self.play_trajectory_topic)
    self._play_trajectory(String(self.play_trajectory_topic), String(recording_file_name))

if __name__=="__main__":
  rospy.init_node('spoon_feeder', anonymous=True)
  recording_file_name = rospy.get_param("~recording_file_name")
  s = SpoonFeeder()
  while (True):
    s.follow_trajectory(recording_file_name)
    time.sleep(5)
