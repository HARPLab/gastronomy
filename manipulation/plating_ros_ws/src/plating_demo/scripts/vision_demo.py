#!/usr/bin/env python
import numpy as np
import rospkg
import os

import wait_logic as wl

import feedbot_trajectory_logic.tracker_interface as tracker
from learn_trajectory.srv import PlayTrajectory
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Quaternion
import rospy

class ServingDemo:
  def __init__(self):
    # quaternion is defined in order x,y,z,w
    self.defaultQuat = Quaternion(0.5, 0.5, 0.5, 0.5)
    self.tracker = tracker.TrackerInterface(self.defaultQuat)
    self.play_trajectory_topic = "/spoon/example_poses"
    self._play_trajectory = rospy.ServiceProxy("play_trajectory", PlayTrajectory)

    # z needs to add 1 because of height of table.
    self.locations = { "survey": [0.3,-0.1,1.1]}

    self.recording_files = { "pick_up" : "short_fork_careful_carry.txt",
                             "scoop" : "scoop_cucumber.txt"}
   
    while True: 
      rospy.sleep(1)
      self.survey()

  def survey(self):
    position = self.locations["survey"]
    self.go_to_pose(position,(self.defaultQuat.x, self.defaultQuat.y, self.defaultQuat.z,self.defaultQuat.w))

  def strawberry(self):
    self.go_to_position("strawberry")
    self.play_recording("vanilla", 
           (np.array(self.locations["strawberry"])-self.locations["vanilla"]).tolist())
  
  # plating_request should be an object PlatingRequest from file plating_request
  def go_to_position(self, location_name, offset = [0,0,0]):
    # pick up food
    ingredient_position = self.locations[location_name]
    self.tracker.start_tracking_fixed_target(ingredient_position + np.array(offset))
    wl.wait(wl.State.MOVING_ARM) 
  
  def go_to_pose(self, point, rot_quat):
    # the quatMsg should be x,y,z,w
    quatMsg = Quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
    self.tracker.start_tracking_fixed_pose(point, quatMsg)
    with wl.DistanceBasedWaitLogic("/distance_to_target") as waitLogic:
      waitLogic.wait()

  def play_recording(self, recording_name, offset = [0,0,0]):
    recording_file = self.recording_files[recording_name]
    self.tracker.start_updating_target_to_pose(self.play_trajectory_topic, offset)
    self._play_trajectory(String(self.play_trajectory_topic), String(recording_file))
    wl.wait(wl.State.FOLLOWING_TRAJECTORY) 

if __name__=="__main__":
  rospy.init_node('proper_plating_learner', anonymous=True)
  sd = ServingDemo()
