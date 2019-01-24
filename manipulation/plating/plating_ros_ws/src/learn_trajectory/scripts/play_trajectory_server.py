#!/usr/bin/env python
import rospy
import argparse
import numpy as np

import play_recorded_trajectory as prt

from std_msgs.msg import Bool
from learn_trajectory.srv import PlayTrajectory, PlayTrajectoryResponse 

def main():
  # initialize the ros node 
  rospy.init_node('play_trajectory_server', anonymous=True)
  pts = PlayTrajectoryService()
  s = rospy.Service('play_trajectory', PlayTrajectory, pts.handle_play_request)
  pts.play_when_called()

class PlayTrajectoryService:
  def __init__(self):
    self.should_play = False
    self.pose_topic = None
    self.recording_file_name = None

  def play_when_called(self):
    # keep this loop from going faster than a fixed amount by means of rospy.rate
    # this time gives the max amount of time it takes for this player to start playing
    # but the loop will pause while the player plays
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
      try:
        if self.should_play:
          if self.pose_topic is None:
            rospy.logerr("pose_topic should be set before should_play is set")
          if self.recording_file_name is None:
            rospy.logerr("recording_file_name should be set before should_play is set")
          prt.publish_poses(self.recording_file_name, self.pose_topic)
          self.should_play = False
      except Exception as e:
        rospy.logerr(e)
        # we don't want this service to ever actually throw errors and fail out
        raise
      # We don't need special code to allow any callbacks to run, in case the user has updated the location
      # since in rospy, callbacks are always called in separate threads 
      # however, since sometimes the loop is a no-op, we add a sleep to keep it from going faster than 10hz
      r.sleep()

  # takes in a TrackArm request and calls ada_control 
  # to move the arm based on that request
  def handle_play_request(self, req):
    if self.should_play:
      rospy.logerr("Player was currently playing when another request to play was called. The new request to play will be ignored")
      return PlayTrajectoryResponse(Bool(False))
    self.pose_topic = req.pose_topic.data
    self.recording_file_name = req.recording_file_name.data
    self.should_play = True
    return PlayTrajectoryResponse(Bool(True))

if __name__=="__main__":
  main() 
