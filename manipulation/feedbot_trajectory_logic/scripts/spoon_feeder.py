#!/usr/bin/env python
import rospy

import feedbot_trajectory_logic.tracker_interface as tracker
import numpy as np
from feeding_state_transition_logic import transitionLogicDictionary, State
from feedbot_trajectory_logic.srv import PlayTrajectory
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Quaternion


class SpoonFeeder:
  def __init__(self):
    rospy.logwarn("sleeping for 5 seconds before starting feeding")
    rospy.sleep(5)
    # quaternion is defined in order x,y,z,w
    self.defaultQuat = Quaternion(0.5, 0.5, 0.5, 0.5)
    self.tracker = tracker.TrackerInterface(self.defaultQuat)
    self.play_trajectory_topic = "/Tapo/example_poses"
    self._play_trajectory = rospy.ServiceProxy("play_trajectory", PlayTrajectory)
    rospy.logwarn("TrackerInterface successfully initialized")
    self._set_state(State.WAIT_FOR_SPOON_CALIBRATION)
    self.restart_do_pub = rospy.Publisher('/tracking_face_transform/reinitialization', Empty, queue_size = 1)

    while not rospy.is_shutdown():
      with transitionLogicDictionary[self.state]() as transitionLogic:
        rospy.logwarn("About to wait and return")
        nextState = transitionLogic.wait_and_return_next_state()
      rospy.logwarn("returned")
      self._set_state(nextState)

  def _set_state(self, state):
    rospy.logwarn("State is now %s" % state)
    self.state = state
    self._update_tracker_based_on_state()

  def _update_tracker_based_on_state(self):
    if self.state == State.MOVE_TO_PLATE:
      self.tracker.start_tracking_fixed_target([0.3,0.05,0.17])
      self.is_first_move_to_plate = False
    elif self.state == State.PICK_UP_FOOD:
      #self.restart_do_pub.publish(Empty())
      self.xoffset = 0.03-0.1 + (np.random.uniform()-0.5) * 0.04
      self.yoffset = 0.015+0.3
      self.zoffset = rospy.get_param('~z_height')
      self.tracker.start_updating_target_to_pose(self.play_trajectory_topic,[self.xoffset, self.yoffset, self.zoffset])
      self._play_trajectory(String(self.play_trajectory_topic))
    elif self.state == State.PREPARE_FOR_MOUTH:
      #self.restart_do_pub.publish(Empty())
      self.tracker.start_tracking_fixed_target([0.3,0.15,0.27])
    elif self.state == State.MOVE_TO_MOUTH:
      follow_mouth = rospy.get_param('~follow_mouth')
      if follow_mouth:
        self.tracker.start_updating_target_to_point(rospy.get_param('~mouth_point_topic'))
      else:  
        self.tracker.start_tracking_fixed_target([0.27,0.25,0.27])
    elif self.state == State.WAIT_IN_MOUTH:
      self.tracker.stop_moving()
    elif self.state == State.PREPARE_FOR_PLATE:
      self.tracker.start_tracking_fixed_target([0.3,0.15,0.27])
    elif self.state == State.WAIT_FOR_SPOON_CALIBRATION:
      pass
    else:
      rospy.logerr("The state %s is not known"%self.state)


if __name__=="__main__":
  rospy.init_node('spoon_feeder', anonymous=True)
  s = SpoonFeeder()
