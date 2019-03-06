#!/usr/bin/python
"""Shows how to set a physics engine and send torque commands to the robot
"""
import rospy
import numpy as np

from feedbot_trajectory_logic.srv import TrackPose
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, PointStamped, Quaternion 

class TrackerInterface:
  def __init__(self, defaultQuat, service_name='update_pose_target'):
    rospy.wait_for_service(service_name, timeout=None)
    self._update_target = rospy.ServiceProxy(service_name, TrackPose)
    self.pose_target_listener = None
    self.mouth_target_listener = None
    self.defaultQuat = defaultQuat 

  #### PUBLIC METHODS
  # we try to make it so that all of these methods are idempotent 
  # and can be called from any state
  def start_updating_target_to_pose(self, target_pose_topic, robot_coord_offset=[0,0,0]):
    self.stop_moving() 
    self.pose_target_listener = rospy.Subscriber(target_pose_topic, Pose, self._update_target_pose_robot_frame, (robot_coord_offset))

  def start_tracking_fixed_pose(self, robot_coord_point, robot_coord_quat):
    self.stop_moving() 
    # you just send the target point, you don't need to continually update it
    self._update_target(target=Pose(Point(robot_coord_point[0], robot_coord_point[1], robot_coord_point[2]), robot_coord_quat))
  
  def start_tracking_fixed_target(self, robot_coord_point):
    self.start_tracking_fixed_pose(robot_coord_point, self.defaultQuat) 

  def stop_moving(self):
    self._stop_updating_target()
    self._update_target(stopMotion=True)


  #### PRIVATE METHODS #####
  def _stop_updating_target(self):
    if (self.mouth_target_listener is not None):
      self.mouth_target_listener.unregister()
    if (self.pose_target_listener is not None):
      self.pose_target_listener.unregister()
  
  # move toward the target spoon pose
  def _update_target_pose_robot_frame(self, target_pose, robot_coord_offset = [0,0,0]):
    newPosition = Point(target_pose.position.x + robot_coord_offset[0],
                        target_pose.position.y + robot_coord_offset[1],
                        target_pose.position.z + robot_coord_offset[2])
    self._update_target(target=Pose(newPosition, target_pose.orientation))



