#!/usr/bin/python
"""Shows how to set a physics engine and send torque commands to the robot
"""
import rospy
import numpy as np

from feedbot_trajectory_logic.srv import TrackPose
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, PointStamped, Quaternion 
from camera_calibration import CameraCalibration

class TrackerInterface:
  def __init__(self, defaultQuat, service_name='update_pose_target'):
    self.cameraCalib = CameraCalibration()
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

  def start_updating_target_to_point(self, mouth_point_topic, robot_coord_offset=[0,0,0]):
    self.stop_moving() 
    self.mouth_target_listener = rospy.Subscriber(mouth_point_topic, PointStamped, self._update_target_camera_frame, (robot_coord_offset))
  
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

  # return an np.array of the [x,y,z] target mouth location
  # in the coordinate frame of the robot base
  def _convert_camera_to_robot_frame(self, mouth_pos):
    t = mouth_pos.point
    point_in_camera_frame = np.array([t.x, t.y, t.z, 1])
    point_in_robot_frame = self.cameraCalib.convert_to_robot_frame(point_in_camera_frame)
    return point_in_robot_frame[0:3]

  # compute and move toward the target mouth location
  # only move for at most timeoutSecs,   
  def _update_target_camera_frame(self, mouth_pos, robot_coord_offset = [0,0,0]):
    endLoc = self._convert_camera_to_robot_frame(mouth_pos) + np.array(robot_coord_offset)
    self._update_target(target=Pose(Point(endLoc[0], endLoc[1], endLoc[2]), self.defaultQuat))
  
  # move toward the target spoon pose
  def _update_target_pose_robot_frame(self, target_pose, robot_coord_offset = [0,0,0]):
    newPosition = Point(target_pose.position.x + robot_coord_offset[0],
                        target_pose.position.y + robot_coord_offset[1],
                        target_pose.position.z + robot_coord_offset[2])
    self._update_target(target=Pose(newPosition, target_pose.orientation))



