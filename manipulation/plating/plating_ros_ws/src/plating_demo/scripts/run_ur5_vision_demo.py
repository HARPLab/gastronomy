#!/usr/bin/env python
import numpy as np
import rospkg
import os

import wait_logic as wl

import feedbot_trajectory_logic.tracker_interface as tracker
from learn_trajectory.srv import PlayTrajectory
from std_msgs.msg import String, Empty
from sensor_msgs.msg import JointState 
from geometry_msgs.msg import Quaternion, PointStamped
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController
import rospy

class ServingDemo:
  def __init__(self):
    # quaternion is defined in order x,y,z,w
    #self.defaultQuat = Quaternion(0.5, 0.5, 0.5, 0.5)
    self.defaultQuat = Quaternion(0, 1, 0, 0)
    self.tracker = tracker.TrackerInterface(self.defaultQuat)
    self.play_trajectory_topic = "/spoon/example_poses"
    self._play_trajectory = rospy.ServiceProxy("play_trajectory", PlayTrajectory)
    self.switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
    self.direct_joint_command_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)

    self.locations = { 
                       "survey" : [-0.15, 0.55, 0.3]
                     }

    self.quats = { "vert" : Quaternion(0.707, 0.707, 0, 0),
                   "carry" : Quaternion(0.5, 0.5, -0.5, -0.5)}

    self.recording_files = { "pickup" : "calibrated_pickup.txt",
                             "dropoff" : "calibrated_dropoff.txt"}

    pickup_recording_offset = np.array([0.02, -0.01, -0.01])


    food_loc = [(i % 2, 0.68-i*0.02) for i in range(8)]
    # ONLY CHEESE
    #food_loc = [(1, 0.68-i*0.02) for i in range(8)]

    for (food, y) in food_loc:
      rospy.logwarn("Moving food %d to position %s" % (food, y))
      self.go_to_named_pose("survey", "vert")
      rospy.sleep(0.5)
      food_msg = rospy.wait_for_message("/food%d" % food, PointStamped)
      food_point = np.array([food_msg.point.x, food_msg.point.y, food_msg.point.z])
      pickup_point = food_point + [0.0, 0.0, 0.0]
      above_pickup = pickup_point + [0.0, 0.0, 0.2]

      rospy.logwarn("About to act")
      self.go_to_pose(above_pickup, self.quats["vert"])
      rospy.logwarn("picking up")
      self.play_recording("pickup", pickup_point + pickup_recording_offset)
      self.go_to_pose(pickup_point + [0.05, 0.0, 0.2], self.quats["carry"])
      
      dropoff_point = np.array([0.13, y, 0.03])
      above_dropoff = dropoff_point + [0.0, 0.0, 0.2] 

      self.go_to_pose(above_dropoff, self.quats["carry"])
      self.play_recording("dropoff", dropoff_point)
      self.jiggle()
    rospy.spin()

  def jiggle(self, jiggle_duration = 0.05):
    rospy.logwarn("About to jiggle in 1 seconds")
    rospy.sleep(1)
    # get the current joint state from ros
    joint_msg = rospy.wait_for_message("/joint_states", JointState)
    joint_angles = np.array(joint_msg.position)
    joint_names = joint_msg.name
    rospy.logwarn(joint_angles)
    should_ind = joint_names.index("shoulder_lift_joint")
    elbow_ind = joint_names.index("elbow_joint")
    wrist_ind = joint_names.index("wrist_1_joint")
    up_angles = np.array(joint_angles)
    up_angles[should_ind] = up_angles[should_ind] - 0.02
    up_angles[elbow_ind] = up_angles[elbow_ind] - 0.02
    up_angles[wrist_ind] = up_angles[wrist_ind] - 0.04
    rospy.logwarn(up_angles)
    joint_angles = joint_angles.tolist()
    up_angles = up_angles.tolist()
    
    # smaller numbers mean more jiggle
    rospy.logwarn("I'm jiggling!")
    self.set_joints_manually(up_angles, joint_names, 0.2)
    rospy.sleep(0.5) 
    for _ in range(6):
      self.set_joints_manually(joint_angles, joint_names, jiggle_duration)
      rospy.sleep(2*jiggle_duration) 
      self.set_joints_manually(up_angles, joint_names, jiggle_duration)
      rospy.sleep(2*jiggle_duration) 
    rospy.logwarn("Done jiggling")
  
  # go to the input position and orientation (with potential translation offset) 
  def go_to_named_pose(self, location_name, quaternion_name, offset = [0,0,0]):
    position = self.locations[location_name] + np.array(offset)
    quat = self.quats[quaternion_name]
    self.go_to_pose(position, quat)

  def go_to_pose(self, position, quat):
    # clear out any numpy-arrayness of position
    position = np.array(position)
    position = position.tolist()
    self.tracker.start_tracking_fixed_pose(position, quat)
    wl.wait(wl.State.MOVING_ARM) 

  def play_recording(self, recording_name, offset = [0,0,0]):
    # clear out any numpy-arrayness of offset
    offset = np.array(offset)
    offset = offset.tolist()
    recording_file = self.recording_files[recording_name]
    self.tracker.start_updating_target_to_pose(self.play_trajectory_topic, offset)
    self._play_trajectory(String(self.play_trajectory_topic), String(recording_file))
    wl.wait(wl.State.FOLLOWING_TRAJECTORY) 
  
  def set_joints_manually(self, joint_angles, joint_names, time):
    joint_angles_np = np.array(joint_angles)
    zeros_list = (joint_angles_np * 0).tolist()
    joint_angles_list = joint_angles_np.tolist()
    h = Header()
    h.stamp = rospy.Time.now()# Note you need to call rospy.init_node() before this will work. This needs to be any number in the future
    pos = joint_angles 
    vel = zeros_list 
    acc = zeros_list 
    eff = zeros_list 
    dur = rospy.Duration(time)
    point = JointTrajectoryPoint(pos, vel, acc, eff, dur)
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names 
    trajectory.points = [point]
    trajectory.header = h
    self.direct_joint_command_pub.publish(trajectory)

if __name__=="__main__":
  rospy.init_node('proper_plating_learner', anonymous=True)
  sd = ServingDemo()
