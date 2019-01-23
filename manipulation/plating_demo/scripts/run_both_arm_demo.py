#!/usr/bin/env python
import rospy
import rospkg
import time
import tf

import wait_logic as wl

import feedbot_trajectory_logic.tracker_interface as tracker
import numpy as np
from learn_trajectory.srv import PlayTrajectory
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Quaternion
import transforms3d as t3d


class SpoonFeeder:
  def __init__(self):
    # here, unlike usual, we store as w,x,y,z
    self.rot_to_domus = [1,0,0,0]
    # quaternion is defined in order x,y,z,w
    self.defaultQuat = Quaternion(0.5, 0.5, 0.5, 0.5)
    self.tracker = tracker.TrackerInterface(self.defaultQuat)
    self.trackertoo = tracker.TrackerInterface(self.defaultQuat, '/domusromus/update_pose_target')
    self.play_trajectory_topic = "trained_poses"
    self._play_trajectory = rospy.ServiceProxy("play_trajectory", PlayTrajectory)
    # the static transform publisher wasn't working robustly enough for my tastes
    self.br = tf.TransformBroadcaster()
    rospy.Timer(rospy.Duration(1), self.broadcastTransform)

    self.recording_files = { "pick_up" : "acquire_tomato.txt",
                             "deposit" : "deposit_tomato.txt"}

    pickup_list = [
        np.array([-0.04,0.05,0]),
        np.array([0.04,0.05,0]),
        np.array([-0.04,0,0]),
        np.array([0.04,0,0])]
    deposit_list = [
        np.array([0.03,-0.15,0]),
        np.array([-0.03,-0.05,0]),
        np.array([-0.03,-0.15,0]),
        np.array([0.03,-0.05,0])]
    
    niryo_start = self.get_niryo_start("pick_up")
    domus_end= self.get_domus_end("pick_up")

    for i in range(4):
      # plate coordinates are those of niryo (blue). z should be 0.
      plate_placement = deposit_list[i]
      pickup_placement = pickup_list[i]
      self.move_niryo_to_pose(niryo_start[0:3]+[0,0,0.10] + pickup_placement, [0.5,0.5,0.5,0.5])
      self.move_niryo_to_pose(niryo_start[0:3]+[0,0,0.10] + pickup_placement, niryo_start[3:7])
      self.prepare_and_play_recording("pick_up", ([0,0,0] + pickup_placement).tolist(), [0,0,0])
      self.prepare_and_play_recording("deposit", ([0,0,0.03] + plate_placement).tolist(), ([-0.01,-0.035,-0.01] - plate_placement).tolist())
      self.move_domus_to_pose(domus_end[0:3] + [-0.02,-0.04,0.04], domus_end[3:7])


  def prepare_and_play_recording(self, recording_name, offset_niryo = [0,0,0], offset_domus = [0,0,0]):
    niryo_start = self.get_niryo_start(recording_name)
    self.move_niryo_to_pose(niryo_start[0:3] + offset_niryo, niryo_start[3:7])
    domus_start = self.get_domus_start(recording_name)
    self.move_domus_to_pose(domus_start[0:3] + offset_domus, domus_start[3:7])
    self.play_recording(recording_name, offset_niryo, offset_domus)

  def play_recording(self, recording_name, offset_niryo = [0,0,0], offset_domus = [0,0,0]):
    self.tracker.start_updating_target_to_pose(self.play_trajectory_topic, offset_niryo)
    self.trackertoo.start_updating_target_to_pose("/domusromus/"+self.play_trajectory_topic,offset_domus)
    self._play_trajectory(String(self.play_trajectory_topic), String(self.recording_files[recording_name]))
    wl.wait(wl.State.FOLLOWING_TRAJECTORY) 
  
  def move_niryo_to_pose(self, point, quat):
    # the msg should be x,y,z,w
    # that matches the ordering in my stored textfiles
    rot_quat_transform3d  = t3d.quaternions.qmult(self.rot_to_domus,[quat[3], quat[0], quat[1], quat[2]])
    rot_quat = rot_quat_transform3d[1],rot_quat_transform3d[2],rot_quat_transform3d[3],rot_quat_transform3d[0]
    quatMsg = Quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
    self.tracker.start_tracking_fixed_pose(point, quatMsg)
    with wl.DistanceBasedWaitLogic("/distance_to_target") as waitLogic:
      waitLogic.wait()
  
  def move_domus_to_pose(self, point, quat):
    rot_quat_transform3d  = t3d.quaternions.qmult(self.rot_to_domus,[quat[3], quat[0], quat[1], quat[2]])
    rot_quat = rot_quat_transform3d[1],rot_quat_transform3d[2],rot_quat_transform3d[3],rot_quat_transform3d[0]
    quatMsg = Quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
    self.trackertoo.start_tracking_fixed_pose(point, quatMsg)
    with wl.DistanceBasedWaitLogic("/domusromus/distance_to_target") as waitLogic:
      waitLogic.wait()

  def get_niryo_start(self, recording_name):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("learn_trajectory")
    motion_data = np.loadtxt(package_path + "/data/" + self.recording_files[recording_name],  delimiter=",")
    return(motion_data[0,8:15])
  
  def get_domus_start(self, recording_name):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("learn_trajectory")
    motion_data = np.loadtxt(package_path + "/data/" + self.recording_files[recording_name],  delimiter=",")
    return(motion_data[0,1:8])
  def get_domus_end(self, recording_name):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("learn_trajectory")
    motion_data = np.loadtxt(package_path + "/data/" + self.recording_files[recording_name],  delimiter=",")
    return(motion_data[-1,1:8])


  def broadcastTransform(self, event):
    #rospy.logwarn("broadcasting tf")
    self.br.sendTransform([0.63,0.0,0.0],
                          [0.0,0.0,1.0,0.0],
                          rospy.Time.now(),
                          "base_link",
                          "deep_doop_base_link")

if __name__=="__main__":
  rospy.init_node('spoon_feeder', anonymous=True)
  s = SpoonFeeder()
