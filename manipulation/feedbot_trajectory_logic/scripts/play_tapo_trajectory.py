#!/usr/bin/env python
import numpy as np
import rospkg
import rospy

import transforms3d as t3d

from std_msgs.msg import Header, Bool
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

def load_position_list(filename):
  rospack = rospkg.RosPack()
  ada_tut_path = rospack.get_path("feedbot_trajectory_logic")
  motion_data = np.loadtxt(ada_tut_path + "/data/" + filename, skiprows=1, delimiter=",")
  sampled_position_motion_data = get_sampled_pos_data(motion_data)
  return(sampled_position_motion_data)

def get_sampled_pos_data(rawData):
  datShape = rawData.shape
  sampleIndices = np.arange(0,datShape[0],1)
  # time, px, py, pz, ox, oy, oz
  datIndices = np.array([0,7,8,9,10,11,12])
  return(rawData[sampleIndices,:][:,datIndices])

# in tapo's data, ox is row[10], oy is row[11], and oz is row[12]
def get_quat(ox, oy, oz):
  quat = t3d.euler.euler2quat(oz, oy, ox, 'rzyx')
  return quat

def publish_poses(poseFile, pose_topic):
  pos_list = load_position_list(poseFile)
  starttime = rospy.Time.now().to_sec() - 2.5 # skip the first chunk of time
  slowdown_factor = 5
  lasttime = pos_list[-1,0] * slowdown_factor # in seconds
  # only run the first 2.8/4 of the trajectory
  lasttime = lasttime  * 2.8/4.0
  curTime = rospy.Time.now().to_sec() - starttime
  target_pub = rospy.Publisher(pose_topic, Pose, queue_size=10)
  pos_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size=10)
  r = rospy.Rate(10) # publish at max 10hz
  while curTime < lasttime and not rospy.is_shutdown():
    #print(curTime, lasttime)
    curRow = np.argmax((pos_list[:,0]  * slowdown_factor) > curTime) - 1
    #print((pos_list[:,0]  * slowdown_factor) < curTime)
    #print(curRow)
    curQuat = get_quat(pos_list[curRow,4],
                       pos_list[curRow,5],
                       pos_list[curRow,6])
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = "world"
    pos = pos_list[curRow,1:4]
    curRot = t3d.quaternions.quat2mat(curQuat)
    pos2 = pos - curRot[:,1] * 0.3

    #rospy.logwarn(curRot)

    curQuat = t3d.quaternions.qmult(curQuat,[0,0,1,0])
    # I wish I had documented the black magic I did here to estimate
    # the prong location of the fork
    pose = Pose(Point(0.15 + pos2[0], 0.15-pos2[2], pos2[1]),
                Quaternion(curQuat[1],curQuat[2],curQuat[3],curQuat[0]))

    poseStamped = PoseStamped(h,pose)
    target_pub.publish(pose)
    pos_pub.publish(poseStamped)
    r.sleep()
    curTime = rospy.Time.now().to_sec() - starttime
  rospy.logwarn("Done playing trajectory. Now we publish the fact that we're done to food_acquired")
  food_acquired_pub = rospy.Publisher("/food_acquired", Bool, queue_size=10)
  # https://github.com/ros/ros_comm/issues/176
  rospy.sleep(1)
  food_acquired_pub.publish(Bool(True))

if __name__=="__main__":
  rospy.init_node("simulate_spoon")
  while not rospy.is_shutdown():
    for i in range(2,3):
      poseFile = "subject11_potato_salad/%d.csv"%i
      #poseFile = "subject10_banana/%d.csv"%i
      #poseFile = "subject11_noodle/%d.csv"%i
      publish_poses(poseFile, "/Tapo/example_poses")
