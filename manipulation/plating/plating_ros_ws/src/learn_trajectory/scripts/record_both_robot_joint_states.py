#!/usr/bin/env python
import rospy
import time
import rospkg
from sensor_msgs.msg import JointState 

domusromus = None

# take in a message msg and a file to write to f
def callback(msg, f):
  global domusromus
  if domusromus:
    # https://stackoverflow.com/questions/44778/how-would-you-make-a-comma-separated-string-from-a-list-of-strings
    f.write(str(float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs)/1000000000) 
         + ',' + ','.join(map(str, msg.position))
         + ',' + ','.join(map(str, domusromus)) + "\n")

def callbacktwo(msg, f):
  global domusromus
  domusromus = msg.position

def main():
  rospy.init_node('joint_state_listener', anonymous=True)
  timestr = time.strftime("%Y%m%d-%H%M%S")
  r = rospkg.RosPack()
  path = r.get_path("learn_trajectory")
  recording_file = path + "/data/joints_data_both.txt"
  with open(recording_file, "w+") as f:
    rospy.Subscriber("/joint_states", JointState, callback, (f))
    rospy.Subscriber("/domusromus/joint_states", JointState, callbacktwo, (f))
    rospy.spin();

if __name__ == '__main__':
    try:
       main()
    except rospy.ROSInterruptException:
       pass
