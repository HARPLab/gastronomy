#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped


rospy.init_node("dummy_publisher")

point_stamped_topic = rospy.get_param('~mouth_point_topic')#point_topic + "Stamped"
#pub = rospy.Publisher(point_topic, Point, queue_size=10)
pubStamped = rospy.Publisher(point_stamped_topic, PointStamped, queue_size=10)
h = Header()
h.stamp = rospy.Time.now()
times = np.array(range(100))
x_dist = 0.0
y_dist = 0.195
z_dist = 1.15

poses = [[ x_dist + 0.1 * np.sin(t), y_dist + 0.1 * np.cos(t), z_dist] for t in times]

while True:
  for pose in poses+poses+poses:
    mesg = Point(pose[0], pose[1], pose[2])
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = "camera_rgb_optical_frame"
    mesgStamped = PointStamped(h,mesg)
#    pub.publish(mesg)
    pubStamped.publish(mesgStamped)
    rospy.sleep(4)
