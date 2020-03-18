#!/usr/bin/env python

import rospy
from magnetic_stickers.msg import MagneticData

def callback(data):
    rospy.loginfo("[" + str(data.ref_x) + "," + str(data.ref_y) + "," + str(data.ref_z) + "," + str(data.ref_t) + 
    			  str(data.x) + "," + str(data.y) + "," + str(data.z) + "," + str(data.t) + "]")

def run_main():

	sub = rospy.Subscriber('/magnetic_stickers/magnetic_data', MagneticData, callback)
	rospy.init_node('magnetic_stickers_subscriber')

	rospy.spin()

if __name__ == '__main__':
    run_main()