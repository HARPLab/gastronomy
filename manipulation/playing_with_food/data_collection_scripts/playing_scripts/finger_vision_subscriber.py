#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='video file to store recording to')
args = parser.parse_args()

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    import cv2  # Make sure cv2 is loaded before it is used in the callback
    assert cv2  # avoid "imported but unused" message (W0611)

    rospy.init_node('finger_vision_subscriber')

    cv_bridge = CvBridge()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    finger_vision_1 = cv2.VideoWriter(args.filename+'_1.avi', fourcc, 30.0, (640,480))
    finger_vision_2 = cv2.VideoWriter(args.filename+'_2.avi', fourcc, 30.0, (640,480))

    def callback1(msg):
        try:
            cv_image = cv_bridge.imgmsg_to_cv2(msg)
            rgb_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            finger_vision_1.write(rgb_cv_image)
        except CvBridgeError as e:
            print(e)

    def callback2(msg):
        try:
            cv_image = cv_bridge.imgmsg_to_cv2(msg)
            rgb_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            finger_vision_2.write(rgb_cv_image)
        except CvBridgeError as e:
            print(e)

    finger_vision_1_sub = rospy.Subscriber('/finger_vision1/image_raw', Image, callback1)
    finger_vision_2_sub = rospy.Subscriber('/finger_vision2/image_raw', Image, callback2)

    rospy.spin()

except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    finger_vision_1.release()
    finger_vision_2.release()
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))