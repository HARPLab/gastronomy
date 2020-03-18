#!/usr/bin/env python

import time
import signal
import sys
import serial

import rospy
from magnetic_stickers.msg import MagneticData

def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if raw_input("\nAre you sure you want to quit? (y/n)> ").lower().startswith('y'):
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, exit_gracefully)



def run_main():

    pub = rospy.Publisher('/magnetic_stickers/magnetic_data', MagneticData, queue_size=1)
    rospy.init_node('magnetic_stickers_publisher')

    r = rospy.Rate(100) # 60hz

    while not rospy.is_shutdown():
        with serial.Serial('/dev/ttyACM0', 250000, timeout=1) as ser:
            line = ser.readline()  
            try:
                magnetic_data = [float(i) for i in line.split(' ')]

                if(len(magnetic_data) == 8):
                    magnetic_data_msg = MagneticData()
                    magnetic_data_msg.header.stamp = rospy.Time.now()
                    magnetic_data_msg.header.frame_id = "/right_fingertip"
                    magnetic_data_msg.ref_x = magnetic_data[0]
                    magnetic_data_msg.ref_y = magnetic_data[1]
                    magnetic_data_msg.ref_z = magnetic_data[2]
                    magnetic_data_msg.ref_t = magnetic_data[3]
                    magnetic_data_msg.x = magnetic_data[4]
                    magnetic_data_msg.y = magnetic_data[5]
                    magnetic_data_msg.z = magnetic_data[6]
                    magnetic_data_msg.t = magnetic_data[7]

                    pub.publish(magnetic_data_msg)
                    r.sleep()
            except:
                print(line)

if __name__ == '__main__':
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)
    run_main()