#!/bin/bash

#roslaunch realsense2_camera rs_rgbd.launch

roslaunch franka_action_lib rs_three_devices.launch serial_no_camera1:=827112071890 serial_no_camera2:=827112070582 serial_no_camera3:=819612071257
