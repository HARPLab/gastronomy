# UR5e Control PC Setup Guide

This guide describes how to setup the software for the computer to specifically control the UR5e Robot Arm. It assumes that the [control pc ubuntu setup guide](./control_pc_ubuntu_setup_guide.md) was successfully completed.

## Setting Up the Robot
If this is the first time you are using the robot, please follow the UR5e Robot Setup instructions located in the binder that accompanies the robot and install the gripper on the robot. 

## Editing the Ethernet Network Configuration
1. Insert the ethernet cable from the UR5e Control box to the Control PC.
2. Turn on the UR5e using the teach pendant.
3. Go to Edit Connections in the Ubuntu Network Connections Menu.
4. Select the Ethernet connection that corresponds to the port that you plugged the ethernet cable into and then click edit.
5. Go to the IPv4 Settings Tab and switch from Automatic (DHCP) to Manual.
6. Add an address of 192.168.1.11 with netmask 255.255.255.0 and 0.0.0.0 for the Gateway then click save.
7. Check to see if you can ping 192.168.1.10 from any terminal.
8. If any issues arise, refer to http://wiki.ros.org/universal_robot/Tutorials/Getting%20Started%20with%20a%20Universal%20Robot%20and%20ROS-Industrial

## UR5e ROS Installation
1. Run the following terminal command to install the universal-robot package:
`sudo apt-get install ros-kinetic-universal-robot`
2. Create a catkin_ws according to instructions [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace).
3. Go to the src directory and clone the following repositories.
	```
	cd ~/catkin_ws/src
	git clone https://github.com/dniewinski/universal_robot
	git clone https://github.com/dniewinski/ur_modern_driver.git
	```
	Make sure that you are on branch `ur_e` for the universal_robot package and branch `kinetic-devel` for the ur_modern_driver package.
4. Go back to the catkin_ws directory: `cd ~/catkin_ws`.
5. Install ROS dependencies for both packages using the following command: 
	`rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y`
6. Catkin make both packages with the command `catkin_make`
7. Make sure the UR5e is set to remote control. You can again consult http://wiki.ros.org/universal_robot/Tutorials/Getting%20Started%20with%20a%20Universal%20Robot%20and%20ROS-Industrial for how to do that.
8. Open 2 terminals and run `source ~/catkin_ws/devel/setup.bash` in both.
9. Run `roslaunch ur_modern_driver ur5e_bringup.launch robot_ip:=192.168.1.10` in one of the two terminals.
10. Run `rosrun ur_driver test_move.py` in the other and the robot should move.
11. If any issues occur, additional information is located here: http://wiki.ros.org/universal_robot/Tutorials/Getting%20Started%20with%20a%20Universal%20Robot%20and%20ROS-Industrial