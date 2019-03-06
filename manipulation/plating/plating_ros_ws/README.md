# gastronomy/manipulation/plating/plating_ros_ws

This plating_ros_ws folder is a catkin workspace to connect to a (possibly simulated) robot and to execute motions based on demonstrations that are used in the plating demo. It can be used with any robot that has a ROS joint position controller interface. It also includes code to project the 2D pixel location of a detected food item in the camera frame onto a known 3D plane (the table).

## Installation
In order to simulate the robot, please install Gazebo (http://gazebosim.org/tutorials?tut=ros_installing#Introduction)

In order to connect to the RealSense camera, please install the RealSense SDK available at https://realsense.intel.com/sdk-2/

Please make a copy of the folder `gastronomy/manipulation/plating/plating_ros_ws`. Within your copy of that folder (and after installing ROS and setting up wstool and rosdep) please run the following commands:

The `wstool up` command will read the file plating_ros_ws/src/.rosinstall and will use wstool to copy third-party code used to connect to the UR5e robot. This code to connect to a UR5 is located at https://github.com/dniewinski/universal_robot.git and at https://github.com/dniewinski/ur_modern_driver.git. Additionally, the `wstool up` command installs the ros drivers for the intel realsense camera we attached to the end-effector of the robot.  The `rosdep install ...` command will install common ROS packages like MoveIt https://moveit.ros.org/install/ by using rosdep. The final set of commands compiles the catkin workspace
```
cd plating_ros_ws/src
source /opt/ros/kinetic/devel/setup.bash
wstool up
cd ..
rosdep update
rosdep install --from-paths src --ignore-src
catkin_make
source devel/setup.bash
```

## Execution Examples
The following example executes plating motions using vision on a simulated UR5e robot on a table with a picture of food. 
```
roslaunch plating_demo ur5_vision_demo.launch sim:=true
```

To run the code on a real UR5, please first turn on the UR5, set it to "remote control", and run the following command

```
roslaunch plating_demo ur5_vision_demo.launch sim:=false robot_ip:=192.168.1.10
```
