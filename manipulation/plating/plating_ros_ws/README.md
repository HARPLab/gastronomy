# gastronomy/manipulation/plating/plating_ros_ws

This plating_ros_ws folder is a catkin workspace to connect to a (possibly simulated) robot and to execute motions based on demonstrations that are used in the plating demo. It can be used with any robot that has a ROS joint position controller interface. It also includes code to project the 2D pixel location of a detected food item in the camera frame onto a known 3D plane (the table).

## Installation
In order to simulate the robot, please install Gazebo (http://gazebosim.org/tutorials?tut=ros_installing#Introduction)

Please make a copy of the folder `gastronomy/manipulation/plating\_ros\_ws`. Within your copy of that folder (and after installing ROS and setting up wstool and rosdep) please run the following commands:

The first set of commands will read the file plating\_ros\_ws/src/.rosinstall and will use wstool to copy third-party code used to connect to the UR5e robot. This code is located at https://github.com/dniewinski/universal\_robot.git and at https://github.com/dniewinski/universal\_robot.git. The second set of commands will install common ROS packages like MoveIt https://moveit.ros.org/install/ by using rosdep. The final set of commands compiles the catkin workspace
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
The following example executes plating motions using a simulated UR5e robot
```
roslaunch plating_demo ur5_default.launch sim:=true
```
The following example runs serving motions using a simulated Niryo One arm.

```
roslaunch plating_demo default.launch sim:=true
```
The following example uses vision and kinematic information to track the 3D position of a tomato slice on the table.

```
roslaunch plating_demo vision.launch
```
