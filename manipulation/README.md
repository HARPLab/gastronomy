# gastronomy/manipulation

The manipulation repository consists of 3 packages.

## Robot-Interface

The Robot-Interface package in the robot-interface folder was created by the Intelligent Manipulation Lab to control the Franka Panda Research Robot at 1kHz. 

## DMP

The DMP package in the dmp folder was created by the Intelligent Manipulation Lab to train new dmp skills using trajectories recorded by the robot-interface package above.

## Plating

The Plating package in the plating folder was created by Travers Rhodes to learn trajectories from demonstrations that are used in the plating demo. It can be used with any robot that has a ROS joint controller interface. It includes code to segment demonstrations using a changepoint detection algorithm, and it contains code to translate and execute those trajectories on a robot.
