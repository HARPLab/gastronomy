# gastronomy/manipulation

The manipulation repository consists of 5 packages.

## Robot-Interface

The Robot-Interface package in the robot-interface folder was created by the Intelligent Manipulation Lab to control the Franka Panda Research Robot at 1kHz. Installation instructions are located in the README.md file in the robot-interface folder. 

## DMP

The DMP package in the dmp folder was created by the Intelligent Manipulation Lab to train new dmp skills using trajectories recorded by the robot-interface package above.

## DMP Weight Clustering and GAN

## Magnetic Stickers

The magnetic stickers folder contains the code needed to run the localization and forcefeedback of the magnetic stickers according to the paper that is also located in the repository. The instructions to install and run everything is located in the folder's README.md. 

## Plating (No longer supported)

The Plating package in the plating folder was created by Travers Rhodes to learn trajectories from demonstrations that are used in the plating demo. It can be used with any robot that has a ROS joint controller interface. It includes code to segment demonstrations using a changepoint detection algorithm, and it contains code to translate and execute those trajectories on a robot.
