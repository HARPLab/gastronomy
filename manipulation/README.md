# gastronomy/manipulation

The manipulation repository consists of 7 packages.

## Robot-Interface

The Robot-Interface package in the robot-interface folder was created by the Intelligent Manipulation Lab to control the Franka Panda Research Robot at 1kHz. Installation instructions are located in the README.md file in the robot-interface folder. 

## DMP

The DMP package in the dmp folder was created by the Intelligent Manipulation Lab to train new dmp skills using trajectories recorded by the robot-interface package above.

## DMP Weight Clustering and GAN

The DMP weight clustering and GAN folder contains work by Ami Sawhney that is detailed in the Cutting section of the Sony-CMU Gastronomy Project Y2 Deliverables document.

## Action Relation

The action relation folder contains work by Mohit Sharma that is detailed in the Skill Preconditions section of the Sony-CMU Gastronomy Project Y2 Deliverables document.

## Plating

The plating folder contains code by Steven Lee that is detailed in the plating section of the Sony-CMU Gastronomy Project Y2 Deliverables document. In addition, it contains previous code by Travers Rhodes to learn trajectories from demonstrations that are used in the plating demo. It can be used with any robot that has a ROS joint controller interface. It includes code to segment demonstrations using a changepoint detection algorithm, and it contains code to translate and execute those trajectories on a robot.

## Magnetic Stickers

The magnetic stickers folder contains the code by Kevin Zhang needed to run the localization and force feedback of the magnetic stickers according to the paper that is also located in the repository. The instructions to install and run everything is located in the folder's README.md. 

## Azure Kinect Calibration

The Azure Kinect Calibration folder contains the calibration scripts needed in order to calibrate the Azure Kinect cameras for use in most of the other packages. It depends on a modified perception package located here: https://github.com/iamlab-cmu/perception. 
