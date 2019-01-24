# gastronomy/manipulation/plating

This folder contains two sets of code useful for plating food.

## changepoint_detection 

This folder contains our Python implementation of a changepoint detection algorithm made available in Matlab by Ryan Turner from
```
@INPROCEEDINGS{Turner2009,
author = {Ryan Turner and Yunus Saat\c{c}i and Carl Edward Rasmussen},
title = {Adaptive Sequential {B}ayesian Change Point Detection},
booktitle = {Temporal Segmentation Workshop at NIPS 2009},
year = {2009},
editor = {Zaid Harchaoui},
address = {Whistler, BC, Canada},
month = {December},
url = {http://mlg.eng.cam.ac.uk/rdturner/BOCPD.pdf}
}
```

## plating_ros_ws

The plating_ros_ws folder is a catkin workspace created by Travers Rhodes to connect to a robot and to execute from demonstrations that are used in the plating demo. It can be used with any robot that has a ROS joint position controller interface.
