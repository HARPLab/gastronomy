# gastronomy/hri

Overview
======

The goal of this project is to gather both passive and active social signals of humans in the setting of restaurants in order to model and determine their states of Neediness (how much they would like help/attention) and Interruptibility (whether they are open to being helped) over time. These metrics will be useful for monitoring restaurant efficiency, and helping robotics waitstaff to plan optimal service.

This code seeks to take input in the form of 2D RGB video from one angle in a restaurant, both live and archived. Features are extracted from these by OpenPose and OpenFace. These are then combined to provide more complex relationships between features, and used to classify customers into different categories of Neediness and Interruptibility.

Library Installation
======

OpenFace
---------
This library can originally be found at https://github.com/TadasBaltrusaitis/OpenFace. It should be cloned into the hri directory, and all standard install instructions followed.

An overview of its capabilities and installation instructions for various environments can be found at https://github.com/TadasBaltrusaitis/OpenFace/wiki

OpenPose
---------
This library can originally be found at https://github.com/CMU-Perceptual-Computing-Lab/openpose. It should be cloned into the hri directory, and all standard install instructions followed.

An overview of its capabilities and installation instructions for various environments can be found at https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/README.md

Running Demos
======
Once these libraries and their dependencies are properly installed, they can be used to analyze videos or live webcam footage.


### The directory structure is as follows:
  hri
    ├── ...
    ├── customer-analysis
    ├── customer-simulator
    ├── gastronomy_web_cam_analysis
    ├── sony-hri-2019                  # All HRI code and samples from 2019
    │   ├── OpenFace         # OpenFace source code and compiled libraries
    │   ├── openpose         # OpenPose source code and compiled libraries
    │   └── sony-hri         # HRI deliverables
    │          ├── example-inputs         # Video samples for analysis
    │          ├── outputs                # Output directories
    │          └── demos                  # Scripts to run analysis on videos or webcam
    │   
    └── ...

> For all video names passed to demo scripts, the short name of any video contained in the example-inputs folder is all that is required, not a full file path.




