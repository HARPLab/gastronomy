# gastronomy/hri

Overview
======
The goal of this project is to gather both passive and active social signals of humans in the setting of restaurants in order to model and determine their needs and the urgency of those needs over time. These metrics will be useful for monitoring restaurant efficiency, and helping robotics waitstaff to plan optimal service.

Below we give the tools we used to investigate Human-Robot Interaction in restaurant environments. This approach includes:
* Collecting videos of natural restaurant interactions from online webstreams.
* Automatically extracting features from these videos using OpenPose and object recognition, as well as noting events such as waiter visits.
* Creating human-generated labels for the state of users and groups of users over time.
* Building machine learning models to analyze trends and discover underlying patterns in these interations.
* Developing a simulator that can play out restaurant scenarios, permute them to add noise to these scenes, and export a CSV file of the physical structure of these activities for testing our machine learning techniques on.

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




