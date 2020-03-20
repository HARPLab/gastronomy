# A PyTorch implementation of a YOLO v3 Object Detector
Python wrapper for Darknet/YOLOv3.

## References
This repository is mainly based on [this repository](https://github.com/ayooshkathuria/pytorch-yolo-v3) by Ayoosh Kathruria.

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2).

If training a custom data set, it is recommended to obtain trained weights by using the [original code](https://pjreddie.com/darknet/). The original code was implemented in C and is much faster and efficient than using python. This code is being used to retreive the bounding box information through python. If you would like to retrieve the bounding box information using the C code, I would recommend using [this](https://github.com/leggedrobotics/darknet_ros) ROS implementation instead

# Installation

## Optional: Create and source virtual environment

### Create virtual environment for this application. Note that Python3.6 is the expected Python version used for this library.
### Replace <path to virtual env>
`virtualenv --system-site-packages -p /usr/bin/python3.6 <path to virtual env>`

### Install some packages
`pip install --upgrade setuptools wheel tqdm twine`

### Activate virtual environment
`source <path to virtual env>/bin/activate`

## Install package with requirements (cd to directory containing setup.py)
`pip install .`

### IMPORTANT: This code is assuming you have already pip installed the rnns package from HARPLab/gastronomy/manipulation/plating/RNNs

### If you are having issues with the matplotlib not properly displaying when using the virtual environment, try creating the virtual environment using the below command instead:
`virtualenv -p /usr/bin/python3.6 <path to virtual env>`
### and proceed with the other instructions as documented.
Matplotlib has had issues when being used from within a python virtual environment, see [here](https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/)
or [here](https://github.com/pypa/virtualenv/issues/609).