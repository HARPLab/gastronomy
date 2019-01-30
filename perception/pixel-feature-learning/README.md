# Pixel feature learning for food
This repository contains code for learning pixel feature in the food domain. It is adapted from the Dense Object Net code (https://github.com/RobotLocomotion/pytorch-dense-correspondence). It learns pixel embedding via learning pixel-level correspondence through two data settings:
- images taken from multiple viewpoints in simulation (multi-view setting)
- images augmented from realistic food images via affine transformation, pixel intensity change, color shift, etc.. (image augmentation setting)

For a more detailed idea of the structure of this codebase, please refer to the official tutorial available in the Dense Object Net repository.

### 1. Setup
The following setup procedure should provide you with most dependencies, but please install necessary dependencies in case the following is not complete.

Under current (```pixel-feature-learning```) directory:
```
git clone https://github.com/warmspringwinds/pytorch-segmentation-detection.git
mkdir pdc

sudo apt install --no-install-recommends wget libglib2.0-dev libqt4-dev libqt4-opengl libx11-dev libxext-dev libxt-dev mesa-utils libglu1-mesa-dev python-dev python-lxml python-numpy python-scipy python-yaml
sudo apt-get install libyaml-dev python-tk ffmpeg

# make virtualenv if you like
cd ~/virtualenvs
virtualenv --system-site-packages pci_venv 
source ~/virtualenvs/pci_venv/bin/activate

# Edit config/setup_environment.sh if needed
source config/setup_environment.sh

pip install matplotlib testresources pybullet visdom requests scipy imageio scikit-image tensorboard sklearn opencv-contrib-python tensorboard_logger tensorflow

pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl torchvision torchnet 

pip install jupyter opencv-python plyfile pandas
```

### 2. File structure
- ```config```: this folder contains all the configuration you would need to specify for training a model.
- ```docker```: you can ignore this since we are not using docker.
- ```dense_correspondence```: main directory for training code.
- ```modules```: here are some utility functions required for training the model.
- ```pdc``` and ```data```: this folders are where we put data. You can look into the training code to see example data path.


### 3. Pipeline
Source these two file before running:
```
source ~/virtualenvs/pci_venv/bin/activate
source ~/git/pytorch-dense-correspondence/config/setup_environment.sh
```
Note: For the following codes please change all the paths in the codes according to your setting.

#### 3.1 Data

##### Generate Data for Multi-view setting
Code for generating data for the multi-view setting is under folder ```data_generation```. Either ```generate_demo_data.py``` or ```generate_training_data.py``` is a complete data generation pipeline. Note that you need to manually prepare .obj files that you want to take picture of, and specify the data path accordingly.

##### Data for image augmentation setting
You do not need to do any pre-processing for this type of data.

#### 3.2 Training
Edit/check these before training:
* ```get_default_K_matrix``` in ```dense_correspondence/correspondence_tools/correspondence_finder.py``` This is the camera intrinsic matrix computed based on your camera setting (for multi-view setting only).
* Training config file in ```config/dense_correspondence/training/```
* Dataset config file in ```config/dense_correspondence/dataset/```
* training file. e.g. ```training_tutorial_cooking.py```

The followings are two example files which serve as templates that you can use for either data settings. They are essentially just a wrapper, can share similar training pipeline inside.

For the multi-view setting:
```
cd dense_correspondence/training
python training_tutorial_cooking.py
```

For the image augmentation setting:
```
cd dense_correspondence/salad_training
python run_training.py
```

#### 3.3 Demo
After you have trained the model, you can use the resulted pixel features for ingredient type/configuration retrieval simply using nearest neighbour search.
