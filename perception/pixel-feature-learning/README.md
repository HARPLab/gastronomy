## Pixel feature learning for food
This repository contains code for learning pixel feature in the food domain. It is adapted from the dense object net code (https://github.com/RobotLocomotion/pytorch-dense-correspondence). It learns pixel embedding via learning pixel-level correspondence through two data settings:
- images taken from multiple viewpoints in simulation (multi-view setting)
- images augmented from realistic food images via affine transformation, pixel intensity change, color shift, etc.. (image augmentation setting)

### 1. Setup
Under current (```pixel-feature-learning```) directory:
```
git clone https://github.com/warmspringwinds/pytorch-segmentation-detection.git
mkdir pdc

sudo apt install --no-install-recommends wget libglib2.0-dev libqt4-dev libqt4-opengl libx11-dev libxext-dev libxt-dev mesa-utils libglu1-mesa-dev python-dev python-lxml python-numpy python-scipy python-yaml
sudo apt-get install libyaml-dev python-tk ffmpeg

cd ~/virtualenvs
virtualenv --system-site-packages pci_venv 
source ~/virtualenvs/pci_venv/bin/activate

# Edit config/setup_environment.sh if needed
source config/setup_environment.sh

pip install matplotlib testresources pybullet

pip install visdom requests scipy imageio scikit-image tensorboard sklearn opencv-contrib-python tensorboard_logger tensorflow

pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl torchvision torchnet 

pip install jupyter opencv-python plyfile pandas tensorflow

# optional
pip install visualization
install meshpy via source
```

### 3. Pipeline
Source these two file before running:
```
source ~/virtualenvs/pci_venv/bin/activate
source ~/git/pytorch-dense-correspondence/config/setup_environment.sh
```
Note: For the following codes please change the paths inside.

#### 3.1 Generate Data for Multi-view setting
Code for generating data for the multi-view setting is under folder ```data_generation```.

#### 3.2 Training
Edit/check these before training:
* ```get_default_K_matrix``` in ```dense_correspondence/correspondence_tools/correspondence_finder.py```
* Training config file in ```config/dense_correspondence/training/```
* Dataset config file in ```config/dense_correspondence/dataset/```
* training file. e.g. ```training_tutorial_toy.py```

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