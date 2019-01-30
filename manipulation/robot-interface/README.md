# Robot Interface

This is a software package used for controlling and learning skills on the Franka Emika Panda Research Robot Arm and UR5e.

## Requirements

* A computer with the Ubuntu 16.04 Realtime Kernel and at least 1 ethernet port.
* ROS Kinetic 

## Computer Setup Instructions

1. The instructions for setting up a computer with the Ubuntu 16.04 Realtime Kernel from scratch are located here: [control pc ubuntu setup guide](docs/control_pc_ubuntu_setup_guide.md)
2. Instructions for setting up the computer specifically for Franka Robots is located here: [franka control pc setup guide](docs/franka_control_pc_setup_guide.md)
3. Instructions for setting up the computer specifically for the UR5e robots is located here: [ur5e control pc setup guide](docs/ur5e_control_pc_setup_guide.md)

## Installation

1. Clone Repo and its Submodules:

```bash
git clone --recurse-submodules https://github.com/iamlab-cmu/robot-interface.git
```
All directories below are given relative to `/robot-interface`.

2. Build LibFranka
```bash
bash make_libfranka.sh
```

3. Build iam_robolib
```bash
bash make_iam_robolib.sh
```
Once it has finished building, you should see an application named `main_iam_robolib` in the build folder.

4. Build ROS Node franka_action_lib
Make sure that you have installed ROS Kinetic already and have added the `source /opt/ros/kinetic/setup.bash` into your `~/.bashrc` file.

```bash
cd src/catkin_ws
catkin_make
```
Once catkin_make has finished there should be a build and devel folder in the catkin_ws folder.

5. Install FrankaPy
```bash
cd src/catkin_ws/src/franka_action_lib
pip install -e .
```

## Running on the Franka Robot

1. Make sure that both the user stop and the brakes of the Franka robot have been unlocked in the Franka Desk GUI.
2. Open up 3 separate terminals.

Terminal 1:
```bash
bash run_iam_robolib.sh
```

Terminal 2:
```bash
source src/catkin_ws/devel/setup.sh
roslaunch franka_action_lib franka_ros_interface.launch
```

Terminal 3:
```bash
source src/catkin_ws/devel/setup.sh
```
Now in terminal 3 you can run any of the scripts in `src/catkin_ws/src/examples` and `src/catkin_ws/src/scripts`.

See `src/catkin_ws/src_scripts/reset_arm.py` for an example of how to use the `FrankaPy` python package.
