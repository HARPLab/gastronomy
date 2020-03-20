# Robot Interface

This is a software package used for controlling and learning skills on the Franka Emika Panda Research Robot Arm and UR5e.

## Requirements

* A computer with the Ubuntu 16.04 Realtime Kernel and at least 1 ethernet port.
* ROS Kinetic 
* [Protocol Buffers](https://github.com/protocolbuffers/protobuf)

## Computer Setup Instructions

1. The instructions for setting up a computer with the Ubuntu 16.04 Realtime Kernel from scratch are located here: [control pc ubuntu setup guide](docs/control_pc_ubuntu_setup_guide.md)
2. Instructions for setting up the computer specifically for Franka Robots is located here: [franka control pc setup guide](docs/franka_control_pc_setup_guide.md)
3. Instructions for setting up the computer specifically for the UR5e robots is located here: [ur5e control pc setup guide](docs/ur5e_control_pc_setup_guide.md)

## Install ProtoBufs

1. Read installation instructions here https://github.com/protocolbuffers/protobuf/blob/master/src/README.md.

2. We use both C++ and Python versions of protobufs so you would need to install Protobufs from source. In short you will have to do the following. NOTE: However, make to read protobuf installation instructions once.

Do `nproc` to find out how many cores you have, and use that as the `N` number in the `make` command below:

```shell
sudo apt-get install autoconf automake libtool curl make g++ unzip
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-all-3.11.4.zip
unzip protobuf-all-3.11.4.zip
cd protobuf-3.11.4
./configure
make -jN
make check -jN
sudo make install
sudo ldconfig
```

## Installation

1. Clone Repo and its Submodules:

   ```bash
   git submodule update --init --recursive
   ```
All directories below are given relative to `/robot-interface`.

2. Build LibFranka
   ```bash
   bash ./bash_scripts/make_libfranka.sh
   ```

3. Build iam_robolib
   ```bash
   bash ./bash_scripts/make_iam_robolib.sh
   ```
   Once it has finished building, you should see an application named `main_iam_robolib` in the build folder.

4. Build ROS Node franka_action_lib

   Make sure that you have installed ROS Kinetic already and have added the `source /opt/ros/kinetic/setup.bash` into your `~/.bashrc` file.

   ```bash
   cd catkin_ws
   catkin_make
   ```
Once catkin_make has finished there should be a build and devel folder in the catkin_ws folder.

5. Install FrankaPy. First source into your virtualenv or conda env (should be Python 3.6). Then:
   ```bash
   cd catkin_ws/src/franka_action_lib
   pip install -e .
   pip install rospkg
   ```
   
6. To make the protobufs on the Python (FrankaPy) side (i.e. on workstation-PC) use the following script `<repo>/catkin_ws/src/franka_action_lib/frankapy/proto/compile_proto.sh`.

## Setting Up SSH Key to Control PC
1. Generate an ssh key by executing the following commands or reading the instructions here: https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   [Press enter]
   [Press enter]
   [Press enter]
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
   ```
2. Upload your public ssh key to the control pc.
   1. In a new terminal, ssh to the control PC.
      ```bash
      ssh iam-lab@iam-[control-pc-name]
      Input password to control-pc.
      ```
   2. Use your favorite text editor to open the authorized_keys file.
      ```bash
      vim ~/.ssh/authorized_keys
      ```
   3. In a separate terminal on your Workhorse PC, use your favorite text editor to open your id_rsa.pub file.
      ```bash
      vim ~/.ssh/id_rsa.pub
      ```
   4. Copy the contents from your id_rsa.pub file to a new line on the authorized_keys file on the Control PC. Then save. 
   5. Open a new terminal and try sshing to the control PC and it should no longer require a password. 
3. (Optional) Upload your ssh key to github by following instructions here: https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account

## Unlocking the Franka Robot
1. In a new terminal, ssh to the control PC with option -X.
   ```bash
   ssh -X iam-lab@iam-[control-pc-name]
   ```
2. Open a web browser, either firefox or google chrome.
   ```bash
   firefox
   ```
3. Go to 172.16.0.2 in the web browser.
4. (Optional) Input the username admin and the password to login to the Franka Desk GUI.
5. Unlock the robot by clicking the unlock button on the bottom right of the web interface.
6. If the robot has pink lights, press down on the e-stop and then release it and the robot should turn blue. If the robot is white, just release the e-stop and it should also turn blue.

## Running the Franka Robot

1. Make sure that both the user stop and the brakes of the Franka robot have been unlocked in the Franka Desk GUI.
2. Open up a new terminal and go to the robot-interface directory.
   ```bash
   bash ./bash_scripts/start_control_pc.sh -i iam-[control-pc-name]
   ```
3. Open up a new terminal and go to the robot-interface directory.
   ```bash
   source catkin_ws/devel/setup.bash
   cd catkin_ws/src/franka_action_lib/scripts
   ```
4. Be in the same virtualenv or Conda env that FrankaPy was installed in. Place your hand on top of the e-stop. Reset the robot pose with the following command.
   ```bash
   python reset_arm.py
   ```
   
See `catkin_ws/src/franka_action_lib/scripts/reset_arm.py` for an example of how to use the `FrankaPy` python package.



## Issues

#### LibPoco issue

libFranka requires libPoco, which can be installed using `sudo apt-get install libpoco-doc libpoco-dev`. However, trying to build libFranka might still fail since `CMAKE` cannot run ` find_package(Poco)` since there doesn't exist `/usr/local/lib/cmake/Poco/PocoConfig.cmake`. This is a peculiarity of libPoco which installs in a weird way without providing an option for us to link against it. 

To fix this we have copied the `libPoco.cmake` file in `{robot-interface-dir}/cmake`, and we add the following line to the CMakeLists.txt

`list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake`

Do the following if you run run into `libfranka: Cannot load model library: Cannot load library`:

```sh
mkdir -p /usr/local/lib/cmake/Poco/
cp cmake/FindPoco.cmake /usr/local/lib/cmake/FindPoco.cmake
```
