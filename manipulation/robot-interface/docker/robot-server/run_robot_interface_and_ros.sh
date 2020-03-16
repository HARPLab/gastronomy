cd robot-interface
source catkin_ws/devel/setup.bash

echo Launching main_iam_robolib
./build/main_iam_robolib --logdir=/external_logs/logs &> /external_logs/main_iam_robolib.log &

sleep 2

echo Launching franka_ros_interface
roslaunch franka_action_lib franka_ros_interface.launch &> /external_logs/franka_action_lib.log &

echo Spinning...
sleep infinity
