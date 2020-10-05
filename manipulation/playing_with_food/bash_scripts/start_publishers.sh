#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start rosmaster in a new gnome-terminal if not already running
if ! pgrep -x "roscore" > /dev/null
then
    start_rosmaster_path="$DIR/start_rosmaster.sh"
    echo "Will start ROS master in new terminal."$start_rosmaster_path
    gnome-terminal -e "bash $start_rosmaster_path"
    sleep 3
    echo "Did start ROS master in new terminal."
else
    echo "Roscore is already running"
fi

start_azure_kinect_ros_driver_path="$DIR/start_azure_kinect_ros_driver.sh"
echo "Will start azure_kinect_ros_driver in new terminal."
gnome-terminal -e "bash $start_azure_kinect_ros_driver_path "
echo "Done"
sleep 3

start_realsense_path="$DIR/start_realsense.sh"
echo "Will start realsense in new terminal."
gnome-terminal -e "bash $start_realsense_path "
echo "Done"
sleep 3

start_finger_vision_path="$DIR/start_finger_vision.sh"
echo "Will start finger_vision in new terminal."
gnome-terminal -e "bash $start_finger_vision_path "
echo "Done"
sleep 3

start_sound_publishing_path="$DIR/start_sound_publishing.sh"
echo "Will start sound_publishing in new terminal."
gnome-terminal -e "bash $start_sound_publishing_path "
echo "Done"
sleep 3
