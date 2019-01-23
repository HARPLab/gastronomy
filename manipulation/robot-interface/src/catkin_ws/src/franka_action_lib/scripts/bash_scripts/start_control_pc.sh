#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

usage="$(basename "$0") [-h] [-a xxx.xxx.xxx.xxx] [-u iam-lab] [-p 0] -- Start control PC from workstation

where:
    -h show this help text
    -a IP address for the control PC.
    -u Username on control PC.
    -p Use password [0/1]
    
    ./start_control_pc.sh -a 128.237.176.200 -u mohit -p 0 (Does not require passwd)
    ./start_control_pc.sh -a 128.237.176.200 -u iam-lab -p 1 (Requires passwd)
    "

control_pc_uname="iam-lab"
control_pc_use_passwd=1
control_pc_robot_lib_path="~/projects/robot-interface"

while getopts ':h:a:u:p:' option; do
  case "${option}" in
    h) echo "$usage"
       exit
       ;;
    a) control_pc_ip_address=$OPTARG
       ;;
    u) control_pc_uname=$OPTARG
       ;;
    p) control_pc_use_passwd=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

workstation_ip_address="`hostname -I`"

# Notify the IP addresses being used.
echo "Control PC IP uname/address: "$control_pc_uname", "$control_pc_ip_address
echo "Workstation IP address: "$workstation_ip_address
if [ "$control_pc_use_passwd" -eq 0 ]; then
  echo "Will not use password to ssh into control-pc."
else
  echo "Will use default password to ssh into control-pc."
fi


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start rosmaster in a new gnome-terminal
start_rosmaster_path="$DIR/start_rosmaster.sh"
echo "Will start ROS master in new terminal."$start_rosmaster_path
gnome-terminal -e "bash $start_rosmaster_path"
echo "Did start ROS master in new terminal."

sleep 5

echo "Should start iam_robolib? Enter [0/1]"
read should_start_iam_robolib

if [ $should_start_iam_robolib -eq 1 ]; then
# ssh to the control pc and start iam_robolib in a new gnome-terminal
start_iam_robolib_on_control_pc_path="$DIR/start_iam_robolib_on_control_pc.sh"
echo "Will ssh to control PC and start iam_roblib..."$start_iam_robolib_on_control_pc_path
gnome-terminal -e "bash $start_iam_robolib_on_control_pc_path $control_pc_uname $control_pc_ip_address $workstation_ip_address $control_pc_use_passwd $control_pc_robot_lib_path"
echo "Done"
sleep 5
else
echo "Will not start iam_robolib on the control pc."
fi

# ssh to the control pc and start ROS action server in a new gnome-terminal
start_ros_action_lib_on_control_pc_path="$DIR/start_ros_action_lib_on_control_pc.sh"
echo "Will ssh to control PC and start ROS action server..."$start_ros_action_lib_on_control_pc_path
gnome-terminal -e "bash $start_ros_action_lib_on_control_pc_path $control_pc_uname $control_pc_ip_address $workstation_ip_address $control_pc_use_passwd $control_pc_robot_lib_path"
echo "Done"

sleep 5

echo "Will start realsense camera on workstation..."
# Start realsense camera on the workstation pc in a new gnome-terminal
start_realsense_on_workstation_path="$DIR/start_realsense_on_workstation.sh"
gnome-terminal -e "bash $start_realsense_on_workstation_path"
echo "Done"

