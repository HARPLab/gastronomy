#!/bin/bash

control_pc_uname=$1
control_pc_ip_address=$2
workstation_ip_address=$3
control_pc_use_passwd=$4
control_pc_robot_lib_path=$5

if [ "$control_pc_use_passwd" = "0" ]; then
ssh -T $control_pc_uname@$control_pc_ip_address << EOSSH
~/projects/robot-interface/build/test_iam_robolib
EOSSH
else
sshpass -p "!@m-guardians" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
~/Documents/robot-interface/build/test_iam_robolib
EOSSH
fi
