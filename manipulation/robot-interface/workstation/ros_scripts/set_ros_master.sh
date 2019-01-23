#!/bin/bash

ros_master_ip_address=$1
IP_ADDR="$(ifconfig wlp3s0 >/dev/null|awk '/inet addr:/ {print $2}'|sed 's/addr://')"

export ROS_MASTER_URI="http://$ros_master_ip_address:11311"
export ROS_HOSTNAME="${IP_ADDR}"
export ROS_IP="${IP_ADDR}"
