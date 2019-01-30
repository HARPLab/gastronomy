#include <ros/ros.h>
#include "franka_action_lib/robot_state_publisher.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "robot_state_publisher_node", ros::init_options::AnonymousName);

  franka_action_lib::RobotStatePublisher robot_state_publisher("robot_state");

  return 0;
}