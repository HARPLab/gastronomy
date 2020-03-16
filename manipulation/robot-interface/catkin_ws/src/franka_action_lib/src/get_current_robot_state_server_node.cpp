#include <ros/ros.h>
#include "franka_action_lib/get_current_robot_state_server.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "get_robot_state_server", ros::init_options::AnonymousName);

  franka_action_lib::GetCurrentRobotStateServer get_current_robot_state_service("get_robot_state_server");

  return 0;
}