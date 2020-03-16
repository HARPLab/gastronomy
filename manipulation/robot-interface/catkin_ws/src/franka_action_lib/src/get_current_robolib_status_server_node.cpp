#include <ros/ros.h>
#include "franka_action_lib/get_current_robolib_status_server.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "get_robolib_status_server", ros::init_options::AnonymousName);

  franka_action_lib::GetCurrentRobolibStatusServer get_current_robolib_status_service("get_robolib_status_server");

  return 0;
}