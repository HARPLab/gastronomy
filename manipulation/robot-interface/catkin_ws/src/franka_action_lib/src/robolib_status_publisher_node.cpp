#include <ros/ros.h>
#include "franka_action_lib/robolib_status_publisher.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "robolib_status_publisher_node", ros::init_options::AnonymousName);

  franka_action_lib::RobolibStatusPublisher robolib_status_publisher("robolib_status");

  return 0;
}