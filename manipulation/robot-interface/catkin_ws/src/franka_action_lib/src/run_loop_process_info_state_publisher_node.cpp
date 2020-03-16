#include <ros/ros.h>
#include "franka_action_lib/run_loop_process_info_state_publisher.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "run_loop_process_info_state_publisher_node", ros::init_options::AnonymousName);

  franka_action_lib::RunLoopProcessInfoStatePublisher run_loop_process_info_state_publisher("run_loop_process_info_state");

  return 0;
}