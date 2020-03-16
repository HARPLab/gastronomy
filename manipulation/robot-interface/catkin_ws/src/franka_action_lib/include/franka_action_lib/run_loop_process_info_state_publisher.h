#ifndef FRANKA_ACTION_LIB_RUN_LOOP_PROCESS_INFO_STATE_PUBLISHER_H
#define FRANKA_ACTION_LIB_RUN_LOOP_PROCESS_INFO_STATE_PUBLISHER_H

#include <iostream>
#include <thread>
#include <array>
#include <chrono>
#include <vector>
#include <ros/ros.h>
#include <franka_action_lib/RunLoopProcessInfoState.h> 

#include "franka_action_lib/shared_memory_handler.h"

namespace franka_action_lib  
{ 
  class RunLoopProcessInfoStatePublisher
  {
    protected:

      ros::NodeHandle nh_;
      ros::Publisher run_loop_process_info_state_pub_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
      std::string topic_name_;

      double publish_frequency_;
      franka_action_lib::SharedMemoryHandler shared_memory_handler_;
      
      bool has_seen_one_run_loop_process_info_state_;
      franka_action_lib::RunLoopProcessInfoState last_run_loop_process_info_state_;

    public:

      RunLoopProcessInfoStatePublisher(std::string name);

      ~RunLoopProcessInfoStatePublisher(void){}

  };
}

#endif // FRANKA_ACTION_LIB_RUN_LOOP_PROCESS_INFO_STATE_PUBLISHER_H