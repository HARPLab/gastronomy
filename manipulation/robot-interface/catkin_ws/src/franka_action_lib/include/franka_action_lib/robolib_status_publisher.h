#ifndef FRANKA_ACTION_LIB_ROBOT_STATE_PUBLISHER_H
#define FRANKA_ACTION_LIB_ROBOT_STATE_PUBLISHER_H

#include <iostream>
#include <thread>
#include <array>
#include <chrono>
#include <vector>
#include <ros/ros.h>

#include "franka_action_lib/RobolibStatus.h"
#include "franka_action_lib/shared_memory_handler.h"

namespace franka_action_lib  
{ 
  class RobolibStatusPublisher
  {
    protected:

      ros::NodeHandle nh_;
      ros::Publisher robolib_status_pub_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
      std::string topic_name_;

      RobolibStatus last_robolib_status_;
      bool has_seen_one_robolib_status_;
      int stale_count = 0;

      double publish_frequency_;
      franka_action_lib::SharedMemoryHandler *shared_memory_handler_;

    public:

      RobolibStatusPublisher(std::string name);

      ~RobolibStatusPublisher(void){}

  };
}

#endif // FRANKA_ACTION_LIB_ROBOT_STATE_PUBLISHER_H