#ifndef FRANKA_ACTION_LIB_ROBOT_STATE_PUBLISHER_H
#define FRANKA_ACTION_LIB_ROBOT_STATE_PUBLISHER_H

#include <iostream>
#include <thread>
#include <array>
#include <chrono>
#include <vector>
#include <ros/ros.h>
#include <franka_action_lib/RobotState.h> // Note: "Action" is appended

#include "franka_action_lib/shared_memory_handler.h"

namespace franka_action_lib  
{ 
  class RobotStatePublisher
  {
    protected:

      ros::NodeHandle nh_;
      ros::Publisher robot_state_pub_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
      std::string topic_name_;

      double publish_frequency_;
      franka_action_lib::SharedMemoryHandler shared_memory_handler_;

    public:

      RobotStatePublisher(std::string name);

      ~RobotStatePublisher(void){}

  };
}

#endif // FRANKA_ACTION_LIB_ROBOT_STATE_PUBLISHER_H