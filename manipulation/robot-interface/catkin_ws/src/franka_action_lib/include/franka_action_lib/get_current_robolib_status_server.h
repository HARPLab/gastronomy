#ifndef FRANKA_ACTION_LIB_GET_CURRENT_ROBOLIB_STATUS_SERVER_H
#define FRANKA_ACTION_LIB_GET_CURRENT_ROBOLIB_STATUS_SERVER_H

#include <iostream>
#include <thread>
#include <mutex>
#include <ros/ros.h>

#include "franka_action_lib/GetCurrentRobolibStatusCmd.h"
#include "franka_action_lib/RobolibStatus.h"

#include <boost/circular_buffer.hpp>

namespace franka_action_lib  
{ 
  class GetCurrentRobolibStatusServer
  {
    protected:

      ros::NodeHandle nh_;
      ros::ServiceServer server; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
      std::string robolib_status_topic_name_;
      static std::mutex current_robolib_status_mutex_;
      static franka_action_lib::RobolibStatus current_robolib_status_;

    public:

      GetCurrentRobolibStatusServer(std::string name);

      ~GetCurrentRobolibStatusServer(void){}

      static bool get_current_robolib_status(GetCurrentRobolibStatusCmd::Request &req, GetCurrentRobolibStatusCmd::Response &res);
      static void robolib_status_sub_cb(const franka_action_lib::RobolibStatus& robolib_status);

  };
}

#endif // FRANKA_ACTION_LIB_GET_CURRENT_ROBOLIB_STATUS_SERVER_H