#ifndef FRANKA_ACTION_LIB_SENSOR_SUBSCRIBER_HANDLER_H
#define FRANKA_ACTION_LIB_SENSOR_SUBSCRIBER_HANDLER_H

#include <iostream>
#include <thread>
#include <array>
#include <chrono>
#include <vector>
#include <ros/ros.h>
//#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include "franka_action_lib/SensorData.h"

#include "franka_action_lib/RobolibStatus.h"
#include "franka_action_lib/shared_memory_handler.h"

namespace franka_action_lib  
{ 
  class SensorSubscriberHandler
  {
    protected:

      ros::NodeHandle nh_;

      double write_frequency_;
      SharedMemoryHandler shared_memory_handler_;

    public:

      SensorSubscriberHandler(ros::NodeHandle& nh);
      ~SensorSubscriberHandler(){};
      
      void dummyTimeCallback(const franka_action_lib::SensorData::ConstPtr& msg);


  };
}

#endif // 