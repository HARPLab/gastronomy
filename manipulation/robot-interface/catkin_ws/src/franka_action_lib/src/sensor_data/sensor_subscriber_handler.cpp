#include "franka_action_lib/sensor_data/sensor_subscriber_handler.h"
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>

namespace franka_action_lib
{
  SensorSubscriberHandler::SensorSubscriberHandler(ros::NodeHandle& nh)
  : shared_memory_handler_  ()
  , nh_                     (nh) {
    ROS_INFO("created_sensor_subscriber_handler");
  }


    //void SensorSubscriberHandler::dummyTimeCallback(const std_msgs::Float64::ConstPtr& f_data)
    void SensorSubscriberHandler::dummyTimeCallback(const franka_action_lib::SensorData::ConstPtr& f_data)
  {
    ROS_INFO("got a message");
    ROS_INFO("%d",f_data->size);
    //ROS_INFO("%s",&f_data->sensorDataInfo.c_str());
    ROS_INFO_STREAM(f_data->sensorDataInfo);

    shared_memory_handler_.tryToLoadSensorDataIntoSharedMemory(f_data);
  }


}
