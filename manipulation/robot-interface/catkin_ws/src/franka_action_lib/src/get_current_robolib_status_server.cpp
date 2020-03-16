#include <ros/ros.h>
#include "franka_action_lib/get_current_robolib_status_server.h"

namespace franka_action_lib
{
  std::mutex GetCurrentRobolibStatusServer::current_robolib_status_mutex_;
  franka_action_lib::RobolibStatus GetCurrentRobolibStatusServer::current_robolib_status_;

  GetCurrentRobolibStatusServer::GetCurrentRobolibStatusServer(std::string name) :  nh_("~")
  {
    nh_.param("robolib_status_topic_name", robolib_status_topic_name_, std::string("/robolib_status_publisher_node/robolib_status"));

    ros::Subscriber sub = nh_.subscribe(robolib_status_topic_name_, 10, robolib_status_sub_cb);
    ros::ServiceServer service = nh_.advertiseService("get_current_robolib_status_server", get_current_robolib_status);
    ROS_INFO("Get Current Robolib Status Server Started");
    ros::spin();
  }

  void GetCurrentRobolibStatusServer::robolib_status_sub_cb(const franka_action_lib::RobolibStatus& robolib_status)
  {
    if (current_robolib_status_mutex_.try_lock()) {
      current_robolib_status_ = robolib_status;
      current_robolib_status_mutex_.unlock();
    }
  }

  bool GetCurrentRobolibStatusServer::get_current_robolib_status(GetCurrentRobolibStatusCmd::Request &req, GetCurrentRobolibStatusCmd::Response &res)
  {
    ROS_DEBUG("Get Current Robolib Status Server request received.");
    current_robolib_status_mutex_.lock();
    res.robolib_status = current_robolib_status_;
    current_robolib_status_mutex_.unlock();
    ROS_DEBUG("Get Current Robolib Status Servier request processed.");
    
    return true;
  }
}
