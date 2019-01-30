#include "franka_action_lib/robot_state_publisher.h"

namespace franka_action_lib
{
  RobotStatePublisher::RobotStatePublisher(std::string name) :  nh_("~"),
                                                                topic_name_(name),
                                                                shared_memory_handler_()
  {
    nh_.param("publish_frequency", publish_frequency_, (double) 100.0);
    robot_state_pub_ = nh_.advertise<franka_action_lib::RobotState>(topic_name_, 100);

    ROS_INFO("Robot State Publisher Started");

    ros::Rate loop_rate(publish_frequency_);
    while (ros::ok())
    {
        franka_action_lib::RobotState robot_state_ = shared_memory_handler_.getRobotState();
        robot_state_pub_.publish(robot_state_);

        ros::spinOnce();
        loop_rate.sleep();
    }
  }
}
