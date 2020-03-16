#include "franka_action_lib/robolib_status_publisher.h"

namespace franka_action_lib
{
  RobolibStatusPublisher::RobolibStatusPublisher(std::string name) :  nh_("~"),
                                                                topic_name_(name)
  {
    nh_.param("publish_frequency", publish_frequency_, (double) 100.0);
    robolib_status_pub_ = nh_.advertise<franka_action_lib::RobolibStatus>(topic_name_, 100);

    shared_memory_handler_ = new franka_action_lib::SharedMemoryHandler();

    ROS_INFO("Robolib Status Publisher Started");

    ros::Rate loop_rate(publish_frequency_);
    while (ros::ok())
    {
        // get robolib
        RobolibStatus robolib_status_ = shared_memory_handler_->getRobolibStatus();

        if (robolib_status_.is_fresh) {
          last_robolib_status_ = robolib_status_;
          stale_count = 0;
          has_seen_one_robolib_status_ = true;
        } else {
          // use previous robolib_status if available
          if (has_seen_one_robolib_status_) {
            robolib_status_ = last_robolib_status_;
            stale_count++;
          }
        }

        // only proceed if received at least 1 robolib status
        if (has_seen_one_robolib_status_) {
          // TODO(jacky): MAGIC NUMBER - Signal not ok if 10 consecutive robolib statuses have been stale. Roughly 100 ms
          if (stale_count > 10) {
            robolib_status_.is_ready = false;
            robolib_status_.is_fresh = false;
          }

          // publish
          robolib_status_pub_.publish(robolib_status_);

          // increment watchdog counter
          shared_memory_handler_->incrementWatchdogCounter();
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
  }
}
