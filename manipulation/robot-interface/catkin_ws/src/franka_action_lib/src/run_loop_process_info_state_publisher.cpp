#include "franka_action_lib/run_loop_process_info_state_publisher.h"

namespace franka_action_lib
{
  RunLoopProcessInfoStatePublisher::RunLoopProcessInfoStatePublisher(std::string name) :  nh_("~"),
                                                                                          topic_name_(name),
                                                                                          shared_memory_handler_()
  {
    nh_.param("publish_frequency", publish_frequency_, (double) 100.0);
    run_loop_process_info_state_pub_ = nh_.advertise<franka_action_lib::RunLoopProcessInfoState>(topic_name_, 100);

    ROS_INFO("Run Loop Process Info State Publisher Started");

    ros::Rate loop_rate(publish_frequency_);
    while (ros::ok())
    {
        franka_action_lib::RunLoopProcessInfoState run_loop_process_info_state_ = shared_memory_handler_.getRunLoopProcessInfoState();

        if (run_loop_process_info_state_.is_fresh) {
          run_loop_process_info_state_pub_.publish(run_loop_process_info_state_);

          has_seen_one_run_loop_process_info_state_ = true;
          last_run_loop_process_info_state_ = run_loop_process_info_state_;
          last_run_loop_process_info_state_.is_fresh = false;
        } else {
          if (has_seen_one_run_loop_process_info_state_) {
            run_loop_process_info_state_pub_.publish(last_run_loop_process_info_state_);
          }
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
  }
}
