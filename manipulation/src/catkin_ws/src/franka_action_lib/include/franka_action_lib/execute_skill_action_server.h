#ifndef FRANKA_ACTION_LIB_EXECUTE_SKILL_ACTION_SERVER_H
#define FRANKA_ACTION_LIB_EXECUTE_SKILL_ACTION_SERVER_H

#include <iostream>
#include <thread>
#include <array>
#include <chrono>
#include <vector>
#include <ros/ros.h>
#include <franka_action_lib/ExecuteSkillAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>

#include "franka_action_lib/shared_memory_handler.h"

namespace franka_action_lib  
{ 
  class ExecuteSkillActionServer
  {
    protected:

      ros::NodeHandle nh_;
      actionlib::SimpleActionServer<franka_action_lib::ExecuteSkillAction> as_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
      std::string action_name_;

      double publish_frequency_;

      // create messages that are used to published feedback/result
      franka_action_lib::ExecuteSkillFeedback feedback_;
      franka_action_lib::ExecuteSkillResult result_;

      franka_action_lib::SharedMemoryHandler shared_memory_handler_;

    public:

      ExecuteSkillActionServer(std::string name);

      ~ExecuteSkillActionServer(void){}

      void executeCB(const franka_action_lib::ExecuteSkillGoalConstPtr &goal);

  };
}

#endif // FRANKA_ACTION_LIB_EXECUTE_SKILL_ACTION_SERVER_H