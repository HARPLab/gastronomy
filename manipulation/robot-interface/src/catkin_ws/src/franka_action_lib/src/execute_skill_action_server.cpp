#include "franka_action_lib/execute_skill_action_server.h"

namespace franka_action_lib
{
  ExecuteSkillActionServer::ExecuteSkillActionServer(std::string name) :  nh_("~"),
                                                                          as_(nh_, name, boost::bind(&ExecuteSkillActionServer::executeCB, this, _1), false),
                                                                          action_name_(name),
                                                                          shared_memory_handler_() {
    nh_.param("publish_frequency", publish_frequency_, (double) 10.0);

    as_.start();

    ROS_INFO("Action Lib Started");

  }

  void ExecuteSkillActionServer::executeCB(const franka_action_lib::ExecuteSkillGoalConstPtr &goal) {
    ros::Rate r(publish_frequency_);

    ROS_INFO("=======");

    ROS_INFO("New Skill received: %s", goal->skill_description.c_str());

    // Load skill parameters into shared memory and returns the skill_id
    int skill_id = shared_memory_handler_.loadSkillParametersIntoSharedMemory(goal);

    while(skill_id == -1) {
      r.sleep();

      skill_id = shared_memory_handler_.loadSkillParametersIntoSharedMemory(goal);
    }

    ROS_INFO("New Skill id = %d", skill_id);

    // Loop until skill is complete from shared memory or is preempted
    int done_skill_id = shared_memory_handler_.getDoneSkillIdInSharedMemory();

    // Ensure preempted flag is set to false
    shared_memory_handler_.setSkillPreemptedFlagInSharedMemory(false);
    while (done_skill_id < skill_id) {
      // check that preempt has not been requested by the client
      if (as_.isPreemptRequested() || !ros::ok()) {
        ROS_INFO("%s: Preempted", action_name_.c_str());
        // set the action state to preempted
        as_.setPreempted();

        shared_memory_handler_.setSkillPreemptedFlagInSharedMemory(true);

        break;
      }

      ROS_DEBUG("get_new_skill_available = %d", shared_memory_handler_.getNewSkillAvailableFlagInSharedMemory());
      ROS_DEBUG("get_done_skill_id = %d", shared_memory_handler_.getDoneSkillIdInSharedMemory());
      ROS_DEBUG("get_new_skill_id = %d", shared_memory_handler_.getNewSkillIdInSharedMemory());

      // TODO fill in execution_feedback from shared memory
      feedback_ = shared_memory_handler_.getSkillFeedback();

      // publish the feedback
      as_.publishFeedback(feedback_);

      // this sleep is not necessary, the execution_feedback is computed at 10 Hz for demonstration purposes
      r.sleep();

      done_skill_id = shared_memory_handler_.getDoneSkillIdInSharedMemory();
      ROS_DEBUG("done skill id = %d", done_skill_id);
    }
    ROS_INFO("Skill terminated id = %d", skill_id);

    if (done_skill_id == skill_id || done_skill_id == skill_id + 1) {
      // Get execution result from shared memory
      result_ = shared_memory_handler_.getSkillResult(skill_id);
      ROS_INFO("%s: Succeeded", action_name_.c_str());
      // set the action state to succeeded
      as_.setSucceeded(result_);
    } else if (as_.isPreemptRequested()) {
      ROS_INFO("Skill was preempted. skill_id = %d", skill_id);
      result_ = shared_memory_handler_.getSkillResult(skill_id);
    } else {
      ROS_ERROR("done_skill_id error: done_skill_id = %d, skill_id = %d", done_skill_id, skill_id);
    }
  }
}
