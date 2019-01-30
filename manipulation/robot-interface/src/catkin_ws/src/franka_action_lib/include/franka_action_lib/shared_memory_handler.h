#ifndef FRANKA_ACTION_LIB_SHARED_MEMORY_HANDLER_H
#define FRANKA_ACTION_LIB_SHARED_MEMORY_HANDLER_H

#include <franka_action_lib/ExecuteSkillAction.h> // Note: "Action" is appended
#include <franka_action_lib/RobotState.h>

#include "ros/ros.h" // For ROS::ERROR messages

#include <array>
#include <vector>
#include <algorithm>
#include <cassert>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include <iam_robolib_common/SharedMemoryInfo.h>
#include <iam_robolib_common/run_loop_process_info.h>

namespace franka_action_lib
{
  class SharedMemoryHandler {
    public:
      SharedMemoryHandler();

      ~SharedMemoryHandler(void){}

      int loadSkillParametersIntoSharedMemory(const franka_action_lib::ExecuteSkillGoalConstPtr &goal);

      // void startSensorSubscribers(const franka_action_lib::ExecuteSkillGoalConstPtr &goal);

      bool getSkillRunningFlagInSharedMemory();

      int getDoneSkillIdInSharedMemory();

      void setSkillPreemptedFlagInSharedMemory(bool skill_preempted_flag);

      franka_action_lib::ExecuteSkillFeedback getSkillFeedback();

      franka_action_lib::ExecuteSkillResult getSkillResult(int skill_id);

      franka_action_lib::RobotState getRobotState();
      
      bool getNewSkillAvailableFlagInSharedMemory();

      int getNewSkillIdInSharedMemory();

    private:

      SharedMemoryInfo shared_memory_info_;

      boost::interprocess::managed_shared_memory managed_shared_memory_;

      RunLoopProcessInfo *run_loop_process_info_;
      boost::interprocess::interprocess_mutex *run_loop_info_mutex_;

      boost::interprocess::interprocess_mutex *shared_memory_object_0_mutex_;
      boost::interprocess::interprocess_mutex *shared_memory_object_1_mutex_;

      boost::interprocess::interprocess_mutex *shared_sensor_data_0_mutex_;
      boost::interprocess::interprocess_mutex *shared_sensor_data_1_mutex_;

      boost::interprocess::interprocess_mutex *shared_execution_response_0_mutex_;
      boost::interprocess::interprocess_mutex *shared_execution_response_1_mutex_;

      boost::interprocess::interprocess_mutex *shared_current_robot_state_mutex_;

      boost::interprocess::shared_memory_object shared_memory_object_0_;
      boost::interprocess::shared_memory_object shared_memory_object_1_;
      boost::interprocess::shared_memory_object shared_sensor_data_0_;
      boost::interprocess::shared_memory_object shared_sensor_data_1_;
      boost::interprocess::shared_memory_object shared_execution_result_0_;
      boost::interprocess::shared_memory_object shared_execution_result_1_;
      boost::interprocess::shared_memory_object shared_current_robot_state_;

      boost::interprocess::mapped_region region_traj_params_0_;
      boost::interprocess::mapped_region region_feedback_controller_params_0_;
      boost::interprocess::mapped_region region_termination_params_0_;
      boost::interprocess::mapped_region region_timer_params_0_;

      boost::interprocess::mapped_region region_traj_params_1_;
      boost::interprocess::mapped_region region_feedback_controller_params_1_;
      boost::interprocess::mapped_region region_termination_params_1_;
      boost::interprocess::mapped_region region_timer_params_1_;

      boost::interprocess::mapped_region region_traj_sensor_data_0_;
      boost::interprocess::mapped_region region_feedback_controller_sensor_data_0_;
      boost::interprocess::mapped_region region_termination_sensor_data_0_;
      boost::interprocess::mapped_region region_timer_sensor_data_0_;

      boost::interprocess::mapped_region region_traj_sensor_data_1_;
      boost::interprocess::mapped_region region_feedback_controller_sensor_data_1_;
      boost::interprocess::mapped_region region_termination_sensor_data_1_;
      boost::interprocess::mapped_region region_timer_sensor_data_1_;

      boost::interprocess::mapped_region execution_feedback_region_0_;
      boost::interprocess::mapped_region execution_result_region_0_;
      boost::interprocess::mapped_region execution_feedback_region_1_;
      boost::interprocess::mapped_region execution_result_region_1_;

      boost::interprocess::mapped_region shared_current_robot_region_;

      float *traj_gen_buffer_0_;
      float *feedback_controller_buffer_0_;
      float *termination_buffer_0_;
      float *timer_buffer_0_;
      float *traj_gen_buffer_1_;
      float *feedback_controller_buffer_1_;
      float *termination_buffer_1_;
      float *timer_buffer_1_;

      float *traj_gen_sensor_buffer_0_;
      float *feedback_controller_sensor_buffer_0_;
      float *termination_sensor_buffer_0_;
      float *timer_sensor_buffer_0_;

      float *traj_gen_sensor_buffer_1_;
      float *feedback_controller_sensor_buffer_1_;
      float *termination_sensor_buffer_1_;
      float *timer_sensor_buffer_1_;

      float *execution_feedback_buffer_0_;
      float *execution_result_buffer_0_;
      float *execution_feedback_buffer_1_;
      float *execution_result_buffer_1_;

      float *current_robot_state_buffer_;

      int getCurrentFreeSharedMemoryIndexInSharedMemoryUnprotected();
      int getCurrentSkillIdInSharedMemoryUnprotected();
      void setCurrentSkillIdInSharedMemoryUnprotected(int current_skill_id);
      int getDoneSkillIdInSharedMemoryUnprotected();
      bool getNewSkillAvailableFlagInSharedMemoryUnprotected();
      void setNewSkillFlagInSharedMemoryUnprotected(bool new_skill_flag);
      int getNewSkillIdInSharedMemoryUnprotected();
      void setNewSkillIdInSharedMemoryUnprotected(int new_skill_id);
      void setNewSkillDescriptionInSharedMemoryUnprotected(std::string description);
      void setNewSkillTypeInSharedMemoryUnprotected(int new_skill_type);
      void setNewMetaSkillIdInSharedMemoryUnprotected(int new_meta_skill_id);
      void setNewMetaSkillTypeInSharedMemoryUnprotected(int new_meta_skill_type);
      void setResultSkillIdInSharedMemoryUnprotected(int result_skill_id);

      void loadSensorDataUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index);
      void loadTrajGenParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index);
      void loadFeedbackControllerParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index);
      void loadTerminationParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index);
      void loadTimerParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index);
      
  };
}


#endif // FRANKA_ACTION_LIB_SHARED_MEMORY_HANDLER_H
