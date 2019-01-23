#include "franka_action_lib/shared_memory_handler.h"

namespace franka_action_lib
{

  SharedMemoryHandler::SharedMemoryHandler() : shared_memory_info_()
  {
    managed_shared_memory_ = boost::interprocess::managed_shared_memory(boost::interprocess::open_only,
                                                                        shared_memory_info_.getSharedMemoryNameForObjects().c_str());

    // Get RunLoopProcessInfo from the the shared memory segment.
    std::pair<RunLoopProcessInfo*, std::size_t> run_loop_process_info_pair = managed_shared_memory_.find<RunLoopProcessInfo>
                                                                             (shared_memory_info_.getRunLoopInfoObjectName().c_str());
    run_loop_process_info_ = run_loop_process_info_pair.first;

    // Make sure the process info object can be found in memory.
    assert(run_loop_process_info_ != 0);

    // Get mutex for ProcessInfo from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> run_loop_info_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getRunLoopInfoMutexName().c_str());
    run_loop_info_mutex_ = run_loop_info_mutex_pair.first;
    assert(run_loop_info_mutex_ != 0);

    // Get mutex for buffer 0 from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> shared_memory_object_0_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getParameterMemoryMutexName(0).c_str());
    shared_memory_object_0_mutex_ = shared_memory_object_0_mutex_pair.first;
    assert(shared_memory_object_0_mutex_ != 0);

    /**
     * Open shared memory region for parameter buffer 0.
     */
    shared_memory_object_0_ = boost::interprocess::shared_memory_object(
        boost::interprocess::open_only,
        shared_memory_info_.getSharedMemoryNameForParameters(0).c_str(),
        boost::interprocess::read_write
    );

    // Allocate regions for each parameter array
    region_traj_params_0_ = boost::interprocess::mapped_region(
        shared_memory_object_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTrajectoryParameters(),
        shared_memory_info_.getSizeForTrajectoryParameters()
        );
    traj_gen_buffer_0_ = reinterpret_cast<float *>(region_traj_params_0_.get_address());
    region_feedback_controller_params_0_ = boost::interprocess::mapped_region(
        shared_memory_object_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForFeedbackControllerParameters(),
        shared_memory_info_.getSizeForFeedbackControllerParameters()
        );
    feedback_controller_buffer_0_ = reinterpret_cast<float *>(region_feedback_controller_params_0_.get_address());
    region_termination_params_0_ = boost::interprocess::mapped_region(
        shared_memory_object_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTerminationParameters(),
        shared_memory_info_.getSizeForTerminationParameters()
    );
    termination_buffer_0_ = reinterpret_cast<float *>(region_termination_params_0_.get_address());
    region_timer_params_0_ = boost::interprocess::mapped_region(
        shared_memory_object_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTimerParameters(),
        shared_memory_info_.getSizeForTimerParameters()
    );
    timer_buffer_0_ = reinterpret_cast<float *>(region_timer_params_0_.get_address());

    // Get mutex for buffer 1 from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> shared_memory_object_1_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getParameterMemoryMutexName(1).c_str());
    shared_memory_object_1_mutex_ = shared_memory_object_1_mutex_pair.first;
    assert(shared_memory_object_1_mutex_ != 0);

    /**
     * Open shared memory region for parameter buffer 1.
     */
    shared_memory_object_1_ = boost::interprocess::shared_memory_object(
        boost::interprocess::open_only,
        shared_memory_info_.getSharedMemoryNameForParameters(1).c_str(),
        boost::interprocess::read_write
        );

    // Allocate regions for each parameter array
    region_traj_params_1_ =  boost::interprocess::mapped_region(
        shared_memory_object_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTrajectoryParameters(),
        sizeof(float) * shared_memory_info_.getSizeForTrajectoryParameters()
        );
    traj_gen_buffer_1_ = reinterpret_cast<float *>(region_traj_params_1_.get_address());
    region_feedback_controller_params_1_ = boost::interprocess::mapped_region(
        shared_memory_object_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForFeedbackControllerParameters(),
        shared_memory_info_.getSizeForFeedbackControllerParameters()
        );
    feedback_controller_buffer_1_ = reinterpret_cast<float *>(region_feedback_controller_params_1_.get_address());
    region_termination_params_1_ = boost::interprocess::mapped_region(
        shared_memory_object_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTerminationParameters(),
        shared_memory_info_.getSizeForTerminationParameters()
        );
    termination_buffer_1_ = reinterpret_cast<float *>(region_termination_params_1_.get_address());
    region_timer_params_1_ = boost::interprocess::mapped_region(
        shared_memory_object_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTimerParameters(),
        shared_memory_info_.getSizeForTimerParameters()
        );
    timer_buffer_1_ = reinterpret_cast<float *>(region_timer_params_1_.get_address());

    // Get mutex for sensor data buffer 0 from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> shared_sensor_data_0_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getSensorDataMutexName(0).c_str());
    shared_sensor_data_0_mutex_ = shared_sensor_data_0_mutex_pair.first;
    assert(shared_sensor_data_0_mutex_ != 0);

    /**
     * Open shared memory region for sensor data buffer 0.
     */
    shared_sensor_data_0_ = boost::interprocess::shared_memory_object(
        boost::interprocess::open_only,
        shared_memory_info_.getSharedMemoryNameForSensorData(0).c_str(),
        boost::interprocess::read_write
    );

    region_traj_sensor_data_0_ =  boost::interprocess::mapped_region(
        shared_sensor_data_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTrajectorySensorData(),
        shared_memory_info_.getSizeForTrajectorySensorData()
    );
    traj_gen_sensor_buffer_0_ = reinterpret_cast<float *>(region_traj_sensor_data_0_.get_address());
    region_feedback_controller_sensor_data_0_= boost::interprocess::mapped_region(
        shared_sensor_data_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForFeedbackControllerSensorData(),
        shared_memory_info_.getSizeForFeedbackControllerSensorData()
    );
    feedback_controller_sensor_buffer_0_ = reinterpret_cast<float *>(region_feedback_controller_sensor_data_0_.get_address());
    region_termination_sensor_data_0_ = boost::interprocess::mapped_region(
        shared_sensor_data_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTerminationSensorData(),
        shared_memory_info_.getSizeForTerminationSensorData()
    );
    termination_sensor_buffer_0_ = reinterpret_cast<float *>(region_termination_sensor_data_0_.get_address());
    region_timer_sensor_data_0_= boost::interprocess::mapped_region(
        shared_sensor_data_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTimerParameters(),
        shared_memory_info_.getSizeForTimerParameters()
    );
    timer_sensor_buffer_0_ = reinterpret_cast<float *>(region_timer_sensor_data_0_.get_address());

    // Get mutex for sensor data buffer 0 from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> shared_sensor_data_1_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getSensorDataMutexName(1).c_str());
    shared_sensor_data_1_mutex_ = shared_sensor_data_1_mutex_pair.first;
    assert(shared_sensor_data_1_mutex_ != 0);

    /**
     * Open shared memory region for sensor data buffer 1.
     */
    shared_sensor_data_1_ = boost::interprocess::shared_memory_object(
        boost::interprocess::open_only,
        shared_memory_info_.getSharedMemoryNameForSensorData(1).c_str(),
        boost::interprocess::read_write
    );

    region_traj_sensor_data_1_ =  boost::interprocess::mapped_region(
        shared_sensor_data_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTrajectorySensorData(),
        shared_memory_info_.getSizeForTrajectorySensorData()
    );
    traj_gen_sensor_buffer_1_ = reinterpret_cast<float *>(region_traj_sensor_data_1_.get_address());
    region_feedback_controller_sensor_data_1_= boost::interprocess::mapped_region(
        shared_sensor_data_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForFeedbackControllerSensorData(),
        shared_memory_info_.getSizeForFeedbackControllerSensorData()
    );
    feedback_controller_sensor_buffer_1_ = reinterpret_cast<float *>(region_feedback_controller_sensor_data_1_.get_address());
    region_termination_sensor_data_1_ = boost::interprocess::mapped_region(
        shared_sensor_data_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTerminationSensorData(),
        shared_memory_info_.getSizeForTerminationSensorData()
    );
    termination_sensor_buffer_1_ = reinterpret_cast<float *>(region_termination_sensor_data_1_.get_address());
    region_timer_sensor_data_1_= boost::interprocess::mapped_region(
        shared_sensor_data_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForTimerParameters(),
        shared_memory_info_.getSizeForTimerParameters()
    );
    timer_sensor_buffer_1_ = reinterpret_cast<float *>(region_timer_sensor_data_1_.get_address());

    // Get mutex for execution response buffer 0 from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> shared_execution_response_0_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getExecutionResponseMutexName(0).c_str());
    shared_execution_response_0_mutex_ = shared_execution_response_0_mutex_pair.first;
    assert(shared_execution_response_0_mutex_ != 0);

    /**
     * Open shared memory region for execution response buffer 0.
     */
    shared_execution_result_0_ = boost::interprocess::shared_memory_object(
        boost::interprocess::open_only,
        shared_memory_info_.getSharedMemoryNameForResults(0).c_str(),
        boost::interprocess::read_write
    );

    execution_feedback_region_0_ =  boost::interprocess::mapped_region(
        shared_execution_result_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForExecutionFeedbackData(),
        shared_memory_info_.getSizeForExecutionFeedbackData()
    );
    execution_feedback_buffer_0_ = reinterpret_cast<float *>(execution_feedback_region_0_.get_address());
    execution_result_region_0_ = boost::interprocess::mapped_region(
        shared_execution_result_0_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForExecutionReturnData(),
        shared_memory_info_.getSizeForExecutionReturnData()
        );
    execution_result_buffer_0_ = reinterpret_cast<float *>(execution_result_region_0_.get_address());

    // Get mutex for execution response buffer 1 from the shared memory segment.
    std::pair<boost::interprocess::interprocess_mutex *, std::size_t> shared_execution_response_1_mutex_pair = \
                                managed_shared_memory_.find<boost::interprocess::interprocess_mutex>
                                (shared_memory_info_.getExecutionResponseMutexName(1).c_str());
    shared_execution_response_1_mutex_ = shared_execution_response_1_mutex_pair.first;
    assert(shared_execution_response_1_mutex_ != 0);

    /**
     * Open shared memory region for execution response buffer 1.
     */
    shared_execution_result_1_ = boost::interprocess::shared_memory_object(
        boost::interprocess::open_only,
        shared_memory_info_.getSharedMemoryNameForResults(1).c_str(),
        boost::interprocess::read_write
    );

    execution_feedback_region_1_ =  boost::interprocess::mapped_region(
        shared_execution_result_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForExecutionFeedbackData(),
        shared_memory_info_.getSizeForExecutionFeedbackData()
    );
    execution_feedback_buffer_1_ = reinterpret_cast<float *>(execution_feedback_region_1_.get_address());
    execution_result_region_1_ = boost::interprocess::mapped_region(
        shared_execution_result_1_,
        boost::interprocess::read_write,
        shared_memory_info_.getOffsetForExecutionReturnData(),
        shared_memory_info_.getSizeForExecutionReturnData()
    );
    execution_result_buffer_1_ = reinterpret_cast<float *>(execution_result_region_1_.get_address());

  }

  int SharedMemoryHandler::loadSkillParametersIntoSharedMemory(const franka_action_lib::ExecuteSkillGoalConstPtr &goal)
  {
    // Grab lock of run_loop_info_mutex_ to see if we can load the new skill parameters into the shared memory
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> run_loop_info_lock(*run_loop_info_mutex_);

    bool new_skill_flag = getNewSkillFlagInSharedMemoryUnprotected();

    // Return -1 if the new_skill_flag is already set to true, which means that another new skill has already been loaded
    // into the shared memory.
    if(new_skill_flag)
    {
      return -1;
    }

    int current_skill_id = getCurrentSkillIdInSharedMemoryUnprotected();
    int new_skill_id = getNewSkillIdInSharedMemoryUnprotected();

    if(current_skill_id != new_skill_id)
    {
      ROS_ERROR("Error with the current_skill_id and new_skill_id. current_skill_id = %d, new_skill_id = %d", current_skill_id, new_skill_id);
      return -1;
    }

    new_skill_id += 1;

    int current_free_shared_memory_index = getCurrentFreeSharedMemoryIndexInSharedMemoryUnprotected();

    // Grab lock of the free shared memory
    if(current_free_shared_memory_index == 0)
    {
      // Grab lock of shared_memory_object_0_mutex_ to make sure no one else can modify shared_memory_0_
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> shared_memory_object_0_lock(*shared_memory_object_0_mutex_);

      // Load all of the data into shared_memory_0_
      loadSensorDataUnprotected(goal, 0);
      loadTrajGenParamsUnprotected(goal, 0);
      loadFeedbackControllerParamsUnprotected(goal, 0);
      loadTerminationParamsUnprotected(goal, 0);
      loadTimerParamsUnprotected(goal, 0);

      // The lock of the shared_memory_object_0_mutex_ should be released automatically
    }
    else if (current_free_shared_memory_index == 1)
    {
      // Grab lock of shared_memory_object_1_mutex_ to make sure no one else can modify shared_memory_1_
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> shared_memory_object_1_lock(*shared_memory_object_1_mutex_);

      // Load all of the data into shared_memory_1_
      loadSensorDataUnprotected(goal, 1);
      loadTrajGenParamsUnprotected(goal, 1);
      loadFeedbackControllerParamsUnprotected(goal, 1);
      loadTerminationParamsUnprotected(goal, 1);
      loadTimerParamsUnprotected(goal, 1);

      // The lock of the shared_memory_object_1_mutex_ should be released automatically
    }

    setNewSkillIdInSharedMemoryUnprotected(new_skill_id);
    setNewSkillTypeInSharedMemoryUnprotected(goal->skill_type);
    setNewMetaSkillTypeInSharedMemoryUnprotected(goal->meta_skill_type);
    setNewMetaSkillIdInSharedMemoryUnprotected(goal->meta_skill_id);

    // Set the new skill flag in shared memory to true to signal that a new skill has been loaded into the current free shared memory.
    setNewSkillFlagInSharedMemoryUnprotected(true);

    // Return the skill_id of the current skill
    return new_skill_id;

    // The lock of the run_loop_info_mutex_ should be released automatically
  }

  bool SharedMemoryHandler::getSkillRunningFlagInSharedMemory()
  {
    bool skill_running_flag;
    {
      // Grab the lock of the run_loop_info_mutex_
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> run_loop_info_lock(*run_loop_info_mutex_);

      skill_running_flag = run_loop_process_info_->get_is_running_skill();

      // The lock of the run_loop_info_mutex_ should be released automatically
    }

    // Return the skill_running_flag
    return skill_running_flag;
  }

  int SharedMemoryHandler::getDoneSkillIdInSharedMemory()
  {
    // Grab the lock of the run_loop_info_mutex_
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> run_loop_info_lock(*run_loop_info_mutex_);

    // Return the done_skill_id
    return getDoneSkillIdInSharedMemoryUnprotected();

    // The lock of the run_loop_info_mutex_ should be released automatically
  }

  void SharedMemoryHandler::setSkillPreemptedFlagInSharedMemory(bool skill_preempted_flag)
  {
    // Grab the lock of the run_loop_info_mutex_
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> run_loop_info_lock(*run_loop_info_mutex_);

    // Set skill_preempted_ in run_loop_process_info_ to the input skill_preempted_flag
    run_loop_process_info_->set_skill_preempted(skill_preempted_flag);

    // The lock of the run_loop_info_mutex_ should be released automatically
  }

  franka_action_lib::ExecuteSkillFeedback SharedMemoryHandler::getSkillFeedback()
  {
    franka_action_lib::ExecuteSkillFeedback feedback;

    // Grab the lock of the run_loop_info_mutex_
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> run_loop_info_lock(*run_loop_info_mutex_);

    int current_free_shared_feedback_index = run_loop_process_info_->get_current_free_shared_feedback_index();

    if(current_free_shared_feedback_index == 0)
    {
      // Grab lock of shared_execution_response_0_mutex_ to make sure no one else can modify 0
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> shared_execution_response_0_lock(*shared_execution_response_0_mutex_);

      int num_execution_feedback = static_cast<int>(execution_feedback_buffer_0_[0]);

      std::vector<float> execution_feedback(execution_feedback_buffer_0_ + 1, execution_feedback_buffer_0_ + num_execution_feedback + 1);

      feedback.num_execution_feedback = num_execution_feedback;
      feedback.execution_feedback = execution_feedback;

      // The lock of the shared_execution_response_0_mutex_ should be released automatically
    }
    else if(current_free_shared_feedback_index == 1)
    {
      // Grab lock of shared_execution_response_1_mutex_ to make sure no one else can modify 1
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> shared_execution_response_1_lock(*shared_execution_response_1_mutex_);

      int num_execution_feedback = static_cast<int>(execution_feedback_buffer_1_[0]);

      std::vector<float> execution_feedback(execution_feedback_buffer_1_ + 1, execution_feedback_buffer_1_ + num_execution_feedback + 1);

      feedback.num_execution_feedback = num_execution_feedback;
      feedback.execution_feedback = execution_feedback;

      // The lock of the shared_execution_response_1_mutex_ should be released automatically
    }

    return feedback;

    // The lock of the run_loop_info_mutex_ should be released automatically
  }

  franka_action_lib::ExecuteSkillResult SharedMemoryHandler::getSkillResult(int skill_id)
  {
    franka_action_lib::ExecuteSkillResult result;

    // Grab the lock of the run_loop_info_mutex_
    boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> run_loop_info_lock(*run_loop_info_mutex_);

    int result_memory_index = skill_id % 2;

    if(result_memory_index == 0)
    {
      int num_execution_result = static_cast<int>(execution_result_buffer_0_[0]);

      std::vector<float> execution_result(execution_result_buffer_0_ + 1, execution_result_buffer_0_ + num_execution_result + 1);

      result.num_execution_result = num_execution_result;
      result.execution_result = execution_result;

      // The lock of the shared_execution_response_0_mutex_ should be released automatically
    }
    else if(result_memory_index == 1)
    {
      int num_execution_result = static_cast<int>(execution_result_buffer_1_[0]);

      std::vector<float> execution_result(execution_result_buffer_1_ + 1, execution_result_buffer_1_ + num_execution_result + 1);

      result.num_execution_result = num_execution_result;
      result.execution_result = execution_result;

      // The lock of the shared_execution_response_1_mutex_ should be released automatically
    }

    setResultSkillIdInSharedMemoryUnprotected(skill_id);

    return result;

    // The lock of the run_loop_info_mutex_ should be released automatically
  }

  // ALL UNPROTECTED FUNCTIONS BELOW REQUIRE A MUTEX OVER THE RUN_LOOP_INFO

  int SharedMemoryHandler::getCurrentFreeSharedMemoryIndexInSharedMemoryUnprotected()
  {
    // Return the current_free_shared_memory_index from run_loop_process_info_
    return run_loop_process_info_->get_current_free_shared_memory_index();
  }

  int SharedMemoryHandler::getCurrentSkillIdInSharedMemoryUnprotected()
  {
    // Return the current_skill_id from run_loop_process_info_
    return run_loop_process_info_->get_current_skill_id();
  }

  void SharedMemoryHandler::setCurrentSkillIdInSharedMemoryUnprotected(int current_skill_id)
  {
    // Set current_skill_id_ in run_loop_process_info_ to the input current_skill_id
    run_loop_process_info_->set_current_skill_id(current_skill_id);
  }

  int SharedMemoryHandler::getDoneSkillIdInSharedMemoryUnprotected()
  {
    // Return the done_skill_id from run_loop_process_info_
    return run_loop_process_info_->get_done_skill_id();
  }

  bool SharedMemoryHandler::getNewSkillFlagInSharedMemoryUnprotected()
  {
    // Return the new_skill_available_flag
    return run_loop_process_info_->get_new_skill_available();
  }

  void SharedMemoryHandler::setNewSkillFlagInSharedMemoryUnprotected(bool new_skill_flag)
  {
    // Set new_skill_available_ in run_loop_process_info_ to the input new_skill_flag
    run_loop_process_info_->set_new_skill_available(new_skill_flag);
  }

  int SharedMemoryHandler::getNewSkillIdInSharedMemoryUnprotected()
  {
    // Return the new_skill_id from run_loop_process_info_
    return run_loop_process_info_->get_new_skill_id();
  }

  void SharedMemoryHandler::setNewSkillIdInSharedMemoryUnprotected(int new_skill_id)
  {
    // Set new_skill_id_ in run_loop_process_info_ to the input new_skill_id
    run_loop_process_info_->set_new_skill_id(new_skill_id);
  }

  void SharedMemoryHandler::setNewSkillTypeInSharedMemoryUnprotected(int new_skill_type)
  {
    // Set new_skill_type_ in run_loop_process_info_ to the input new_skill_type
    run_loop_process_info_->set_new_skill_type(new_skill_type);
  }

  void SharedMemoryHandler::setNewMetaSkillIdInSharedMemoryUnprotected(int new_meta_skill_id)
  {
    // Set new_meta_skill_id_ in run_loop_process_info_ to the input new_meta_skill_id
    run_loop_process_info_->set_new_meta_skill_id(new_meta_skill_id);
  }

  void SharedMemoryHandler::setNewMetaSkillTypeInSharedMemoryUnprotected(int new_meta_skill_type)
  {
    // Set new_meta_skill_type_ in run_loop_process_info_ to the input new_meta_skill_type
    run_loop_process_info_->set_new_meta_skill_type(new_meta_skill_type);
  }

  void SharedMemoryHandler::setResultSkillIdInSharedMemoryUnprotected(int result_skill_id)
  {
    // Set result_skill_id_ in run_loop_process_info_ to the input result_skill_id
    run_loop_process_info_->set_result_skill_id(result_skill_id);
  }

  // Loads sensor data into the designated sensor memory buffer
  // Requires a lock on the mutex of the designated sensor memory buffer
  void SharedMemoryHandler::loadSensorDataUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index)
  {
    if(current_free_shared_memory_index == 0)
    {
      // Currently ignoring sensor names and putting everything into the traj_gen_sensor_buffer
      traj_gen_sensor_buffer_0_[0] = static_cast<float>(goal->sensor_value_sizes[0]);
      memcpy(traj_gen_sensor_buffer_0_ + 1, &goal->initial_sensor_values[0], goal->sensor_value_sizes[0] * sizeof(float));
    }
    else if(current_free_shared_memory_index == 1)
    {
      // Currently ignoring sensor names and putting everything into the traj_gen_sensor_buffer
      traj_gen_sensor_buffer_1_[0] = static_cast<float>(goal->sensor_value_sizes[0]);
      memcpy(traj_gen_sensor_buffer_1_ + 1, &goal->initial_sensor_values[0], goal->sensor_value_sizes[0] * sizeof(float));
    }
  }

  // Loads traj gen parameters into the designated current_free_shared_memory_index buffer
  // Requires a lock on the mutex of the designated current_free_shared_memory_index buffer
  void SharedMemoryHandler::loadTrajGenParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index)
  {
    if(current_free_shared_memory_index == 0)
    {
      traj_gen_buffer_0_[0] = static_cast<float>(goal->traj_gen_type);
      traj_gen_buffer_0_[1] = static_cast<float>(goal->num_traj_gen_params);
      memcpy(traj_gen_buffer_0_ + 2, &goal->traj_gen_params[0], goal->num_traj_gen_params * sizeof(float));
    }
    else if(current_free_shared_memory_index == 1)
    {
      traj_gen_buffer_1_[0] = static_cast<float>(goal->traj_gen_type);
      traj_gen_buffer_1_[1] = static_cast<float>(goal->num_traj_gen_params);
      memcpy(traj_gen_buffer_1_ + 2, &goal->traj_gen_params[0], goal->num_traj_gen_params * sizeof(float));
    }
  }

  // Loads feedback controller parameters into the designated current_free_shared_memory_index buffer
  // Requires a lock on the mutex of the designated current_free_shared_memory_index buffer
  void SharedMemoryHandler::loadFeedbackControllerParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index)
  {
    if(current_free_shared_memory_index == 0)
    {
      feedback_controller_buffer_0_[0] = static_cast<float>(goal->feedback_controller_type);
      feedback_controller_buffer_0_[1] = static_cast<float>(goal->num_feedback_controller_params);
      memcpy(feedback_controller_buffer_0_ + 2, &goal->feedback_controller_params[0], goal->num_feedback_controller_params * sizeof(float));
    }
    else if(current_free_shared_memory_index == 1)
    {
      feedback_controller_buffer_1_[0] = static_cast<float>(goal->feedback_controller_type);
      feedback_controller_buffer_1_[1] = static_cast<float>(goal->num_feedback_controller_params);
      memcpy(feedback_controller_buffer_1_ + 2, &goal->feedback_controller_params[0], goal->num_feedback_controller_params * sizeof(float));
    }
  }

  // Loads termination parameters into the designated current_free_shared_memory_index buffer
  // Requires a lock on the mutex of the designated current_free_shared_memory_index buffer
  void SharedMemoryHandler::loadTerminationParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index)
  {
    if(current_free_shared_memory_index == 0)
    {
      termination_buffer_0_[0] = static_cast<float>(goal->termination_type);
      termination_buffer_0_[1] = static_cast<float>(goal->num_termination_params);
      memcpy(termination_buffer_0_ + 2, &goal->termination_params[0], goal->num_termination_params * sizeof(float));
    }
    else if(current_free_shared_memory_index == 1)
    {
      termination_buffer_1_[0] = static_cast<float>(goal->termination_type);
      termination_buffer_1_[1] = static_cast<float>(goal->num_termination_params);
      memcpy(termination_buffer_1_ + 2, &goal->termination_params[0], goal->num_termination_params * sizeof(float));
    }
  }

  // Loads timer parameters into the designated current_free_shared_memory_index buffer
  // Requires a lock on the mutex of the designated current_free_shared_memory_index buffer
  void SharedMemoryHandler::loadTimerParamsUnprotected(const franka_action_lib::ExecuteSkillGoalConstPtr &goal, int current_free_shared_memory_index)
  {
    if(current_free_shared_memory_index == 0)
    {
      timer_buffer_0_[0] = static_cast<float>(goal->timer_type);
      timer_buffer_0_[1] = static_cast<float>(goal->num_timer_params);
      memcpy(timer_buffer_0_ + 2, &goal->timer_params[0], goal->num_timer_params * sizeof(float));
    }
    else if(current_free_shared_memory_index == 1)
    {
      timer_buffer_1_[0] = static_cast<float>(goal->timer_type);
      timer_buffer_1_[1] = static_cast<float>(goal->num_timer_params);
      memcpy(timer_buffer_1_ + 2, &goal->timer_params[0], goal->num_timer_params * sizeof(float));
    }
  }

}
