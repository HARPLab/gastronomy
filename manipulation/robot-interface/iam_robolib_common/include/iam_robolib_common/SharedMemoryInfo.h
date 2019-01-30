//
// Created by mohit on 11/25/18.
//

#pragma once

#include <string>

class SharedMemoryInfo {
 public:
  SharedMemoryInfo();

  std::string getSharedMemoryNameForParameters(int index);

  std::string getSharedMemoryNameForObjects();

  std::string getSharedMemoryNameForSensorData(int index);

  /**
   * Shared Memory name for results buffer.
   */
  std::string getSharedMemoryNameForResults(int index);

  std::string getRunLoopInfoObjectName();

  std::string getSharedMemoryNameForCurrentRobotState();

  /**
   * Declare mutex names.
   */
  std::string getRunLoopInfoMutexName();
  std::string getParameterMemoryMutexName(int index);
  std::string getSensorDataMutexName(int index);
  std::string getExecutionResponseMutexName(int index);
  std::string getCurrentRobotStateMutexName();

  /**
   * Get sizes for different buffers.
   */
  int getParameterMemorySize(int index);
  int getSensorDataMemorySize();
  int getExecutionResponseMemorySize();
  int getCurrentRobotStateMemorySize();

  int getObjectMemorySize();

  int getSizeForTrajectoryParameters();
  int getOffsetForTrajectoryParameters();
  int getSizeForTrajectorySensorData();
  int getOffsetForTrajectorySensorData();

  int getSizeForFeedbackControllerParameters();
  int getOffsetForFeedbackControllerParameters();
  int getSizeForFeedbackControllerSensorData();
  int getOffsetForFeedbackControllerSensorData();

  int getSizeForTerminationParameters();
  int getOffsetForTerminationParameters();
  int getSizeForTerminationSensorData();
  int getOffsetForTerminationSensorData();

  int getSizeForTimerParameters();
  int getOffsetForTimerParameters();
  int getSizeForTimerSensorData();
  int getOffsetForTimerSensorData();

  int getOffsetForExtraSensorData();
  int getSizeForExtraSensorData();
 
  int getSizeForCurrentRobotState();
  int getOffsetForCurrentRobotState();

  /**
   * Buffers for execution response, i.e. execution feedback and execution result.
   */
  int getSizeForExecutionFeedbackData();
  int getOffsetForExecutionFeedbackData();
  int getSizeForExecutionReturnData();
  int getOffsetForExecutionReturnData();


 private:
  const std::string params_memory_name_0_ = "run_loop_shared_obj_0";
  const std::string params_memory_name_1_ = "run_loop_shared_obj_1";
  const std::string objects_memory_name_ = "run_loop_shared_memory";
  const std::string sensor_data_memory_name_0_ = "run_loop_sensor_data_0";
  const std::string sensor_data_memory_name_1_ = "run_loop_sensor_data_1";
  const std::string execution_response_name_0_ = "run_loop_execution_response_0";
  const std::string execution_response_name_1_ = "run_loop_execution_response_1";
  const std::string current_robot_state_name_ = "current_robot_state";

  // Object names
  const std::string run_loop_info_name_ = "run_loop_info";
  const std::string run_loop_info_mutex_name_ = "run_loop_info_mutex";

  // Declare mutexes
  const std::string params_memory_mutex_name_0_ = "run_loop_shared_obj_0_mutex";
  const std::string params_memory_mutex_name_1_ = "run_loop_shared_obj_1_mutex";
  const std::string sensor_data_mutex_name_0_ = "run_loop_sensor_data_0_mutex";
  const std::string sensor_data_mutex_name_1_ = "run_loop_sensor_data_1_mutex";
  const std::string execution_response_mutex_name_0_ = "run_loop_execution_response_0_mutex";
  const std::string execution_response_mutex_name_1_ = "run_loop_execution_response_1_mutex";
  const std::string current_robot_state_mutex_name_ = "current_robot_state_mutex";

  // Declare sizes
  const int params_memory_size_0_ = 4 * 1024 * sizeof(float);
  const int params_memory_size_1_ = 4 * 1024 * sizeof(float);
  const int objects_memory_size_ = 1024 * sizeof(float);
  const int sensor_buffer_size_ = 5 * 1024 * sizeof(float);

  const int trajectory_params_buffer_size_= 1024 * sizeof(float);
  const int feedback_controller_params_buffer_size_= 1024 * sizeof(float);
  const int termination_params_buffer_size_= 1024 * sizeof(float);
  const int timer_params_buffer_size_= 1024 * sizeof(float);

  const int trajectory_sensor_data_buffer_size_= 1024 * sizeof(float);
  const int feedback_controller_sensor_data_buffer_size_= 1024 * sizeof(float);
  const int termination_sensor_data_buffer_size_= 1024 * sizeof(float);
  const int timer_sensor_data_buffer_size_ = 1024 * sizeof(float);
  const int extra_sensor_data_buffer_size_ = 1024 * sizeof(float);

  const int execution_response_feedback_size_= 1024 * sizeof(float);
  const int execution_response_return_size_= 1024 * sizeof(float);

  const int current_robot_state_size_= 300 * sizeof(float);

};

