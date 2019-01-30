//
// Created by mohit on 11/25/18.
//

#include <iam_robolib_common/SharedMemoryInfo.h>

#include <cassert>

SharedMemoryInfo::SharedMemoryInfo() {
  // pass
}


std::string SharedMemoryInfo::getSharedMemoryNameForParameters(int index) {
  if (index == 0) {
    return params_memory_name_0_;
  } else if (index == 1) {
    return params_memory_name_1_;
  } else {
    assert(false);
    return "";
  }
}

std::string SharedMemoryInfo::getSharedMemoryNameForObjects() {
  return objects_memory_name_;
}

std::string SharedMemoryInfo::getSharedMemoryNameForSensorData(int index) {
  if (index == 0) {
    return sensor_data_memory_name_0_;
  } else if (index == 1) {
    return sensor_data_memory_name_1_;
  } else {
    assert(false);
    return "";
  }
}

std::string SharedMemoryInfo::getSharedMemoryNameForResults(int index) {
  if (index == 0) {
    return execution_response_name_0_;
  } else if (index == 1) {
    return execution_response_name_1_;
  } else {
    assert(false);
    return "";
  }
}

std::string SharedMemoryInfo::getSharedMemoryNameForCurrentRobotState() {
  return current_robot_state_name_;
}

std::string SharedMemoryInfo::getRunLoopInfoObjectName() {
  return run_loop_info_name_;
}

std::string SharedMemoryInfo::getRunLoopInfoMutexName() {
  return run_loop_info_mutex_name_;
}

std::string SharedMemoryInfo::getParameterMemoryMutexName(int index) {
  if (index == 0) {
    return params_memory_mutex_name_0_;
  } else if (index == 1) {
    return params_memory_mutex_name_1_;
  } else {
    assert(false);
    return "";
  }
}

std::string SharedMemoryInfo::getSensorDataMutexName(int index) {
  if (index == 0) {
    return sensor_data_mutex_name_0_;
  } else if (index == 1) {
    return sensor_data_mutex_name_1_;
  } else {
    assert(false);
    return "";
  }
}

std::string SharedMemoryInfo::getExecutionResponseMutexName(int index) {
  if (index == 0) {
    return execution_response_mutex_name_0_;
  } else if (index == 1) {
    return execution_response_mutex_name_1_;
  } else {
    assert(false);
    return "";
  }
}

std::string SharedMemoryInfo::getCurrentRobotStateMutexName() {
  return current_robot_state_mutex_name_;
}

int SharedMemoryInfo::getParameterMemorySize(int index) {
  if (index == 0) {
    return params_memory_size_0_;
  } else if (index == 1) {
    return params_memory_size_1_;
  } else {
    assert(false);
    return 0;
  }
}

int SharedMemoryInfo::getSensorDataMemorySize() {
  return sensor_buffer_size_;
}

int SharedMemoryInfo::getObjectMemorySize() {
  return objects_memory_size_;
}

int SharedMemoryInfo::getExecutionResponseMemorySize() {
  return execution_response_feedback_size_ + execution_response_return_size_;
}

int SharedMemoryInfo::getCurrentRobotStateMemorySize() {
  return current_robot_state_size_;
}

int SharedMemoryInfo::getSizeForTrajectoryParameters() {
  return trajectory_params_buffer_size_;
}

int SharedMemoryInfo::getOffsetForTrajectoryParameters() {
  return 0;
}

int SharedMemoryInfo::getSizeForTrajectorySensorData() {
  return trajectory_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getOffsetForTrajectorySensorData() {
  return 0;
}

int SharedMemoryInfo::getSizeForFeedbackControllerParameters() {
  return feedback_controller_params_buffer_size_;
}

int SharedMemoryInfo::getOffsetForFeedbackControllerParameters() {
  return trajectory_params_buffer_size_;
}

int SharedMemoryInfo::getSizeForFeedbackControllerSensorData() {
  return feedback_controller_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getOffsetForFeedbackControllerSensorData() {
  return trajectory_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getSizeForTerminationParameters() {
  return termination_params_buffer_size_;
}

int SharedMemoryInfo::getOffsetForTerminationParameters() {
  return trajectory_params_buffer_size_ + feedback_controller_params_buffer_size_;
}

int SharedMemoryInfo::getSizeForTerminationSensorData() {
  return termination_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getOffsetForTerminationSensorData() {
  return trajectory_sensor_data_buffer_size_ + feedback_controller_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getSizeForTimerParameters() {
  return timer_params_buffer_size_;
}

int SharedMemoryInfo::getOffsetForTimerParameters() {
  return (trajectory_params_buffer_size_
    + feedback_controller_params_buffer_size_
    + termination_params_buffer_size_);
}

int SharedMemoryInfo::getSizeForTimerSensorData() {
  return timer_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getOffsetForTimerSensorData() {
  return trajectory_sensor_data_buffer_size_
      + feedback_controller_sensor_data_buffer_size_
      + termination_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getSizeForExtraSensorData() {
  return extra_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getOffsetForExtraSensorData() {
  return trajectory_sensor_data_buffer_size_
      + feedback_controller_sensor_data_buffer_size_
      + termination_sensor_data_buffer_size_
      + timer_sensor_data_buffer_size_;
}

int SharedMemoryInfo::getSizeForExecutionFeedbackData() {
  return execution_response_feedback_size_;
}

int SharedMemoryInfo::getOffsetForExecutionFeedbackData() {
  return 0;
}

int SharedMemoryInfo::getSizeForExecutionReturnData() {
  return execution_response_return_size_;
}

int SharedMemoryInfo::getOffsetForExecutionReturnData() {
  return execution_response_feedback_size_;
}

int SharedMemoryInfo::getSizeForCurrentRobotState() {
  return current_robot_state_size_;
}

int SharedMemoryInfo::getOffsetForCurrentRobotState() {
  return 0;
}