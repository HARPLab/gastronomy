#ifndef IAM_ROBOLIB_RUN_LOOP_SHARED_MEMORY_HANDLER_H_
#define IAM_ROBOLIB_RUN_LOOP_SHARED_MEMORY_HANDLER_H_

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include <iam_robolib_common/iam_robolib_state_info.h>
#include <iam_robolib_common/run_loop_process_info.h>
#include <iam_robolib_common/SharedMemoryInfo.h>

#include <franka/robot.h>
#include <franka/gripper.h>

#include "iam_robolib_common/definitions.h"

class RunLoopSharedMemoryHandler {
 public:

  /**
   *  Start the RunLoop.
   *
   *  This will allocate the shared memory buffers i.e., shared memory object and
   *  shared memory segment used to communicate between the actionlib interface and
   *  the real time loop.
   */
  void start();

  RunLoopProcessInfo* getRunLoopProcessInfo();

  IAMRobolibStateInfo* getIAMRobolibStateInfo();

  boost::interprocess::interprocess_mutex* getRunLoopProcessInfoMutex();

  boost::interprocess::interprocess_mutex* getCurrentRobotStateBufferMutex();

  boost::interprocess::interprocess_mutex* getIAMRobolibStateInfoMutex();

  boost::interprocess::interprocess_mutex* getSensorDataBufferMutex();

  SharedBufferTypePtr getTrajectoryGeneratorBuffer(int memory_region);

  SharedBufferTypePtr getFeedbackControllerBuffer(int memory_region);

  SharedBufferTypePtr getTerminationParametersBuffer(int memory_region);

  SharedBufferTypePtr getExecutionResultBuffer(int memory_region);

  SharedBufferTypePtr getFeedbackResultBuffer(int memory_region);

  SharedBufferTypePtr getCurrentRobotStateBuffer();

  SensorBufferTypePtr getSensorDataBuffer(int memory_region);


  void clearAllBuffers();

 private:
  SharedMemoryInfo shared_memory_info_=SharedMemoryInfo();

  boost::interprocess::interprocess_mutex* run_loop_info_mutex_= nullptr;
  RunLoopProcessInfo* run_loop_info_= nullptr;

  boost::interprocess::interprocess_mutex* iam_robolib_state_info_mutex_= nullptr;
  IAMRobolibStateInfo* iam_robolib_state_info_= nullptr;

  // Managed memory segments
  boost::interprocess::managed_shared_memory managed_shared_memory_{};

  // Managed memory objects
  boost::interprocess::shared_memory_object shared_memory_object_0_{};
  boost::interprocess::shared_memory_object shared_memory_object_1_{};
  boost::interprocess::interprocess_mutex* shared_memory_mutex_0_= nullptr;
  boost::interprocess::interprocess_mutex* shared_memory_mutex_1_= nullptr;


  boost::interprocess::mapped_region region_traj_params_0_{};
  boost::interprocess::mapped_region region_feedback_controller_params_0_{};
  boost::interprocess::mapped_region region_termination_params_0_{};
  boost::interprocess::mapped_region region_timer_params_0_{};

  boost::interprocess::mapped_region region_traj_params_1_{};
  boost::interprocess::mapped_region region_feedback_controller_params_1_{};
  boost::interprocess::mapped_region region_termination_params_1_{};
  boost::interprocess::mapped_region region_timer_params_1_{};

  SharedBufferTypePtr traj_gen_buffer_0_=0;
  SharedBufferTypePtr feedback_controller_buffer_0_=0;
  SharedBufferTypePtr termination_buffer_0_=0;
  SharedBufferTypePtr timer_buffer_0_=0;

  SensorBufferTypePtr sensor_data_buffer_0_=0;

  SharedBufferTypePtr traj_gen_buffer_1_=0;
  SharedBufferTypePtr feedback_controller_buffer_1_=0;
  SharedBufferTypePtr termination_buffer_1_=0;
  SharedBufferTypePtr timer_buffer_1_=0;

  boost::interprocess::interprocess_mutex *sensor_data_mutex_0_= nullptr;

  boost::interprocess::shared_memory_object shared_execution_result_0_{};
  boost::interprocess::shared_memory_object shared_execution_result_1_{};
  boost::interprocess::interprocess_mutex *shared_execution_result_mutex_0_= nullptr;
  boost::interprocess::interprocess_mutex *shared_execution_result_mutex_1_= nullptr;

  boost::interprocess::mapped_region region_execution_feedback_buffer_0{};
  boost::interprocess::mapped_region region_execution_result_buffer_0_{};
  boost::interprocess::mapped_region region_execution_feedback_buffer_1_{};
  boost::interprocess::mapped_region region_execution_result_buffer_1_{};

  boost::interprocess::mapped_region region_sensor_data_0_{};

  SharedBufferTypePtr execution_feedback_buffer_0_=0;
  SharedBufferTypePtr execution_result_buffer_0_=0;
  SharedBufferTypePtr execution_feedback_buffer_1_=0;
  SharedBufferTypePtr execution_result_buffer_1_=0;

  boost::interprocess::shared_memory_object shared_current_robot_state_{};
  boost::interprocess::mapped_region region_current_robot_state_buffer_{};
  SharedBufferTypePtr current_robot_state_buffer_=0;
  boost::interprocess::interprocess_mutex *shared_current_robot_state_mutex_ = nullptr;

};

#endif  // IAM_ROBOLIB_RUN_LOOP_SHARED_MEMORY_HANDLER_H_