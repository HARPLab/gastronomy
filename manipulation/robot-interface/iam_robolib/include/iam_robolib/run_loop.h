#ifndef IAM_ROBOLIB_RUN_LOOP_H_
#define IAM_ROBOLIB_RUN_LOOP_H_

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include <iam_robolib_common/run_loop_process_info.h>
#include <iam_robolib_common/SharedMemoryInfo.h>

#include <franka/robot.h>
#include <franka/gripper.h>

// TODO(Mohit): Fix this, CANNOT do private imports in public headers. FML.
#include "iam_robolib/skill_info_manager.h"
#include "iam_robolib/skills/skill_info.h"
#include "iam_robolib/run_loop_logger.h"
#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/run_loop_shared_memory_handler.h"
#include "iam_robolib/trajectory_generator_factory.h"
#include "iam_robolib/feedback_controller_factory.h"
#include "iam_robolib/termination_handler_factory.h"
#include "iam_robolib/definitions.h"

//class BaseSkill;

// Set thread to real time priority.
void setCurrentThreadToRealtime(bool throw_on_error);

// TODO(Mohit): Add a namespace to these declarations.

class run_loop {
 public:
  run_loop(std::mutex& logger_mutex,
          std::mutex& robot_loop_data_mutex) : limit_rate_(false),
                                                 cutoff_frequency_(0.0),
                                                 logger_(logger_mutex),
                                                 elapsed_time_(0.0),
                                                 process_info_requires_update_(false),
                                                 robot_("172.16.0.2"),
                                                 gripper_("172.16.0.2") {
    robot_state_data_ = new RobotStateData(robot_loop_data_mutex);
  };

  // Todo(Mohit): Implement this!!! We should free up the shared memory correctly.
  // ~run_loop();

  bool init();

  /**
   *  Start the RunLoop.
   *
   *  This will allocate the shared memory buffers i.e., shared memory object and
   *  shared memory segment used to communicate between the actionlib interface and
   *  the real time loop.
   */
  void start();

  void stop();

  /**
   *  Update the currently executing task. Maybe we should pass in the TaskInfo or
   *  it should return some task info from it.
   */
  bool update();

  /**
   *  Start running the real time loop.
   */
  void run();

  /**
   *  Start running the real time loop on franka
   */
  void run_on_franka();

  /**
   * Get SkillInfo manager.
   */
  SkillInfoManager* getSkillInfoManager();

  /**
   * Did finish skill in meta skill.
   * @param skill
   */
  void didFinishSkillInMetaSkill(BaseSkill* skill);

  /**
   * Start executing new skill.
   * @param new_skill New skill to start.
   */
  void start_new_skill(BaseSkill* new_skill);

  /**
   *  Finish current executing skill.
   */
  void finish_current_skill(BaseSkill* skill);

  // TODO(jacky): this isn't actually being used. should implement this properly by introducing exit conditions on threads.
  static std::atomic<bool> running_skills_;

  bool start_time;

 private:

  franka::Robot robot_;
  franka::Gripper gripper_;

  std::thread print_thread_{};
  std::thread current_robot_state_io_thread_{};

  RunLoopSharedMemoryHandler* shared_memory_handler_ = nullptr;
  SkillInfoManager skill_manager_{};
  RunLoopLogger logger_;
  // This logs the robot state data by using robot readState and within control loops.
  RobotStateData *robot_state_data_=nullptr;

  // If this flag is true at every loop we will try to get the lock and update process info.
  bool process_info_requires_update_;
  const bool limit_rate_;  // NOLINT(readability-identifier-naming)

  const double cutoff_frequency_; // NOLINT(readability-identifier-naming)
  uint32_t elapsed_time_;

  TrajectoryGeneratorFactory traj_gen_factory_={};
  FeedbackControllerFactory feedback_controller_factory_={};
  TerminationHandlerFactory termination_handler_factory_={};

  /**
   * Check if new skill should be started or not. Starting a new skill
   * initializes it's trajectory generator, feedback controller and other
   * associated things.
   *
   * @param old_skill
   * @param new_skill
   * @return True if new skill should be started else false.
   */
  bool should_start_new_skill(BaseSkill* old_skill, BaseSkill* new_skill);

  /**
   * Update process info in the shared memory to reflect run-loop's
   * current status.
   */
  void update_process_info();

  /**
   * Setup thread to print data from the real time control loop thread.
   */
  void setup_save_robot_state_thread();

  /**
   * Setup thread to save current robot state data to shared memory buffer.
   */
  void setup_current_robot_state_io_thread();

  /**
   * Setup default collision behavior for robot.
   */
  void setup_robot_default_behavior();

  /**
   * Setup data loggers for logging robot state and larger control loop data.
   */
  void setup_data_loggers();

  /**
   * Log skill description to file logger. Only logs once when the skill begins.
   * @param skill Skill to log.
   */
  void log_skill_info(BaseSkill* skill);
};

#endif  // IAM_ROBOLIB_ROBOT_STATE_DATA_H_