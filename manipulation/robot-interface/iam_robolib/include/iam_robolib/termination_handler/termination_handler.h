#ifndef IAM_ROBOLIB_TERMINATION_HANDLER_TERMINATION_HANDLER_H_
#define IAM_ROBOLIB_TERMINATION_HANDLER_TERMINATION_HANDLER_H_

#include <franka/robot_state.h>

#include <iam_robolib_common/run_loop_process_info.h>
#include "iam_robolib/trajectory_generator/trajectory_generator.h"
#include "iam_robolib/robots/franka_robot.h"

class TerminationHandler {
 public:
  explicit TerminationHandler(float *p, RunLoopProcessInfo *r) : params_{p}, run_loop_info_{r} {};

  /**
   * Parse parameters from memory.
   */
  virtual void parse_parameters() = 0;

  /**
   * Initialize termination handler after parameter parsing.
   */
  virtual void initialize_handler() = 0;

  /**
   * Initialize termination handler after parameter parsing.
   */
  virtual void initialize_handler_on_franka(FrankaRobot *robot) = 0;

  /**
   * Should we terminate the current skill.
   */
  virtual bool should_terminate(TrajectoryGenerator *traj_generator) = 0;

  /**
   * Should we terminate the current skill.
   */
  virtual bool should_terminate_on_franka(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) = 0;

  /**
   * Sets done_ to true if preempt flag is true.
   */
  void check_terminate_preempt();

  bool done_ = false;
 protected:
  float *params_ = 0;
  RunLoopProcessInfo *run_loop_info_ = nullptr;
};

#endif  // IAM_ROBOLIB_TERMINATION_HANDLER_TERMINATION_HANDLER_H_