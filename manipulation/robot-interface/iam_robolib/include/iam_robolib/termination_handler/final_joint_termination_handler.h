#ifndef IAM_ROBOLIB_TERMINATION_HANDLER_FINAL_JOINT_TERMINATION_HANDLER_H_
#define IAM_ROBOLIB_TERMINATION_HANDLER_FINAL_JOINT_TERMINATION_HANDLER_H_

#include "iam_robolib/termination_handler/termination_handler.h"

class FinalJointTerminationHandler :public TerminationHandler{
 public:
  using TerminationHandler::TerminationHandler;

  /**
   * Parse parameters from memory.
   */
  void parse_parameters() override;

  /**
   * Initialize termination handler after parameter parsing.
   */
  void initialize_handler() override;

  /**
   * Initialize termination handler after parameter parsing.
   */
  void initialize_handler_on_franka(FrankaRobot *robot) override;

  /**
   * Should we terminate the current skill.
   */
  bool should_terminate(TrajectoryGenerator *traj_generator) override;

  /**
   * Should we terminate the current skill.
   */
  bool should_terminate_on_franka(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) override;

 private:
  double buffer_time_ = 0.0;
  int num_params_;

};

#endif  // IAM_ROBOLIB_TERMINATION_HANDLER_FINAL_JOINT_TERMINATION_HANDLER_H_