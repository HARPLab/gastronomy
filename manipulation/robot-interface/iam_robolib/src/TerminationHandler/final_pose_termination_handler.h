#pragma once

#include "termination_handler.h"

#include <franka/robot.h>

class FinalPoseTerminationHandler : public TerminationHandler {
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
  void initialize_handler(franka::Robot *robot) override;

  /**
   * Should we terminate the current skill.
   */
  bool should_terminate(TrajectoryGenerator *traj_generator) override;

  /**
   * Should we terminate the current skill.
   */
  bool should_terminate(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) override;

 private:
  std::array<double, 16> pose_final_{};
};