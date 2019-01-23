//
// Created by mohit on 11/26/18.
//

#pragma once

#include "termination_handler.h"

class NoopTerminationHandler : public TerminationHandler {
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

};
