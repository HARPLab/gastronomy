//
// Created by mohit on 11/25/18.
//

#pragma once

#include <array>
#include <franka/robot_state.h>
#include <franka/model.h>
#include "TrajectoryGenerator/trajectory_generator.h"

class FeedbackController {
 public:
  explicit FeedbackController(float *p) : params_{p} {};

  /**
   * Parse parameters from memory.
   */
  virtual void parse_parameters() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_controller() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_controller(franka::Model *model) = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step() = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) = 0;

  std::array<double, 7> tau_d_array_{};

 protected:
  float *params_=0;
};
