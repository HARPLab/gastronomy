//
// Created by mohit on 11/30/18.
//

#include "linear_joint_trajectory_controller.h"

#include <cassert>
#include <iostream>
#include <memory.h>

void LinearJointTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  if(num_params != 7) {
    std::cout << "Incorrect number of params given: " << num_params << std::endl;
  }

  memcpy(deltas_, &params_[2], 7 * sizeof(float));
}

void LinearJointTrajectoryGenerator::initialize_trajectory() {
  // assert(false);
}

void LinearJointTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  joint_desired_ = robot_state.q;
  joint_initial_ = robot_state.q;
}

void LinearJointTrajectoryGenerator::get_next_step() {
  // double delta_angle = M_PI / 8.0 * (1 - std::cos(M_PI / 0.625 * time_));
  double delta_angle = (M_PI / 8.0) * (time_ / 4.0);
  double max_delta = (M_PI / 8.0);
  if (delta_angle > max_delta) {
    delta_angle = max_delta;
  }
  //double delta_angle = 0;
  joint_desired_[3] = joint_initial_[3] + delta_angle;
  // joint_desired_[4] = joint_initial_[4] + delta_angle;
  joint_desired_[6] = joint_initial_[6] + delta_angle;
}
  