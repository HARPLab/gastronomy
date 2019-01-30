//
// Created by jacky on 1/26/19.
//

#include "iam_robolib/trajectory_generator/impulse_trajectory_generator.h"

#include <cassert>
#include <iostream>
#include <memory.h>

void ImpulseTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);
  if (num_params == 7) {
    run_time_ = static_cast<double>(params_[2]);
    for (int i = 0; i < force_torque_desired_.size(); i++) {
      force_torque_desired_[i] = static_cast<double>(params_[i + 3]);
    }
  } else {
    std::cout << "Incorrect number of params given: " << num_params << std::endl;
  }
}

void ImpulseTrajectoryGenerator::initialize_trajectory() {
  // pass
}

void ImpulseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  // pass
}

void ImpulseTrajectoryGenerator::get_next_step() {
  // pass
}
  