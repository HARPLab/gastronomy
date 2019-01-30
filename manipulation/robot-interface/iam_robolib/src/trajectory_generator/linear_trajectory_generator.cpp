//
// Created by Kevin on 11/29/18.
//

#include "iam_robolib/trajectory_generator/linear_trajectory_generator.h"

#include <cassert>

void LinearTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  if(num_params != 16) {
    std::cout << "Incorrect number of params given: " << num_params << std::endl;
  }

  memcpy(deltas_, &params_[2], 16 * sizeof(float));
  assert(deltas_[13] < 5.0 && deltas_[14] < 5.0);
}

void LinearTrajectoryGenerator::initialize_trajectory() {
  // assert(false);
}

void LinearTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  pose_desired_ = robot_state.O_T_EE;
}

void LinearTrajectoryGenerator::get_next_step() {
  if(velocity_ < vel_max_ && time_ < run_time_) {
    velocity_ += dt_ * std::fabs(vel_max_ / acceleration_time_);
  }
  if(velocity_ > 0.0 && time_ > run_time_) {
    velocity_ -= dt_ * std::fabs(vel_max_ / acceleration_time_);
  }
  velocity_ = std::fmax(velocity_, 0.0);
  velocity_ = std::fmin(velocity_, vel_max_);

  double period = std::sin(M_PI * time_ / 0.5);  
  pose_desired_[12] += (velocity_ * dt_) * period * deltas_[12];
  for(int i = 13; i < 15; i++) {
    // pose_desired_[i] += (1 - std::cos(velocity_ * dt_)) * deltas_[i];
    double factor = 1.0;
    if (time_ > 3.0) {
      factor = 0.2;
    }
    pose_desired_[i] += (velocity_ * dt_) * deltas_[i] * factor;
  }
}

