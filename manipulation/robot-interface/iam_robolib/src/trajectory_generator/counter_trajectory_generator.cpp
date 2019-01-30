//
// Created by mohit on 11/21/18.
//

#include "iam_robolib/trajectory_generator/counter_trajectory_generator.h"

#include <iostream>
#include <cassert>

void CounterTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  // Shared test app
//  start_ = (int)params_[1];
//  end_ = (int)params_[2];
//  delta_ = (int)params_[3];
  start_point_[0] = params_[2];
  start_point_[1] = params_[3];
  start_point_[2] = params_[4];
}

void CounterTrajectoryGenerator::initialize_trajectory() {
  // current_val_ = start_;
  for (int i=0; i < 3; i++) {
    current_point_[i] = start_point_[i];
  }
  std::cout << "Initial traj point: " << current_point_[0] << ", " << current_point_[1] << ", "
    << current_point_[2] << std::endl;
}

void CounterTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state)
{
  assert(false);
}

void CounterTrajectoryGenerator::get_next_step() {
  std::cout << "delta: " << delta_ << std::endl;
  std::cout << "Current location: ";
  for (int i=0; i < 3; i++) {
    current_point_[i] += delta_;
    std::cout << current_point_[i] << ", ";
  }
  std::cout << "\n";
}

