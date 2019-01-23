//
// Created by Kevin on 11/30/18.
//

#include "goal_linear_trajectory_generator.h"

void GoalLinearTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  if(num_params != 32) {
    std::cout << "Incorrect number of params given: " << num_params << std::endl;
  }

  memcpy(goal_pos_, &params_[2], 16 * sizeof(float));
  memcpy(deltas_, &params_[18], 16 * sizeof(float));
}

void GoalLinearTrajectoryGenerator::initialize_trajectory() {

}

void GoalLinearTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  pose_desired_ = robot_state.O_T_EE_c;
}

void GoalLinearTrajectoryGenerator::get_next_step() {

  for(int i = 0; i < 16; i++) 
  {
    if(velocities_[i] < vel_max_ && time_ < acceleration_time_ && std::fabs(goal_pos_[i] - pose_desired_[i]) > 0.0001) {
      velocities_[i] += dt_ * std::fabs(vel_max_ / acceleration_time_);
    }
    if(velocities_[i] > 0.0 && std::fabs(goal_pos_[i] - pose_desired_[i]) < 0.0001) {
      velocities_[i] -= dt_ * std::fabs(vel_max_ / acceleration_time_);
    }
    velocities_[i] = std::fmax(velocities_[i], 0.0);
    velocities_[i] = std::fmin(velocities_[i], vel_max_);

    pose_desired_[i] += (velocities_[i] * dt_) * deltas_[i];
  }
}

