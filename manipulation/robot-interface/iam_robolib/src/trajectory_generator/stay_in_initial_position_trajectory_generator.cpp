//
// Created by Kevin on 11/29/18.
//

#include "iam_robolib/trajectory_generator/stay_in_initial_position_trajectory_generator.h"

#include <cassert>

void StayInInitialPositionTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  if(num_params == 0) 
  {
    std::cout << "StayInInitialPositionTrajectoryGenerator: No parameters provided. Using default run_time. " << std::endl;
  } 
  // Time
  else if(num_params == 1) 
  {
    run_time_ = static_cast<double>(params_[2]);
  } 
  else
  {
    std::cout << "StayInInitialPositionTrajectoryGenerator: Invalid number of params provided: " << num_params << std::endl;
  }
}

void StayInInitialPositionTrajectoryGenerator::initialize_trajectory() {
  // assert(false);
}

void StayInInitialPositionTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  initial_position_ = Eigen::Vector3d(initial_transform.translation());
  initial_orientation_ = Eigen::Quaterniond(initial_transform.linear());
}

void StayInInitialPositionTrajectoryGenerator::get_next_step() {
  desired_position_ = initial_position_;
  desired_orientation_ = initial_orientation_;
}

