//
// Created by Kevin on 11/29/18.
//

#include "iam_robolib/trajectory_generator/stay_in_initial_pose_trajectory_generator.h"

#include <iostream>

void StayInInitialPoseTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 0:
      std::cout << "StayInInitialPoseTrajectoryGenerator: No parameters provided. Using default run_time." << std::endl;
      break;
    case 1:
      // Time (1)
      run_time_ = static_cast<double>(params_[params_idx++]);
      break;
    default:
      std::cout << "StayInInitialPoseTrajectoryGenerator: Invalid number of params provided: " << num_params << std::endl;
  }
}

void StayInInitialPoseTrajectoryGenerator::get_next_step() {
  desired_position_ = initial_position_;
  desired_orientation_ = initial_orientation_;
}