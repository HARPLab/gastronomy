//
// Created by Kevin on 4/1/19.
//

#include "iam_robolib/trajectory_generator/stay_in_initial_joints_trajectory_generator.h"

#include <iostream>

void StayInInitialJointsTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 0:
      std::cout << "StayInInitialJointsTrajectoryGenerator: No parameters provided. Using default run_time." << std::endl;
      break;
    case 1:
      // Time (1)
      run_time_ = static_cast<double>(params_[params_idx++]);
      break;
    default:
      std::cout << "StayInInitialJointsTrajectoryGenerator: Invalid number of params provided: " << num_params << std::endl;
  }
}

void StayInInitialJointsTrajectoryGenerator::get_next_step() {
  desired_joints_ = initial_joints_;
}