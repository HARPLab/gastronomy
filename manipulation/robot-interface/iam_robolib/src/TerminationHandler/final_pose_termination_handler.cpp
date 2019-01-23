//
// Created by mohit on 11/29/18.
//

#include "final_pose_termination_handler.h"

#include <iostream>

#include "TrajectoryGenerator/linear_trajectory_generator.h"

void FinalPoseTerminationHandler::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  if(num_params != 16) {
    std::cout << "Incorrect number of params given: " << num_params << std::endl;
  }

  for (size_t i=0; i < pose_final_.size(); i++) {
    pose_final_[i] = static_cast<double>(params_[2 + i]);
  }
}

void FinalPoseTerminationHandler::initialize_handler() {
  // pass
}

void FinalPoseTerminationHandler::initialize_handler(franka::Robot *robot) {
  // pass
}

bool FinalPoseTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  if(!done_) {
    LinearTrajectoryGenerator *linear_traj_generator =
          static_cast<LinearTrajectoryGenerator *>(trajectory_generator);
    for(size_t i = 0; i < 16; i++) {
      if(fabs(pose_final_[i] - linear_traj_generator->pose_desired_[i]) > 0.0001) {
        return false;
      }
    }
    done_ = true;
  }
  
  return done_;
}

bool FinalPoseTerminationHandler::should_terminate(const franka::RobotState &robot_state, TrajectoryGenerator *trajectory_generator) {
  if(!done_){
    LinearTrajectoryGenerator *linear_traj_generator =
          static_cast<LinearTrajectoryGenerator *>(trajectory_generator);
    for(size_t i = 0; i < 16; i++) {
      if(fabs(pose_final_[i] - linear_traj_generator->pose_desired_[i]) > 0.0001) {
        return false;
      }
    }
    done_ = true;
  }
  
  return done_;
}
