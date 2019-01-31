//
// Created by mohit on 11/30/18.
//

#include "iam_robolib/termination_handler/final_joint_termination_handler.h"

#include <iostream>

#include "iam_robolib/trajectory_generator/linear_joint_trajectory_generator.h"

void FinalJointTerminationHandler::parse_parameters() {
  num_params_ = static_cast<int>(params_[1]);

  if(num_params_ == 0)
  {
    std::cout << "No parameters given, using default buffer time and error thresholds." << std::endl;
  }
  // buffer_time(1) 
  else if(num_params_ == 1)
  {
    buffer_time_ = static_cast<double>(params_[2]);
  }
}

void FinalJointTerminationHandler::initialize_handler() {
  // pass
}

void FinalJointTerminationHandler::initialize_handler_on_franka(FrankaRobot *robot) {
  // pass
}

bool FinalJointTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();

  if (!done_) {
    LinearJointTrajectoryGenerator *linear_joint_traj_generator =
        static_cast<LinearJointTrajectoryGenerator *>(trajectory_generator);

    if(linear_joint_traj_generator->time_ > linear_joint_traj_generator->run_time_ + buffer_time_)
    {
      done_ = true;
      return true;
    }

    for(size_t i = 0; i < linear_joint_traj_generator->joint_goal_.size(); i++) {
      if(fabs(linear_joint_traj_generator->joint_goal_[i] - linear_joint_traj_generator->joint_desired_[i]) > 0.0001) {
        return false;
      }
    }
  }
  return done_;
}

bool FinalJointTerminationHandler::should_terminate_on_franka(const franka::RobotState &_, 
                                                              TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();
  
  if(!done_)
  {
    LinearJointTrajectoryGenerator *linear_joint_traj_generator =
        static_cast<LinearJointTrajectoryGenerator *>(trajectory_generator);

    if(linear_joint_traj_generator->time_ > linear_joint_traj_generator->run_time_ + buffer_time_)
    {
      done_ = true;
      return true;
    }

    for(size_t i = 0; i < linear_joint_traj_generator->joint_goal_.size(); i++) {
      if(fabs(linear_joint_traj_generator->joint_goal_[i] - linear_joint_traj_generator->joint_desired_[i]) > 0.0001) {
        return false;
      }
    }
  }
  return done_;
}

