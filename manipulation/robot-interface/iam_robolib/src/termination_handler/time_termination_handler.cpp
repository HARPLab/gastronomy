//
// Created by mohit on 11/26/18.
//

#include "iam_robolib/termination_handler/time_termination_handler.h"

#include <iostream>

void TimeTerminationHandler::parse_parameters() {
  
  int num_params = static_cast<int>(params_[1]);

  if(num_params == 0) {
    buffer_time_ = 0.0;
    std::cout << "No parameters given, using default buffer time." << std::endl;
  } else if(num_params == 1) {
    buffer_time_ = static_cast<double>(params_[2]);
  } else {
    buffer_time_ = 0.0;
  	std::cout << "TimeTerminationHandler Error: invalid number of params provided: " << num_params << std::endl;
  }
}

void TimeTerminationHandler::initialize_handler() {
  // pass
}

void TimeTerminationHandler::initialize_handler_on_franka(FrankaRobot *robot) {
  // pass
}

bool TimeTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();

  if(!done_) {
    // Terminate if the skill time_ has exceeded the provided run_time_ + buffer_time_
    if(trajectory_generator->time_ > trajectory_generator->run_time_ + buffer_time_) {
      done_= true;
    }
  }
  
  return done_;
}

bool TimeTerminationHandler::should_terminate_on_franka(const franka::RobotState &robot_state, TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();
  
  if(!done_) {
    // Terminate if the skill time_ has exceeded the provided run_time_ + buffer_time_
    if(trajectory_generator->time_ > trajectory_generator->run_time_ + buffer_time_) {
      done_= true;
    }
  }
  
  return done_;
}
