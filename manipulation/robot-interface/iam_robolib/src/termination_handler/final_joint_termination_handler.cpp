//
// Created by mohit on 11/30/18.
//

#include "iam_robolib/termination_handler/final_joint_termination_handler.h"

#include <iostream>
#include <exception>

#include "iam_robolib/trajectory_generator/joint_trajectory_generator.h"

void FinalJointTerminationHandler::parse_parameters() {
  num_params_ = static_cast<int>(params_[1]);

  if(num_params_ == 0) {
    std::cout << "No parameters given, using default buffer time and error thresholds." << std::endl;
  }
  // buffer_time(1) 
  else if(num_params_ == 1) {
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
    JointTrajectoryGenerator *joint_traj_generator =
        dynamic_cast<JointTrajectoryGenerator *>(trajectory_generator);

    if(joint_traj_generator == nullptr) {
      throw std::bad_cast();
    }

    if(joint_traj_generator->time_ > joint_traj_generator->run_time_ + buffer_time_) {
      done_ = true;
      return true;
    }

    std::array<double, 7> desired_joints = joint_traj_generator->get_desired_joints();
    std::array<double, 7> goal_joints = joint_traj_generator->get_goal_joints();

    for(size_t i = 0; i < goal_joints.size(); i++) {
      if(fabs(goal_joints[i] - desired_joints[i]) > 0.0001) {
        return false;
      }
    }

    done_ = true;
  }
  
  return done_;
}

bool FinalJointTerminationHandler::should_terminate_on_franka(const franka::RobotState &robot_state, 
                                                              franka::Model *model,
                                                              TrajectoryGenerator *trajectory_generator) {

  check_terminate_virtual_wall_collisions(robot_state, model);
  return should_terminate(trajectory_generator);
}

