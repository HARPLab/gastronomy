//
// Created by mohit on 11/29/18.
//

#include "iam_robolib/termination_handler/final_pose_termination_handler.h"

#include <iostream>
#include <exception>
#include <Eigen/Dense>

#include "iam_robolib/trajectory_generator/pose_trajectory_generator.h"

void FinalPoseTerminationHandler::parse_parameters() {
  // First parameter is reserved for the type

  num_params_ = static_cast<int>(params_[1]);

  if(num_params_ == 0) {
    std::cout << "No parameters given, using default buffer time and error thresholds." << std::endl;
  }
  // buffer_time(1) 
  else if(num_params_ == 1) {
    buffer_time_ = static_cast<double>(params_[2]);
  }
  // position_error_threshold (1) + orientation_error_threshold (1)
  else if(num_params_ == 2) {
    position_threshold_ = static_cast<double>(params_[2]);
    orientation_threshold_ = static_cast<double>(params_[3]);
  }
  // buffer_time (1) + position_error_threshold (1) + orientation_error_threshold (1)
  else if(num_params_ == 3) {
    buffer_time_ = static_cast<double>(params_[2]);
    position_threshold_ = static_cast<double>(params_[3]);
    orientation_threshold_ = static_cast<double>(params_[4]);
  }  
  // position_error_threshold (3) + orientation_error_threshold (3)
  else if(num_params_ == 6) {
    position_thresholds_[0] = static_cast<double>(params_[2]);
    position_thresholds_[1] = static_cast<double>(params_[3]);
    position_thresholds_[2] = static_cast<double>(params_[4]);
    orientation_thresholds_[0] = static_cast<double>(params_[5]);
    orientation_thresholds_[1] = static_cast<double>(params_[6]);
    orientation_thresholds_[2] = static_cast<double>(params_[7]);
  }
  // buffer_time (1) + position_error_threshold (3) + orientation_error_threshold (3)
  else if(num_params_ == 7) {
    buffer_time_ = static_cast<double>(params_[2]);
    position_thresholds_[0] = static_cast<double>(params_[3]);
    position_thresholds_[1] = static_cast<double>(params_[4]);
    position_thresholds_[2] = static_cast<double>(params_[5]);
    orientation_thresholds_[0] = static_cast<double>(params_[6]);
    orientation_thresholds_[1] = static_cast<double>(params_[7]);
    orientation_thresholds_[2] = static_cast<double>(params_[8]);
  }
  else {
    std::cout << "Invalid number of params provided: " << num_params_ << std::endl;
  }
}

void FinalPoseTerminationHandler::initialize_handler() {
  // pass
}

void FinalPoseTerminationHandler::initialize_handler_on_franka(FrankaRobot *robot) {
  // pass
}

// WARNING since this function does not have robot state, it is using the desired position and orientation from the 
// trajectory generator to check for termination. If you would like to use the actual position and orientation from the
// robot state to check for termination, use the should_terminate function with robot state below.
bool FinalPoseTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();
  
  if(!done_){
    PoseTrajectoryGenerator *pose_trajectory_generator =
          dynamic_cast<PoseTrajectoryGenerator *>(trajectory_generator);

    if(pose_trajectory_generator == nullptr) {
      throw std::bad_cast();
    }

    if(pose_trajectory_generator->time_ > pose_trajectory_generator->run_time_ + buffer_time_) {
      done_ = true;
      return true;
    }

    Eigen::Vector3d position_error = pose_trajectory_generator->get_goal_position() - 
                                     pose_trajectory_generator->get_desired_position();
    
    Eigen::Quaterniond goal_orientation(pose_trajectory_generator->get_goal_orientation());
    Eigen::Quaterniond desired_orientation(pose_trajectory_generator->get_desired_orientation());

    if (goal_orientation.coeffs().dot(desired_orientation.coeffs()) < 0.0) {
      desired_orientation.coeffs() << -desired_orientation.coeffs();
    }
    Eigen::Quaterniond error_quaternion(desired_orientation * goal_orientation.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    // compute "orientation error"
    Eigen::Vector3d orientation_error = error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();
    
    if(num_params_ == 6) {
      for(int i=0; i<3; i++) {
        if(std::abs(position_error[i]) > position_thresholds_[i] || 
           std::abs(orientation_error[i]) > orientation_thresholds_[i]) {
          return false;
        }
      }
      done_ = true;
    }
    else if(num_params_ == 2 && position_error.norm() < position_threshold_ && 
            orientation_error.norm() < orientation_threshold_) {
      done_ = true;
    }
    else {
      return false;
    }
  }
  return done_;
}


bool FinalPoseTerminationHandler::should_terminate_on_franka(const franka::RobotState &robot_state, 
                                                            franka::Model *model,
                                                             TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();
  check_terminate_virtual_wall_collisions(robot_state, model);

  if(!done_){
    PoseTrajectoryGenerator *pose_trajectory_generator =
          dynamic_cast<PoseTrajectoryGenerator *>(trajectory_generator);

    if(pose_trajectory_generator == nullptr) {
      throw std::bad_cast();
    }

    // Terminate if the skill time_ has exceeded the provided run_time_ + buffer_time_ 
    if(pose_trajectory_generator->time_ > pose_trajectory_generator->run_time_ + buffer_time_) {
      done_ = true;
      return true;
    }


    // Terminate immediately if collision is detected
    std::array<double, 6> cartesian_collision = robot_state.cartesian_collision;

    for(int i = 0; i < 6; i++) {
      if(cartesian_collision[i] != 0) {
        done_ = true;
        return true;
      }
    }

    Eigen::Affine3d current_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    Eigen::Vector3d current_position(current_transform.translation());
    Eigen::Quaterniond current_orientation(current_transform.linear());

    Eigen::Vector3d position_error = pose_trajectory_generator->get_goal_position() - current_position;
    
    Eigen::Quaterniond goal_orientation(pose_trajectory_generator->get_goal_orientation());

    if (goal_orientation.coeffs().dot(current_orientation.coeffs()) < 0.0) {
      current_orientation.coeffs() << -current_orientation.coeffs();
    }

    Eigen::Quaterniond error_quaternion(current_orientation * goal_orientation.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    // compute "orientation error"
    Eigen::Vector3d orientation_error = error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();
    
    if(num_params_ == 6) {
      for(int i=0; i<3; i++) {
        if(std::abs(position_error[i]) > position_thresholds_[i] || 
           std::abs(orientation_error[i]) > orientation_thresholds_[i]) {
          return false;
        }
      }
      done_ = true;
    }
    else if(num_params_ == 2 && position_error.norm() < position_threshold_ && 
            orientation_error.norm() < orientation_threshold_) {
      done_ = true;
    }
    else {
      return false;
    }
  }
  return done_;
  
}

