//
// Created by kevin on 3/27/19.
//

#include "iam_robolib/feedback_controller/set_internal_impedance_feedback_controller.h"

#include <iostream>

void SetInternalImpedanceFeedbackController::parse_parameters() {
  // First parameter is reserved for the type

  int param_idx = 1;
  int num_params = static_cast<int>(params_[param_idx++]);

  switch (num_params) {
    case 0:
      std::cout << "No parameters given, using default internal joint and cartesian impedances."
                << std::endl;
      set_joint_impedance_ = true;
      set_cartesian_impedance_ = true;
      K_theta_ = k_K_theta_;
      K_x_ = k_K_x_;
      break;
    case 6:
      // Cartesian Impedances (6)
      for(size_t i = 0; i < K_x_.size(); i++) {
        K_x_[i] = static_cast<double>(params_[param_idx++]);
      }
      set_joint_impedance_ = false;
      set_cartesian_impedance_ = true;
      break;
    case 7:
      // Joint Impedances (7)
      for(size_t i = 0; i < K_theta_.size(); i++) {
        K_theta_[i] = static_cast<double>(params_[param_idx++]);
      }
      set_joint_impedance_ = true;
      set_cartesian_impedance_ = false;
      break;
    case 13:
      // Joint (7) and Cartesian (6) Impedances
      for(size_t i = 0; i < K_theta_.size(); i++) {
        K_theta_[i] = static_cast<double>(params_[param_idx++]);
      }
      for(size_t i = 0; i < K_x_.size(); i++) {
        K_x_[i] = static_cast<double>(params_[param_idx++]);
      }
      set_joint_impedance_ = true;
      set_cartesian_impedance_ = true;
      break;
    default:
      std::cout << "Invalid number of params provided: " << num_params << std::endl;
  }
}

void SetInternalImpedanceFeedbackController::initialize_controller() {
  // pass
}

void SetInternalImpedanceFeedbackController::initialize_controller(FrankaRobot *robot) {
  if(set_joint_impedance_) {
    bool joint_impedance_values_valid = true;
    for(size_t i = 0; i < K_theta_.size(); i++) {
      if(K_theta_[i] < 0.0 or K_theta_[i] > max_joint_impedance_) {
        joint_impedance_values_valid = false;
      }
    }
    if(joint_impedance_values_valid) {
      robot->setJointImpedance(K_theta_);
    }
  }
  if(set_cartesian_impedance_) {
    bool cartesian_impedance_values_valid = true;
    for(size_t i = 0; i < K_x_.size(); i++) {
      if(K_x_[i] < 0.0 or K_x_[i] > max_cartesian_impedance_) {
        cartesian_impedance_values_valid = false;
      }
    }
    if(cartesian_impedance_values_valid) {
      robot->setCartesianImpedance(K_x_);
    }
  }
}

void SetInternalImpedanceFeedbackController::get_next_step() {
  // pass
}

void SetInternalImpedanceFeedbackController::get_next_step(const franka::RobotState &robot_state, 
                                                           TrajectoryGenerator *traj_generator) {
  // pass
}