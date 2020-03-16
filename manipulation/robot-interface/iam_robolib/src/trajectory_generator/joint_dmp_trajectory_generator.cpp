//
// Created by mohit on 12/3/18.
//

#include "iam_robolib/trajectory_generator/joint_dmp_trajectory_generator.h"

#include <cmath>
#include <iostream>

void JointDmpTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 109:
      // Run time(1) + Tau (1) + alpha(1) + beta(1) + num_basis = 6 (1) + 
      // num_sensor_values = 2 (1) + basis_mean (6) + basis_std(6) + 
      // initial_y0(7) + weights (7 joints * 6 basis functions * 2 sensor inputs)
      {
        run_time_ = static_cast<double>(params_[params_idx++]);
        tau_ = static_cast<double>(params_[params_idx++]);
        alpha_  = static_cast<double>(params_[params_idx++]);
        beta_ = static_cast<double>(params_[params_idx++]);
        num_basis_ = static_cast<int>(params_[params_idx++]);
        num_sensor_values_ = static_cast<int>(params_[params_idx++]);

        // Get the mean and std for the basis functions
        for (int i = 0; i < num_basis_; i++) {
          basis_mean_[i] = static_cast<double>(params_[params_idx++]);
        }

        for (int i = 0; i < num_basis_; i++) {
          basis_std_[i] = static_cast<double>(params_[params_idx++]);
        }

        for (size_t i = 0; i < y0_.size(); i++) {
          y0_[i] = static_cast<double>(params_[params_idx++]);
        }

        for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_sensor_values_; j++) {
            for (int k = 0; k < num_basis_; k++) {
              weights_[i][j][k] = static_cast<double>(params_[params_idx++]);
            }
          }
        }

        // TODO(Mohit): We need to start using sensor values in our trajectory generator and feedback controller.
        for (int i = 0; i < num_sensor_values_; i++) {
          initial_sensor_values_[i] = 1.0;
        }
      }
      break;
    default:
      std::cout << "JointDmpTrajectoryGenerator: Invalid number of parameters: " << num_params << std::endl;
  }
}

void JointDmpTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  initialize_initial_and_desired_joints(robot_state, SkillType::JointPositionSkill);
  
  y0_ = robot_state.q_d;
  y_ = robot_state.q_d;
  dy_ = robot_state.dq_d;
  x_ = 1.0;
}

void JointDmpTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state,
                                                        SkillType skill_type) {
  initialize_initial_and_desired_joints(robot_state, skill_type);
  
  y0_ = robot_state.q_d;
  y_ = robot_state.q_d;
  dy_ = robot_state.dq_d;
  x_ = 1.0;
}


void JointDmpTrajectoryGenerator::get_next_step() {
  static int i, j, k;
  static double ddy, t;

  static std::array<double, 20> factor{};
  static double den = 0.;
  static double net_sensor_force;
  static double sensor_feature;

  // Calculate feature values.
  den = 0.;
  for (k = 0; k < num_basis_; k++) {
    factor[k] = exp(-basis_std_[k] * pow((x_ - basis_mean_[k]), 2));
    // first basis is for the min-jerk feature
    if (k > 0) {
      den += factor[k];
    }
  }

  for (k = 1; k < num_basis_; k++) {
    // factor[k] = (factor[k] * x_) / (den * basis_mean_[k]);
    factor[k] = (factor[k] * x_) / (den + 1e-8);
  }
  t = fmin(-log(x_)/tau_, 1);
  // TODO(Mohit): Shouldn't the below index be 0?
  factor[0] = pow(t, 3) * (6*pow(t, 2) - 15 * t + 10);

  for (i = 0; i < num_dims_; i++) {
    ddy = (alpha_ * (beta_ * (y0_[i] - y_[i]) - tau_ * dy_[i]));
    net_sensor_force = 0;
    for (j = 0; j < num_sensor_values_; j++) {
      sensor_feature = 0;
      for (k=0; k < num_basis_; k++) {
        sensor_feature += (factor[k] * weights_[i][j][k]);
      }
      net_sensor_force += (initial_sensor_values_[j] * sensor_feature);
    }
    ddy += (alpha_ * beta_ * net_sensor_force);
    ddy /= (tau_ * tau_);
    dy_[i] += (ddy * dt_);
    y_[i] += (dy_[i] * dt_);
  }

  // Update canonical system.
  x_ -= (x_ * tau_) * dt_;

  // Finally set the joints we want.
  desired_joints_ = y_;
}

void JointDmpTrajectoryGenerator::getInitialMeanAndStd() {
  std::array<double, 10> basis_mean{};
  std::array<double, 10> basis_std{};
  for (int i = 0; i < num_basis_; i++)  {
    basis_mean[i] = exp(-(i)*0.5/(num_basis_-1));
  }
  for (int i = 0; i < num_basis_ - 1; i++) {
    basis_std[i] = 0.5 / (0.65 * pow(basis_mean[i+1] - basis_mean[i], 2));
  }
  basis_std[num_basis_ - 1] = basis_std[num_basis_ - 2];
}