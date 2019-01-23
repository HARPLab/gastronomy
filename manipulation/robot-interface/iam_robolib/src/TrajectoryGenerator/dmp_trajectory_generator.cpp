//
// Created by mohit on 12/3/18.
//

#include "dmp_trajectory_generator.h"

#include <cstdlib>
#include <cmath>
#include <iostream>

void DmpTrajectoryGenerator::getInitialMeanAndStd() {
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

void DmpTrajectoryGenerator::parse_parameters() {
  int num_params = static_cast<int>(params_[1]);

  // Tau (1) + num_basis = 7 (1) + num_sensor_values = 10 (1) + initial_y0(7) + weights (7 joints * 20 basis functions * 10 sensor inputs)
  if(num_params == 109) {
    run_time_ = params_[2];
    tau_ = static_cast<double>(params_[3]);
    alpha_  = static_cast<double>(params_[4]);
    beta_ = static_cast<double>(params_[5]);
    num_basis_ = static_cast<int>(params_[6]);
    num_sensor_values_ = static_cast<int>(params_[7]);

    // Get the mean and std for the basis functions
    memcpy(&basis_mean_, &params_[8], num_basis_ * sizeof(float));
    memcpy(&basis_std_, &params_[8+num_basis_], num_basis_ * sizeof(float));
    memcpy(&y0_, &params_[8 + 2*num_basis_], 7 * sizeof(float));

    // memcpy(&weights_, &params_[8 + 2*num_basis_ + 7], num_dims_ * num_sensor_values_ * num_basis_ * sizeof(float));
    int params_start_idx = 8 + 2*num_basis_ + 7;
    for (int i = 0; i < num_dims_; i++) {
      for (int j = 0; j < num_sensor_values_; j++) {
        for (int k = 0; k < num_basis_;k++) {
          weights_[i][j][k] = params_[params_start_idx];
          params_start_idx += 1;
        }
      }
    }

    // TODO(Mohit): We need to start using sensor values in our trajectory generator and feedback controller.
    for (int i = 0; i < num_sensor_values_; i++) {
      initial_sensor_values_[i] = 1.0;
    }
  } else {
    std::cout << "DmpTrajectoryGenerator Invalid number of parameters: " << num_params << std::endl;
  }
}

void DmpTrajectoryGenerator::initialize_trajectory() {
  // assert(false);
}

void DmpTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  // TODO: Should we use desired joint values here?
  for (size_t i = 0; i < y0_.size(); i++) {
    y0_[i] = static_cast<float>(robot_state.q_d[i]);
  }
  y_ = robot_state.q;
  dy_ = robot_state.dq;
  x_ = 1.0;
}


void DmpTrajectoryGenerator::get_next_step() {
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
  joint_desired_ = y_;
}

