//
// Created by mohit on 12/6/18.
//

#include "custom_gain_torque_controller.h"

#include <iostream>

#include <franka/rate_limiting.h>

void CustomGainTorqueController::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  // No parameters given, using default translational stiffness and rotational stiffness
  if (num_params == 0) {
    std::cout << "No parameters given, using default translational and rotational stiffness."
              << std::endl;
  }
  // translational_stiffness(1) and rotational_stiffness(1) were given
  if (num_params == 14) {
    memcpy(&k_gains_, &params_[2], 7 * sizeof(float));
    memcpy(&d_gains_, &params_[2 + 7], 7 * sizeof(float));
  } else {
    std::cout << "Invalid number of params provided: " << num_params << std::endl;
  }
}

void CustomGainTorqueController::initialize_controller() {}

void CustomGainTorqueController::initialize_controller(franka::Model *model) {
  model_ = model;
}

void CustomGainTorqueController::get_next_step() {}

void CustomGainTorqueController::get_next_step(const franka::RobotState &robot_state,
                                               TrajectoryGenerator *traj_generator) {

  // Read current coriolis terms from model.
  std::array<double, 7> coriolis = model_->coriolis(robot_state);

  // Compute torque command from joint impedance control law.
  // Note: The answer to our Cartesian pose inverse kinematics is always in state.q_d with one
  // time step delay.
  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; i++) {
    tau_d_calculated[i] = k_gains_[i] * (robot_state.q_d[i] - robot_state.q[i])
          - d_gains_[i] * robot_state.dq[i] + coriolis[i];
  }

  // The following line is only necessary if rate limiting is not activate. If we activated
  // rate limiting for the control loop (activated by default), the torque would anyway be
  // adjusted!
  std::array<double, 7> tau_d_rate_limited =
      franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, robot_state.tau_J_d);

  tau_d_array_ = tau_d_rate_limited;
}
