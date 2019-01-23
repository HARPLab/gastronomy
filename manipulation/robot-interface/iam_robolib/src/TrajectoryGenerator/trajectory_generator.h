//
// Created by mohit on 11/21/18.
//

#pragma once

#include <array>
#include <Eigen/Dense>
#include <franka/robot_state.h>

class TrajectoryGenerator {
 public:
  explicit TrajectoryGenerator(float *p) : params_{p} {};

  /**
   * Parse parameters from memory.
   */
  virtual void parse_parameters() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_trajectory() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_trajectory(const franka::RobotState &robot_state) = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step() = 0;

  std::array<double, 16> pose_desired_{};
  std::array<double, 7> joint_desired_{};
  Eigen::Vector3d desired_position_;
  Eigen::Quaterniond desired_orientation_;

  const double acceleration_time_ = 5.0;
  float run_time_ = 20.5;
  float dt_ = 0.001;
  double time_ = 0.0;

 protected:
  float *params_=0;

};

