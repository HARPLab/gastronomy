//
// Created by Kevin on 11/29/18.
//

#pragma once

#include <array>
#include <cstring>
#include <iostream>
#include <franka/robot.h>
#include <Eigen/Dense>

#include "linear_trajectory_generator_with_time_and_goal.h"

class RelativeLinearTrajectoryGenerator : public LinearTrajectoryGeneratorWithTimeAndGoal {
 public:
  using LinearTrajectoryGeneratorWithTimeAndGoal::LinearTrajectoryGeneratorWithTimeAndGoal;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

 private:
  Eigen::Vector3d initial_position_;
  Eigen::Quaterniond initial_orientation_;
  Eigen::Vector3d relative_position_;
  Eigen::Quaterniond relative_orientation_;
  double t_ = 0.0;
};

