//
// Created by Kevin on 11/29/18.
//

#pragma once

#include <array>
#include <cstring>
#include <iostream>
#include <franka/robot.h>
#include <Eigen/Dense>

#include "trajectory_generator.h"

class LinearTrajectoryGeneratorWithTimeAndGoal : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

  Eigen::Vector3d goal_position_;
  Eigen::Quaterniond goal_orientation_;

 private:
  Eigen::Vector3d initial_position_;
  Eigen::Quaterniond initial_orientation_;
  double t_ = 0.0;
};

