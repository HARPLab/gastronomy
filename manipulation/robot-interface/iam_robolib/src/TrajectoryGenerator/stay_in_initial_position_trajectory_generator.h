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

class StayInInitialPositionTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

 private:
  Eigen::Vector3d initial_position_;
  Eigen::Quaterniond initial_orientation_;
};

