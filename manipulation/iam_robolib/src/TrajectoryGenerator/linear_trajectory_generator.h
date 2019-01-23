//
// Created by Kevin on 11/29/18.
//

#pragma once

#include <cstring>
#include <iostream>

#include "trajectory_generator.h"

class LinearTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

  const float vel_max_ = 0.25;
  float deltas_[16]={};

 private:
  float velocity_ = 0.0;
};

