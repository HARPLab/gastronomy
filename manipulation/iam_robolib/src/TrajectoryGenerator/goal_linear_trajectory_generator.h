//
// Created by Kevin on 11/29/18.
//

#pragma once

#include <cstring>
#include <iostream>
#include <franka/robot.h>

#include "linear_trajectory_generator.h"

class GoalLinearTrajectoryGenerator : public LinearTrajectoryGenerator {
 public:
  using LinearTrajectoryGenerator::LinearTrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

 private:
  float goal_pos_[16]={};
  float velocities_[16] = {};
};

