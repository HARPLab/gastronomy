//
// Created by mohit on 11/30/18.
//

#pragma once

#include "trajectory_generator.h"

class LinearJointTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

 private:
  float deltas_[7]={};
  std::array<double, 7>joint_initial_={};
};

