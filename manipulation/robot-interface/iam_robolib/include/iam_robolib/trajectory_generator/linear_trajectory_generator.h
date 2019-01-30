#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_LINEAR_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_LINEAR_TRAJECTORY_GENERATOR_H_

#include <cstring>
#include <iostream>

#include "iam_robolib/trajectory_generator/trajectory_generator.h"

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

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_LINEAR_TRAJECTORY_GENERATOR_H_