#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_POSITION_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_POSITION_TRAJECTORY_GENERATOR_H_

#include <array>
#include <cstring>
#include <iostream>
#include <franka/robot.h>
#include <Eigen/Dense>

#include "iam_robolib/trajectory_generator/trajectory_generator.h"

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

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_POSITION_TRAJECTORY_GENERATOR_H_