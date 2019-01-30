#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_RELATIVE_LINEAR_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_RELATIVE_LINEAR_TRAJECTORY_GENERATOR_H_

#include <array>
#include <cstring>
#include <iostream>
#include <franka/robot.h>
#include <Eigen/Dense>

#include "iam_robolib/trajectory_generator/linear_trajectory_generator_with_time_and_goal.h"

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

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_RELATIVE_LINEAR_TRAJECTORY_GENERATOR_H_