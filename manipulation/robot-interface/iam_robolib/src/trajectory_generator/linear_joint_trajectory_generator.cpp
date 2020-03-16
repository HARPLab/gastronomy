//
// Created by mohit on 11/30/18.
//

#include "iam_robolib/trajectory_generator/linear_joint_trajectory_generator.h"

void LinearJointTrajectoryGenerator::get_next_step() {
  t_ = std::min(std::max(time_ / run_time_, 0.0), 1.0);

  for (size_t i = 0; i < desired_joints_.size(); i++) {
    desired_joints_[i] = initial_joints_[i] * (1 - t_) + goal_joints_[i] * t_;
  }
}
  