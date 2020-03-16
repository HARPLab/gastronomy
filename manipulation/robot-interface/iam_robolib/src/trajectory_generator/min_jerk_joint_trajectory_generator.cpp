//
// Created by jacky on 2/11/19.
// From https://mika-s.github.io/python/control-theory/trajectory-generation/2017/12/06/trajectory-generation-with-a-minimum-jerk-trajectory.html
//

#include "iam_robolib/trajectory_generator/min_jerk_joint_trajectory_generator.h"

#include <cmath>

void MinJerkJointTrajectoryGenerator::get_next_step() {
  t_ = std::min(std::max(time_ / run_time_, 0.0), 1.0);
  
  for (size_t i = 0; i < desired_joints_.size(); i++) {
    desired_joints_[i] = initial_joints_[i] + (goal_joints_[i] - initial_joints_[i]) * (
            10 * std::pow(t_, 3) - 15 * std::pow(t_, 4) + 6 * std::pow(t_, 5)
        );
  }
}
  