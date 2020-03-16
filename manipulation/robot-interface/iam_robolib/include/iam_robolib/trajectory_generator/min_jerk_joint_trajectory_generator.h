#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_MIN_JERK_JOINT_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_MIN_JERK_JOINT_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/joint_trajectory_generator.h"

class MinJerkJointTrajectoryGenerator : public JointTrajectoryGenerator {
 public:
  using JointTrajectoryGenerator::JointTrajectoryGenerator;

  void get_next_step() override;

};

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_MIN_JERK_JOINT_TRAJECTORY_GENERATOR_H_