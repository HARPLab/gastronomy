#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_SINE_JOINT_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_SINE_JOINT_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/joint_trajectory_generator.h"

class SineJointTrajectoryGenerator : public JointTrajectoryGenerator {
 public:
  using JointTrajectoryGenerator::JointTrajectoryGenerator;

  void get_next_step() override;

  double sine_t_ = 0.0;

};

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_SINE_JOINT_TRAJECTORY_GENERATOR_H_