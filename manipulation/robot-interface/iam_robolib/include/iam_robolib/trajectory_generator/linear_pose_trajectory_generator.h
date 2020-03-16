#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_LINEAR_POSE_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_LINEAR_POSE_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/pose_trajectory_generator.h"

class LinearPoseTrajectoryGenerator : public PoseTrajectoryGenerator {
 public:
  using PoseTrajectoryGenerator::PoseTrajectoryGenerator;

  void get_next_step() override;
};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_LINEAR_POSE_TRAJECTORY_GENERATOR_H_