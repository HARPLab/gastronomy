#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_RELATIVE_MIN_JERK_POSE_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_RELATIVE_MIN_JERK_POSE_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/relative_pose_trajectory_generator.h"

class RelativeMinJerkPoseTrajectoryGenerator : public RelativePoseTrajectoryGenerator {
 public:
  using RelativePoseTrajectoryGenerator::RelativePoseTrajectoryGenerator;

  void get_next_step() override;

 private:
  double slerp_t_ = 0.0;
};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_RELATIVE_MIN_JERK_POSE_TRAJECTORY_GENERATOR_H_