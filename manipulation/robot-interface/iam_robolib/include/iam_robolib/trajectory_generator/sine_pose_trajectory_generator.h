#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_SINE_POSE_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_SINE_POSE_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/pose_trajectory_generator.h"

class SinePoseTrajectoryGenerator : public PoseTrajectoryGenerator {
 public:
  using PoseTrajectoryGenerator::PoseTrajectoryGenerator;

  void get_next_step() override;
  
 private:
  double sine_t_ = 0.0;
};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_SINE_POSE_TRAJECTORY_GENERATOR_H_