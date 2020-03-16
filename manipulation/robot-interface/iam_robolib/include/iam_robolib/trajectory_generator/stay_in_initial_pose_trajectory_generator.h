#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_POSE_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_POSE_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/pose_trajectory_generator.h"

class StayInInitialPoseTrajectoryGenerator : public PoseTrajectoryGenerator {
 public:
  using PoseTrajectoryGenerator::PoseTrajectoryGenerator;

  void parse_parameters() override;

  void get_next_step() override;
};

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_POSE_TRAJECTORY_GENERATOR_H_