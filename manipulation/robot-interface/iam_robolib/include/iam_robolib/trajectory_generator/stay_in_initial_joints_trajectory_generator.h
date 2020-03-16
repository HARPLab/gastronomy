#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_JOINTS_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_JOINTS_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/joint_trajectory_generator.h"

class StayInInitialJointsTrajectoryGenerator : public JointTrajectoryGenerator {
 public:
  using JointTrajectoryGenerator::JointTrajectoryGenerator;

  void parse_parameters() override;

  void get_next_step() override;
};

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_STAY_IN_INITIAL_JOINTS_TRAJECTORY_GENERATOR_H_