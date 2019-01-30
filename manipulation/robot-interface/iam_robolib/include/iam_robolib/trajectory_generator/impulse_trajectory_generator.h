#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_IMPULSE_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_IMPULSE_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/trajectory_generator.h"

class ImpulseTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

};

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_IMPULSE_TRAJECTORY_GENERATOR_H_