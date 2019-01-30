#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_COUNTER_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_COUNTER_TRAJECTORY_GENERATOR_H_

#include <franka/robot.h>

#include "iam_robolib/trajectory_generator/trajectory_generator.h"

class CounterTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

  float current_point_[3] = {};
  float delta_ = 0;

 private:
  int start_ = 0;
  int end_ = 0;
  int current_val_ = 0;

  float start_point_[3] = {};
};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_COUNTER_TRAJECTORY_GENERATOR_H_