#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_IMPULSE_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_IMPULSE_TRAJECTORY_GENERATOR_H_

#include <array>
#include <Eigen/Dense>

#include "iam_robolib/trajectory_generator/trajectory_generator.h"

class ImpulseTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void initialize_trajectory(const franka::RobotState &robot_state, SkillType skill_type) override;

  void initialize_initial_states(const franka::RobotState &robot_state, SkillType skill_type);

  void get_next_step() override;

  void check_displacement_cap(const franka::RobotState &robot_state);

  const std::array<double, 6>& get_desired_force_torque() const;

  const Eigen::Vector3d& get_initial_position() const;

  const Eigen::Quaterniond& get_initial_orientation() const;

 private:
  double acc_time_ = 0.0;

  std::array<double, 16> initial_pose_{};

  std::array<double, 6> target_force_torque_{};
  bool should_deacc_ = false;

  double max_translation_{0.0};
  double max_rotation_{0.0}; 

  std::array<double, 6> desired_force_torque_{};
  Eigen::Vector3d initial_position_;
  Eigen::Quaterniond initial_orientation_;

  Eigen::Vector3d current_position_;
  Eigen::Quaterniond current_orientation_;
};

#endif	// IAM_ROBOLIB_TRAJECTORY_GENERATOR_IMPULSE_TRAJECTORY_GENERATOR_H_