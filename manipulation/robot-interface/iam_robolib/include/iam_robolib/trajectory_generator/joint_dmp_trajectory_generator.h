#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_JOINT_DMP_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_JOINT_DMP_TRAJECTORY_GENERATOR_H_

#include "iam_robolib/trajectory_generator/joint_trajectory_generator.h"

class JointDmpTrajectoryGenerator : public JointTrajectoryGenerator {
 public:
  using JointTrajectoryGenerator::JointTrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void initialize_trajectory(const franka::RobotState &robot_state,
                             SkillType skill_type) override;

  void get_next_step() override;

  std::array<double, 7> y_={};
  std::array<double, 7> dy_={};

 private:
  // Variables initialized from shared memory should be doubles.
  double alpha_=5.0;
  double beta_=5.0/4.0;
  double tau_=0.0;
  double x_=1.0;
  int num_basis_=6;
  int num_dims_=7;
  int num_sensor_values_=2;
  std::array<double, 6> basis_mean_{};
  std::array<double, 6> basis_std_{};
  std::array<std::array<std::array<double, 6>, 2>, 7> weights_{};
  std::array<double, 2> initial_sensor_values_{{1.0, 1.0}};
  std::array<double, 7> y0_={};

  void getInitialMeanAndStd();
};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_JOINT_DMP_TRAJECTORY_GENERATOR_H_