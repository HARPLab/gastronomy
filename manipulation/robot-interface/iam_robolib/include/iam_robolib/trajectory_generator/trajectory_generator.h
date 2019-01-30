#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_TRAJECTORY_GENERATOR_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_TRAJECTORY_GENERATOR_H_

#include <array>
#include <Eigen/Dense>
#include <franka/robot_state.h>

class TrajectoryGenerator {
 public:
  explicit TrajectoryGenerator(float *p) : params_{p} {};

  /**
   * Parse parameters from memory.
   */
  virtual void parse_parameters() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_trajectory() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_trajectory(const franka::RobotState &robot_state) = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step() = 0;

  std::array<double, 16> pose_desired_{};
  std::array<double, 7> joint_desired_{};
  std::array<double, 6> force_torque_desired_{};
  Eigen::Vector3d desired_position_;
  Eigen::Quaterniond desired_orientation_;

  const double acceleration_time_ = 5.0;
  double run_time_ = 0.0;
  float dt_ = 0.001f;
  double time_ = 0.0;

 protected:
  float *params_=0;

};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_TRAJECTORY_GENERATOR_H_