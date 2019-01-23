//
// Created by Kevin on 11/29/18.
//

#include "linear_trajectory_generator_with_time_and_goal.h"

#include <cassert>

void LinearTrajectoryGeneratorWithTimeAndGoal::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  // Time + Full Cartesian Pose (std::array<double,16>) was given
  if(num_params == 17) 
  {
    run_time_ = static_cast<double>(params_[2]);

    std::array<double,16> cartesian_pose_goal{};
    for(int i = 0; i < 16; i++)
    {
      cartesian_pose_goal[i] = static_cast<double>(params_[3+i]);
    }
    Eigen::Affine3d goal_transform(Eigen::Matrix4d::Map(cartesian_pose_goal.data()));
    goal_position_ = Eigen::Vector3d(goal_transform.translation());
    goal_orientation_ = Eigen::Quaterniond(goal_transform.linear());
  } 
  // Time + x,y,z + quaternion was given
  else if(num_params == 8)
  {
    run_time_ = static_cast<double>(params_[2]);

    goal_position_[0] = static_cast<double>(params_[3]);
    goal_position_[1] = static_cast<double>(params_[4]);
    goal_position_[2] = static_cast<double>(params_[5]);

    std::array<double,4> goal_quaternion{};
    for(int i = 0; i < 4; i++)
    {
      goal_quaternion[i] = static_cast<double>(params_[6+i]);
    }
    goal_orientation_ = Eigen::Quaterniond(goal_quaternion[0], goal_quaternion[1], goal_quaternion[2], goal_quaternion[3]);
  } 
  // Time + x,y,z + axis angle was given
  else if(num_params == 7)
  {
    run_time_ = static_cast<double>(params_[2]);

    goal_position_[0] = static_cast<double>(params_[3]);
    goal_position_[1] = static_cast<double>(params_[4]);
    goal_position_[2] = static_cast<double>(params_[5]);

    Eigen::Vector3d goal_axis_angle;
    for(int i = 0; i < 3; i++)
    {
      goal_axis_angle[i] = static_cast<double>(params_[6+i]);
    }

    double angle = goal_axis_angle.norm();
    double sin_angle_divided_by_2 = std::sin(angle/2);
    double cos_angle_divided_by_2 = std::cos(angle/2);

    goal_orientation_ = Eigen::Quaterniond(goal_axis_angle[0] * sin_angle_divided_by_2,
                                           goal_axis_angle[1] * sin_angle_divided_by_2,
                                           goal_axis_angle[2] * sin_angle_divided_by_2,
                                           cos_angle_divided_by_2);
  }
  else
  {
    std::cout << "Invalid number of params provided: " << num_params << std::endl;
  }
}

void LinearTrajectoryGeneratorWithTimeAndGoal::initialize_trajectory() {
  // assert(false);
}

void LinearTrajectoryGeneratorWithTimeAndGoal::initialize_trajectory(const franka::RobotState &robot_state) {
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  initial_position_ = Eigen::Vector3d(initial_transform.translation());
  initial_orientation_ = Eigen::Quaterniond(initial_transform.linear());
}

void LinearTrajectoryGeneratorWithTimeAndGoal::get_next_step() {
  t_ = std::min(std::max(time_ / run_time_, 0.0), 1.0);

  desired_position_ = initial_position_ + (goal_position_ - initial_position_) * t_;
  desired_orientation_ = initial_orientation_.slerp(t_, goal_orientation_);
}

