#include "iam_robolib/trajectory_generator/relative_pose_trajectory_generator.h"

#include <iostream>

void RelativePoseTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 17:
      // Time + Full Cartesian Pose (std::array<double,16>) was given
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        for(size_t i = 0; i < relative_pose_.size(); i++) {
          relative_pose_[i] = static_cast<double>(params_[params_idx++]);
        }
        Eigen::Affine3d goal_transform(Eigen::Matrix4d::Map(relative_pose_.data()));
        relative_position_ = Eigen::Vector3d(goal_transform.translation());
        relative_orientation_ = Eigen::Quaterniond(goal_transform.linear());
      }
      break;
    case 8:
      // Time + x,y,z + quaternion was given
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        relative_position_[0] = static_cast<double>(params_[params_idx++]);
        relative_position_[1] = static_cast<double>(params_[params_idx++]);
        relative_position_[2] = static_cast<double>(params_[params_idx++]);

        std::array<double,4> relative_quaternion{};
        for(size_t i = 0; i < relative_quaternion.size(); i++) {
          relative_quaternion[i] = static_cast<double>(params_[params_idx++]);
        }
        relative_orientation_ = Eigen::Quaterniond(relative_quaternion[0],
                                                   relative_quaternion[1],
                                                   relative_quaternion[2],
                                                   relative_quaternion[3]);
      }
      break;
    case 7:
      // Time + x,y,z + axis angle was given
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        relative_position_[0] = static_cast<double>(params_[params_idx++]);
        relative_position_[1] = static_cast<double>(params_[params_idx++]);
        relative_position_[2] = static_cast<double>(params_[params_idx++]);

        Eigen::Vector3d relative_axis_angle;
        for(int i = 0; i < 3; i++) {
          relative_axis_angle[i] = static_cast<double>(params_[params_idx++]);
        }

        double angle = relative_axis_angle.norm();
        double sin_angle_divided_by_2 = std::sin(angle/2);
        double cos_angle_divided_by_2 = std::cos(angle/2);

        relative_orientation_ = Eigen::Quaterniond(relative_axis_angle[0] * sin_angle_divided_by_2,
                                                   relative_axis_angle[1] * sin_angle_divided_by_2,
                                                   relative_axis_angle[2] * sin_angle_divided_by_2,
                                                   cos_angle_divided_by_2);
      }
      break;
    default:
      std::cout << "RelativePoseTrajectoryGenerator: Invalid number of params provided: " << num_params << std::endl;
  }
}

void RelativePoseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  initialize_initial_and_desired_poses(robot_state, SkillType::ImpedanceControlSkill);
  goal_position_ = initial_position_ + relative_position_;
  goal_orientation_ = initial_orientation_ * relative_orientation_;

  fix_goal_quaternion();
}

void RelativePoseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state,
                                                            SkillType skill_type) {
  initialize_initial_and_desired_poses(robot_state, skill_type);
  goal_position_ = initial_position_ + relative_position_;
  goal_orientation_ = initial_orientation_ * relative_orientation_;

  fix_goal_quaternion();
}