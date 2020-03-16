//
// Created by jacky on 1/26/19.
//

#include "iam_robolib/trajectory_generator/impulse_trajectory_generator.h"

#include <cassert>
#include <type_traits>
#include <iostream>
#include <memory.h>

#include <iam_robolib_common/definitions.h>

void ImpulseTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 10:
      // Run Time, Acceleration Time, Max Translation, Max Rotation,
      // and Target Force Torques (6)
      {
        run_time_ = static_cast<double>(params_[params_idx++]);
        acc_time_ = static_cast<double>(params_[params_idx++]);
        max_translation_ = static_cast<double>(params_[params_idx++]);
        max_rotation_ = static_cast<double>(params_[params_idx++]);
        for (size_t i = 0; i < target_force_torque_.size(); i++) {
          target_force_torque_[i] = static_cast<double>(params_[params_idx++]);
        }
      }
      break;
    default:
      std::cout << "ImpulseTrajectoryGenerator: Incorrect number of params given: " << num_params << std::endl;
  }
}

void ImpulseTrajectoryGenerator::initialize_trajectory() {
  // pass
}

void ImpulseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  initialize_initial_states(robot_state, SkillType::ForceTorqueSkill);
}

void ImpulseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state,
                                                       SkillType skill_type) {
  initialize_initial_states(robot_state, skill_type);
}

void ImpulseTrajectoryGenerator::initialize_initial_states(const franka::RobotState &robot_state,
                                                           SkillType skill_type) {
  switch(skill_type) {
    case SkillType::ForceTorqueSkill:
      initial_pose_ = robot_state.O_T_EE;
      break;
    default:
      initial_pose_ = robot_state.O_T_EE;
      std::cout << "Invalid Skill Type provided: " << static_cast<std::underlying_type<SkillType>::type>(skill_type) << "\n";
      std::cout << "Using default O_T_EE" << std::endl;
  }

  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  initial_position_ = Eigen::Vector3d(initial_transform.translation());
  initial_orientation_ = Eigen::Quaterniond(initial_transform.linear());
}

void ImpulseTrajectoryGenerator::get_next_step() {
  t_ = time_;

  double coef = 0.0;
  if (!should_deacc_) {
    if (t_ >= 0 && t_ < acc_time_) {
      coef = t_/acc_time_;
    } else if (t_ >= acc_time_ && t_ < run_time_ - acc_time_) {
      coef = 1.0;
    } else if (t_ >= run_time_ - acc_time_ && t_ < run_time_) {
      coef = (run_time_ - t_)/acc_time_;
    } else {
      coef = 0.0;
    }
  }

  for (size_t i = 0; i < target_force_torque_.size(); i++) {
    desired_force_torque_[i] = coef * target_force_torque_[i];
  }
}

void ImpulseTrajectoryGenerator::check_displacement_cap(const franka::RobotState& robot_state) {
  // check if max translation and rotation caps are reached
  if (!should_deacc_) {
    Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    
    if (max_translation_ > 0) {
      current_position_ = transform.translation();
      if ((current_position_ - initial_position_).norm() > max_translation_) {
        should_deacc_ = true;
      }
    }

    if (max_rotation_ > 0) {
      current_orientation_ = transform.linear();
      if (current_orientation_.coeffs().dot(initial_orientation_.coeffs()) < 0.0) {
        current_orientation_.coeffs() << -current_orientation_.coeffs();
      }
      Eigen::Quaterniond Q_delta(initial_orientation_ * current_orientation_.inverse());
      Eigen::AngleAxisd A_delta(Q_delta);
      if (A_delta.angle() > max_rotation_) {
        should_deacc_ = true;
      }
    }
  }
}
  
const std::array<double, 6>& ImpulseTrajectoryGenerator::get_desired_force_torque() const {
  return desired_force_torque_;
}

const Eigen::Vector3d& ImpulseTrajectoryGenerator::get_initial_position() const {
  return initial_position_;
}

const Eigen::Quaterniond& ImpulseTrajectoryGenerator::get_initial_orientation() const {
  return initial_orientation_;
}