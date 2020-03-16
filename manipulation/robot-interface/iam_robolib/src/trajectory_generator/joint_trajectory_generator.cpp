#include "iam_robolib/trajectory_generator/joint_trajectory_generator.h"

#include <iostream>

void JointTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 8:
      // Run time + 7 joints
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        for (size_t i = 0; i < goal_joints_.size(); i++) {
          goal_joints_[i] = static_cast<double>(params_[params_idx++]);
        }
      }
      break;
    default:
      std::cout << "JointTrajectoryGenerator: Incorrect number of params given: " << num_params << std::endl;
  }
}

void JointTrajectoryGenerator::initialize_trajectory() {
  // pass
}

void JointTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  initialize_initial_and_desired_joints(robot_state, SkillType::JointPositionSkill);
}

void JointTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state,
                                                     SkillType skill_type) {
  initialize_initial_and_desired_joints(robot_state, skill_type);
}

void JointTrajectoryGenerator::initialize_initial_and_desired_joints(const franka::RobotState &robot_state,
                                                                     SkillType skill_type) {
  switch(skill_type) {
    case SkillType::JointPositionSkill:
      initial_joints_ = robot_state.q_d;
      desired_joints_ = robot_state.q_d;
      break;
    case SkillType::ImpedanceControlSkill:
      initial_joints_ = robot_state.q;
      desired_joints_ = robot_state.q;
      break;
    default:
      initial_joints_ = robot_state.q_d;
      desired_joints_ = robot_state.q_d;
  }
}
void JointTrajectoryGenerator::setGoalJoints(const std::array<double, 7> joints) {
  for (int i = 0; i < 7; i++) {
    goal_joints_[i] = static_cast<double>(joints[i]);
  }
}

void JointTrajectoryGenerator::setInitialJoints(const std::array<double, 7> joints) {
  for (int i = 0; i < 7; i++) {
    initial_joints_[i] = static_cast<double>(joints[i]);
  }
}

const std::array<double, 7>& JointTrajectoryGenerator::get_desired_joints() const {
  return desired_joints_;
}

const std::array<double, 7>& JointTrajectoryGenerator::get_goal_joints() const {
  return goal_joints_;
}