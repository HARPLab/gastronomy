//
// Created by iamlab on 12/1/18.
//

#include "iam_robolib/trajectory_generator/gripper_trajectory_generator.h"

#include <iostream>

void GripperTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 2:
      // Width and Speed
      {
        width_ = static_cast<double >(params_[params_idx++]);
        speed_ = static_cast<double >(params_[params_idx++]);
        is_grasp_skill_ = false;
      }
      break;
    case 3:
      // Width, Speed, and Force
      {
        width_ = static_cast<double >(params_[params_idx++]);
        speed_ = static_cast<double >(params_[params_idx++]);
        force_ = static_cast<double> (params_[params_idx++]);
        is_grasp_skill_ = true;
      }
      break;
    default:
      std::cout << "GripperTrajectoryGenerator: Incorrect number of params given: " << num_params << std::endl;
  }
}

void GripperTrajectoryGenerator::initialize_trajectory() {
  // pass
}

void GripperTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  // pass
}

void GripperTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state,
                                                       SkillType skill_type) {
  // pass
}

void GripperTrajectoryGenerator::get_next_step() {
  // pass
}

double GripperTrajectoryGenerator::getWidth() {
  return width_;
}

double GripperTrajectoryGenerator::getSpeed()  {
  return speed_;
}

double GripperTrajectoryGenerator::getForce()  {
  return force_;
}

bool GripperTrajectoryGenerator::isGraspSkill(){
  return is_grasp_skill_;
}
