//
// Created by iamlab on 12/2/18.
//

#include "iam_robolib/skills/base_skill.h"

#include <iostream>

#include "iam_robolib/feedback_controller/feedback_controller.h"
#include "iam_robolib/termination_handler/termination_handler.h"
#include "iam_robolib/trajectory_generator/trajectory_generator.h"

int BaseSkill::get_skill_id() {
  return skill_idx_;
}

int BaseSkill::get_meta_skill_id() {
  return meta_skill_idx_;
}

std::string BaseSkill::get_description() {
  return description_;
}

void BaseSkill::set_skill_status(SkillStatus status) {
  // TODO(Mohit): Maybe add checks such that task status progresses
  // in one direction.
  skill_status_ = status;
}

SkillStatus BaseSkill::get_current_skill_status() {
  return skill_status_;
}

TrajectoryGenerator* BaseSkill::get_trajectory_generator() {
  return traj_generator_;
}

FeedbackController* BaseSkill::get_feedback_controller() {
  return feedback_controller_;
}

TerminationHandler* BaseSkill::get_termination_handler() {
  return termination_handler_;
}

void BaseSkill::start_skill(Robot* robot,
                            TrajectoryGenerator *traj_generator,
                            FeedbackController *feedback_controller,
                            TerminationHandler *termination_handler) {
  skill_status_ = SkillStatus::TO_START;
  traj_generator_ = traj_generator;
  traj_generator_->initialize_trajectory();
  feedback_controller_ = feedback_controller;
  feedback_controller_->initialize_controller();
  termination_handler_ = termination_handler;
  switch(robot->robot_type_)
  {
    case RobotType::FRANKA:
      termination_handler_->initialize_handler_on_franka(dynamic_cast<FrankaRobot *>(robot));
      break;
    case RobotType::UR5E:
      break;
    default:
      termination_handler_->initialize_handler();
  }
}

bool BaseSkill::should_terminate() {
  return termination_handler_->should_terminate(traj_generator_);
}

void BaseSkill::write_result_to_shared_memory(float *result_buffer) {
  std::cout << "Should write result to shared memory\n";
}

void BaseSkill::write_result_to_shared_memory(float *result_buffer, FrankaRobot* robot) {
  std::cout << "Should write result to shared memory\n";
}

void BaseSkill::write_result_to_shared_memory(float *result_buffer, Robot* robot) {
  std::cout << "Should write result to shared memory\n";
}

void BaseSkill::write_feedback_to_shared_memory(float *feedback_buffer) {
  std::cout << "Should write feedback to shared memory\n";
}

