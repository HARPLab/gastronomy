//
// Created by mohit on 11/25/18.
//

#include "noop_feedback_controller.h"

#include <iostream>

void NoopFeedbackController::parse_parameters() {
  // pass
}

void NoopFeedbackController::initialize_controller() {
  // pass
}

void NoopFeedbackController::initialize_controller(franka::Model *model) {
  // pass
}

void NoopFeedbackController::get_next_step() {
  // pass
}

void NoopFeedbackController::get_next_step(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) {
  // pass
}