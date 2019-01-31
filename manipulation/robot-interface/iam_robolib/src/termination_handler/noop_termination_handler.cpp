//
// Created by mohit on 11/26/18.
//

#include "iam_robolib/termination_handler/noop_termination_handler.h"

void NoopTerminationHandler::parse_parameters() {
  // pass
}

void NoopTerminationHandler::initialize_handler() {
  // pass
}

void NoopTerminationHandler::initialize_handler_on_franka(FrankaRobot *robot) {
  // pass
}

bool NoopTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();
}

bool NoopTerminationHandler::should_terminate_on_franka(const franka::RobotState &robot_state, TrajectoryGenerator *trajectory_generator) {
  check_terminate_preempt();
}
