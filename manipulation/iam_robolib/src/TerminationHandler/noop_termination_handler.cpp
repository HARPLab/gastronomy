//
// Created by mohit on 11/26/18.
//

#include "noop_termination_handler.h"

void NoopTerminationHandler::parse_parameters() {
  // pass
}

void NoopTerminationHandler::initialize_handler() {
  // pass
}

void NoopTerminationHandler::initialize_handler(franka::Robot *robot) {
  // pass
}

bool NoopTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  // pass
}

bool NoopTerminationHandler::should_terminate(const franka::RobotState &robot_state, TrajectoryGenerator *trajectory_generator) {
  // pass
}
