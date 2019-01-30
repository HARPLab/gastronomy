//
// Created by mohit on 12/18/18.
//

#include "iam_robolib/termination_handler_factory.h"

#include "iam_robolib/termination_handler/final_joint_termination_handler.h"
#include "iam_robolib/termination_handler/final_pose_termination_handler.h"
#include "iam_robolib/termination_handler/noop_termination_handler.h"
#include "iam_robolib/termination_handler/contact_termination_handler.h"
#include "iam_robolib/termination_handler/linear_trajectory_generator_with_time_and_goal_termination_handler.h"
#include "iam_robolib/termination_handler/time_termination_handler.h"

#include <iostream>

TerminationHandler* TerminationHandlerFactory::getTerminationHandlerForSkill(SharedBuffer buffer) {
  int termination_handler_id = static_cast<int>(buffer[0]);

  std::cout << "Termination Handler id: " << termination_handler_id << "\n";

  TerminationHandler *termination_handler = nullptr;
  if (termination_handler_id == 1) {
    // Create Counter based trajectory.
    termination_handler = new NoopTerminationHandler(buffer);
  } else if (termination_handler_id == 2) {
    termination_handler = new FinalPoseTerminationHandler(buffer);
  } else if (termination_handler_id == 3) {
    termination_handler = new FinalJointTerminationHandler(buffer);
  } else if (termination_handler_id == 4) {
    termination_handler = new LinearTrajectoryGeneratorWithTimeAndGoalTerminationHandler(buffer);
  } else if (termination_handler_id == 5) {
    termination_handler = new ContactTerminationHandler(buffer);
  } else if (termination_handler_id == 6) {
    termination_handler = new TimeTerminationHandler(buffer);
  } else {
    // Cannot create Trajectory generator for this skill. Throw error
    std::cout << "Cannot create TerminationHandler with class_id: " << termination_handler_id <<
      "\n" ;
    return nullptr;
  }
  termination_handler->parse_parameters();
  return termination_handler;
}
