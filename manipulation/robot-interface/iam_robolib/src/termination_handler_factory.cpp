//
// Created by mohit on 12/18/18.
//

#include "iam_robolib/termination_handler_factory.h"

#include <iostream>
#include <iam_robolib_common/definitions.h>

#include "iam_robolib/termination_handler/contact_termination_handler.h"
#include "iam_robolib/termination_handler/final_joint_termination_handler.h"
#include "iam_robolib/termination_handler/final_pose_termination_handler.h"
#include "iam_robolib/termination_handler/noop_termination_handler.h"
#include "iam_robolib/termination_handler/time_termination_handler.h"

TerminationHandler* TerminationHandlerFactory::getTerminationHandlerForSkill(SharedBufferTypePtr buffer, RunLoopProcessInfo *run_loop_info) {
  TerminationHandlerType termination_handler_type = static_cast<TerminationHandlerType>(buffer[0]);

  std::cout << "Termination Handler Type: " << 
  static_cast<std::underlying_type<TerminationHandlerType>::type>(termination_handler_type) << 
  "\n";

  TerminationHandler *termination_handler = nullptr;
  switch (termination_handler_type) {
    case TerminationHandlerType::ContactTerminationHandler:
      termination_handler = new ContactTerminationHandler(buffer, run_loop_info);
      break;
    case TerminationHandlerType::FinalJointTerminationHandler:
      termination_handler = new FinalJointTerminationHandler(buffer, run_loop_info);
      break;
    case TerminationHandlerType::FinalPoseTerminationHandler:
      termination_handler = new FinalPoseTerminationHandler(buffer, run_loop_info);
      break;
    case TerminationHandlerType::NoopTerminationHandler:
      termination_handler = new NoopTerminationHandler(buffer, run_loop_info);
      break;
    case TerminationHandlerType::TimeTerminationHandler:
      termination_handler = new TimeTerminationHandler(buffer, run_loop_info);
      break;
    default:
      std::cout << "Cannot create Termination Handler with type: " << 
      static_cast<std::underlying_type<TerminationHandlerType>::type>(termination_handler_type) <<
      "\n" ;
      return nullptr;
  }

  termination_handler->parse_parameters();
  return termination_handler;
}
