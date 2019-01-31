#ifndef IAM_ROBOLIB_TERMINATION_HANDLER_FACTORY_H_
#define IAM_ROBOLIB_TERMINATION_HANDLER_FACTORY_H_

#include <iam_robolib_common/run_loop_process_info.h>
#include "iam_robolib/definitions.h"

class TerminationHandler;

class TerminationHandlerFactory {
 public:
  TerminationHandlerFactory() {};

  /**
   * Get termination handler for skill.
   *
   * @param memory_region  Region of the memory where the parameters
   * will be stored.
   * @return TermatinationHanndler instance for this skill
   */
  TerminationHandler* getTerminationHandlerForSkill(SharedBuffer buffer, RunLoopProcessInfo *run_loop_info);

};

#endif  // IAM_ROBOLIB_TERMINATION_HANDLER_FACTORY_H_