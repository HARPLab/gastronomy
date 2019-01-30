#ifndef IAM_ROBOLIB_TERMINATION_HANDLER_FACTORY_H_
#define IAM_ROBOLIB_TERMINATION_HANDLER_FACTORY_H_

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
  TerminationHandler* getTerminationHandlerForSkill(SharedBuffer buffer);

};

#endif  // IAM_ROBOLIB_TERMINATION_HANDLER_FACTORY_H_