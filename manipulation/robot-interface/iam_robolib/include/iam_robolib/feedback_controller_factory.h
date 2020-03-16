#ifndef IAM_ROBOLIB_FEEDBACK_CONTROLLER_FACTORY_H_
#define IAM_ROBOLIB_FEEDBACK_CONTROLLER_FACTORY_H_

#include "iam_robolib_common/definitions.h"

class FeedbackController;

class FeedbackControllerFactory {
 public:
  FeedbackControllerFactory() {};

  /**
   * Get feedback controller for skill.
   *
   * @param memory_region  Region of the memory where the parameters
   * will be stored.
   * @return FeedbackController instance for this skill
   */
  FeedbackController* getFeedbackControllerForSkill(SharedBufferTypePtr buffer);

};

#endif  // IAM_ROBOLIB_FEEDBACK_CONTROLLER_FACTORY_H_