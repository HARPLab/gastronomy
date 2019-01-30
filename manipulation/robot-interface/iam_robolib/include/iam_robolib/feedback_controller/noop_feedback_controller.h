#ifndef IAM_ROBOLIB_FEEDBACK_CONTROLLER_NOOP_FEEDBACK_CONTROLLER_H_
#define IAM_ROBOLIB_FEEDBACK_CONTROLLER_NOOP_FEEDBACK_CONTROLLER_H_

#include "iam_robolib/feedback_controller/feedback_controller.h"

class NoopFeedbackController : public FeedbackController {
 public:
  using FeedbackController::FeedbackController;

  void parse_parameters() override;

  void initialize_controller() override;

  void initialize_controller(franka::Model *model) override;

  void get_next_step() override;

  void get_next_step(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) override;

  float delta_=0.0;
};

#endif  // IAM_ROBOLIB_FEEDBACK_CONTROLLER_NOOP_FEEDBACK_CONTROLLER_H_