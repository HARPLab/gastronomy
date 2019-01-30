#ifndef IAM_ROBOLIB_FEEDBACK_CONTROLLER_TORQUE_FEEDBACK_CONTROLLER_H_
#define IAM_ROBOLIB_FEEDBACK_CONTROLLER_TORQUE_FEEDBACK_CONTROLLER_H_

#include <Eigen/Dense>

#include "iam_robolib/feedback_controller/feedback_controller.h"

class TorqueFeedbackController : public FeedbackController {
 public:
  using FeedbackController::FeedbackController;

  void parse_parameters() override;

  void initialize_controller() override;

  void initialize_controller(franka::Model *model) override;

  void get_next_step() override;

  void get_next_step(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) override;

 private:
  const franka::Model *model_;

  double translational_stiffness_ = 600;
  double rotational_stiffness_ = 50;
  Eigen::MatrixXd stiffness_;
  Eigen::MatrixXd damping_;
};

#endif  // IAM_ROBOLIB_FEEDBACK_CONTROLLER_TORQUE_FEEDBACK_CONTROLLER_H_