//
// Created by mohit on 11/25/18.
//

#pragma once

#include <Eigen/Dense>
#include "feedback_controller.h"

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
