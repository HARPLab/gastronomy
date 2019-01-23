#pragma once

#include "feedback_controller.h"

#include <array>

class CustomGainTorqueController : public FeedbackController {
 public:
  using FeedbackController::FeedbackController;

  void parse_parameters() override;

  void initialize_controller() override;

  void initialize_controller(franka::Model *model) override;

  void get_next_step() override;

  void get_next_step(const franka::RobotState &robot_state,
                     TrajectoryGenerator *traj_generator) override;

 private:
  const franka::Model *model_;
  // Stiffness
  std::array<double, 7> k_gains_ = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    // Damping
  std::array<double, 7> d_gains_ = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
};

