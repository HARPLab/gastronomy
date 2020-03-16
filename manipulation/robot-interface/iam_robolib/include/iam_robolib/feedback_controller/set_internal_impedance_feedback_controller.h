#ifndef IAM_ROBOLIB_FEEDBACK_CONTROLLER_SET_INTERNAL_IMPEDANCE_FEEDBACK_CONTROLLER_H_
#define IAM_ROBOLIB_FEEDBACK_CONTROLLER_SET_INTERNAL_IMPEDANCE_FEEDBACK_CONTROLLER_H_

#include "iam_robolib/feedback_controller/feedback_controller.h"

class SetInternalImpedanceFeedbackController : public FeedbackController {
 public:
  using FeedbackController::FeedbackController;

  void parse_parameters() override;

  void initialize_controller() override;

  void initialize_controller(FrankaRobot *robot) override;

  void get_next_step() override;

  void get_next_step(const franka::RobotState &robot_state, 
                     TrajectoryGenerator *traj_generator) override;

 private:
  bool set_joint_impedance_ = false;
  bool set_cartesian_impedance_ = false;

  // Max Joint and Cartesian Impedance Values. 
  // TODO: Check to see what they actually are.
  // Kevin simply guessed 10000.0
  double max_joint_impedance_ = 10000.0;
  double max_cartesian_impedance_ = 10000.0;

  std::array<double, 7> K_theta_ = {{3000, 3000, 3000, 2500, 2500, 2000, 2000}};
  std::array<double, 6> K_x_ = {{3000, 3000, 3000, 300, 300, 300}};

  // Constant values
  static constexpr std::array<double, 7> k_K_theta_ = {{3000, 3000, 3000, 2500, 2500, 2000, 2000}};
  static constexpr std::array<double, 6> k_K_x_ = {{3000, 3000, 3000, 300, 300, 300}};
};

#endif  // IAM_ROBOLIB_FEEDBACK_CONTROLLER_SET_INTERNAL_IMPEDANCE_FEEDBACK_CONTROLLER_H_