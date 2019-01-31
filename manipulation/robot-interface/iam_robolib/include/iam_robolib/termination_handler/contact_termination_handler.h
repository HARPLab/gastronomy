#ifndef IAM_ROBOLIB_TERMINATION_HANDLER_CONTACT_TERMINATION_HANDLER_H_
#define IAM_ROBOLIB_TERMINATION_HANDLER_CONTACT_TERMINATION_HANDLER_H_

#include "iam_robolib/termination_handler/termination_handler.h"

class ContactTerminationHandler : public TerminationHandler {
 public:
  using TerminationHandler::TerminationHandler;

  /**
   * Parse parameters from memory.
   */
  void parse_parameters() override;

  /**
   * Initialize termination handler after parameter parsing.
   */
  void initialize_handler() override;

  /**
   * Initialize termination handler after parameter parsing.
   */
  void initialize_handler_on_franka(FrankaRobot *robot) override;

  /**
   * Should we terminate the current skill.
   */
  bool should_terminate(TrajectoryGenerator *traj_generator) override;

  /**
   * Should we terminate the current skill.
   */
  bool should_terminate_on_franka(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) override;

 private:
  double buffer_time_ = 0.0;

  std::array<double, 6> cartesian_contacts_to_use_{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  std::array<double, 7> lower_torque_thresholds_acceleration_{};
  std::array<double, 7> upper_torque_thresholds_acceleration_{};
  std::array<double, 7> lower_torque_thresholds_nominal_{};
  std::array<double, 7> upper_torque_thresholds_nominal_{};
  std::array<double, 6> lower_force_thresholds_acceleration_{};
  std::array<double, 6> upper_force_thresholds_acceleration_{};
  std::array<double, 6> lower_force_thresholds_nominal_{};
  std::array<double, 6> upper_force_thresholds_nominal_{};

  const std::array<double, 7> default_lower_torque_thresholds_acceleration_{{20.0,20.0,18.0,18.0,16.0,14.0,12.0}};
  const std::array<double, 7> default_upper_torque_thresholds_acceleration_{{120.0,120.0,118.0,118.0,116.0,114.0,112.0}};
  const std::array<double, 7> default_lower_torque_thresholds_nominal_{{20.0,20.0,18.0,18.0,16.0,14.0,12.0}};
  const std::array<double, 7> default_upper_torque_thresholds_nominal_{{120.0,120.0,118.0,118.0,116.0,114.0,112.0}};
  const std::array<double, 6> default_lower_force_thresholds_acceleration_{{20.0,20.0,20.0,25.0,25.0,25.0}};
  const std::array<double, 6> default_upper_force_thresholds_acceleration_{{120.0,120.0,120.0,125.0,125.0,125.0}};
  const std::array<double, 6> default_lower_force_thresholds_nominal_{{20.0,20.0,20.0,25.0,25.0,25.0}};
  const std::array<double, 6> default_upper_force_thresholds_nominal_{{120.0,120.0,120.0,125.0,125.0,125.0}};
};

#endif  // IAM_ROBOLIB_TERMINATION_HANDLER_CONTACT_TERMINATION_HANDLER_H_