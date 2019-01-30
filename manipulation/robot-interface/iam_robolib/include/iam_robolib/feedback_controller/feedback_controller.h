#ifndef IAM_ROBOLIB_FEEDBACK_CONTROLLER_FEEDBACK_CONTROLLER_H_
#define IAM_ROBOLIB_FEEDBACK_CONTROLLER_FEEDBACK_CONTROLLER_H_

#include <array>
#include <franka/robot_state.h>
#include <franka/model.h>

#include "iam_robolib/trajectory_generator/trajectory_generator.h"

class FeedbackController {
 public:
  explicit FeedbackController(float *p) : params_{p} {};

  /**
   * Parse parameters from memory.
   */
  virtual void parse_parameters() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_controller() = 0;

  /**
   * Initialize trajectory generation after parameter parsing.
   */
  virtual void initialize_controller(franka::Model *model) = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step() = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step(const franka::RobotState &robot_state, TrajectoryGenerator *traj_generator) = 0;

  std::array<double, 7> tau_d_array_{};

 protected:
  float *params_=0;
};

#endif  // IAM_ROBOLIB_FEEDBACK_CONTROLLER_FEEDBACK_CONTROLLER_H_