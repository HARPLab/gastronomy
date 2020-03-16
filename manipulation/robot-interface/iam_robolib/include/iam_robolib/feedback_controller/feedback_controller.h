#ifndef IAM_ROBOLIB_FEEDBACK_CONTROLLER_FEEDBACK_CONTROLLER_H_
#define IAM_ROBOLIB_FEEDBACK_CONTROLLER_FEEDBACK_CONTROLLER_H_

#include <array>
#include <franka/robot_state.h>
#include <iam_robolib_common/definitions.h>

#include "iam_robolib/trajectory_generator/trajectory_generator.h"
#include "iam_robolib/robots/franka_robot.h"

class FeedbackController {
 public:
  explicit FeedbackController(SharedBufferTypePtr p) : params_{p} {};

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
  virtual void initialize_controller(FrankaRobot *robot) = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step() = 0;

  /**
   *  Get next trajectory step.
   */
  virtual void get_next_step(const franka::RobotState &robot_state, 
                             TrajectoryGenerator *traj_generator) = 0;

  std::array<double, 7> tau_d_array_{};

 protected:
  SharedBufferTypePtr params_=0;
};

#endif  // IAM_ROBOLIB_FEEDBACK_CONTROLLER_FEEDBACK_CONTROLLER_H_