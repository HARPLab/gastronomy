#pragma once

#include "trajectory_generator.h"

/**
 * Used in GripperOpen skill. Specifies 3 parameters in the following order
 *
 *      1) gripper open width
 *      2) gripper open speed
 *      3) wait time for gripper to open in milliseconds.
 */
class GripperOpenTrajectoryGenerator : public TrajectoryGenerator {
 public:
  using TrajectoryGenerator::TrajectoryGenerator;

  void parse_parameters() override;

  void initialize_trajectory() override;

  void initialize_trajectory(const franka::RobotState &robot_state) override;

  void get_next_step() override;

  /**
   * Get width to open the gripper to.
   * @return width speed.
   */
  double getWidth();

  /**
   * Get speed to open the gripper.
   * @return gripper speed.
   */
  double getSpeed();

  /**
   * Get Force to grasp an object.
   * @return
   */
  double getForce();

  /**
   * Get time to wait for the skill.
   * @return
   */
  double getWaitTimeInMilliseconds();

  /**
   * Check if the skill requires to grasp the object.
   * @return True if the skill requires to grasp the object, returns false if it does not.
   */
  bool isGraspSkill();

 private:
  double width_=1.0;
  double speed_=0.0;
  double force_=0.0;
  double wait_time_in_milliseconds_=3000.0;
  bool is_grasp_skill_{false};
};

