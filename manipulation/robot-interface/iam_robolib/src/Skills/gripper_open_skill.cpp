#include "gripper_open_skill.h"

#include <cassert>
#include <iostream>
#include <thread>

#include "TrajectoryGenerator/gripper_open_trajectory_generator.h"

void GripperOpenSkill::execute_skill() {
  assert(false);
}

void GripperOpenSkill::execute_skill_on_franka(franka::Robot *robot, franka::Gripper* gripper,
                                               ControlLoopData *control_loop_data) {
  // Check for the maximum grasping width.
  franka::GripperState gripper_state = gripper->readOnce();
  GripperOpenTrajectoryGenerator *gripper_traj_generator = static_cast<
      GripperOpenTrajectoryGenerator *>(traj_generator_);
  double open_width = gripper_traj_generator->getWidth();
  if (gripper_state.max_width < open_width) {
    std::cout << "Object is too large for the current fingers on the gripper." << std::endl;
    return_status_ = false;
    return;
  }

  double open_speed = gripper_traj_generator->getSpeed();
  if (gripper_traj_generator->isGraspSkill()) {
    // TOOD(Mohit): Maybe stop the gripper before trying to grip again?
    franka::GripperState gripper_state = gripper->readOnce();
    if (!gripper_state.is_grasped) {
      return_status_ = gripper->grasp(open_width, open_speed, gripper_traj_generator->getForce(),
			              0.1, 0.1);
    }
  } else {
    return_status_ = gripper->move(open_width, open_speed);
  }

  double wait_time = gripper_traj_generator->getWaitTimeInMilliseconds();
  std::cout << "Gripper wait time: " << wait_time << "\n";
  // Block on this thread to allow gripper to execute skill.
  std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(
      wait_time));
}

bool GripperOpenSkill::should_terminate() {
  // Wait for some time before terminating this skill.
  return true;
}
