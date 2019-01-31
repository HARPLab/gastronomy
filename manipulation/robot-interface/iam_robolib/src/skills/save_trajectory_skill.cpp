//
// Created by mohit on 12/6/18.
//

#include "iam_robolib/skills/save_trajectory_skill.h"

#include <cassert>
#include <franka/robot.h>
#include <iostream>

#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/termination_handler/termination_handler.h"

void SaveTrajectorySkill::execute_skill() {
  assert(false);
}

void SaveTrajectorySkill::execute_skill_on_franka(FrankaRobot *robot,
                                                  RobotStateData *robot_state_data) {
  if (!running_skill_) {
    std::cout << "Will execute SaveTrajectory skill\n" << std::endl;

    running_skill_ = true;
    save_traj_thread_ = std::thread([=]() {
      franka::RobotState robot_state = robot->getRobotState();
      franka::Duration start_time(robot_state.time), curr_time(robot_state.time);
      int print_rate  = 500;

      while (running_skill_) {
        robot_state = robot->getRobotState();
        curr_time = robot_state.time;
        double time = curr_time.toSec() - start_time.toSec();

        robot_state_data->log_pose_desired(traj_generator_->pose_desired_);
        robot_state_data->log_robot_state(robot_state, time);

        traj_generator_->time_ = time;

        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0))));
      }
    });
  }
}

bool SaveTrajectorySkill::should_terminate() {
  bool should_terminate = termination_handler_->should_terminate(traj_generator_);
  if (should_terminate) {
    running_skill_ = false;
    save_traj_thread_.join();
  }
  return should_terminate;
}

