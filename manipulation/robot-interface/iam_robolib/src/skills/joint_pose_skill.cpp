//
// Created by mohit on 12/6/18.
//

#include "iam_robolib/skills/joint_pose_skill.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <array>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>

#include "iam_robolib/run_loop.h"
#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/termination_handler/termination_handler.h"
#include "iam_robolib/trajectory_generator/trajectory_generator.h"

void JointPoseSkill::execute_skill() {
  assert(false);
}

void JointPoseSkill::execute_skill_on_franka(FrankaRobot* robot,
                                             RobotStateData *robot_state_data) {

  try {
    double time = 0.0;
    int log_counter = 0;

    std::cout << "Will run the control loop\n";

    franka::Model model = robot->getModel();

    std::function<franka::JointPositions(const franka::RobotState&, franka::Duration)>
        joint_pose_callback = [=, &time, &log_counter](
        const franka::RobotState& robot_state,
        franka::Duration period) -> franka::JointPositions {
      if (time == 0.0) {
        traj_generator_->initialize_trajectory(robot_state);
      }
      time += period.toSec();
      traj_generator_->time_ = time;
      traj_generator_->dt_ = period.toSec();
      traj_generator_->get_next_step();

      bool done = termination_handler_->should_terminate(traj_generator_);
      franka::JointPositions joint_desired(traj_generator_->joint_desired_);

      log_counter += 1;
      if (log_counter % 1 == 0) {
        robot_state_data->log_pose_desired(traj_generator_->pose_desired_);
        robot_state_data->log_robot_state(robot_state, time);
      }

      if(done) {
        return franka::MotionFinished(joint_desired);
      }
      return joint_desired;
    };

    robot->robot_.control(joint_pose_callback);

  } catch (const franka::Exception& ex) {
    run_loop::running_skills_ = false;
    std::cerr << ex.what() << std::endl;
    // Make sure we don't lose data.
    robot_state_data->writeCurrentBufferData();

    // print last 50 values
    robot_state_data->printGlobalData(50);
    robot_state_data->file_logger_thread_.join();
  }
}

