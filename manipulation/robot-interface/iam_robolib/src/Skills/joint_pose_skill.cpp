//
// Created by mohit on 12/6/18.
//

#include "joint_pose_skill.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <array>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>

#include <iam_robolib/run_loop.h>
#include "TerminationHandler/termination_handler.h"
#include "TrajectoryGenerator/trajectory_generator.h"
#include "control_loop_data.h"

void JointPoseSkill::execute_skill() {
  assert(false);
}

void JointPoseSkill::execute_skill_on_franka(franka::Robot* robot,
                                             franka::Gripper* gripper,
                                             ControlLoopData *control_loop_data) {

  try {
    double time = 0.0;
    int log_counter = 0;

    std::cout << "Will run the control loop\n";

    franka::Model model = robot->loadModel();

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
        control_loop_data->log_pose_desired(traj_generator_->pose_desired_);
        control_loop_data->log_robot_state(robot_state, time);
      }

      if(done) {
        return franka::MotionFinished(joint_desired);
      }
      return joint_desired;
    };

    robot->control(joint_pose_callback);

  } catch (const franka::Exception& ex) {
    run_loop::running_skills_ = false;
    std::cerr << ex.what() << std::endl;
    // Make sure we don't lose data.
    control_loop_data->writeCurrentBufferData();

    // print last 50 values
    control_loop_data->printGlobalData(50);
    control_loop_data->file_logger_thread_.join();
  }
}

