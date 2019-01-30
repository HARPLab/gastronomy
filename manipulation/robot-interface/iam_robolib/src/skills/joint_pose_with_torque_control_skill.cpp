#include "iam_robolib/skills/joint_pose_with_torque_control_skill.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <array>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>
#include <franka/rate_limiting.h>

#include "iam_robolib/run_loop.h"
#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/feedback_controller/feedback_controller.h"
#include "iam_robolib/termination_handler/termination_handler.h"
#include "iam_robolib/trajectory_generator/trajectory_generator.h"

void JointPoseWithTorqueControlSkill::execute_skill() {
  assert(false);
}

void JointPoseWithTorqueControlSkill::execute_skill_on_franka(
    franka::Robot* robot, franka::Gripper* gripper, RobotStateData *robot_state_data) {

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
        robot_state_data->log_pose_desired(traj_generator_->pose_desired_);
        robot_state_data->log_robot_state(robot_state, time);
      }

      if(done or time >= traj_generator_->run_time_) {
        return franka::MotionFinished(joint_desired);
      }
      return joint_desired;
    };

    std::function<franka::Torques(const franka::RobotState&,
        franka::Duration)> impedance_control_callback = [&](
          const franka::RobotState& state, franka::Duration) -> franka::Torques {
          feedback_controller_->get_next_step();
          return feedback_controller_->tau_d_array_;
        };

    robot->control(impedance_control_callback, joint_pose_callback);

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

void JointPoseWithTorqueControlSkill::execute_meta_skill_on_franka(franka::Robot *robot,
                                                                   franka::Gripper *gripper,
                                                                   RobotStateData *robot_state_data) {
  std::cout << "Not implemented\n" << std::endl;
  assert(false);
}

