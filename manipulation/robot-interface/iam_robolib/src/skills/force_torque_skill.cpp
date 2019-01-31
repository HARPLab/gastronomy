//
// Created by jacky on 1/26/19.
//

#include "iam_robolib/skills/force_torque_skill.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <array>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>

#include "iam_robolib/termination_handler/termination_handler.h"
#include "iam_robolib/trajectory_generator/trajectory_generator.h"
#include "iam_robolib/run_loop.h"
#include "iam_robolib/robot_state_data.h"

void ForceTorqueSkill::execute_skill() {
  assert(false);
}

void ForceTorqueSkill::execute_skill_on_franka(FrankaRobot* robot,
                                               RobotStateData *robot_state_data) {

  try {
    double time = 0.0;
    int log_counter = 0;

    std::cout << "Will run the control loop\n";

    franka::Model model = robot->getModel();

    // get torque offset from gravity
    franka::RobotState initial_state = robot->getRobotState();
    Eigen::VectorXd initial_tau_ext(7);
    std::array<double, 7> gravity_array = model.gravity(initial_state);
    Eigen::Map<Eigen::Matrix<double, 7, 1> > initial_tau_measured(initial_state.tau_J.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1> > initial_gravity(gravity_array.data());
    initial_tau_ext = initial_tau_measured - initial_gravity;

    auto force_control_callback = [&](const franka::RobotState& robot_state,
                                      franka::Duration period) -> franka::Torques {
      if (time == 0.0) {
        traj_generator_->initialize_trajectory(robot_state);
      }
      time += period.toSec();
      traj_generator_->time_ = time;
      traj_generator_->dt_ = period.toSec();
      traj_generator_->get_next_step();

      bool done = termination_handler_->should_terminate(traj_generator_);
      double* force_torque_desired_ptr = &(traj_generator_->force_torque_desired_[0]);
      Eigen::Map<Eigen::VectorXd> desired_force_torque(force_torque_desired_ptr, 6);

      log_counter += 1;
      if (log_counter % 1 == 0) {
        robot_state_data->log_pose_desired(traj_generator_->pose_desired_);
        robot_state_data->log_robot_state(robot_state, time);
      }

      // get jacobian
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
      Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());

      // compute control torques
      Eigen::VectorXd tau_d(7);
      tau_d << jacobian.transpose() * desired_force_torque;

      std::array<double, 7> tau_d_array{};
      Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

      if (done) {
        franka::Torques torques(tau_d_array);
        return franka::MotionFinished(torques);
      }

      return tau_d_array;
    };

    robot->robot_.control(force_control_callback);

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

