//
// Created by mohit on 11/20/18.
//

#include "skill_info.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <array>

#include <Eigen/Dense>

#include <franka/exception.h>

#include <iam_robolib/run_loop.h>
#include "control_loop_data.h"

void SkillInfo::execute_skill() {
  assert(traj_generator_ != 0);
  // HACK
  std::string skill_status_string = "Running";
  if (skill_status_ == SkillStatus::FINISHED) {
    skill_status_string = "Finished";
  }
  std::cout << "Will execute skill with status: " << skill_status_string << "\n";
  traj_generator_->get_next_step();
}

void SkillInfo::execute_skill_on_franka(franka::Robot* robot, franka::Gripper* gripper,
                                        ControlLoopData *control_loop_data) {

  try {
    double time = 0.0;
    int log_counter = 0;

    std::cout << "Will run the control loop\n";

    franka::Model model = robot->loadModel();

    // define callback for the torque control loop
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&, &time](const franka::RobotState& robot_state,
                                                franka::Duration period) -> franka::Torques {

      if (time == 0.0) {
        traj_generator_->initialize_trajectory(robot_state);
        feedback_controller_->initialize_controller(&model);
      }

      if (control_loop_data->mutex_.try_lock()) {
        control_loop_data->counter_ += 1;
        control_loop_data->time_ =  period.toSec();
        control_loop_data->has_data_ = true;
        control_loop_data->mutex_.unlock();
      }

      traj_generator_->dt_ = period.toSec();
      traj_generator_->time_ += period.toSec();
      time += period.toSec();
      log_counter += 1;

      traj_generator_->get_next_step();

      if (log_counter % 1 == 0) {
        control_loop_data->log_pose_desired(traj_generator_->pose_desired_);
        control_loop_data->log_robot_state(robot_state, time);
      }

      feedback_controller_->get_next_step(robot_state, traj_generator_);

      bool done = termination_handler_->should_terminate(robot_state, traj_generator_);

      if(done) {
        return franka::MotionFinished(franka::Torques(feedback_controller_->tau_d_array_));
      }


      return feedback_controller_->tau_d_array_;
    };

    robot->control(impedance_control_callback);

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

void SkillInfo::execute_meta_skill_on_franka(franka::Robot *robot, franka::Gripper *gripper,
                                             ControlLoopData *control_loop_data) {
  std::cout << "Not implemented\n" << std::endl;
  assert(false);
}

void SkillInfo::execute_skill_on_franka_joint_base(franka::Robot* robot, franka::Gripper* gripper,
                                                   ControlLoopData *control_loop_data) {

  try {
    double time = 0.0;
    int log_counter = 0;

    std::cout << "Will run the control loop\n";

    franka::Model model = robot->loadModel();

    // define callback for the torque control loop
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&, &time](const franka::RobotState& robot_state,
                                                franka::Duration period) -> franka::Torques {

      if (time == 0.0) {
        traj_generator_->initialize_trajectory(robot_state);
        feedback_controller_->initialize_controller(&model);
      }

      if (control_loop_data->mutex_.try_lock()) {
        control_loop_data->counter_ += 1;
        control_loop_data->time_ =  period.toSec();
        control_loop_data->has_data_ = true;
        control_loop_data->mutex_.unlock();
      }

      traj_generator_->dt_ = period.toSec();
      traj_generator_->time_ += period.toSec();
      time += period.toSec();
      log_counter += 1;

      traj_generator_->get_next_step();

      if (log_counter % 1 == 0) {
        control_loop_data->log_pose_desired(traj_generator_->pose_desired_);
        control_loop_data->log_robot_state(robot_state, time);
      }

      feedback_controller_->get_next_step(robot_state, traj_generator_);

      bool done = termination_handler_->should_terminate(robot_state, traj_generator_);

      if(done) {
        return franka::MotionFinished(franka::Torques(feedback_controller_->tau_d_array_));
      }


      return feedback_controller_->tau_d_array_;
    };

    robot->control(impedance_control_callback);

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

void SkillInfo::execute_skill_on_franka_temp2(franka::Robot* robot, franka::Gripper* gripper,
                                              ControlLoopData *control_loop_data) {
  const double translational_stiffness{500.0};
  const double rotational_stiffness{35.0};
  Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
      Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
      Eigen::MatrixXd::Identity(3, 3);

  try {
    double time = 0.0;
    int log_counter = 0;

    std::cout << "Will run the control loop\n";

    franka::Model model = robot->loadModel();

    // define callback for the torque control loop
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&, &time](const franka::RobotState& robot_state,
                                                franka::Duration period) -> franka::Torques {

      if (time == 0.0) {
        traj_generator_->initialize_trajectory(robot_state);
      }

      if (control_loop_data->mutex_.try_lock()) {
        control_loop_data->counter_ += 1;
        control_loop_data->time_ =  period.toSec();
        control_loop_data->has_data_ = true;
        control_loop_data->mutex_.unlock();
      }

      traj_generator_->dt_ = period.toSec();
      traj_generator_->time_ += period.toSec();
      time += period.toSec();
      log_counter += 1;

      traj_generator_->get_next_step();

      bool done = termination_handler_->should_terminate(traj_generator_);

      if (log_counter % 1 == 0) {
        control_loop_data->log_pose_desired(traj_generator_->pose_desired_);
        control_loop_data->log_robot_state(robot_state, time);
      }

      Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(traj_generator_->pose_desired_.data()));
      Eigen::Vector3d position_d(initial_transform.translation());
      Eigen::Quaterniond orientation_d(initial_transform.linear());

      // get state variables
      std::array<double, 7> coriolis_array = model.coriolis(robot_state);
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

      // convert to Eigen
      Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());
      Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.linear());

      // compute error to desired equilibrium pose
      // position error
      Eigen::Matrix<double, 6, 1> error;
      error.head(3) << position - position_d;

      // orientation error
      // "difference" quaternion
      if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() << -orientation.coeffs();
      }
      Eigen::Quaterniond error_quaternion(orientation * orientation_d.inverse());
      // convert to axis angle
      Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
      // compute "orientation error"
      error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

      // compute control
      Eigen::VectorXd tau_task(7), tau_d(7);

      // Spring damper system with damping ratio=1
      tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
      tau_d << tau_task + coriolis;

      std::array<double, 7> tau_d_array{};
      Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
      return tau_d_array;
    };

    robot->control(impedance_control_callback);

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

void SkillInfo::execute_skill_on_franka_temp(franka::Robot* robot, franka::Gripper* gripper,
                                             ControlLoopData *control_loop_data) {
  try {
    double time = 0.0;
    int log_counter = 0;


    std::cout << "Will run the control loop\n";
    std::function<franka::CartesianPose(const franka::RobotState&, franka::Duration)> cartesian_pose_callback =
        [=, &time, &log_counter](const franka::RobotState& robot_state,
                                 franka::Duration period) -> franka::CartesianPose {
          if (time == 0.0) {
            traj_generator_->initialize_trajectory(robot_state);
          }

          if (control_loop_data->mutex_.try_lock()) {
            control_loop_data->counter_ += 1;
            control_loop_data->time_ =  period.toSec();
            control_loop_data->has_data_ = true;
            control_loop_data->mutex_.unlock();
          }

          traj_generator_->dt_ = period.toSec();
          traj_generator_->time_ += period.toSec();
          time += period.toSec();
          log_counter += 1;

          traj_generator_->get_next_step();

          bool done = termination_handler_->should_terminate(traj_generator_);

          franka::CartesianPose pose_desired(traj_generator_->pose_desired_);
          if (log_counter % 1 == 0) {
            control_loop_data->log_pose_desired(traj_generator_->pose_desired_);
            control_loop_data->log_robot_state(robot_state, time);
          }

          if(done or time >= traj_generator_->run_time_ + traj_generator_->acceleration_time_)
          {
            return franka::MotionFinished(pose_desired);
          }

          return pose_desired;
        };

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

      if(done or time >= traj_generator_->run_time_) {
        return franka::MotionFinished(joint_desired);
      }
      return joint_desired;
    };

    franka::Model model = robot->loadModel();

    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)> impedance_control_callback =
        [=, &model, &q_goal](
            const franka::RobotState& state, franka::Duration /*period*/) -> franka::Torques {
          // Read current coriolis terms from model.
          std::array<double, 7> coriolis = model.coriolis(state);

          // Compute torque command from joint impedance control law.
          // Note: The answer to our Cartesian pose inverse kinematics is always in state.q_d with one
          // time step delay.
          std::array<double, 7> tau_d_calculated;
          for (size_t i = 0; i < 7; i++) {
            tau_d_calculated[i] =
                k_gains_[i] * (state.q_d[i] - state.q[i]) - d_gains_[i] * state.dq[i] + coriolis[i];
            // tau_d_calculated[i] =
            //     k_gains_[i] * (q_goal[i] - state.q[i]) - d_gains_[i] * state.dq[i] + coriolis[i];
          }

          // The following line is only necessary for printing the rate limited torque. As we activated
          // rate limiting for the control loop (activated by default), the torque would anyway be
          // adjusted!
          std::array<double, 7> tau_d_rate_limited =
              franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

          // Send torque command.
          return tau_d_rate_limited;
        };


    /*int memory_index = run_loop_info_->get_current_shared_memory_index();
    SharedBuffer buffer = execution_feedback_buffer_0_;
    if (memory_index == 1) {
      buffer = execution_feedback_buffer_1_;
    }
    skill->write_feedback_to_shared_memory(buffer);

    robot.control()*/

    // robot->control(impedance_control_callback, cartesian_pose_callback);
    // robot->control(cartesian_pose_callback, franka::ControllerMode::kCartesianImpedance, true, 1000.0);
    // robot->control(impedance_control_callback, joint_pose_callback);
    robot->control(impedance_control_callback);

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

bool SkillInfo::should_terminate() {
  return termination_handler_->should_terminate(traj_generator_);
}

void SkillInfo::write_result_to_shared_memory(float *result_buffer) {
  std::cout << "Should write result to shared memory\n";
}

void SkillInfo::write_result_to_shared_memory(float *result_buffer, franka::Robot* robot) {
  std::cout << "Writing final robot state to shared memory\n";
  franka::RobotState robot_state = robot->readOnce();

  result_buffer[0] = static_cast<float>(16+7+7+7+7);

  int result_buffer_index = 1;
  for(int i = 0; i < 16; i++){
    result_buffer[result_buffer_index] = static_cast<float>(robot_state.O_T_EE[i]);
    result_buffer_index++;
  }

  for(int i = 0; i < 7; i++){
    result_buffer[result_buffer_index] = static_cast<float>(robot_state.tau_J[i]);
    result_buffer_index++;
  }
  for(int i = 0; i < 7; i++){
    result_buffer[result_buffer_index] = static_cast<float>(robot_state.dtau_J[i]);
    result_buffer_index++;
  }
  for(int i = 0; i < 7; i++){
    result_buffer[result_buffer_index] = static_cast<float>(robot_state.q[i]);
    result_buffer_index++;
  }
  for(int i = 0; i < 7; i++){
    result_buffer[result_buffer_index] = static_cast<float>(robot_state.dq[i]);
    result_buffer_index++;
  }
}

void SkillInfo::write_feedback_to_shared_memory(float *feedback_buffer) {
  std::cout << "Should write feedback to shared memory\n";
}

