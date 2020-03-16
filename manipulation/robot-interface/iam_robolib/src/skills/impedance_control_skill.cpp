//
// Created by mohit on 11/20/18.
//

#include "iam_robolib/skills/impedance_control_skill.h"

#include <cassert>
#include <iostream>

#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/run_loop.h"
#include "iam_robolib/run_loop_shared_memory_handler.h"

#include <iam_robolib_common/definitions.h>
#include <iam_robolib_common/run_loop_process_info.h>

void ImpedanceControlSkill::execute_skill() {
  assert(traj_generator_ != 0);
  // HACK
  std::string skill_status_string = "Running";
  if (skill_status_ == SkillStatus::FINISHED) {
    skill_status_string = "Finished";
  }
  std::cout << "Will execute skill with status: " << skill_status_string << "\n";
  traj_generator_->get_next_step();
}

void ImpedanceControlSkill::execute_skill_on_franka(run_loop* run_loop, 
                                                    FrankaRobot* robot,
                                                    RobotStateData *robot_state_data) {

  double time = 0.0;
  int log_counter = 0;
  std::array<double, 16> pose_desired;

  RunLoopSharedMemoryHandler* shared_memory_handler = run_loop->get_shared_memory_handler();
  RunLoopProcessInfo* run_loop_info = shared_memory_handler->getRunLoopProcessInfo();
  boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(
                                  *(shared_memory_handler->getRunLoopProcessInfoMutex()),
                                  boost::interprocess::defer_lock);

  std::cout << "Will run the control loop\n";

  // define callback for the torque control loop
  std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
      impedance_control_callback = [&](const franka::RobotState& robot_state,
                                              franka::Duration period) -> franka::Torques {

    current_period_ = period.toSec();
    time += current_period_;

    if (time == 0.0) {
      traj_generator_->initialize_trajectory(robot_state, SkillType::ImpedanceControlSkill);
      try {
        if (lock.try_lock()) {
          run_loop_info->set_time_skill_started_in_robot_time(robot_state.time.toSec());
          lock.unlock();
        } 
      } catch (boost::interprocess::lock_exception) {
        // Do nothing
      }
    }

    if (robot_state_data->mutex_.try_lock()) {
      robot_state_data->counter_ += 1;
      robot_state_data->time_ =  period.toSec();
      robot_state_data->has_data_ = true;
      robot_state_data->mutex_.unlock();
    }

    traj_generator_->time_ = time;
    traj_generator_->dt_ = current_period_;
    time += period.toSec();
    log_counter += 1;

    if(time > 0.0) {
      traj_generator_->get_next_step();
    }

    if (log_counter % 1 == 0) {
      pose_desired = robot_state.O_T_EE_d;
      robot_state_data->log_robot_state(pose_desired, robot_state, robot->getModel(), time);
    }

    // This code was added for dynamic interpolation skill, but this is not the right way to change
    // the controller. Hence, for now just comment this code out.

//    JointSensorInfo new_joint_sensor_info;
//
//    SensorDataManagerReadStatus sensor_msg_status = sensor_data_manager->readJointSensorInfoMessage(
//        new_joint_sensor_info);
//    if (sensor_msg_status == SensorDataManagerReadStatus::SUCCESS) {
//      assert(new_joint_sensor_info.IsInitialized());
//      std::array<double, 7> new_goal_joints = {
//          new_joint_sensor_info.q1(),
//          new_joint_sensor_info.q2(),
//          new_joint_sensor_info.q3(),
//          new_joint_sensor_info.q4(),
//          new_joint_sensor_info.q5(),
//          new_joint_sensor_info.q6(),
//          new_joint_sensor_info.q7(),
//      };
//      joint_trajectory_generator->setGoalJoints(new_goal_joints);
//      std::cout << "Updated new goal joints: ";
//      for (int i = 0; i < new_goal_joints.size(); i++) {
//        std::cout << new_goal_joints[i] << ", ";
//      }
//      std::cout << std::endl;
//    }

    feedback_controller_->get_next_step(robot_state, traj_generator_);

    bool done = termination_handler_->should_terminate_on_franka(robot_state, model_, traj_generator_);

    if(done && time > 0.0) {
      try {
        if (lock.try_lock()) {
          run_loop_info->set_time_skill_finished_in_robot_time(robot_state.time.toSec());
          lock.unlock();
        } 
      } catch (boost::interprocess::lock_exception) {
        // Do nothing
      }
      
      return franka::MotionFinished(franka::Torques(feedback_controller_->tau_d_array_));
    }

    return feedback_controller_->tau_d_array_;
  };

  robot->robot_.control(impedance_control_callback);
}

void ImpedanceControlSkill::write_result_to_shared_memory(SharedBufferTypePtr result_buffer) {
  std::cout << "Should write result to shared memory\n";
}

void ImpedanceControlSkill::write_result_to_shared_memory(SharedBufferTypePtr result_buffer, FrankaRobot* robot) {
  franka::GripperState gripper_state = robot->getGripperState();
  franka::RobotState robot_state = robot->getRobotState();

  int result_buffer_idx = 0;

  result_buffer[result_buffer_idx++] = static_cast<double>(16+16+16+16+
                                                           1+9+3+
                                                           1+9+3+
                                                           1+9+3+
                                                           2+2+2+2+2+
                                                           7+7+7+7+7+7+7+7+
                                                           7+6+7+6+
                                                           7+6+6+6+16+6+6+
                                                           7+7+37+37+
                                                           1+1+1+
                                                           1+1+1+1+1); // 344

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_T_EE.data(), robot_state.O_T_EE.size() * sizeof(double));
  result_buffer_idx += robot_state.O_T_EE.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_T_EE_d.data(), robot_state.O_T_EE_d.size() * sizeof(double));
  result_buffer_idx += robot_state.O_T_EE_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.F_T_EE.data(), robot_state.F_T_EE.size() * sizeof(double));
  result_buffer_idx += robot_state.F_T_EE.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.EE_T_K.data(), robot_state.EE_T_K.size() * sizeof(double));
  result_buffer_idx += robot_state.EE_T_K.size();

  result_buffer[result_buffer_idx++] = robot_state.m_ee;

  memcpy(&result_buffer[result_buffer_idx], robot_state.I_ee.data(), robot_state.I_ee.size() * sizeof(double));
  result_buffer_idx += robot_state.I_ee.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.F_x_Cee.data(), robot_state.F_x_Cee.size() * sizeof(double));
  result_buffer_idx += robot_state.F_x_Cee.size();

  result_buffer[result_buffer_idx++] = robot_state.m_load;

  memcpy(&result_buffer[result_buffer_idx], robot_state.I_load.data(), robot_state.I_load.size() * sizeof(double));
  result_buffer_idx += robot_state.I_load.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.F_x_Cload.data(), robot_state.F_x_Cload.size() * sizeof(double));
  result_buffer_idx += robot_state.F_x_Cload.size();

  result_buffer[result_buffer_idx++] = robot_state.m_total;

  memcpy(&result_buffer[result_buffer_idx], robot_state.I_total.data(), robot_state.I_total.size() * sizeof(double));
  result_buffer_idx += robot_state.I_total.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.F_x_Ctotal.data(), robot_state.F_x_Ctotal.size() * sizeof(double));
  result_buffer_idx += robot_state.F_x_Ctotal.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.elbow.data(), robot_state.elbow.size() * sizeof(double));
  result_buffer_idx += robot_state.elbow.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.elbow_d.data(), robot_state.elbow_d.size() * sizeof(double));
  result_buffer_idx += robot_state.elbow_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.elbow_c.data(), robot_state.elbow_c.size() * sizeof(double));
  result_buffer_idx += robot_state.elbow_c.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.delbow_c.data(), robot_state.delbow_c.size() * sizeof(double));
  result_buffer_idx += robot_state.delbow_c.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.ddelbow_c.data(), robot_state.ddelbow_c.size() * sizeof(double));
  result_buffer_idx += robot_state.ddelbow_c.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.tau_J.data(), robot_state.tau_J.size() * sizeof(double));
  result_buffer_idx += robot_state.tau_J.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.tau_J_d.data(), robot_state.tau_J_d.size() * sizeof(double));
  result_buffer_idx += robot_state.tau_J_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.dtau_J.data(), robot_state.dtau_J.size() * sizeof(double));
  result_buffer_idx += robot_state.dtau_J.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.q.data(), robot_state.q.size() * sizeof(double));
  result_buffer_idx += robot_state.q.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.q_d.data(), robot_state.q_d.size() * sizeof(double));
  result_buffer_idx += robot_state.q_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.dq.data(), robot_state.dq.size() * sizeof(double));
  result_buffer_idx += robot_state.dq.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.dq_d.data(), robot_state.dq_d.size() * sizeof(double));
  result_buffer_idx += robot_state.dq_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.ddq_d.data(), robot_state.ddq_d.size() * sizeof(double));
  result_buffer_idx += robot_state.ddq_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.joint_contact.data(), robot_state.joint_contact.size() * sizeof(double));
  result_buffer_idx += robot_state.joint_contact.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.cartesian_contact.data(), robot_state.cartesian_contact.size() * sizeof(double));
  result_buffer_idx += robot_state.cartesian_contact.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.joint_collision.data(), robot_state.joint_collision.size() * sizeof(double));
  result_buffer_idx += robot_state.joint_collision.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.cartesian_collision.data(), robot_state.cartesian_collision.size() * sizeof(double));
  result_buffer_idx += robot_state.cartesian_collision.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.tau_ext_hat_filtered.data(), robot_state.tau_ext_hat_filtered.size() * sizeof(double));
  result_buffer_idx += robot_state.tau_ext_hat_filtered.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_F_ext_hat_K.data(), robot_state.O_F_ext_hat_K.size() * sizeof(double));
  result_buffer_idx += robot_state.O_F_ext_hat_K.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.K_F_ext_hat_K.data(), robot_state.K_F_ext_hat_K.size() * sizeof(double));
  result_buffer_idx += robot_state.K_F_ext_hat_K.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_dP_EE_d.data(), robot_state.O_dP_EE_d.size() * sizeof(double));
  result_buffer_idx += robot_state.O_dP_EE_d.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_T_EE_c.data(), robot_state.O_T_EE_c.size() * sizeof(double));
  result_buffer_idx += robot_state.O_T_EE_c.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_dP_EE_c.data(), robot_state.O_dP_EE_c.size() * sizeof(double));
  result_buffer_idx += robot_state.O_dP_EE_c.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.O_ddP_EE_c.data(), robot_state.O_ddP_EE_c.size() * sizeof(double));
  result_buffer_idx += robot_state.O_ddP_EE_c.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.theta.data(), robot_state.theta.size() * sizeof(double));
  result_buffer_idx += robot_state.theta.size();

  memcpy(&result_buffer[result_buffer_idx], robot_state.dtheta.data(), robot_state.dtheta.size() * sizeof(double));
  result_buffer_idx += robot_state.dtheta.size();

  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.self_collision_avoidance_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_velocity_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_velocity_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.force_control_safety_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_reflex ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_reflex ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.max_goal_pose_deviation_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.max_path_pose_deviation_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_velocity_profile_safety_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_position_motion_generator_start_pose_invalid ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_motion_generator_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_motion_generator_velocity_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_motion_generator_velocity_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_motion_generator_acceleration_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_position_motion_generator_start_pose_invalid ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_elbow_limit_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_velocity_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_velocity_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_acceleration_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_elbow_sign_inconsistent ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_start_elbow_invalid ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_joint_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_joint_velocity_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_joint_velocity_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_motion_generator_joint_acceleration_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.cartesian_position_motion_generator_invalid_frame ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.force_controller_desired_force_tolerance_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.controller_torque_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.start_elbow_sign_inconsistent ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.communication_constraints_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.power_limit_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_p2p_insufficient_torque_for_planning ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.tau_j_range_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.instability_detected ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.current_errors.joint_move_in_wrong_direction ? 1.0 : 0.0;

  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.self_collision_avoidance_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_velocity_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_velocity_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.force_control_safety_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_reflex ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_reflex ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.max_goal_pose_deviation_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.max_path_pose_deviation_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_velocity_profile_safety_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_position_motion_generator_start_pose_invalid ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_motion_generator_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_motion_generator_velocity_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_motion_generator_velocity_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_motion_generator_acceleration_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_position_motion_generator_start_pose_invalid ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_elbow_limit_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_velocity_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_velocity_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_acceleration_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_elbow_sign_inconsistent ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_start_elbow_invalid ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_joint_position_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_joint_velocity_limits_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_joint_velocity_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_motion_generator_joint_acceleration_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.cartesian_position_motion_generator_invalid_frame ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.force_controller_desired_force_tolerance_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.controller_torque_discontinuity ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.start_elbow_sign_inconsistent ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.communication_constraints_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.power_limit_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_p2p_insufficient_torque_for_planning ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.tau_j_range_violation ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.instability_detected ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = robot_state.last_motion_errors.joint_move_in_wrong_direction ? 1.0 : 0.0;

  result_buffer[result_buffer_idx++] = robot_state.control_command_success_rate;
  result_buffer[result_buffer_idx++] = static_cast<double>(static_cast<uint8_t>(robot_state.robot_mode));
  result_buffer[result_buffer_idx++] = robot_state.time.toSec();

  result_buffer[result_buffer_idx++] = gripper_state.width;
  result_buffer[result_buffer_idx++] = gripper_state.max_width;
  result_buffer[result_buffer_idx++] = gripper_state.is_grasped ? 1.0 : 0.0;
  result_buffer[result_buffer_idx++] = static_cast<double>(gripper_state.temperature);
  result_buffer[result_buffer_idx++] = gripper_state.time.toSec();
}

void ImpedanceControlSkill::write_result_to_shared_memory(SharedBufferTypePtr result_buffer, Robot* robot) {
  std::cout << "Writing final robot state to shared memory\n";

  switch(robot->robot_type_)
  {
    case RobotType::FRANKA:
      write_result_to_shared_memory(result_buffer, dynamic_cast<FrankaRobot *>(robot));
      break;
    case RobotType::UR5E:
      break;
  }
  
}

void ImpedanceControlSkill::write_feedback_to_shared_memory(SharedBufferTypePtr feedback_buffer) {
  std::cout << "Should write feedback to shared memory\n";
}
