//
// Created by jacky on 1/26/19.
//

#include "iam_robolib/skills/force_torque_skill.h"

#include <cassert>
#include <iostream>
#include <array>

#include <franka/robot.h>

#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/run_loop.h"
#include "iam_robolib/run_loop_shared_memory_handler.h"
#include "iam_robolib/trajectory_generator/impulse_trajectory_generator.h"

#include <iam_robolib_common/definitions.h>
#include <iam_robolib_common/run_loop_process_info.h>

void ForceTorqueSkill::execute_skill() {
  assert(false);
}

void ForceTorqueSkill::execute_skill_on_franka(run_loop* run_loop,
                                               FrankaRobot* robot,
                                               RobotStateData *robot_state_data) {
  double time = 0.0;
  int log_counter = 0;
  std::array<double, 16> pose_desired;

  std::cout << "Will run the control loop\n";

  RunLoopSharedMemoryHandler* shared_memory_handler = run_loop->get_shared_memory_handler();
  RunLoopProcessInfo* run_loop_info = shared_memory_handler->getRunLoopProcessInfo();
  boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(
                                  *(shared_memory_handler->getRunLoopProcessInfoMutex()),
                                  boost::interprocess::defer_lock);

  ImpulseTrajectoryGenerator* impulse_trajectory_generator = dynamic_cast<ImpulseTrajectoryGenerator*>(traj_generator_);

  if(impulse_trajectory_generator == nullptr) {
    throw std::bad_cast();
  } 

  auto force_control_callback = [&](const franka::RobotState& robot_state, 
                    franka::Duration period) -> franka::Torques {
    
    current_period_ = period.toSec();
    time += current_period_;

    if (time == 0.0) {
      impulse_trajectory_generator->initialize_trajectory(robot_state, SkillType::ForceTorqueSkill);
      try {
        if (lock.try_lock()) {
          run_loop_info->set_time_skill_started_in_robot_time(robot_state.time.toSec());
          lock.unlock();
        } 
      } catch (boost::interprocess::lock_exception) {
        // Do nothing
      }
    }
    log_counter += 1;
    if (log_counter % 1 == 0) {
      pose_desired = robot_state.O_T_EE_d;
      robot_state_data->log_robot_state(pose_desired, robot_state, robot->getModel(), time);
    }

    bool done = termination_handler_->should_terminate_on_franka(robot_state, model_,
                                                                 traj_generator_);
    if (done && time > 0.0) {
      try {
        if (lock.try_lock()) {
          run_loop_info->set_time_skill_finished_in_robot_time(robot_state.time.toSec());
          lock.unlock();
        } 
      } catch (boost::interprocess::lock_exception) {
        // Do nothing
      }

      // return 0 torques to finish
      std::array<double, 7> tau_d_array{};
      franka::Torques torques(tau_d_array);
      return franka::MotionFinished(torques);
    }
    
    impulse_trajectory_generator->check_displacement_cap(robot_state);
    traj_generator_->time_ = time;
    traj_generator_->dt_ = current_period_;
    if(time > 0.0) {
      traj_generator_->get_next_step();
    }

    feedback_controller_->get_next_step(robot_state, traj_generator_);

    return feedback_controller_->tau_d_array_;
  };

  robot->robot_.control(force_control_callback);
}

