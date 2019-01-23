//
// Created by mohit on 12/9/18.
//

#include "joint_pose_continuous_skill.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <array>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>

#include <iam_robolib/run_loop.h>

#include "base_skill.h"
#include "control_loop_data.h"
#include "TrajectoryGenerator/dmp_trajectory_generator.h"
#include "TerminationHandler/termination_handler.h"
#include "TrajectoryGenerator/trajectory_generator.h"

bool JointPoseContinuousSkill::isComposableSkill() {
  return true;
}


void JointPoseContinuousSkill::execute_skill_on_franka(run_loop *run_loop, franka::Robot* robot,
    franka::Gripper* gripper, ControlLoopData *control_loop_data) {
  try {
    double time = 0.0;
    double current_skill_time = 0.0;
    int log_counter = 0;

    SkillInfoManager *skill_info_manager = run_loop->getSkillInfoManager();
    BaseSkill *current_skill = skill_info_manager->get_current_skill();

    std::cout << "Will run JointPoseContinuousSkill control loop\n";

    franka::Model model = robot->loadModel();
    std::array<double, 7> last_dmp_q = robot->readOnce().q;
    std::array<double, 7> last_dmp_dq = robot->readOnce().dq;

    std::function<franka::JointPositions(const franka::RobotState&, franka::Duration)>
        joint_pose_callback = [&](
        const franka::RobotState& robot_state,
        franka::Duration period) -> franka::JointPositions {

      DmpTrajectoryGenerator* traj_generator = static_cast<DmpTrajectoryGenerator *>(
          current_skill->get_trajectory_generator());
      if (current_skill_time == 0.0) {
        traj_generator->initialize_trajectory(robot_state);
        traj_generator->y_ = last_dmp_q;
        traj_generator->dy_ = last_dmp_dq;
      }

      double period_in_seconds = period.toSec();
      time += period_in_seconds;
      current_skill_time += period_in_seconds;
      traj_generator->time_ = current_skill_time;
      traj_generator->dt_ = static_cast<float>(period_in_seconds);
      traj_generator->get_next_step();

      log_counter += 1;
      if (log_counter % 1 == 0) {
        control_loop_data->log_pose_desired(traj_generator->pose_desired_);
        control_loop_data->log_robot_state(robot_state, time);
      }

      TerminationHandler *termination_handler = current_skill->get_termination_handler();
      bool done = termination_handler->should_terminate(traj_generator);
      franka::JointPositions joint_desired(traj_generator->joint_desired_);

      if(done) {
        // Finish current skill and update RunLoopProcessInfo.
        run_loop->didFinishSkillInMetaSkill(current_skill);
        // Get new skill, the above update might have found a new skill.
        BaseSkill *new_skill = skill_info_manager->get_current_skill();

        if (new_skill->get_skill_id() == current_skill->get_skill_id()) {
          // No new skill, let's just continue with the current skill.
          // current_skill_time = 0.0;
          last_dmp_q = traj_generator->y_;
          last_dmp_dq = traj_generator->dy_;

          std::cout << "Meta skill continuing with old skill: " << current_skill->get_skill_id() << "\n";
        } else {
          if (current_skill->get_meta_skill_id() != new_skill->get_meta_skill_id()) {
            // New meta skill, so we should stop this.
            skill_info_manager->get_current_meta_skill()->setMetaSkillStatus(SkillStatus::FINISHED);
            last_dmp_q = traj_generator->y_;
            last_dmp_dq = traj_generator->dy_;
            std::cout << "Meta skill finished: " << skill_info_manager->get_current_meta_skill()->getMetaSkillId() << "\n";
            return franka::MotionFinished(franka::JointPositions(robot_state.q_d));
          } else {
            // Same meta skill, let's continue
            run_loop->start_new_skill(new_skill);
            current_skill_time = 0.0;

            last_dmp_q = traj_generator->y_;
            last_dmp_dq = traj_generator->dy_;

            current_skill = new_skill;
            std::cout << "New skill: " << new_skill->get_skill_id() << ", found for meta skill: " <<
              skill_info_manager->get_current_meta_skill()->getMetaSkillId() << "\n";
          }
        }
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

