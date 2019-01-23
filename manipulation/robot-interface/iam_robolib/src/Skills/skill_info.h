//
// Created by mohit on 11/20/18.
//

#pragma once

#include "base_skill.h"

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include "FeedbackController/feedback_controller.h"
#include "TerminationHandler/termination_handler.h"
#include "TrajectoryGenerator/trajectory_generator.h"

class SkillInfo : public BaseSkill {
  public:
    SkillInfo(int skill_idx, int meta_skill_idx): BaseSkill(skill_idx, meta_skill_idx) {};


    void execute_skill() override;

    void execute_skill_on_franka(franka::Robot *robot, franka::Gripper *gripper,
                                 ControlLoopData *control_loop_data) override;

    void execute_meta_skill_on_franka(franka::Robot *robot, franka::Gripper *gripper,
                                      ControlLoopData *control_loop_data);

    void execute_skill_on_franka_temp(franka::Robot *robot, franka::Gripper *gripper,
        ControlLoopData *control_loop_data);

    void execute_skill_on_franka_temp2(franka::Robot *robot, franka::Gripper *gripper,
        ControlLoopData *control_loop_data);

    void execute_skill_on_franka_joint_base(franka::Robot* robot, franka::Gripper* gripper,
        ControlLoopData *control_loop_data);

    bool should_terminate() override;

    /**
     * Write result to the shared memory after skill is done.
     * @param result_buffer
     */
    void write_result_to_shared_memory(float *result_buffer) override;

    /**
     * Write result to the shared memory after skill is done.
     * @param result_buffer
     */
    void write_result_to_shared_memory(float *result_buffer, franka::Robot *robot) override;


    /**
     * Write feedback result to the shared memory as feedback for a skill.
     * @param feedback_buffer
     */
    void write_feedback_to_shared_memory(float *feedback_buffer) override;

 protected:
    const std::array<double, 7> k_gains_ = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    // Damping
    const std::array<double, 7> d_gains_ = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
};

