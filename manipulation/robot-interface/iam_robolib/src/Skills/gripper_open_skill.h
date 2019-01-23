#pragma once

#include "skill_info.h"

class TrajectoryGenerator;
class FeedbackController;
class TerminationHandler;

class GripperOpenSkill : public BaseSkill {
 public:
  GripperOpenSkill(int skill_idx, int meta_skill_idx): BaseSkill(skill_idx, meta_skill_idx) {};

  void execute_skill() override;

  void execute_skill_on_franka(franka::Robot* robot, franka::Gripper* gripper,
                               ControlLoopData *control_loop_data) override;

  bool should_terminate() override;

 private:
  bool return_status_{false};
};

