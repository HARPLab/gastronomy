#pragma once

#include "base_meta_skill.h"

class run_loop;

class JointPoseContinuousSkill : public BaseMetaSkill {
 public:
  JointPoseContinuousSkill(int skill_idx): BaseMetaSkill(skill_idx) {
    is_composable_ = true;
  };

  bool isComposableSkill() override;
  
  void execute_skill_on_franka(run_loop *run_loop, franka::Robot* robot, franka::Gripper* gripper,
                               ControlLoopData *control_loop_data) override;

 private:
  bool return_status_{false};
};
