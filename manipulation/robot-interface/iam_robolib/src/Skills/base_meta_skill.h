#pragma once

#include "base_skill.h"

class run_loop;

class BaseMetaSkill {
 public:
  BaseMetaSkill(int skill_idx): skill_idx_(skill_idx),
                                skill_status_(SkillStatus::TO_START) {};

  int getMetaSkillId();

  virtual bool isComposableSkill();

  void setMetaSkillStatus(SkillStatus new_task_status);

  SkillStatus getCurrentMetaSkillStatus();

  virtual void execute_skill_on_franka(run_loop* run_loop, franka::Robot *robot,
      franka::Gripper *gripper, ControlLoopData *control_loop_data);

 protected:
  int skill_idx_;
  SkillStatus skill_status_;
  bool is_composable_{false};

};