//
// Created by mohit on 12/9/18.
//

#include "base_meta_skill.h"

#include <iam_robolib/run_loop.h>

int BaseMetaSkill::getMetaSkillId() {
  return skill_idx_;
}

bool BaseMetaSkill::isComposableSkill() {
  return is_composable_;
}

void BaseMetaSkill::setMetaSkillStatus(SkillStatus new_status) {
  skill_status_ = new_status;
}

SkillStatus BaseMetaSkill::getCurrentMetaSkillStatus() {
  return skill_status_;
}

void BaseMetaSkill::execute_skill_on_franka(run_loop* run_loop, franka::Robot *robot,
    franka::Gripper *gripper, ControlLoopData *control_loop_data) {
  BaseSkill* skill = run_loop->getSkillInfoManager()->get_current_skill();
  if (skill != nullptr) {
    skill->execute_skill_on_franka(robot, gripper, control_loop_data);
    run_loop->finish_current_skill(skill);
  }
}
