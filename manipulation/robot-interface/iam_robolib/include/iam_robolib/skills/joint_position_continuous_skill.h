#ifndef IAM_ROBOLIB_SKILLS_JOINT_POSITION_CONTINUOUS_SKILL_H_
#define IAM_ROBOLIB_SKILLS_JOINT_POSITION_CONTINUOUS_SKILL_H_

#include "iam_robolib/skills/base_meta_skill.h"

class JointPositionContinuousSkill : public BaseMetaSkill {
 public:
  JointPositionContinuousSkill(int skill_idx): BaseMetaSkill(skill_idx) {
    is_composable_ = true;
  };

  bool isComposableSkill() override;
  
  void execute_skill_on_franka(run_loop *run_loop, 
                               FrankaRobot* robot, 
                               RobotStateData* robot_state_data) override;

 private:
  bool return_status_{false};
};

#endif  // IAM_ROBOLIB_SKILLS_JOINT_POSITION_CONTINUOUS_SKILL_H_