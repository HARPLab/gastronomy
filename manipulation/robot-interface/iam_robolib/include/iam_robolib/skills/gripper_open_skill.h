#ifndef IAM_ROBOLIB_SKILLS_GRIPPER_OPEN_SKILL_H_
#define IAM_ROBOLIB_SKILLS_GRIPPER_OPEN_SKILL_H_

#include "iam_robolib/skills/base_skill.h"

class TrajectoryGenerator;
class FeedbackController;
class TerminationHandler;

class GripperOpenSkill : public BaseSkill {
 public:
  GripperOpenSkill(int skill_idx, int meta_skill_idx, std::string description): BaseSkill(skill_idx,
      meta_skill_idx, description) {};

  void execute_skill() override;

  void execute_skill_on_franka(FrankaRobot* robot,
                               RobotStateData *robot_state_data) override;

  bool should_terminate() override;

 private:
  bool return_status_{false};
};

#endif  // IAM_ROBOLIB_SKILLS_GRIPPER_OPEN_SKILL_H_