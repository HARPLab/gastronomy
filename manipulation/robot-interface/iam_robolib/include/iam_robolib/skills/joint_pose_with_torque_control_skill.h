#ifndef IAM_ROBOLIB_SKILLS_JOINT_POSE_WITH_TORQUE_CONTROL_SKILL_H_
#define IAM_ROBOLIB_SKILLS_JOINT_POSE_WITH_TORQUE_CONTROL_SKILL_H_

#include "iam_robolib/skills/base_skill.h"

class JointPoseWithTorqueControlSkill : public BaseSkill {
 public:
  JointPoseWithTorqueControlSkill(int skill_idx, int meta_skill_idx, std::string description): BaseSkill(
      skill_idx, meta_skill_idx, description) {};

  void execute_skill() override;

  void execute_skill_on_franka(franka::Robot* robot, franka::Gripper* gripper,
                               RobotStateData *robot_state_data) override;

  void execute_meta_skill_on_franka(franka::Robot *robot, franka::Gripper *gripper,
                                    RobotStateData *robot_state_data);

 private:
  bool return_status_{false};
};

#endif  // IAM_ROBOLIB_SKILLS_JOINT_POSE_WITH_TORQUE_CONTROL_SKILL_H_