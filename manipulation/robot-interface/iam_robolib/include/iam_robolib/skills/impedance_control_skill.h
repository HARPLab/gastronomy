#ifndef IAM_ROBOLIB_SKILLS_IMPEDANCE_CONTROL_SKILL_H_
#define IAM_ROBOLIB_SKILLS_IMPEDANCE_CONTROL_SKILL_H_

#include "iam_robolib/skills/base_skill.h"

class ImpedanceControlSkill : public BaseSkill {
 public:
  ImpedanceControlSkill(int skill_idx, int meta_skill_idx, std::string description) : 
            BaseSkill(skill_idx, meta_skill_idx, description) 
  {};

  void execute_skill() override;

  void execute_skill_on_franka(run_loop* run_loop, 
                               FrankaRobot* robot, 
                               RobotStateData* robot_state_data) override;

  /**
   * Write result to the shared memory after skill is done.
   * @param result_buffer
   */
  void write_result_to_shared_memory(SharedBufferTypePtr result_buffer) override;

  /**
   * Write result to the shared memory after skill is done.
   * @param result_buffer
   */
  void write_result_to_shared_memory(SharedBufferTypePtr result_buffer, 
                                     FrankaRobot *robot) override;

  /**
   * Write result to the shared memory after skill is done.
   * @param result_buffer
   */
  void write_result_to_shared_memory(SharedBufferTypePtr result_buffer, 
                                     Robot *robot) override;


  /**
   * Write feedback result to the shared memory as feedback for a skill.
   * @param feedback_buffer
   */
  void write_feedback_to_shared_memory(SharedBufferTypePtr feedback_buffer) 
                                                                      override;

 protected:
  const std::array<double, 7> k_gains_ = {{600.0, 600.0, 600.0, 600.0, 
                                           250.0, 150.0, 50.0}};
  // Damping
  const std::array<double, 7> d_gains_ = {{50.0, 50.0, 50.0, 50.0, 
                                           30.0, 25.0, 15.0}};
};

#endif  // IAM_ROBOLIB_SKILLS_IMPEDANCE_CONTROL_SKILL_H_