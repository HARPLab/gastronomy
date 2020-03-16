#ifndef IAM_ROBOLIB_SKILLS_BASE_SKILL_H_
#define IAM_ROBOLIB_SKILLS_BASE_SKILL_H_

#include <iam_robolib_common/definitions.h>

#include "iam_robolib/robots/franka_robot.h"

#include "iam_robolib/feedback_controller/feedback_controller.h"
#include "iam_robolib/termination_handler/termination_handler.h"
#include "iam_robolib/trajectory_generator/trajectory_generator.h"

class run_loop;

class RobotStateData;

namespace franka {
  class RobotState;
  class Duration;
}

class BaseSkill {
 public:
  BaseSkill(int skill_idx, int meta_skill_idx, std::string description) : 
                                          skill_idx_(skill_idx),
                                          meta_skill_idx_(meta_skill_idx),
                                          skill_status_(SkillStatus::TO_START),
                                          description_(description) 
  {};

  /**
   * Get skill id.
   */
  int get_skill_id();

  /**
   * Get meta-skill id for this skill id.
   */
  int get_meta_skill_id();

  /**
   * Get skill description.
   * @return
   */
  std::string get_description();

  /**
   * Update skill status;
   */
  void set_skill_status(SkillStatus new_task_status);

  /**
   * Get current skill status.
   */
  SkillStatus get_current_skill_status();

  TrajectoryGenerator* get_trajectory_generator();
  FeedbackController* get_feedback_controller();
  TerminationHandler* get_termination_handler();

  /**
   * Start skill. Initiliazes and parses the parameters for different skill 
   * components.
   * @param robot
   * @param traj_generator
   * @param feedback_controller
   * @param termination_handler
   */
  void start_skill(Robot* robot,
                   TrajectoryGenerator* traj_generator,
                   FeedbackController* feedback_controller,
                   TerminationHandler* termination_handler);

  virtual void execute_skill() = 0;

  /**
   * Execute skill on franka with the given robot and gripper configuration.
   * @param robot
   * @param gripper
   * @param robot_state_data
   */
  virtual void execute_skill_on_franka(run_loop* run_loop,
                                       FrankaRobot* robot,
                                       RobotStateData* robot_state_data) = 0;

  virtual bool has_terminated(Robot* robot);
  // Check if termination was due to collision with virtual walls.
  virtual bool has_terminated_by_virt_coll();


  virtual void write_result_to_shared_memory(SharedBufferTypePtr result_buffer);
  virtual void write_result_to_shared_memory(SharedBufferTypePtr result_buffer, 
                                             FrankaRobot* robot);
  virtual void write_result_to_shared_memory(SharedBufferTypePtr result_buffer, 
                                             Robot* robot);

  virtual void write_feedback_to_shared_memory(SharedBufferTypePtr feedback_buffer);

 protected:
  int skill_idx_;
  int meta_skill_idx_;
  SkillStatus skill_status_;
  std::string description_;

  double current_period_;

  TrajectoryGenerator* traj_generator_= nullptr;
  FeedbackController* feedback_controller_= nullptr;
  TerminationHandler* termination_handler_= nullptr;
  franka::Model* model_= nullptr;
};

#endif  // IAM_ROBOLIB_SKILLS_BASE_SKILL_H_