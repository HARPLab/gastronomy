#ifndef IAM_ROBOLIB_SKILLS_BASE_SKILL_H_
#define IAM_ROBOLIB_SKILLS_BASE_SKILL_H_

#include <franka/gripper.h>
#include <franka/robot.h>

enum class SkillStatus { TO_START, RUNNING, FINISHED };  // enum class

class RobotStateData;
class FeedbackController;
class TerminationHandler;
class TrajectoryGenerator;
namespace franka {
  class RobotState;
  class Duration;
}

class BaseSkill {
 public:
  BaseSkill(int skill_idx, int meta_skill_idx, std::string description): skill_idx_(skill_idx),
                                                                         meta_skill_idx_(meta_skill_idx),
                                                                         skill_status_(SkillStatus::TO_START),
                                                                         description_(description){};

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
   * Start skill. Initiliazes and parses the parameters for different skill components.
   * @param robot
   * @param traj_generator
   * @param feedback_controller
   * @param termination_handler
   */
  void start_skill(franka::Robot* robot,
                   TrajectoryGenerator *traj_generator,
                   FeedbackController *feedback_controller,
                   TerminationHandler *termination_handler);

  virtual void execute_skill() = 0;

  /**
   * Execute skill on franka with the given robot and gripper configuration.
   * @param robot
   * @param gripper
   * @param robot_state_data
   */
  virtual void execute_skill_on_franka(franka::Robot *robot, franka::Gripper *gripper,
                                       RobotStateData *robot_state_data) = 0;

  virtual bool should_terminate();

  virtual void write_result_to_shared_memory(float *result_buffer);
  virtual void write_result_to_shared_memory(float *result_buffer, franka::Robot *robot);

  virtual void write_feedback_to_shared_memory(float *feedback_buffer);

 protected:
  int skill_idx_;
  int meta_skill_idx_;
  SkillStatus skill_status_;
  std::string description_;

  TrajectoryGenerator *traj_generator_= nullptr;
  FeedbackController *feedback_controller_= nullptr;
  TerminationHandler *termination_handler_= nullptr;
};

#endif  // IAM_ROBOLIB_SKILLS_BASE_SKILL_H_