//
// Created by mohit on 12/18/18.
//

#include "iam_robolib/trajectory_generator_factory.h"

#include "iam_robolib/skills/base_meta_skill.h"
#include "iam_robolib/skills/base_skill.h"
#include "iam_robolib/trajectory_generator/dmp_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/gripper_open_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/linear_joint_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/counter_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/linear_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/linear_trajectory_generator_with_time_and_goal.h"
#include "iam_robolib/trajectory_generator/relative_linear_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/stay_in_initial_position_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/impulse_trajectory_generator.h"

TrajectoryGenerator* TrajectoryGeneratorFactory::getTrajectoryGeneratorForSkill(
    SharedBuffer buffer) {
  int traj_gen_id = static_cast<int>(buffer[0]);
  TrajectoryGenerator *traj_generator = nullptr;

  std::cout << "Trajectory generator id: " << traj_gen_id << "\n";

  if (traj_gen_id == 1) {
    // Create Counter based trajectory.
    traj_generator = new CounterTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 2) {
    traj_generator = new LinearTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 3) {
    traj_generator = new LinearJointTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 4) {
    traj_generator = new LinearTrajectoryGeneratorWithTimeAndGoal(buffer);
  } else if (traj_gen_id == 5){
    traj_generator = new GripperOpenTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 6) {
    traj_generator = new StayInInitialPositionTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 7) {
    traj_generator = new DmpTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 8) {
    traj_generator = new RelativeLinearTrajectoryGenerator(buffer);
  } else if (traj_gen_id == 9) {
    traj_generator = new ImpulseTrajectoryGenerator(buffer);
  } else {
    // Cannot create Trajectory generator for this skill. Throw error
    std::cout << "Cannot create TrajectoryGenerator with class_id:" << traj_gen_id << "\n";
    return nullptr;
  }
  traj_generator->parse_parameters();
  return traj_generator;
}

