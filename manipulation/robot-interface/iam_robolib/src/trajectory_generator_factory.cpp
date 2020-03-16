//
// Created by mohit on 12/18/18.
//

#include "iam_robolib/trajectory_generator_factory.h"

#include <iostream>
#include <iam_robolib_common/definitions.h>

#include "iam_robolib/skills/base_meta_skill.h"
#include "iam_robolib/skills/base_skill.h"
#include "iam_robolib/trajectory_generator/goal_pose_dmp_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/gripper_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/impulse_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/joint_dmp_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/linear_pose_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/linear_joint_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/min_jerk_joint_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/min_jerk_pose_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/pose_dmp_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/relative_linear_pose_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/relative_min_jerk_pose_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/sine_joint_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/sine_pose_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/stay_in_initial_joints_trajectory_generator.h"
#include "iam_robolib/trajectory_generator/stay_in_initial_pose_trajectory_generator.h"

TrajectoryGenerator* TrajectoryGeneratorFactory::getTrajectoryGeneratorForSkill(
    SharedBufferTypePtr buffer) {
  TrajectoryGeneratorType trajectory_generator_type = static_cast<TrajectoryGeneratorType>(buffer[0]);
  TrajectoryGenerator *trajectory_generator = nullptr;

  std::cout << "Trajectory Generator Type: " << 
  static_cast<std::underlying_type<TrajectoryGeneratorType>::type>(trajectory_generator_type) << 
  "\n";

  switch (trajectory_generator_type) {
    case TrajectoryGeneratorType::GoalPoseDmpTrajectoryGenerator:
      trajectory_generator = new GoalPoseDmpTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::GripperTrajectoryGenerator:
      trajectory_generator = new GripperTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::ImpulseTrajectoryGenerator:
      trajectory_generator = new ImpulseTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::JointDmpTrajectoryGenerator:
      trajectory_generator = new JointDmpTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::LinearJointTrajectoryGenerator:
      trajectory_generator = new LinearJointTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::LinearPoseTrajectoryGenerator:
      trajectory_generator = new LinearPoseTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::MinJerkJointTrajectoryGenerator:
      trajectory_generator = new MinJerkJointTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::MinJerkPoseTrajectoryGenerator:
      trajectory_generator = new MinJerkPoseTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::PoseDmpTrajectoryGenerator:
      trajectory_generator = new PoseDmpTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::RelativeLinearPoseTrajectoryGenerator:
      trajectory_generator = new RelativeLinearPoseTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::RelativeMinJerkPoseTrajectoryGenerator:
      trajectory_generator = new RelativeMinJerkPoseTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::SineJointTrajectoryGenerator:
      trajectory_generator = new SineJointTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::SinePoseTrajectoryGenerator:
      trajectory_generator = new SinePoseTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::StayInInitialJointsTrajectoryGenerator:
      trajectory_generator = new StayInInitialJointsTrajectoryGenerator(buffer);
      break;
    case TrajectoryGeneratorType::StayInInitialPoseTrajectoryGenerator:
      trajectory_generator = new StayInInitialPoseTrajectoryGenerator(buffer);
      break;
    default:
      std::cout << "Cannot create Trajectory Generator with type:" << 
      static_cast<std::underlying_type<TrajectoryGeneratorType>::type>(trajectory_generator_type) << 
      "\n";
      return nullptr;
  }

  trajectory_generator->parse_parameters();
  return trajectory_generator;
}

