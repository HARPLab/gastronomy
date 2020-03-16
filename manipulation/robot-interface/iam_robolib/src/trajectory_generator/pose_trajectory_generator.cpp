#include "iam_robolib/trajectory_generator/pose_trajectory_generator.h"

#include <iostream>

void PoseTrajectoryGenerator::parse_parameters() {
  // First parameter is reserved for the type

  int params_idx = 1;
  int num_params = static_cast<int>(params_[params_idx++]);

  switch(num_params) {
    case 17:
      // Time + Full Cartesian Pose (std::array<double,16>) was given
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        for(size_t i = 0; i < goal_pose_.size(); i++) {
          goal_pose_[i] = static_cast<double>(params_[params_idx++]);
        }
        Eigen::Affine3d goal_transform(Eigen::Matrix4d::Map(goal_pose_.data()));
        goal_position_ = Eigen::Vector3d(goal_transform.translation());
        goal_orientation_ = Eigen::Quaterniond(goal_transform.linear());
      }
      break;
    case 8:
      // Time + x,y,z + quaternion was given
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        goal_position_[0] = static_cast<double>(params_[params_idx++]);
        goal_position_[1] = static_cast<double>(params_[params_idx++]);
        goal_position_[2] = static_cast<double>(params_[params_idx++]);

        std::array<double,4> goal_quaternion{};
        for(size_t i = 0; i < goal_quaternion.size(); i++) {
          goal_quaternion[i] = static_cast<double>(params_[params_idx++]);
        }
        goal_orientation_ = Eigen::Quaterniond(goal_quaternion[0],
                                               goal_quaternion[1],
                                               goal_quaternion[2],
                                               goal_quaternion[3]);
      }
      break;
    case 7:
      // Time + x,y,z + axis angle was given
      {
        run_time_ = static_cast<double>(params_[params_idx++]);

        goal_position_[0] = static_cast<double>(params_[params_idx++]);
        goal_position_[1] = static_cast<double>(params_[params_idx++]);
        goal_position_[2] = static_cast<double>(params_[params_idx++]);

        Eigen::Vector3d goal_axis_angle;
        for(int i = 0; i < 3; i++) {
          goal_axis_angle[i] = static_cast<double>(params_[params_idx++]);
        }

        double angle = goal_axis_angle.norm();
        double sin_angle_divided_by_2 = std::sin(angle/2);
        double cos_angle_divided_by_2 = std::cos(angle/2);

        goal_orientation_ = Eigen::Quaterniond(goal_axis_angle[0] * sin_angle_divided_by_2,
                                               goal_axis_angle[1] * sin_angle_divided_by_2,
                                               goal_axis_angle[2] * sin_angle_divided_by_2,
                                               cos_angle_divided_by_2);
      }
      break;
    default:
      std::cout << "PoseTrajectoryGenerator: Invalid number of params provided: " << num_params << std::endl;
  }
}

void PoseTrajectoryGenerator::initialize_trajectory() {
  // pass
}

void PoseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state) {
  initialize_initial_and_desired_poses(robot_state, SkillType::ImpedanceControlSkill);
  fix_goal_quaternion();
}

void PoseTrajectoryGenerator::initialize_trajectory(const franka::RobotState &robot_state,
                                                    SkillType skill_type) {
  initialize_initial_and_desired_poses(robot_state, skill_type);
  fix_goal_quaternion();
}

void PoseTrajectoryGenerator::initialize_initial_and_desired_poses(const franka::RobotState &robot_state,
                                                                   SkillType skill_type) {
  switch(skill_type) {
    case SkillType::ImpedanceControlSkill:
      // Use O_T_EE as the initial pose for Impedance Control for safety reasons
      initial_pose_ = robot_state.O_T_EE;
      desired_pose_ = robot_state.O_T_EE;
      break;
    case SkillType::CartesianPoseSkill:
      // Use O_T_EE_c as the initial pose for Cartesian Pose Control to 
      // avoid trajectory discontinuity errors
      initial_pose_ = robot_state.O_T_EE_c;
      desired_pose_ = robot_state.O_T_EE_c;
      break;
    default:
      // Default to using O_T_EE as the initial pose for safety reasons 
      initial_pose_ = robot_state.O_T_EE;
      desired_pose_ = robot_state.O_T_EE;
  }

  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  initial_position_ = Eigen::Vector3d(initial_transform.translation());
  initial_orientation_ = Eigen::Quaterniond(initial_transform.linear());
  desired_position_ = Eigen::Vector3d(initial_transform.translation());
  desired_orientation_ = Eigen::Quaterniond(initial_transform.linear());
}

void PoseTrajectoryGenerator::fix_goal_quaternion(){
  // Flip the goal quaternion if the initial orientation dotted with the goal
  // orientation is negative.

  initial_orientation_.normalize();
  goal_orientation_.normalize();

  double quaternion_dot_product = initial_orientation_.coeffs().dot(goal_orientation_.coeffs());
  if (quaternion_dot_product < 0.0) {
    goal_orientation_.coeffs() << -goal_orientation_.coeffs();
  }

  same_orientation = abs(quaternion_dot_product) > quaternion_dist_threshold;
  std::cout << "PoseTrajectoryGenerator: same_orientation = " << same_orientation << std::endl;
}

void PoseTrajectoryGenerator::calculate_desired_pose() {
  if(same_orientation) {
    calculate_desired_position();
  } else {
    Eigen::Affine3d desired_pose_affine = Eigen::Affine3d::Identity();
    desired_pose_affine.translate(desired_position_);
    // Normalize desired orientation quaternion to avoid precision issues
    desired_orientation_.normalize();
    desired_pose_affine.rotate(desired_orientation_);
    Eigen::Matrix4d desired_pose_matrix = desired_pose_affine.matrix();

    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 4; j++) {
        desired_pose_[4*i+j] = desired_pose_matrix(j,i); // Column wise
      }
    }
  }
}

void PoseTrajectoryGenerator::calculate_desired_position() {
  // Just change the desired position and not the orientation.
  for (int i = 0; i < 3; i++) {
    desired_pose_[12 + i] = desired_position_(i);
  }
}

const std::array<double, 16>& PoseTrajectoryGenerator::get_desired_pose() const {
  return desired_pose_;
}

const Eigen::Vector3d& PoseTrajectoryGenerator::get_desired_position() const {
  return desired_position_;
}

const Eigen::Quaterniond& PoseTrajectoryGenerator::get_desired_orientation() const {
  return desired_orientation_;
}

const Eigen::Vector3d& PoseTrajectoryGenerator::get_goal_position() const {
  return goal_position_;
}

const Eigen::Quaterniond& PoseTrajectoryGenerator::get_goal_orientation() const {
  return goal_orientation_;
}