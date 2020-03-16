//
// Created by mohit on 11/25/18.
//

#include "iam_robolib/feedback_controller/cartesian_impedance_feedback_controller.h"

#include <iostream>
#include <exception>

#include "iam_robolib/trajectory_generator/pose_trajectory_generator.h"

void CartesianImpedanceFeedbackController::parse_parameters() {
  // First parameter is reserved for the type

  int param_index = 1;
  int num_params = static_cast<int>(params_[param_index++]);

  switch(num_params) {
    case 0:
      // No parameters given, using default translational stiffness and rotational stiffness
      std::cout << "No parameters given, using default translational and rotational stiffness." << std::endl;
      break;
    case 2:
      // translational_stiffness(1) and rotational_stiffness(1) were given
      for(size_t i = 0; i < translational_stiffnesses_.size(); i++) {
        translational_stiffnesses_[i] = static_cast<double>(params_[2]);
      }
      for(size_t i = 0; i < rotational_stiffnesses_.size(); i++) {
        rotational_stiffnesses_[i] = static_cast<double>(params_[3]);
      }
      break;
    case 6:
      // translational_stiffness(3) and rotational_stiffness(3) were given
      for(size_t i = 0; i < translational_stiffnesses_.size(); i++) {
        translational_stiffnesses_[i] = static_cast<double>(params_[param_index++]);
      }
      for(size_t i = 0; i < rotational_stiffnesses_.size(); i++) {
        rotational_stiffnesses_[i] = static_cast<double>(params_[param_index++]);
      }
      break;
    default:
      std::cout << "Invalid number of params provided: " << num_params << std::endl;
  }
}

void CartesianImpedanceFeedbackController::initialize_controller() {
  std::cout << "No model provided." << std::endl;

  stiffness_ = Eigen::MatrixXd(6,6);
  stiffness_.setZero();
  stiffness_(0,0) = translational_stiffnesses_[0];
  stiffness_(1,1) = translational_stiffnesses_[1];
  stiffness_(2,2) = translational_stiffnesses_[2];
  stiffness_(3,3) = rotational_stiffnesses_[0];
  stiffness_(4,4) = rotational_stiffnesses_[1];
  stiffness_(5,5) = rotational_stiffnesses_[2];

  damping_ = Eigen::MatrixXd(6,6);
  damping_.setZero();
  damping_(0,0) = 2.0 * sqrt(translational_stiffnesses_[0]);
  damping_(1,1) = 2.0 * sqrt(translational_stiffnesses_[1]);
  damping_(2,2) = 2.0 * sqrt(translational_stiffnesses_[2]);
  damping_(3,3) = 2.0 * sqrt(rotational_stiffnesses_[0]);
  damping_(4,4) = 2.0 * sqrt(rotational_stiffnesses_[1]);
  damping_(5,5) = 2.0 * sqrt(rotational_stiffnesses_[2]);
}

void CartesianImpedanceFeedbackController::initialize_controller(FrankaRobot *robot) {
  model_ = robot->getModel();

  stiffness_ = Eigen::MatrixXd(6,6);
  stiffness_.setZero();
  stiffness_(0,0) = translational_stiffnesses_[0];
  stiffness_(1,1) = translational_stiffnesses_[1];
  stiffness_(2,2) = translational_stiffnesses_[2];
  stiffness_(3,3) = rotational_stiffnesses_[0];
  stiffness_(4,4) = rotational_stiffnesses_[1];
  stiffness_(5,5) = rotational_stiffnesses_[2];

  damping_ = Eigen::MatrixXd(6,6);
  damping_.setZero();
  damping_(0,0) = 2.0 * sqrt(translational_stiffnesses_[0]);
  damping_(1,1) = 2.0 * sqrt(translational_stiffnesses_[1]);
  damping_(2,2) = 2.0 * sqrt(translational_stiffnesses_[2]);
  damping_(3,3) = 2.0 * sqrt(rotational_stiffnesses_[0]);
  damping_(4,4) = 2.0 * sqrt(rotational_stiffnesses_[1]);
  damping_(5,5) = 2.0 * sqrt(rotational_stiffnesses_[2]);
}

void CartesianImpedanceFeedbackController::get_next_step() {
  // pass
}

void CartesianImpedanceFeedbackController::get_next_step(const franka::RobotState &robot_state,
                                                         TrajectoryGenerator *traj_generator) {
  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array = model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  // convert to Eigen
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
  Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());

  PoseTrajectoryGenerator* pose_trajectory_generator = dynamic_cast<PoseTrajectoryGenerator*>(traj_generator);

  if(pose_trajectory_generator == nullptr) {
    throw std::bad_cast();
  }

  Eigen::Vector3d position_d(pose_trajectory_generator->get_desired_position());
  Eigen::Quaterniond orientation_d(pose_trajectory_generator->get_desired_orientation());

  // compute error to desired equilibrium pose
  // position error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d;

  // orientation error
  // "difference" quaternion
  if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation * orientation_d.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  // compute "orientation error"
  error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

  // compute control
  Eigen::VectorXd tau_task(7), tau_d(7);

  // Spring damper system with damping ratio=1
  tau_task << jacobian.transpose() * (-stiffness_ * error - damping_ * (jacobian * dq));
  tau_d << tau_task + coriolis;

  Eigen::VectorXd::Map(&tau_d_array_[0], 7) = tau_d;
}