//
// Created by jacky on 2/13/19.
//

#include "iam_robolib/feedback_controller/force_axis_impedence_feedback_controller.h"

#include <iostream>

#include "iam_robolib/trajectory_generator/impulse_trajectory_generator.h"

void ForceAxisImpedenceFeedbackController::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  // No parameters given, using default translational stiffness and rotational stiffness
  if(num_params == 0) {
    std::cout << "No parameters given, using default translational and rotational stiffness." << std::endl;
  }
  // translational_stiffness(1) and rotational_stiffness(1) were given
  if(num_params == 5) {
    translational_stiffness_ = static_cast<double>(params_[2]);
    rotational_stiffness_ = static_cast<double>(params_[3]);
    // This should be a unit vector!!!
    axis_ = Eigen::Vector3d({params_[4], params_[5], params_[6]});
  } else {
    std::cout << "Invalid number of params provided: " << num_params << std::endl;
  }
}

void ForceAxisImpedenceFeedbackController::initialize_controller() {
  stiffness_ = Eigen::MatrixXd(6,6);
  stiffness_.setZero();
  stiffness_.topLeftCorner(3, 3) << translational_stiffness_ * Eigen::MatrixXd::Identity(3, 3);
  stiffness_.bottomRightCorner(3, 3) << rotational_stiffness_ * Eigen::MatrixXd::Identity(3, 3);
  damping_ = Eigen::MatrixXd(6,6);
  damping_.setZero();
  damping_.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness_) *
                                     Eigen::MatrixXd::Identity(3, 3);
  damping_.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness_) *
                                         Eigen::MatrixXd::Identity(3, 3);
}

void ForceAxisImpedenceFeedbackController::initialize_controller(FrankaRobot *robot) {
  model_ = robot->getModel();
  initialize_controller();
}

void ForceAxisImpedenceFeedbackController::get_next_step() {
  // pass
}

void ForceAxisImpedenceFeedbackController::get_next_step(const franka::RobotState &robot_state,
                                             TrajectoryGenerator *traj_generator) {
  
  ImpulseTrajectoryGenerator* impulse_trajectory_generator = dynamic_cast<ImpulseTrajectoryGenerator*>(traj_generator);

  if(impulse_trajectory_generator == nullptr) {
    throw std::bad_cast();
  }  

  std::array<double, 6> desired_force_torque_array = impulse_trajectory_generator->get_desired_force_torque();
  Eigen::Map<Eigen::VectorXd> desired_force_torque(desired_force_torque_array.data(), 6);

  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array = model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  // convert to Eigen
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
  Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  Eigen::Quaterniond orientation_d(impulse_trajectory_generator->get_initial_orientation());
  Eigen::Vector3d position_d(impulse_trajectory_generator->get_initial_position());

  // compute position error to given axis through the initial position
  Eigen::Matrix<double, 6, 1> error;
  Eigen::Vector3d delta_position = position - position_d;
  error.head(3) << delta_position - delta_position.dot(axis_) * axis_;

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
  tau_task << jacobian.transpose() * (-stiffness_ * error - damping_ * (jacobian * dq) + desired_force_torque);
  tau_d << tau_task + coriolis;

  Eigen::VectorXd::Map(&tau_d_array_[0], 7) = tau_d;
}