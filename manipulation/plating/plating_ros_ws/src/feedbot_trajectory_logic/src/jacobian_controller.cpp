//
// This class computes the robot jacobian and can be use to move the robot along a straight-line path
// (in cylindrical coordinates) to the target pose (in cartesian coordinates)
//

#include <feedbot_trajectory_logic/jacobian_controller.h>

const double TRANS_EPSILON = 0.01;
const double QUAT_EPSILON = 0.01;
const double ANGLE_STEP_SIZE = 0.1;
//const double TRANS_STEP_SIZE = 0.01; replaced by step_size_meters
const double MAX_JOINT_STEP = 0.05;
const double MIN_TIME_TO_REACH_NEW = 0.5;
const double TRANS_SPEED = 0.5;
 
//constructor
JacobianController::JacobianController(double trans_step_size_meters,  RobotInterface* robot_interface, ros::NodeHandle* n, std::string link_prefix)
  : robot_model_loader_("robot_description"),
    _trans_step_size_meters(trans_step_size_meters)
{
  link_prefix_ = link_prefix;
  kinematic_model_ = robot_model_loader_.getModel();
  robot_interface_ = robot_interface;
  robot_interface_->InitializeConnection();

  kinematic_state_ = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model_));
  kinematic_state_->setToDefaultValues();

  joint_model_group_ = kinematic_model_->getJointModelGroup(robot_interface_->srdf_group_name_);
  std::cout << "BOY HAVE WE GOT LINKS FOR YOU!!!!!\n"; 
  std::vector<std::string> linkNames = joint_model_group_->getLinkModelNames();
  for (std::string link : linkNames)
  {
    std::cout << link << " ,  ";
  }
  std::cout << "BOY HAVE WE GOT LINKS FOR YOU!!!!!\n"; 

  //std::cout << "Waiting to give time for connection to Arduino to be established" << std::endl;
  //ros::Duration(2).sleep();
  std::cout << "Moving to default position" << std::endl;
  robot_interface->SendTargetAngles(robot_interface_->initial_joint_values_, 3);
  kinematic_state_->setJointGroupPositions(joint_model_group_, robot_interface->initial_joint_values_);  
  current_pose_ = kinematic_state_->getGlobalLinkTransform(link_prefix_ + robot_interface_->end_effector_link_);
  
  std::cout << "Sleeping for 2 seconds to get to initial position";
  ros::Duration(2).sleep();
}

// returns the distance to the target
// discretized to 0 (arrived) or 1 (not there yet) currently...
double
JacobianController::make_step_to_target_pose(const geometry_msgs::Pose &target_pose)
{
  // convert the input ROS topic pose to the associated Eigen representation
  Eigen::Quaterniond target_quat(target_pose.orientation.w,
                              target_pose.orientation.x,
                              target_pose.orientation.y, 
                              target_pose.orientation.z);

  // because this is coming in from a ROS topic, they might not have properly normalized it when typing
  // in the desired quaternion...Travers
  target_quat.normalize();
  Eigen::Translation3d target_trans(target_pose.position.x,
                                    target_pose.position.y,
                                    target_pose.position.z);

  Eigen::Translation3d cur_trans(current_pose_.translation());
  Eigen::Quaterniond cur_quat(current_pose_.rotation());
  double quat_dist = quat_distance(cur_quat, target_quat);
  double trans_dist = distance(cur_trans, target_trans);
  //std::cout << trans_dist << " is the translation distance. " << quat_dist << " is the quat dist" << std::endl;
  //std::cout << target_quat.w() << "," <<  target_quat.vec() << " is the target quat. " << cur_quat.w() << "," << target_quat.vec() << " is the current quat" << std::endl;
  if (trans_dist < TRANS_EPSILON &&
      quat_dist < QUAT_EPSILON)
  {
    // No need to move. Return 0 to say we are at the target.
    return 0.0;
  }
 
  // Compute the translation and rotation change we need 
  Eigen::Translation3d trans_matrix = cur_trans.inverse() * target_trans;
  Eigen::Vector3d trans_diff = trans_matrix.translation();
  Eigen::Quaterniond rot_diff =  target_quat * cur_quat.inverse();
  Eigen::AngleAxisd rot_axis_angle(rot_diff);
  double rot_angle = rot_axis_angle.angle();
  Eigen::Vector3d rot_axis = rot_axis_angle.axis();
  
  // TSR is ashamed that this is not in a testable helper function.
  // Maybe this is just an issue with "old" Eigen... (optimism)
  if (rot_angle < 0)
  {
    rot_angle *= -1.0;
    rot_axis = rot_axis *= -1.0;
  }

  if (rot_angle > M_PI)
  {
    rot_angle = 2 * M_PI - rot_angle;
    rot_axis = rot_axis *= -1;
  }

  // cylindrical_diff is of the form dR, dTheta, dZ
  // where R is sqrt(x^2 + y^2), theta is arctan2(y,x), and z is z
  Eigen::Vector3d cylindrical_diff = get_cylindrical_point_translation(cur_trans.translation(), target_trans.translation());

  // scale down the move if the translation is too big
  if (trans_dist > _trans_step_size_meters)
  {
    double step_scale = _trans_step_size_meters / trans_dist;
    scale_down_step(step_scale, trans_diff, rot_angle, cylindrical_diff);
    trans_dist = _trans_step_size_meters;
  }
 
  // scale down the move if the rotation is too big 
  if (rot_angle > ANGLE_STEP_SIZE)
  {
    double angle_step_scale = ANGLE_STEP_SIZE / rot_angle;
    scale_down_step(angle_step_scale, trans_diff, rot_angle, cylindrical_diff);
    trans_dist = trans_dist * angle_step_scale;
  }

  Eigen::VectorXd joint_delta = get_joint_delta(cylindrical_diff, rot_angle, rot_axis);

  // ensure that no joint rotation is larger than MAX_JOINT_STEP at any given time
  for(int i = 0; i < 6; i++)
  {
    double cur_joint_step = std::abs(joint_delta[i]);
    if (cur_joint_step > MAX_JOINT_STEP)
    {
      double scale = MAX_JOINT_STEP / cur_joint_step;
      for (int j = 0; j < 6; j++)
      {
        joint_delta[j] = joint_delta[j] * scale;
      }
    }
  }

  // actually send the requested angles to the robot
  std::vector<double> current_joint_values;
  kinematic_state_->copyJointGroupPositions(joint_model_group_, current_joint_values);
  std::vector<double> new_joint_values(6);
  for(std::size_t i = 0; i < 6; ++i)
  {
    new_joint_values[i] = joint_delta(i) + current_joint_values[i];
  }

  bool successful_move = robot_interface_->SendTargetAngles(new_joint_values, std::max(MIN_TIME_TO_REACH_NEW, trans_dist / TRANS_SPEED));

  if (successful_move) {
    // update the state of this class to reflect the new robot position
    kinematic_state_->setJointGroupPositions(joint_model_group_, new_joint_values);  
    current_pose_ = kinematic_state_->getGlobalLinkTransform(link_prefix_ + robot_interface_->end_effector_link_);
  }
  // return 1 to say we have not yet arrived at the target 
  return 1.0;
}

// scale down the step by step_scale amount
void
JacobianController::scale_down_step(double step_scale, Eigen::Vector3d &trans_diff, double &rot_angle, Eigen::Vector3d &cylindrical_diff)
{
    trans_diff = trans_diff * step_scale;
    rot_angle = rot_angle * step_scale;
    cylindrical_diff = cylindrical_diff * step_scale;
}

// compute the required joint angles to make the robot end effector move the desired change in position (in cylindrical coordinates)
// and rotation (rot_angle around rot_axis)
// note: cylindrical_diff is of the form dR, dTheta, dZ
//   where R is sqrt(x^2 + y^2), theta is arctan2(y,x), and z is z
// rot_axis is (x,y,z) in the frame of the robot base
Eigen::VectorXd
JacobianController::get_joint_delta(Eigen::Vector3d cylindrical_diff, double rot_angle, Eigen::Vector3d rot_axis)
{
  // get the cylindrical jacobian 
  Eigen::MatrixXd jacobian = get_cylindrical_jacobian();
  //std::cout << "jacobian was computed to be" << jacobian << std::endl; 

  // create a single column vector of the translation and rotation we want to achieve
  Eigen::Matrix<double, 6, 1> target_delta;
  target_delta << cylindrical_diff, rot_angle * rot_axis;

  // use regularized least squares to compute the required joint angle changes
  //https://eigen.tuxfamily.org/dox/group__LeastSquares.html
  // with the extra addition that we use a tiny regularization term to reduce problems due to singularities, so we're solving (J^T J + lambda * I)^-1 J^T Y
  double lambda = 0.02;
  Eigen::VectorXd joint_delta = (jacobian.transpose() * jacobian + (lambda * Eigen::MatrixXd::Identity(6,6))).ldlt().solve(jacobian.transpose() * target_delta);
  return joint_delta;
}

// The jacobian in cylindrical coordinates
Eigen::MatrixXd
JacobianController::get_cylindrical_jacobian()
{
  Eigen::Vector3d reference_point_position(0.0,0.0,0.0);
  Eigen::MatrixXd jacobian;
  const moveit::core::LinkModel *link_model = kinematic_state_->getLinkModel(link_prefix_+ robot_interface_->end_effector_link_);
  kinematic_state_->getJacobian(joint_model_group_,
    link_model,
    reference_point_position,
    jacobian);
  //std::cout << "rect_jacob "<< jacobian << std::endl; 

  Eigen::Vector3d cur_trans(current_pose_.translation());
  Eigen::Matrix<double,6,6> rect_to_cyl_jacob = compute_jacob_from_rect_to_cyl(cur_trans);
  //std::cout << "rect_to_cyl_jacob" << rect_to_cyl_jacob << std::endl; 
  
  Eigen::Matrix<double,6,6> cyl_jacobian = rect_to_cyl_jacob * jacobian;
  return cyl_jacobian;
}
