//
// This class connects to DOMUS 
// and can be used to send target angles for DOMUS to move to
//
#include "feedbot_trajectory_logic/ros_robot_interface.h"

RosRobotInterface::RosRobotInterface(std::string follow_joint_trajectory_name, CustomRobotParams robot_params) : RobotInterface(robot_params)
{
  follow_joint_trajectory_name_ = follow_joint_trajectory_name;
}

void
RosRobotInterface::InitializeConnection()
{
  ac_ = std::make_shared<actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>>(follow_joint_trajectory_name_, true);
  ac_->waitForServer();
}

// move to the target joint_angles and the motion should take you secs seconds.
// return false if we know the robot motion failed
bool
RosRobotInterface::SendTargetAngles(const std::vector<double> &joint_angles, float secs)
{
  for (int i = 0; i < joint_angles.size(); i++) {
    if (joint_angles[i] > max_joint_angles_[i] || joint_angles[i] < min_joint_angles_[i]) 
    {
      ROS_ERROR_STREAM("The requested joint " << i << " was " << joint_angles[i] << " which is past the joint limits.");
      return false;
    }
  }

  // use a mutex (apparently i should use a unique_lock, but i have no idea why...) to ensure only
  // one command sent at a time 
  control_msgs::FollowJointTrajectoryGoal goal;
  trajectory_msgs::JointTrajectory joint_trajectory;
  std::vector<trajectory_msgs::JointTrajectoryPoint> points;

  std::vector<control_msgs::JointTolerance> tols;
  for (int i = 0; i < joint_angles.size(); i++) {
    control_msgs::JointTolerance tol;
    tol.name = joint_names_[i];
    tol.position = 5;
    tol.velocity = 5;
    tol.acceleration = 5;
    tols.push_back(tol);
  }
  
  trajectory_msgs::JointTrajectoryPoint point;
  point.positions = joint_angles;
  point.time_from_start = ros::Duration(secs);
  points.push_back(point);

  // now, finally, fill out the structure
  joint_trajectory.joint_names = joint_names_;
  joint_trajectory.points = points;
  goal.trajectory = joint_trajectory;
  goal.path_tolerance = tols;
  
  std_msgs::Empty empt;
  ac_->sendGoal(goal);
  return true;
}
