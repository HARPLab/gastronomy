//
// This class connects to DOMUS 
// and can be used to send target angles for DOMUS to move to
//
#include "feedbot_trajectory_logic/joint_echoing_interface.h"

JointEchoingInterface::JointEchoingInterface(ros::NodeHandle* n, CustomRobotParams robot_params) : RobotInterface(robot_params)
{
  // set up joint publishing
  joint_pub_ = n->advertise<sensor_msgs::JointState>("joint_states", 1);
}

void
JointEchoingInterface::InitializeConnection()
{
}

// move to the target joint_angles and the motion should take you secs seconds.
// return false if we know the robot motion failed
bool
JointEchoingInterface::SendTargetAngles(const std::vector<double> &joint_angles, float secs)
{
  PublishRobotState(joint_angles);
  return true;
}

void
JointEchoingInterface::PublishRobotState(const std::vector<double> &joint_values)
{
  const std::vector<std::string> &joint_names = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};
  joint_state_.header.stamp = ros::Time::now();
  joint_state_.name.resize(joint_values.size());
  joint_state_.position.resize(joint_values.size());
  for (int i = 0; i < joint_values.size(); i++) {
    joint_state_.name[i] = joint_names[i];
    joint_state_.position[i] = joint_values[i];
  }
  joint_pub_.publish(joint_state_);
  return;
}
