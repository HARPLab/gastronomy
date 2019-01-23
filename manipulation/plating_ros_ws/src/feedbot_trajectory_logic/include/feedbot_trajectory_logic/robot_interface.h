//
// This class connects to DOMUS 
// and can be used to send target angles for DOMUS to move to
//
#ifndef ROBOT_INTERFACE_H_
#define ROBOT_INTERFACE_H_
#include "feedbot_trajectory_logic/custom_robot_params.h"
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

class RobotInterface
{
  public:
    RobotInterface(CustomRobotParams robot_params);
    virtual void InitializeConnection();
    virtual bool SendTargetAngles(const std::vector<double> &joint_angles, float secs);
    std::vector<double> max_joint_angles_, min_joint_angles_, initial_joint_values_;
    std::vector<std::string> joint_names_;
    std::string srdf_group_name_, end_effector_link_;
};
#endif
