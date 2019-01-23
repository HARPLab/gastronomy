//
// This class connects to DOMUS 
// and can be used to send target angles for DOMUS to move to
//
#ifndef JOINT_ECHOING_INTERFACE_H_
#define JOINT_ECHOING_INTERFACE_H_
#include "feedbot_trajectory_logic/robot_interface.h"
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

class JointEchoingInterface : public RobotInterface
{
  public:
    JointEchoingInterface(ros::NodeHandle* n, CustomRobotParams robot_params);
    virtual void InitializeConnection();
    virtual bool SendTargetAngles(const std::vector<double> &joint_angles, float secs);
  protected:
    void PublishRobotState(const std::vector<double> &joint_value);
  private:
    sensor_msgs::JointState joint_state_;
    ros::Publisher joint_pub_;
};
#endif
