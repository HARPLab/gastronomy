//
// This class connects to DOMUS 
// and can be used to send target angles for DOMUS to move to
//
#ifndef CUSTOM_DOMUS_INTERFACE_H_
#define CUSTOM_DOMUS_INTERFACE_H_

#include "feedbot_trajectory_logic/joint_echoing_interface.h"
#include <ros/ros.h>
#include <serial/serial.h>
#include <math.h>
#include <sensor_msgs/JointState.h>

class CustomDomusInterface : public JointEchoingInterface
{
  public:
    CustomDomusInterface(ros::NodeHandle* n, CustomRobotParams robot_params);
    virtual void InitializeConnection();
    virtual bool SendTargetAngles(const std::vector<double> &joint_angles, float secs);
  private:
    serial::Serial ser;
};
#endif
