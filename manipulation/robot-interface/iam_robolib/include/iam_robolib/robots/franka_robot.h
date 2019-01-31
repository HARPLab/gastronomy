#ifndef IAM_ROBOLIB_ROBOTS_FRANKA_ROBOT_H_
#define IAM_ROBOLIB_ROBOTS_FRANKA_ROBOT_H_

#include <franka/robot.h>
#include <franka/gripper.h>
#include <franka/model.h>

#include "iam_robolib/robots/robot.h"

class FrankaRobot : public Robot
{
 public:
  FrankaRobot(std::string &robot_ip, RobotType robot_type) : Robot(robot_ip, robot_type),
                                                             robot_(robot_ip),
                                                             gripper_(robot_ip)
  {

  }

  franka::Model getModel()
  {
    return robot_.loadModel();
  }

  franka::RobotState getRobotState()
  {
    return robot_.readOnce();
  }

  franka::GripperState getGripperState()
  {
    return gripper_.readOnce();
  }

  franka::Robot robot_;
  franka::Gripper gripper_;
};

#endif  // IAM_ROBOLIB_ROBOTS_FRANKA_ROBOT_H_