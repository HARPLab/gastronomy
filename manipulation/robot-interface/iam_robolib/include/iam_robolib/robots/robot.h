#ifndef IAM_ROBOLIB_ROBOTS_ROBOT_H_
#define IAM_ROBOLIB_ROBOTS_ROBOT_H_

#include <string>

#include "iam_robolib/definitions.h"

class Robot
{
 public:
  Robot(std::string &robot_ip, RobotType robot_type) : robot_ip_(robot_ip),
                                                       robot_type_(robot_type)
  {

  }

  virtual ~Robot() = default;

  std::string robot_ip_;
  RobotType robot_type_;

};

#endif  // IAM_ROBOLIB_ROBOTS_ROBOT_H_