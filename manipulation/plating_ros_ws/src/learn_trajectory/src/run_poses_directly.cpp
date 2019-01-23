//
// This class computes the robot jacobian and can be use to move the robot along a straight-line path
// (in cylindrical coordinates) to the target pose (in cartesian coordinates)
//
#include <ros/ros.h>
#include <ros/package.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>
#include "std_msgs/Empty.h"
#include <mutex>
#include <iostream>
#include <fstream>
#include "feedbot_trajectory_logic/custom_domus_interface.h"

void ExecutePose(std::vector<double> &joint_state, RobotInterface &di)
{
  di.SendTargetAngles(joint_state,0.1);  
}

//read in the joints file and write the file
void ExecuteAllPoses(RobotInterface &di)
{
  //https://stackoverflow.com/questions/14516915/read-numeric-data-from-a-text-file-in-c
  std::ifstream myfile;
  std::string path = ros::package::getPath("learn_trajectory");
  myfile.open(path + "/data/" + "joints_data.txt");
  double next_float, time;
  std::vector<double> joint_state;
  int indx_count = 0;
  while (ros::ok() && myfile >> next_float)
  {
    if (indx_count == 0)
      time = next_float;
    else
      joint_state.push_back(next_float);
    
    if (indx_count == 6)
    {
      ExecutePose(joint_state, di);
      joint_state.clear();
      indx_count = -1;
      ros::Duration(0.1).sleep();
    }
    indx_count++;
    if (myfile.peek() == ',')
        myfile.ignore();
  } 
  myfile.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "convert_joint_file_to_poses");
  ros::NodeHandle n;
  NiryoRobotParams robot_params;
  CustomDomusInterface di(&n, robot_params);
  di.InitializeConnection();
  ExecuteAllPoses(di);
}
