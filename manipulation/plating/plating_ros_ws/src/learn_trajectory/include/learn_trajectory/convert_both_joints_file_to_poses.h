//
// This class reads in a joint file and writes out a pose file
// UNIX philosophy, I guess?
//
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <vector>

class ConvertJointFileToPoses 
{
  public:
    ConvertJointFileToPoses(std::string filename);
    void WritePoseFile();
    void WriteLineToFile(double time, std::vector<double> &joint_state, std::vector<double> &joint_state_too, std::ofstream &outfile);

  private:
    std::string joints_file_name_;
    Eigen::Affine3d current_pose_;
    robot_state::RobotStatePtr kinematic_state_;
    const robot_state::JointModelGroup* joint_model_group_;
    robot_model::RobotModelPtr kinematic_model_;
    robot_model_loader::RobotModelLoader robot_model_loader_;
    sensor_msgs::JointState joint_state_;
};
