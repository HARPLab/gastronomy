//
// This ROS node listens on /set_joint_angles and will move DOMUS to the requested position in joint angles
//

#include <ros/ros.h>
#include <feedbot_trajectory_logic/ros_robot_interface.h>
#include <feedbot_trajectory_logic/JointAngles.h>

RobotInterface *domus;

void set_angles_callback(const feedbot_trajectory_logic::JointAngles::ConstPtr& msg)
{
  ROS_INFO_STREAM("I heard" << msg->joint_angles[0] << msg->joint_angles[1] <<"and more");
  //https://stackoverflow.com/questions/6399090/c-convert-vectorint-to-vectordouble
  std::vector<double> joint_angle_vector(msg->joint_angles.begin(), msg->joint_angles.end());
  domus->SendTargetAngles(joint_angle_vector,3);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "domus_controller");
  ros::NodeHandle n;
  NiryoRobotParams robot_params;
  RosRobotInterface niryo_robot("niryo_one_follow_joint_trajectory_controller/follow_joint_trajectory", robot_params);
  domus = &niryo_robot;
  domus->InitializeConnection();
  ros::Subscriber sub = n.subscribe("set_joint_angles", 10, set_angles_callback);
  ros::spin();
  return 0; 
}
