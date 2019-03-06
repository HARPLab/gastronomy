//
// This class computes the robot jacobian and can be use to move the robot along a straight-line path
// (in cylindrical coordinates) to the target pose (in cartesian coordinates)
//

#include <learn_trajectory/convert_joints_file_to_poses.h>

//constructor
ConvertJointFileToPoses::ConvertJointFileToPoses(std::string filename, std::string moveit_group, std::string end_effector_link)
  : robot_model_loader_("robot_description")
{
  kinematic_model_ = robot_model_loader_.getModel();
  kinematic_state_ = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model_));
  kinematic_state_->setToDefaultValues();
  joint_model_group_ = kinematic_model_->getJointModelGroup(moveit_group);

  std::vector<double> initial_joint_values { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  kinematic_state_->setJointGroupPositions(joint_model_group_, initial_joint_values);  
  
  end_effector_link_ = end_effector_link;
  current_pose_ = kinematic_state_->getGlobalLinkTransform(end_effector_link_);
  joints_file_name_ = filename;
}

void ConvertJointFileToPoses::WriteLineToFile(double time, std::vector<double> &joint_state, std::ofstream &outfile)
{
  kinematic_state_->setJointGroupPositions(joint_model_group_, joint_state);  
  current_pose_ = kinematic_state_->getGlobalLinkTransform(end_effector_link_);
  outfile << std::setprecision (15) << time;
  
  for (int i = 0; i < 3; i++)
  {
    outfile << ", " << current_pose_(i,3);
  }

  Eigen::Quaterniond quats(current_pose_.rotation());
  outfile << ", " << quats.x();
  outfile << ", " << quats.y();
  outfile << ", " << quats.z();
  outfile << ", " << quats.w();
  outfile << "\n";
}

//read in the joints file and write the file
void ConvertJointFileToPoses::WritePoseFile()
{
  //https://stackoverflow.com/questions/14516915/read-numeric-data-from-a-text-file-in-c
  std::ifstream myfile;
  std::ofstream outfile;
  std::string path = ros::package::getPath("learn_trajectory");
  myfile.open(path + "/data/" + joints_file_name_);
  outfile.open(path + "/data/pose_data.txt");
  double next_float, time;
  std::vector<double> joint_state;
  int indx_count = 0;
  while (myfile >> next_float)
  {
    if (indx_count == 0)
      time = next_float;
    else
      joint_state.push_back(next_float);
    
    if (indx_count == 6)
    {
      WriteLineToFile(time, joint_state, outfile);
      joint_state.clear();
      indx_count = -1;
    }
    indx_count++;
    if (myfile.peek() == ',')
        myfile.ignore();
  } 
  myfile.close();
  outfile.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "convert_joint_file_to_poses");
  ros::NodeHandle nh;
  std::string moveit_group, end_effector_link;
  ros::param::get("~moveit_group", moveit_group);
  ros::param::get("~end_effector_link", end_effector_link);
  ros::Duration(1).sleep();
  ConvertJointFileToPoses converter("joints_data.txt", moveit_group, end_effector_link);
  converter.WritePoseFile();
}
