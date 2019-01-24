//
// This class computes the robot jacobian and can be use to move the robot along a straight-line path
// (in cylindrical coordinates) to the target pose (in cartesian coordinates)
//

#include <learn_trajectory/convert_both_joints_file_to_poses.h>

//constructor
ConvertJointFileToPoses::ConvertJointFileToPoses(std::string filename)
  : robot_model_loader_("robot_description")
{
  kinematic_model_ = robot_model_loader_.getModel();
  kinematic_state_ = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model_));
  kinematic_state_->setToDefaultValues();
  joint_model_group_ = kinematic_model_->getJointModelGroup("arm");

  std::vector<double> initial_joint_values { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  kinematic_state_->setJointGroupPositions(joint_model_group_, initial_joint_values);  

  current_pose_ = kinematic_state_->getGlobalLinkTransform("spoon_link");
  joints_file_name_ = filename;
}

void ConvertJointFileToPoses::WriteLineToFile(double time, std::vector<double> &joint_state, std::vector<double> &joint_state_too, std::ofstream &outfile)
{
  kinematic_state_->setJointGroupPositions(joint_model_group_, joint_state);  
  current_pose_ = kinematic_state_->getGlobalLinkTransform("spoon_link");
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
  kinematic_state_->setJointGroupPositions(joint_model_group_, joint_state_too);  
  current_pose_ = kinematic_state_->getGlobalLinkTransform("spoon_link");
  for (int i = 0; i < 3; i++)
  {
    outfile << ", " << current_pose_(i,3);
  }

  Eigen::Quaterniond quats2(current_pose_.rotation());
  outfile << ", " << quats2.x();
  outfile << ", " << quats2.y();
  outfile << ", " << quats2.z();
  outfile << ", " << quats2.w();

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
  outfile.open(path + "/data/spoon_poses_both.txt");
  double next_float, time;
  std::vector<double> joint_state;
  std::vector<double> joint_state_too;
  int indx_count = 0;
  while (myfile >> next_float)
  {
    if (indx_count == 0)
      time = next_float;
    else if (indx_count <= 6)
      joint_state.push_back(next_float);
    else 
      joint_state_too.push_back(next_float);
    
    if (indx_count == 12)
    {
      WriteLineToFile(time, joint_state, joint_state_too, outfile);
      joint_state.clear();
      joint_state_too.clear();
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
  ConvertJointFileToPoses converter("joints_data_both.txt");
  converter.WritePoseFile();
}
