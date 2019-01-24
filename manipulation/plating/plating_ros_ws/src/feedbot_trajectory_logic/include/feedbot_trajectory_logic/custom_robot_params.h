#include <vector>
#include <string>

class CustomRobotParams
{
  public:
    std::vector<double> min_joint_angles, max_joint_angles, initial_joint_values;
    std::vector<std::string> joint_names;
    std::string srdf_group_name, end_effector_link; 
};

class NiryoRobotParams : public CustomRobotParams
{
  public:
    NiryoRobotParams()
    {
      std::vector<double> max = { 2.7, 0 + 0.64, 2.1 - 1.35, 2.7, 2.4, 2.5 };
      std::vector<double> min = { -2.7, -2.3 + 0.64, 0 - 1.35, -2.7, -2.4, -2.5 };
      std::vector<double> initial_joint_vals = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      std::vector<std::string> names = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};
      max_joint_angles = max;
      min_joint_angles = min;
      initial_joint_values = initial_joint_vals;
      joint_names = names;
      srdf_group_name = "arm";
      end_effector_link = "spoon_link";
    };
};

class UR5RobotParams : public CustomRobotParams
{
  public:
    UR5RobotParams()
    {
      std::vector<double> max = { 3, 3, 3, 3, 3, 3 }; 
      std::vector<double> min = { -3, -3, -3, -3, -3, -3};
      std::vector<double> initial_joint_vals = { 0.0, -1.5, 1.5, 0.0, 0.0, 0.0};
      std::vector<std::string> names = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
      max_joint_angles = max;
      min_joint_angles = min;
      initial_joint_values = initial_joint_vals;
      joint_names = names;
      srdf_group_name = "manipulator";
      end_effector_link = "ee_link";
    };
};
