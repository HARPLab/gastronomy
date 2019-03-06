#include <vector>
#include <string>

class CustomRobotParams
{
  public:
    std::vector<double> min_joint_angles, max_joint_angles, initial_joint_values;
    std::vector<std::string> joint_names;
    std::string srdf_group_name, end_effector_link; 
};

class UR5RobotParams : public CustomRobotParams
{
  public:
    UR5RobotParams()
    {
      std::vector<double> max = { 6, 6, 6, 6, 6, 6 }; 
      std::vector<double> min = { -6, -6, -6, -6, -6, -6};
      std::vector<double> initial_joint_vals = { 1.5, -1.5, 1.5, -1.5, -1.5, 0.0};
      std::vector<std::string> names = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
      max_joint_angles = max;
      min_joint_angles = min;
      initial_joint_values = initial_joint_vals;
      joint_names = names;
      srdf_group_name = "ur5e_arm";
      end_effector_link = "fork_point";
    };
};
