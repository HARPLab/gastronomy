# Franka robot state
std_msgs/Header header
float32[16] pose_desired
float32[16] pose
float32[7] joint_torques
float32[7] joint_torques_derivative
float32[7] joints
float32[7] joints_desired
float32[7] joint_velocities
float32 time_since_skill_started
float32 gripper_width
bool gripper_is_grasped