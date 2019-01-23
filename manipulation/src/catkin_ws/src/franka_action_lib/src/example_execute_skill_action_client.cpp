#include <vector>
#include <ros/ros.h>
#include <franka_action_lib/ExecuteSkillAction.h> // Note: "Action" is appended
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

typedef actionlib::SimpleActionClient<franka_action_lib::ExecuteSkillAction> Client;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "example_execute_skill_action_client");
  Client client("/execute_skill_action_server_node/execute_skill", true); // true -> don't need ros::spin()
  client.waitForServer();
  franka_action_lib::ExecuteSkillGoal goal;

  // Fill in goal here
  std::vector<float> initial_sensor_values{ 1,3,5,7,9 };
  std::vector<float> traj_gen_params{ 1,1,1,3,3,3 };
  std::vector<float> feedback_controller_params{ 0.1 };
  std::vector<float> termination_params{ 3,3,3 };
  std::vector<float> timer_params{ 1,2,3,4,5 };

  goal.sensor_topics.push_back("/franka_robot/camera/");
  goal.sensor_value_sizes.push_back(initial_sensor_values.size());
  goal.initial_sensor_values = initial_sensor_values;
  goal.traj_gen_type = 1;
  goal.num_traj_gen_params = traj_gen_params.size();
  goal.traj_gen_params = traj_gen_params;
  goal.feedback_controller_type = 1;
  goal.num_feedback_controller_params = feedback_controller_params.size();
  goal.feedback_controller_params = feedback_controller_params;
  goal.termination_type = 1;
  goal.num_termination_params = termination_params.size();
  goal.termination_params = termination_params;
  goal.timer_type = 1;
  goal.num_timer_params = timer_params.size();
  goal.timer_params = timer_params;

  client.sendGoal(goal);
  client.waitForResult(ros::Duration(5.0));

  actionlib::SimpleClientGoalState action_server_state = client.getState();

  while(action_server_state != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    printf("Current State: %s\n", client.getState().toString().c_str());
    client.waitForResult(ros::Duration(5.0));
    action_server_state = client.getState();
  }
  
  printf("Trajectory has been completed.");
  
  return 0;
}