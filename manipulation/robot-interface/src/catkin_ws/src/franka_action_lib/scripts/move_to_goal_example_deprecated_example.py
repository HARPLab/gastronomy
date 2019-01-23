#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal

def feedback_callback(feedback):
    print(feedback)

if __name__ == '__main__':
    rospy.init_node('example_execute_skill_action_client')
    client = actionlib.SimpleActionClient('/execute_skill_action_server_node/execute_skill', ExecuteSkillAction)
    client.wait_for_server()

    goal = ExecuteSkillGoal()
    # Fill in the goal here
    initial_sensor_values = [1,3,5,7,9]
    traj_gen_params = [0,0,0,0,0,0,0]
    # traj_gen_params = [0.997146,0.0727788,0.0196026,0,0.0721817,-0.996942,0.0296162,0,0.0216985,-0.0281173,-0.999369,0,0.526579,0.0200579,0.252545,1]
    # traj_gen_params = [5.0, 0.997146,0.0727788,0.0196026,0,0.0721817,-0.996942,0.0296162,0,0.0216985,-0.0281173,-0.999369,0,0.526579,0.0600579,0.252545,1]
    traj_gen_params = [5.0, 0.99562,0.0868484,0.0343491,0,0.0867175,-0.996209,0.00528489,0,0.0346785,-0.00228312,-0.999396,0,0.52984,0.0645623,0.271222,1]
    feedback_controller_params = [600, 50];
    termination_params = [1.0];
    
    timer_params = [1,2,3,4,5];

    goal.skill_type = 0
    goal.sensor_topics = ["/franka_robot/camera/"]
    goal.sensor_value_sizes = [len(initial_sensor_values)]
    goal.initial_sensor_values = initial_sensor_values
    goal.traj_gen_type = 4
    goal.num_traj_gen_params = len(traj_gen_params)
    goal.traj_gen_params = traj_gen_params
    goal.feedback_controller_type = 2
    goal.num_feedback_controller_params = len(feedback_controller_params)
    goal.feedback_controller_params = feedback_controller_params
    goal.termination_type = 5
    goal.num_termination_params = len(termination_params)
    goal.termination_params = termination_params
    goal.timer_type = 4
    goal.num_timer_params = len(timer_params)
    goal.timer_params = timer_params

    client.send_goal(goal, feedback_cb=feedback_callback)
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))