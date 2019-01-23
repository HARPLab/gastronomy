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
    initial_sensor_values = [1,3,5,7,9];
    traj_gen_params = [0,0,0,0,0,0,0,0,0,0,0,0,0,-0.001,0,0];
    feedback_controller_params = [0.1];
    termination_params = [0,0,0,0,0,0,0,0,0,0,0,0,0,0.3,0,0];
    # termination_params = [0.981755,-0.00891852,0.189888,0,-0.0073292,-0.999922,-0.00907029,0,0.189958,0.00751322,-0.981763,0,0.474874,0.0381648,0.394827,1]
    termination_params = [0.981755,-0.00891852,0.189888,0,-0.0073292,-0.999922,-0.00907029,0,0.189958,0.00751322,-0.981763,0,0.474874,-0.0381648,0.394827,1]
    
    timer_params = [1,2,3,4,5];

    goal.skill_type = 0
    goal.sensor_topics = ["/franka_robot/camera/"]
    goal.sensor_value_sizes = [len(initial_sensor_values)]
    goal.initial_sensor_values = initial_sensor_values
    goal.traj_gen_type = 2
    goal.num_traj_gen_params = len(traj_gen_params)
    goal.traj_gen_params = traj_gen_params
    goal.feedback_controller_type = 1
    goal.num_feedback_controller_params = len(feedback_controller_params)
    goal.feedback_controller_params = feedback_controller_params
    goal.termination_type = 2
    goal.num_termination_params = len(termination_params)
    goal.termination_params = termination_params
    goal.timer_type = 2
    goal.num_timer_params = len(timer_params)
    goal.timer_params = timer_params

    client.send_goal(goal, feedback_cb=feedback_callback)
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))
