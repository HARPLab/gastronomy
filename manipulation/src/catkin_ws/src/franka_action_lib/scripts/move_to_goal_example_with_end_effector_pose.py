#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal

from skill_list import BaseSkill
from skill_list import ArmMoveToGoalWithDefaultSensorSkill

def feedback_callback(feedback):
    print(feedback)

if __name__ == '__main__':
    rospy.init_node('example_execute_skill_action_client')
    client = actionlib.SimpleActionClient('/execute_skill_action_server_node/execute_skill', ExecuteSkillAction)
    client.wait_for_server()

    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, 0.998235,0.0375818,0.0457788,0,0.0378176,-0.999266,-0.00429513,0,0.0455846,0.00601891,-0.998942,0,0.531054,0.0154956,0.301051,1]
)  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print(client.get_result())
