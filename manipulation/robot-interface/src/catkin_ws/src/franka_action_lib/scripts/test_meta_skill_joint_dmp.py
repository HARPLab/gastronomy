#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib
import pickle
import argparse
import numpy as np

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal

from frankapy.skill_list import BaseSkill, JointPoseDMPWithDefaultSensorSkill, ArmRelativeMotionWithDefaultSensorSkill

def feedback_callback(feedback):
    print(feedback)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint DMP Skill Example')
    parser.add_argument('--filename', required=True, help='filename with dmp weights')
    args = parser.parse_args()

    rospy.init_node('joint_dmp_skill_example_client')
    client = actionlib.SimpleActionClient('/execute_skill_action_server_node/execute_skill', ExecuteSkillAction)
    client.wait_for_server()

    file = open(args.filename,"rb")
    dmp_info = pickle.load(file)


    skill = ArmRelativeMotionWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([5.0, 0.0, 0.0, -0.08, 1.0, 0.0, 0.0, 0.0])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print(client.get_result())


    skill = JointPoseDMPWithDefaultSensorSkill()
    skill.add_initial_sensor_values(dmp_info['phi_j'])  # sensor values

    # y0 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    y0 = [-0.282618, -0.18941, 0.0668932, -2.18632, 0.0524845, 1.916, -1.06273]

    print(dmp_info['num_basis'])

    # Run time, tau, alpha, beta, num_basis, num_sensor_values, mu, h, weights
    trajectory_params = [4.0, dmp_info['tau'], dmp_info['alpha'], dmp_info['beta'],\
                         float(dmp_info['num_basis']), float(dmp_info['num_sensors'])] + dmp_info['mu'] + \
                         dmp_info['h'] + y0 + np.array(dmp_info['weights']).reshape(-1).tolist()
    skill.add_trajectory_params(trajectory_params)  
    goal = skill.create_goal()
    print(goal)

    skill.set_meta_skill_id(1)
    skill.set_meta_skill_type(1)   
    skill.add_termination_params([1.0]) 
    goal = skill.create_goal()
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))

    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    skill = ArmRelativeMotionWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([5.0, 0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print(client.get_result())