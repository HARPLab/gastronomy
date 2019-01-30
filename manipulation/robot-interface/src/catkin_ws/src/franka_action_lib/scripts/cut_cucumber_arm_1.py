#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib
import pickle
import argparse
import numpy as np

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal

from frankapy.skill_list import *

def get_move_left_skill(distance_in_m):
    skill = ArmRelativeMotionWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, 0.0, -distance_in_m, 0.0, 1.0, 0.0, 0.0, 0.0])
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    return skill

def create_skill_to_move_to_cucumber(cutting_knife_location_x, last_cutting_knife_location_y):
     # Move left to contact cucumber
    skill = ArmMoveToGoalContactWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random  
    skill.add_trajectory_params(
        [3.0, 0.00193678,0.999977,0.00475145,0,0.9999,-0.00199989,0.0133132,0,
        0.0133227,0.00472528,-0.9999,0,cutting_knife_location_x,last_cutting_knife_location_y,0.024956,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness

    lower_torque_thresholds_acceleration = [10.0,10.0,10.0,10.0,10.0,10.0,10.0]
    upper_torque_thresholds_acceleration = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]
    lower_torque_thresholds_nominal = [10.0,10.0,10.0,10.0,10.0,10.0,10.0]
    upper_torque_thresholds_nominal = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]
    lower_force_thresholds_acceleration = [10.0,3.0,10.0,10.0,10.0,10.0]
    upper_force_thresholds_acceleration = [120.0,120.0,120.0,125.0,125.0,125.0]
    lower_force_thresholds_nominal = [10.0,3.0,10.0,10.0,10.0,10.0]
    upper_force_thresholds_nominal = [120.0,120.0,120.0,125.0,125.0,125.0]  

    collision_termination_params = lower_torque_thresholds_acceleration \
        + upper_torque_thresholds_acceleration \
        + lower_torque_thresholds_nominal \
        + upper_torque_thresholds_nominal \
        + lower_force_thresholds_acceleration \
        + upper_force_thresholds_acceleration \
        + lower_force_thresholds_nominal \
        + upper_force_thresholds_nominal

    skill.add_termination_params([1.0] + collision_termination_params) # buffer time
    return skill

def feedback_callback(feedback):
    print(feedback)

if __name__ == '__main__':
    rospy.init_node('example_execute_skill_action_client')
    client = actionlib.SimpleActionClient('/execute_skill_action_server_node/execute_skill', ExecuteSkillAction)
    client.wait_for_server()

    parser = argparse.ArgumentParser(description='Joint DMP Skill Example')
    parser.add_argument('--filename', required=True, help='filename with dmp weights')
    args = parser.parse_args()

    file = open(args.filename,"rb")
    dmp_info = pickle.load(file)
    cutting_knife_location_x = 0.53
    last_cutting_knife_location_y = 0.07
    initial_cutting_knife_location_y = -0.176568


    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, -0.0372113,0.999006,0.0241583,0,0.998733,0.0379922,-0.0327119,0,-0.0335979,0.0229109,-0.999173,0,cutting_knife_location_x,-0.243681,0.245551,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print(client.get_result())

    # Move to designated position above the cutting board
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, 0.00428579,0.999112,-0.0416815,0,0.999585,-0.00545372,-0.0279469,0,-0.02815,-0.0415452,-0.99874,0,cutting_knife_location_x,initial_cutting_knife_location_y,0.141577,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print(client.get_result())

    # Move down to contact cutting board
    skill = ArmMoveToGoalContactWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, 0.00338211,0.99924,-0.0385855,0,0.999849,-0.00401365,-0.0163014,0,-0.0164442,-0.0385253,-0.999122,0,cutting_knife_location_x,initial_cutting_knife_location_y,0.0293225,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    # Move left to contact cucumber
    skill = ArmMoveToGoalContactWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random  
    skill.add_trajectory_params([3.0, 0.00193678,0.999977,0.00475145,0,0.9999,-0.00199989,0.0133132,0,0.0133227,0.00472528,-0.9999,0,cutting_knife_location_x,last_cutting_knife_location_y,0.024956,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness

    lower_torque_thresholds_acceleration = [10.0,10.0,10.0,10.0,10.0,10.0,10.0]
    upper_torque_thresholds_acceleration = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]
    lower_torque_thresholds_nominal = [10.0,10.0,10.0,10.0,10.0,10.0,10.0]
    upper_torque_thresholds_nominal = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]
    lower_force_thresholds_acceleration = [10.0,3.0,10.0,10.0,10.0,10.0]
    upper_force_thresholds_acceleration = [120.0,120.0,120.0,125.0,125.0,125.0]
    lower_force_thresholds_nominal = [10.0,3.0,10.0,10.0,10.0,10.0]
    upper_force_thresholds_nominal = [120.0,120.0,120.0,125.0,125.0,125.0] 

    collision_termination_params = lower_torque_thresholds_acceleration + upper_torque_thresholds_acceleration + lower_torque_thresholds_nominal + \
                                   upper_torque_thresholds_nominal + lower_force_thresholds_acceleration + upper_force_thresholds_acceleration + \
                                   lower_force_thresholds_nominal + upper_force_thresholds_nominal

    skill.add_termination_params([1.0] + collision_termination_params) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    for blahblaj in range(3):
        # Move up above the cucumber
        skill = ArmRelativeMotionWithDefaultSensorSkill()
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
        skill.add_trajectory_params([3.0, 0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0])  # Run Time (1) and Desired End Effector Pose(16)
        skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
        skill.add_termination_params([1.0]) # buffer time
        goal = skill.create_goal()
        print(goal)
        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        while not rospy.is_shutdown() and done != True:
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        print(client.get_result())

        # Move left above the cucumber
        skill = ArmRelativeMotionWithDefaultSensorSkill()
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
        skill.add_trajectory_params([3.0, 0.0, 0.0075, 0.0, 1.0, 0.0, 0.0, 0.0])  # Run Time (1) and Desired End Effector Pose(16)
        skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
        skill.add_termination_params([1.0]) # buffer time
        goal = skill.create_goal()
        print(goal)
        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        while not rospy.is_shutdown() and done != True:
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        print(client.get_result())

        lower_torque_thresholds_acceleration = [20.0,20.0,20.0,20.0,20.0,20.0,20.0]
        upper_torque_thresholds_acceleration = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]
        lower_torque_thresholds_nominal = [20.0,20.0,20.0,20.0,20.0,20.0,20.0]
        upper_torque_thresholds_nominal = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]
        lower_force_thresholds_acceleration = [20.0,20.0,18.0,20.0,20.0,20.0]
        upper_force_thresholds_acceleration = [120.0,120.0,120.0,125.0,125.0,125.0]
        lower_force_thresholds_nominal = [20.0,20.0,18.0,20.0,20.0,20.0]
        upper_force_thresholds_nominal = [120.0,120.0,120.0,125.0,125.0,125.0] 

        collision_termination_params = lower_torque_thresholds_acceleration + upper_torque_thresholds_acceleration + lower_torque_thresholds_nominal + \
                                       upper_torque_thresholds_nominal + lower_force_thresholds_acceleration + upper_force_thresholds_acceleration + \
                                       lower_force_thresholds_nominal + upper_force_thresholds_nominal

        # Move to contact
        skill = ArmRelativeMotionToContactWithDefaultSensorSkill()
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
        skill.add_trajectory_params([3.0, 0.0, 0.0, -0.08, 1.0, 0.0, 0.0, 0.0])  # Run Time (1) and Desired End Effector Pose(16)
        skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
        skill.add_termination_params([1.0] + collision_termination_params) # buffer time
        goal = skill.create_goal()
        print(goal)
        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        while not rospy.is_shutdown() and done != True:
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        # Start DMP cutting for 3 times
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
        skill.set_meta_skill_id(blahblaj+1)
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

        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))
        while not rospy.is_shutdown() and done != True:
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        skill = get_move_left_skill(0.06)
        goal = skill.create_goal()
        print(goal)
        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        while not rospy.is_shutdown() and done != True:
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        print(client.get_result())         

        skill = create_skill_to_move_to_cucumber(cutting_knife_location_x, last_cutting_knife_location_y)
        goal = skill.create_goal()
        print(goal)
        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

        while not rospy.is_shutdown() and done != True:
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))
