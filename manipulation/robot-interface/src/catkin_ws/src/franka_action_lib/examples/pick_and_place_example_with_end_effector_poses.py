#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal

from skill_list import BaseSkill
from skill_list import ArmMoveToGoalWithDefaultSensorSkill, GripperWithDefaultSensorSkill, ArmMoveToGoalContactWithDefaultSensorSkill, StayInPositionWithDefaultSensorSkill

def feedback_callback(feedback):
    print(feedback)

if __name__ == '__main__':
    rospy.init_node('example_execute_skill_action_client')
    client = actionlib.SimpleActionClient('/execute_skill_action_server_node/execute_skill', ExecuteSkillAction)
    client.wait_for_server()

    # Open the gripper
    skill = GripperWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([0.05, 0.025, 1100])  # Gripper Width, Gripper Speed, Wait Time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    # Move to Picking Position
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, -0.00146913,0.999948,-0.00909845,0,0.999977,0.00142362,-0.00500628,0,-0.00499316,-0.00910576,-0.999946,0,0.652849,0.0717463,0.0183853,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print ('===== ')
    print("Closing the gripper and grasping.")

    # Close the gripper and grasp
    skill = GripperWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([0.075, 0.025, 5, 1100]) # Gripper Width, Gripper Speed, Grasping Force, Wait Time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))


    print ('===== ')
    print("Moving to intermediate position.")

    # Move to Intermediate Position
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, 0.00678993,0.999959,0.00403557,0,0.999118,-0.00661782,-0.0412309,0,-0.0412033,0.00431205,-0.999141,0,0.619531,-0.0983122,0.247968,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print ('===== ')
    print("Moving to contact goal position")

    # Move to contact goal position
    skill = ArmMoveToGoalContactWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0, 0.010964,0.99987,0.0110068,0,0.999883,-0.0108553,-0.00988176,0,-0.00976118,0.011114,-0.999891,0,0.627216,-0.0954885,0.0158609,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print ('===== ')
    print("Stay in Position")

    # Stay in the position for a certain amount of time
    skill = StayInPositionWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([3.0])  # Run Time 
    goal = skill.create_goal()

    print(goal)
    
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print ('===== ')
    print("Open the Gripper")

    # Open the gripper
    skill = GripperWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([0.05, 0.025, 1100])  # Gripper Width, Gripper Speed, Wait Time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    # Move to original position
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([5.0, -0.0268194,0.997366,-0.0672444,0,0.998971,0.0242969,-0.0380534,0,-0.0363201,-0.068197,-0.997011,0,0.551823,-0.0419016,0.265313,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))
