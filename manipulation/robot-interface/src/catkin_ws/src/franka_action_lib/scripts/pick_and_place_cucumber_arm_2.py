#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal, RobotState

from skill_list import BaseSkill
from skill_list import ArmMoveToGoalWithDefaultSensorSkill, GripperWithDefaultSensorSkill, ArmMoveToGoalContactWithDefaultSensorSkill, StayInPositionWithDefaultSensorSkill

def feedback_callback(feedback):
    print(feedback)

def load_result_into_robot_state_msg(result):
    robot_state = RobotState()

    current_result_index = 0

    for i in range(16):
        robot_state.O_T_EE[i] = result.execution_result[current_result_index]
        current_result_index += 1

    for i in range(7):
        robot_state.tau_J[i] = result.execution_result[current_result_index]
        current_result_index += 1

    for i in range(7):
        robot_state.dtau_J[i] = result.execution_result[current_result_index]
        current_result_index += 1

    for i in range(7):
        robot_state.q[i] = result.execution_result[current_result_index]
        current_result_index += 1

    for i in range(7):
        robot_state.dq[i] = result.execution_result[current_result_index]
        current_result_index += 1

    return robot_state


if __name__ == '__main__':
    rospy.init_node('example_execute_skill_action_client')
    client = actionlib.SimpleActionClient('/execute_skill_action_server_node/execute_skill', ExecuteSkillAction)
    client.wait_for_server()
    pub = rospy.Publisher('Arm_2_robot_state', RobotState, queue_size=10)
    

    print ('===== ')
    print("Opening the gripper to prepare for grasping.")

    # Open the gripper
    skill = GripperWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([0.07, 0.025, 1100])  # Gripper Width, Gripper Speed, Wait Time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    print ('===== ')
    print("Moving to the cucumber position.")

    # Move to Picking Position
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    #skill.add_trajectory_params([3.0, 0.997967,0.0553182,0.0313609,0,0.0550471,-0.998429,0.0094417,0,0.0318345,-0.00769632,-0.999464,0,0.603236,0.150695,0.00406402,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_trajectory_params([3.0, 0.821749,0.00614164,-0.569799,0,0.00931894,-0.999943,0.00266152,0,-0.569762,-0.00749717,-0.821776,0,0.639448,0.138796,0.00735826,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    robot_state = load_result_into_robot_state_msg(client.get_result())
    pub.publish(robot_state)

    print ('===== ')
    print("Closing the gripper and grasping.")

    # Close the gripper and grasp
    skill = GripperWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([0.05, 0.025, 80, 1100]) # Gripper Width, Gripper Speed, Grasping Force, Wait Time
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
    #skill.add_trajectory_params([3.0, 0.999077,0.03154,-0.0288332,0,0.0324637,-0.998946,0.0321529,0,-0.0277892,-0.0330599,-0.999067,0,0.595558,0.0450467,0.3142,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_trajectory_params([3.0, 0.886204,0.0441216,-0.461168,0,0.0171246,-0.997884,-0.0625638,0,-0.462962,0.047548,-0.885102,0,0.677055,0.125854,0.195946,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    robot_state = load_result_into_robot_state_msg(client.get_result())
    pub.publish(robot_state)


    print ('===== ')
    print("Moving to intermediate position.")

    # Move to Intermediate Position
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    #skill.add_trajectory_params([3.0, 0.999077,0.03154,-0.0288332,0,0.0324637,-0.998946,0.0321529,0,-0.0277892,-0.0330599,-0.999067,0,0.595558,0.0450467,0.3142,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_trajectory_params([3.0, 0.0174699,0.816579,-0.576953,0,0.98671,-0.107281,-0.121961,0,-0.161491,-0.567166,-0.807616,0,0.651099,-0.336885,0.161649,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    robot_state = load_result_into_robot_state_msg(client.get_result())
    pub.publish(robot_state)

    print ('===== ')
    print("Moving to contact goal position")

    # Move to contact goal position
    skill = ArmMoveToGoalContactWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    #skill.add_trajectory_params([3.0, 0.00189348,-0.999791,-0.0198586,0,-0.999966,-0.0020278,0.00674575,0,-0.00678474,0.0198455,-0.99978,0,0.622919,-0.405382,0.0198905,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_trajectory_params([3.0, -0.0052017,0.837409,-0.546534,0,0.998476,-0.0255863,-0.0487069,0,-0.0547725,-0.545965,-0.836016,0,0.684861,-0.419534,0.0362174,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    robot_state = load_result_into_robot_state_msg(client.get_result())
    pub.publish(robot_state)

    print ('===== ')
    print("Stay in Position")

    # Stay in the position for a certain amount of time
    skill = StayInPositionWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([300.0])  # Run Time 
    skill.add_feedback_controller_params([800, 50]) # translational stiffness, rotational stiffness
    goal = skill.create_goal()

    print(goal)
    
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(1.0))

    while not rospy.is_shutdown() and done != True:
        done = client.wait_for_result(rospy.Duration.from_sec(1.0))
        pub.publish(robot_state)

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

    print ('===== ')
    print("Moving to the original position.")

    # Move to original position
    skill = ArmMoveToGoalWithDefaultSensorSkill()
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params([5.0, 0.99747,0.0476472,-0.0525668,0,0.0488561,-0.998555,0.0219546,0,-0.0514457,-0.0244678,-0.998376,0,0.434247,-0.0252676,0.333927,1])  # Run Time (1) and Desired End Effector Pose(16)
    skill.add_feedback_controller_params([600, 50]) # translational stiffness, rotational stiffness
    skill.add_termination_params([1.0]) # buffer time
    goal = skill.create_goal()
    print(goal)
    client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
    done = client.wait_for_result(rospy.Duration.from_sec(5.0))

    robot_state = load_result_into_robot_state_msg(client.get_result())
    pub.publish(robot_state)
