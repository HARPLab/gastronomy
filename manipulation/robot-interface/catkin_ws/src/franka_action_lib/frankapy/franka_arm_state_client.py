import sys, logging
from time import sleep

import numpy as np
import rospy
from franka_action_lib.srv import GetCurrentRobotStateCmd

from .utils import franka_pose_to_rigid_transform

class FrankaArmStateClient:

    def __init__(self, new_ros_node=True, robot_state_server_name='/get_current_robot_state_server_node/get_current_robot_state_server'):
        if new_ros_node:
            rospy.init_node('FrankaArmStateClient', anonymous=True)

        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)

    def get_data(self):
        '''Get all fields of current robot data in a dict.

        Returns:
            dict of robot state
        '''
        ros_data = self._get_current_robot_state().robot_state

        data = {
            'pose': franka_pose_to_rigid_transform(ros_data.O_T_EE),
            'joint_torques': np.array(ros_data.tau_J),
            'joint_torques_derivative': np.array(ros_data.dtau_J),
            'joints': np.array(ros_data.q),
            'joints_desired': np.array(ros_data.q_d),
            'joint_velocities': np.array(ros_data.dq),
            'gripper_width': ros_data.gripper_width,
            'gripper_is_grasped': ros_data.gripper_is_grasped       
        }

        return data

    def get_pose(self):
        '''Get the current pose.

        Returns:
            RigidTransform
        '''
        return self.get_data()['pose']

    def get_joints(self):
        '''Get the current joint configuration.

        Returns:
            ndarray of shape (7,)
        '''
        return self.get_data()['joints']

    def get_joint_torques(self):
        '''Get the current joint torques.

        Returns:
            ndarray of shape (7,)
        '''
        return self.get_data()['joint_torques']

    def get_joint_velocities(self):
        '''Get the current joint velocities.

        Returns:
            ndarray of shape (7,)
        '''
        return self.get_data()['joint_velocities']

    def get_gripper_width(self):
        '''Get most recent gripper width. Note this value will *not* be updated
        during a control command.

        Returns:
            float of gripper width in meters
        '''
        return self.get_data()['gripper_width']

    def get_gripper_is_grasped(self):
        '''Returns whether or not the gripper is grasping something. Note this
        value will *not* be updated during a control command.

        Returns:
            True if gripper is grasping something. False otherwise
        '''
        return self.get_data()['gripper_is_grasped']
