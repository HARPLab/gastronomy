import sys, logging
from multiprocessing import Queue
try:
    from Queue import Empty
except:
    from queue import Empty
from time import sleep

import numpy as np
import rospy
from franka_action_lib.msg import RobotState

from .franka_constants import FrankaConstants as FC
from .utils import franka_pose_to_rigid_transform

class FrankaArmSubscriber:

    def __init__(self, new_ros_node=True, topic_name='/robot_state_publisher_node/robot_state'):
        self._data_q = Queue(maxsize=1)

        def callback(data):
            if self._data_q.full():
                try:
                    self._data_q.get_nowait()
                except Empty:
                    pass
            self._data_q.put(data)

        if new_ros_node:
            rospy.init_node('FrankaArmSubscriber', anonymous=True)

        rospy.Subscriber(topic_name, RobotState, callback)

    def get_data(self):
        '''Get all fields of current robot data in a dict.

        Returns:
            dict of robot state
        '''
        ros_data = self._data_q.get(block=True)

        data = {
            'pose_desired': franka_pose_to_rigid_transform(ros_data.pose_desired),
            'pose': franka_pose_to_rigid_transform(ros_data.pose),
            'joint_torques': np.array(ros_data.joint_torques),
            'joint_torques_derivative': np.array(ros_data.joint_torques_derivative),
            'joints': np.array(ros_data.joints),
            'joints_desired': np.array(ros_data.joints_desired),
            'joint_velocities': np.array(ros_data.joint_velocities),
            'time_since_skill_started': ros_data.time_since_skill_started,
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
        '''Get most recent gripper width. Note this value will *not* be updated during a control command.

        Returns:
            float of gripper width in meters
        '''
        return self.get_data()['gripper_width']

    def get_gripper_is_grasped(self):
        '''Returns whether or not the gripper is grasping something. Note this value will *not* be updated during a control command.

        Returns:
            True if gripper is grasping something. False otherwise
        '''
        return self.get_data()['gripper_is_grasped']