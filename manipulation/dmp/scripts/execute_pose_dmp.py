import numpy as np
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_dmp_weights_file_path', type=str, default='')
    args = parser.parse_args()

    if not args.pose_dmp_weights_file_path:
        args.pose_dmp_weights_file_path = '/home/sony/logs/robot_state_data_0_pose_weights.pkl'

    pose_dmp_file = open(args.pose_dmp_weights_file_path,"rb")
    pose_dmp_info = pickle.load(pose_dmp_file)

    print('Starting robot')
    fa = FrankaArm()

    fa.reset_pose()

    fa.reset_joints()

    print('Opening Grippers')
    fa.open_gripper()

    # random_position = RigidTransform(rotation=np.array([
    #         [1, 0, 0],
    #         [0, -1, 0],
    #         [0, 0, -1]
    #     ]), translation=np.array([0.3069, 0, 0.2867]),
    # from_frame='franka_tool', to_frame='world')

    # fa.goto_pose_with_cartesian_control(random_position, 10)

    fa.execute_pose_dmp(pose_dmp_info, duration=10, skill_desc='pose_dmp')