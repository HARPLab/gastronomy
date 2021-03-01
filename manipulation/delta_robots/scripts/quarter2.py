#!/usr/bin/env python

from frankapy import FrankaArm
import time
import numpy as np
from DeltaArray import DeltaArray

if __name__ == '__main__':

    fa = FrankaArm()

    height = 0.02

    delta_array = DeltaArray()

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)
    
    time.sleep(3)

    starting_pose = fa.get_pose()
    second_pose = fa.get_pose()

    second_pose.translation -= [0, 0, height]
    fa.goto_pose(second_pose, use_impedance=False)

    delta_position = np.array([0.013, 0.01, 0.01, 0.01, 0.01, 0.013]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(4)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)
    
    time.sleep(3)

    fa.goto_pose(starting_pose, use_impedance=False)

    joints = fa.get_joints()
    joints[6] += np.deg2rad(90)
    fa.goto_joints(joints)

    third_pose = fa.get_pose()
    fourth_pose = fa.get_pose()

    fourth_pose.translation -= [0, 0, height]
    fa.goto_pose(fourth_pose, use_impedance=False)

    delta_position = np.array([0.013, 0.01, 0.01, 0.01, 0.01, 0.013]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(4)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)
    
    time.sleep(3)

    fa.goto_pose(third_pose, use_impedance=False)

    fa.goto_pose(starting_pose, use_impedance=False)

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    delta_velocity = np.array([-0.005, -0.005, -0.005, -0.005, -0.005, -0.005]).reshape((1,6))

    delta_array.move_delta_velocity(delta_velocity,[1.0])    
    