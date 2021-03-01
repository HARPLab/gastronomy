#!/usr/bin/env python

from frankapy import FrankaArm
import time
import numpy as np
from DeltaArray import DeltaArray

if __name__ == '__main__':

    fa = FrankaArm()
    delta_array = DeltaArray()

    height = 0.014

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    delta_velocity = np.array([-0.005, -0.005, -0.005, -0.005, -0.005, -0.005]).reshape((1,6))

    delta_array.move_delta_velocity(delta_velocity,[2.0])

    time.sleep(3)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    
    time.sleep(3)

    starting_pose = fa.get_pose()
    second_pose = fa.get_pose()

    second_pose.translation -= [0, 0, height]
    fa.goto_pose(second_pose, use_impedance=False)

    delta_position = np.array([0.011, 0.01, 0.01, 0.01, 0.01, 0.011]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()

    time.sleep(3)

    fa.goto_pose(starting_pose, use_impedance=False)

    delta_position = np.array([0.012, 0.01, 0.013, 0.01, 0.013, 0.012]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(0.3)

    delta_position = np.array([0.013, 0.011, 0.016, 0.011, 0.016, 0.013]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()

    time.sleep(3)

    delta_position = np.array([0.012, 0.01, 0.013, 0.01, 0.013, 0.012]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(0.3)

    delta_position = np.array([0.011, 0.01, 0.01, 0.01, 0.01, 0.011]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()

    time.sleep(0.3)

    delta_position = np.array([0.012, 0.014, 0.01, 0.013, 0.01, 0.012]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    # delta_position = np.array([0.01, 0.016, 0.011, 0.016, 0.011, 0.01]).reshape((1,6))

    # delta_array.move_delta_position(delta_position)

    # #delta_array.wait_until_done_moving()

    # time.sleep(3)

    # delta_position = np.array([0.01, 0.013, 0.01, 0.013, 0.01, 0.01]).reshape((1,6))

    # delta_array.move_delta_position(delta_position)

    # time.sleep(3)

    delta_position = np.array([0.011, 0.01, 0.01, 0.01, 0.01, 0.011]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()

    time.sleep(3)

    fa.goto_pose(second_pose, use_impedance=False)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    
    time.sleep(3)

    fa.goto_pose(starting_pose, use_impedance=False)

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(1)

    delta_velocity = np.array([-0.005, -0.005, -0.005, -0.005, -0.005, -0.005]).reshape((1,6))

    delta_array.move_delta_velocity(delta_velocity,[1.0])


    #delta_array.wait_until_done_moving()
    
    