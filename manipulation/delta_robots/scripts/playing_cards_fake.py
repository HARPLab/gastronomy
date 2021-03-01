#!/usr/bin/env python

from frankapy import FrankaArm
import time
import numpy as np

if __name__ == '__main__':

    fa = FrankaArm()
    

    starting_pose = fa.get_pose()
    second_pose = fa.get_pose()

    second_pose.translation += [0, 0, 0.05]
    fa.goto_pose(second_pose, use_impedance=False)

    fa.goto_pose(starting_pose, use_impedance=False)

    # delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    # delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    
    #time.sleep(3)
