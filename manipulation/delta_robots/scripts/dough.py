#!/usr/bin/env python

from frankapy import FrankaArm
import time
from autolab_core import RigidTransform
import numpy as np

if __name__ == '__main__':

    fa = FrankaArm()

    delta_pose = RigidTransform(
        translation=[0.0, -0.05, 0.0],
        from_frame='franka_tool', to_frame='franka_tool'
    )
    
    fa.goto_pose_delta(delta_pose, use_impedance=False)


    # joints = fa.get_joints()
    # joints[6] += np.deg2rad(-90)
    # fa.goto_joints(joints)

    # delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    # delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    
    #time.sleep(3)
