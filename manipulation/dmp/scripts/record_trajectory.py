import numpy as np
from frankapy import FrankaArm
from autolab_core import RigidTransform

if __name__ == '__main__':

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

    fa.apply_effector_forces_torques(10, 0, 0, 0)