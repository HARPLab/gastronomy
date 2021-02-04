import numpy as np

from carbongym import gymapi
from carbongym_utils.math_utils import slerp_quat

def min_jerk(xi, xf, t):
    return xi + (xf - xi) * (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)

def min_jerk_gripper_interp(num_time_steps):
    gripper_trajectory = {}

    dt = 1.0/num_time_steps
    for i in range(num_time_steps):
        current_time = i * dt
        gripper_trajectory[i] = (1-current_time) * 0.04
    return gripper_trajectory

def min_jerk_trajectory_interp(start, end, num_time_steps):
    trajectory = {}

    start_position = start.p
    end_position = end.p
    start_quaternion = start.r
    end_quaternion = end.r

    dt = 1.0/num_time_steps

    for i in range(num_time_steps):
        current_time = i * dt
        trajectory[i] = gymapi.Transform(
            p=min_jerk(start_position, end_position, current_time),
            r=slerp_quat(start_quaternion, end_quaternion, current_time)
        )
    return trajectory