import logging
import math
import numpy as np
from autolab_core import RigidTransform

class FrankaConstants:
    '''
    Contains default robot values, as well as robot limits.
    All units are in SI. 
    '''

    LOGGING_LEVEL = logging.INFO

    EMPTY_SENSOR_VALUES = [0]

    # translational stiffness, rotational stiffness
    DEFAULT_TORQUE_CONTROLLER_PARAMS = [600, 50]
    DEFAULT_FORCE_AXIS_CONTROLLER_PARAMS = [600, 20]

    # buffer time
    DEFAULT_TERM_BUFFER_TIME = 0.2

    HOME_JOINTS = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]
    HOME_POSE = RigidTransform(rotation=np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]), translation=np.array([0.3069, 0, 0.4867]),
        from_frame='franka_tool', to_frame='world')
    READY_JOINTS = [0, -math.pi/4, 0, -2.85496998, 0, 2.09382820,  math.pi/4]
    READY_POSE = RigidTransform(rotation=np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]), translation=np.array([0.3069, 0, 0.2867]),
        from_frame='franka_tool', to_frame='world')

    # See https://frankaemika.github.io/docs/control_parameters.html
    JOINT_LIMITS_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    JOINT_LIMITS_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

    GRIPPER_WIDTH_MAX = 0.08
    GRIPPER_WIDTH_MIN = 0
    GRIPPER_MAX_FORCE = 60

    MAX_LIN_MOMENTUM = 20
    MAX_ANG_MOMENTUM = 2
    MAX_LIN_MOMENTUM_CONSTRAINED = 100

    DEFAULT_ROBOLIB_TIMEOUT = 10
    ACTION_WAIT_LOOP_TIME = 0.001

    GRIPPER_CMD_SLEEP_TIME = 0.2

    DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
    DEFAULT_D_GAINS = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
    DEFAULT_TRANSLATIONAL_STIFFNESSES = [600.0, 600.0, 600.0]
    DEFAULT_ROTATIONAL_STIFFNESSES = [50.0, 50.0, 50.0]

    DEFAULT_JOINT_IMPEDANCES = [3000, 3000, 3000, 2500, 2500, 2000, 2000]
    DEFAULT_CARTESIAN_IMPEDANCES = [3000, 3000, 3000, 300, 300, 300]

    DEFAULT_LOWER_TORQUE_THRESHOLDS_ACCEL = [20.0,20.0,18.0,18.0,16.0,14.0,12.0]
    DEFAULT_UPPER_TORQUE_THRESHOLDS_ACCEL = [120.0,120.0,120.0,118.0,116.0,114.0,112.0]
    DEFAULT_LOWER_TORQUE_THRESHOLDS_NOMINAL = [20.0,20.0,18.0,18.0,16.0,14.0,12.0]
    DEFAULT_UPPER_TORQUE_THRESHOLDS_NOMINAL = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]

    DEFAULT_LOWER_FORCE_THRESHOLDS_ACCEL = [10.0,10.0,10.0,10.0,10.0,10.0]
    DEFAULT_UPPER_FORCE_THRESHOLDS_ACCEL = [120.0,120.0,120.0,125.0,125.0,125.0]
    DEFAULT_LOWER_FORCE_THRESHOLDS_NOMINAL = [10.0,10.0,10.0,10.0,10.0,10.0]
    DEFAULT_UPPER_FORCE_THRESHOLDS_NOMINAL = [120.0,120.0,120.0,125.0,125.0,125.0]
