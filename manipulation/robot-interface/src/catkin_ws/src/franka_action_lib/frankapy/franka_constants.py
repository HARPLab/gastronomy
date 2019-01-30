import logging
import math

class FrankaConstants:

    LOGGING_LEVEL = logging.INFO

    EMPTY_SENSOR_VALUES = [0]

    # translational stiffness, rotational stiffness
    DEFAULT_TORQUE_CONTROLLER_PARAMS = [600, 50]

    # buffer time
    DEFAULT_TERM_PARAMS = [1]

    HOME_JOINTS = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]

    # See https://frankaemika.github.io/docs/control_parameters.html
    JOINT_LIMITS_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    JOINT_LIMITS_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

    GRIPPER_WIDTH_MAX = 0.08
    GRIPPER_WIDTH_MIN = 0
    GRIPPER_WAIT_TIME = 0
    GRIPPER_MAX_FORCE = 80

    MAX_FORCE_NORM = 20
    MAX_TORQUE_NORM = 2