import numpy as np
from autolab_core import RigidTransform
from carbongym_utils.math_utils import RigidTransform_to_transform

def get_pan_pose(pan_num, table_height):
	# For Pans 1 and 2
    if pan_num == 1 or pan_num == 2:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.3, table_height, 0.15],
            rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
        ))
    # For Pan 3
    elif pan_num == 3:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.18, table_height, -0.15]
        ))
    # For Pan 4
    elif pan_num == 4:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.33, table_height, -0.15]
        ))
    # For Pan 5
    elif pan_num == 5:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.15, table_height, -0.2]
        ))
    # For Pan 6
    elif pan_num == 6:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.33, table_height, 0.15],
            rotation=RigidTransform.quaternion_from_axis_angle([0, np.pi/2, 0])
        ))
    else:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.15, table_height, -0.2]
        ))

    return pan_pose