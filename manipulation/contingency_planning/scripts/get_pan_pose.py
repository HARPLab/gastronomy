import numpy as np
from autolab_core import RigidTransform
from carbongym_utils.math_utils import RigidTransform_to_transform

def get_pan_pose(urdf_path, table_height):
	# For Pans 1 and 2
    if '1' in urdf_path or '2' in urdf_path:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.3, table_height, 0.15],
            rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
        ))
    # For Pan 3
    elif '3' in urdf_path:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.18, table_height, -0.15]
        ))
    # For Pan 5
    elif '5' in urdf_path:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.15, table_height, -0.2]
        ))
    else:
        pan_pose = RigidTransform_to_transform(RigidTransform(
            translation=[0.15, table_height, -0.2]
        ))

    return pan_pose