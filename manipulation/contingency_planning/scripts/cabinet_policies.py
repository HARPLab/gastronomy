import numpy as np

from carbongym import gymapi
from carbongym_utils.policy import Policy
from carbongym_utils.math_utils import RigidTransform_to_transform, transform_to_RigidTransform, np_to_vec3, rpy_to_quat, vec3_to_np, quat_to_np

from autolab_core import RigidTransform
from min_jerk_interpolation import min_jerk_trajectory_interp

class CabinetOpenPolicy(Policy):

    def __init__(self, franka, franka_name, door, door_name, n_envs, x_y_zs):
        self._franka = franka
        self._franka_name = franka_name
        self._door = door
        self._door_name = door_name
        self._n_envs = n_envs
        self._x_y_zs = x_y_zs

        self._time_horizon = 900

        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._pull_transforms = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def set_x_y_zs(self, x_y_zs):
        self._x_y_zs = x_y_zs

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._pull_transforms = {}

        self._pre_grasp_trajectory = {}
        self._grasp_trajectory = {}
        self._pull_trajectory = {}
        self._post_release_trajectory = {}

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 50:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            door_ah = scene.ah_map[env_index][self._door_name]

            handle_idx = self._door.rb_names_map['door_handle']
            pole_idx = self._door.rb_names_map['cabinet_door']
            handle_transform = self._door.get_rb_transforms(env_ptr, door_ah)[handle_idx]
            pole_transform = self._door.get_rb_transforms(env_ptr, door_ah)[pole_idx]
            handle_pole_dist = np.linalg.norm(vec3_to_np(handle_transform.p - pole_transform.p) * [1, 0, 1]) * 0.8

            x_y_z = self._x_y_zs[self._simulation_num * self._n_envs + env_index]
            grasp_offset = [x_y_z[0], x_y_z[2], -x_y_z[1]]

            self._pre_grasp_transforms[env_index] = gymapi.Transform(
                p=handle_transform.p + np_to_vec3([-0.15, 0, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, np.pi/2, -np.pi/2))
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p + np_to_vec3([0.17, 0, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            # if abs(np.arctan2(x_y_z[2], -x_y_z[0])) < 1.3:
            #     self._pull_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
            #         p=handle_transform.p + np_to_vec3(grasp_offset),
            #         r=self._ee_transforms[env_index].r * rpy_to_quat((0, np.pi/2, -np.pi/2)) 
            #     ), from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
            #         rotation=RigidTransform.quaternion_from_axis_angle([np.arctan2(x_y_z[2], -x_y_z[0]), 0, 0]), 
            #         from_frame='ee_frame',to_frame='ee_frame'))
            # else:
            self._pull_transforms[env_index] = gymapi.Transform(
                p=handle_transform.p + np_to_vec3(grasp_offset),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, np.pi/2, -np.pi/2)) 
            )

            self._pre_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._ee_transforms[env_index], self._pre_grasp_transforms[env_index], 100)
            self._grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._pre_grasp_transforms[env_index], self._grasp_transforms[env_index], 100)
            self._pull_trajectory[env_index] = min_jerk_trajectory_interp(self._grasp_transforms[env_index], self._pull_transforms[env_index], 100)
            self._post_release_trajectory[env_index] = min_jerk_trajectory_interp(self._pull_transforms[env_index], self._ee_transforms[env_index], 100)

        if t_step > 50 and t_step <= 150:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_trajectory[env_index][t_step-51])

        if t_step > 250 and t_step <= 350:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_trajectory[env_index][t_step-251])

        if t_step == 450:
            self._franka.close_grippers(env_ptr, ah)

        if t_step > 500 and t_step <= 600:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pull_trajectory[env_index][t_step-501])

        if t_step == 700:
            self._franka.open_grippers(env_ptr, ah)

        if t_step > 750 and t_step <= 850:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_release_trajectory[env_index][t_step-751])
