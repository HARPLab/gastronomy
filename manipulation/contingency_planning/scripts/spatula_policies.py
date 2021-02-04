import numpy as np

from carbongym import gymapi
from carbongym_utils.policy import Policy
from carbongym_utils.math_utils import RigidTransform_to_transform, transform_to_RigidTransform, quat_to_rot, np_to_vec3, rpy_to_quat, vec3_to_np, quat_to_np
from min_jerk_interpolation import min_jerk_trajectory_interp

from autolab_core import RigidTransform

class PickUpBlockFlatPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_theta_dists):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_dists = x_y_theta_dists

        self._simulation_num = 0

        self._time_horizon = 1000

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}
        self._tilt_transforms = {}

        self._pre_grasp_trajectory = {}
        self._grasp_trajectory = {}
        self._push_trajectory = {}
        self._post_grasp_trajectory = {}
        self._tilt_trajectory = {}
        self._post_release_trajectory = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 0:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

        if t_step == 50:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            block_ah = scene.ah_map[env_index][self._block_name]

            block1_idx = self._block.rb_names_map['block1']
            block1_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block1_idx]
            block2_idx = self._block.rb_names_map['block2']
            block2_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block2_idx]
            block3_idx = self._block.rb_names_map['block3']
            block3_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block3_idx]

            block_center = (vec3_to_np(block1_transform.p) + vec3_to_np(block2_transform.p) + vec3_to_np(block3_transform.p)) / 3

            c = (0.15 - np.abs(block1_transform.p.x - block3_transform.p.x) + np.abs(block2_transform.p.x - block3_transform.p.x) - np.abs(block1_transform.p.x - block2_transform.p.x)) / 2

            if (block1_transform.p.x - block3_transform.p.x) < 0:
                block_center[0] = block3_transform.p.x + c - 0.075
            else:
                block_center[0] = block3_transform.p.x - c + 0.075

            x_y_theta_dist = self._x_y_theta_dists[self._simulation_num * self._n_envs + env_index]
            
            theta = x_y_theta_dist[2]
            displacement = x_y_theta_dist[3]
            backoff_distance = -x_y_theta_dist[1] - 0.3

            pre_grasp_offset = [x_y_theta_dist[0], backoff_distance, 0.0]

            push_offset = [x_y_theta_dist[0], backoff_distance + displacement, 0.0]

            height = -0.013

            self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.2, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist[2]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=pre_grasp_offset, 
                from_frame='ee_frame',to_frame='ee_frame'))

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.2-height, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, height, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist[2]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=push_offset, 
                from_frame='ee_frame',to_frame='ee_frame'))

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                r=self._push_transforms[env_index].r
            )

            self._tilt_transforms[env_index] = gymapi.Transform(
                p=self._ee_transforms[env_index].p,
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, -np.pi/2, 0)))

            self._pre_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._ee_transforms[env_index], self._pre_grasp_transforms[env_index], 100)
            self._grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._pre_grasp_transforms[env_index], self._grasp_transforms[env_index], 100)
            self._push_trajectory[env_index] = min_jerk_trajectory_interp(self._grasp_transforms[env_index], self._push_transforms[env_index], 100)
            self._post_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._push_transforms[env_index], self._post_grasp_transforms[env_index], 100)
            self._tilt_trajectory[env_index] = min_jerk_trajectory_interp(self._post_grasp_transforms[env_index], self._tilt_transforms[env_index], 100)
            self._post_release_trajectory[env_index] = min_jerk_trajectory_interp(self._tilt_transforms[env_index], self._ee_transforms[env_index], 100)

        if t_step > 50 and t_step <= 150:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_trajectory[env_index][t_step-51])

        if t_step > 200 and t_step <= 300:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_trajectory[env_index][t_step-201])

        if t_step > 350 and t_step <= 450:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_trajectory[env_index][t_step-351])

        if t_step > 550 and t_step <= 650:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_trajectory[env_index][t_step-551])

        if t_step > 750 and t_step <= 850:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._tilt_trajectory[env_index][t_step-751])
        
        if t_step > 900 and t_step <= 1000:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_release_trajectory[env_index][t_step-901])


class PickUpBlockTiltedPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_theta_dist_tilts):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_dist_tilts = x_y_theta_dist_tilts

        self._simulation_num = 0

        self._time_horizon = 900

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}
        self._tilt_transforms = {}

        self._pre_grasp_trajectory = {}
        self._grasp_trajectory = {}
        self._post_grasp_trajectory = {}
        self._tilt_trajectory = {}
        self._post_release_trajectory = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 0:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

        if t_step == 50:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            block_ah = scene.ah_map[env_index][self._block_name]

            block1_idx = self._block.rb_names_map['block1']
            block1_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block1_idx]
            block2_idx = self._block.rb_names_map['block2']
            block2_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block2_idx]
            block3_idx = self._block.rb_names_map['block3']
            block3_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block3_idx]

            block_center = (vec3_to_np(block1_transform.p) + vec3_to_np(block2_transform.p) + vec3_to_np(block3_transform.p)) / 3

            c = (0.15 - np.abs(block1_transform.p.x - block3_transform.p.x) + np.abs(block2_transform.p.x - block3_transform.p.x) - np.abs(block1_transform.p.x - block2_transform.p.x)) / 2

            if (block1_transform.p.x - block3_transform.p.x) < 0:
                block_center[0] = block3_transform.p.x + c - 0.075
            else:
                block_center[0] = block3_transform.p.x - c + 0.075

            x_y_theta_dist_tilt = self._x_y_theta_dist_tilts[self._simulation_num * self._n_envs + env_index]
            
            theta = x_y_theta_dist_tilt[2]
            displacement = x_y_theta_dist_tilt[3]
            tilt_angle = x_y_theta_dist_tilt[4]
            backoff_distance = -x_y_theta_dist_tilt[1] - 0.1*np.cos(np.abs(tilt_angle)) - 0.2

            pre_push_offset = [x_y_theta_dist_tilt[0], backoff_distance, 0.0]

            push_offset = [x_y_theta_dist_tilt[0], backoff_distance + displacement, 0.0]

            pre_push_height = 0.1*np.sin(np.abs(tilt_angle)) + 0.03

            height = - 0.013

            self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, pre_push_height, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist_tilt[2]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=pre_push_offset, 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                from_frame='ee_frame',to_frame='ee_frame'
            ))

            self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, height, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist_tilt[2]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=push_offset, 
                from_frame='ee_frame',to_frame='ee_frame'))

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                r=self._push_transforms[env_index].r
            )

            self._tilt_transforms[env_index] = gymapi.Transform(
                p=self._ee_transforms[env_index].p,
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, -np.pi/2, 0)))

            self._pre_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._ee_transforms[env_index], self._pre_grasp_transforms[env_index], 100)
            self._grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._pre_grasp_transforms[env_index], self._push_transforms[env_index], 100)
            self._post_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._push_transforms[env_index], self._post_grasp_transforms[env_index], 100)
            self._tilt_trajectory[env_index] = min_jerk_trajectory_interp(self._post_grasp_transforms[env_index], self._tilt_transforms[env_index], 100)
            self._post_release_trajectory[env_index] = min_jerk_trajectory_interp(self._tilt_transforms[env_index], self._ee_transforms[env_index], 100)

        if t_step > 50 and t_step <= 150:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_trajectory[env_index][t_step-51])

        if t_step > 200 and t_step <= 300:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_trajectory[env_index][t_step-201])

        if t_step > 400 and t_step <= 500:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_trajectory[env_index][t_step-401])

        if t_step > 600 and t_step <= 700:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._tilt_trajectory[env_index][t_step-601])
        
        if t_step > 750 and t_step <= 850:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_release_trajectory[env_index][t_step-751])

class PickUpURDFFlatPolicy(Policy):

    def __init__(self, franka, franka_name, urdf, urdf_name, n_envs, x_y_theta_dists):
        self._franka = franka
        self._franka_name = franka_name
        self._urdf = urdf
        self._urdf_name = urdf_name
        self._n_envs = n_envs
        self._x_y_theta_dists = x_y_theta_dists
        self._settling_t_step = 150

        self._simulation_num = 0

        self._time_horizon = self._settling_t_step + 1150

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}
        self._tilt_transforms = {}
        self._settling_t_step = 150

        self._pre_grasp_trajectory = {}
        self._grasp_trajectory = {}
        self._push_trajectory = {}
        self._post_grasp_trajectory = {}
        self._tilt_trajectory = {}
        self._post_release_trajectory = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def set_x_y_theta_dists(self, x_y_theta_dists):
        self._x_y_theta_dists = x_y_theta_dists

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]
        urdf_ah = scene.ah_map[env_index][self._urdf_name]

        if t_step == 0:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])
            self._starting_urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]

        else:
            if env_index == 0 and self._settling_t_step == 150:
                current_urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]
                if np.linalg.norm(vec3_to_np(current_urdf_transform.p) - vec3_to_np(self._starting_urdf_transform.p)) < 0.00001:
                    self._settling_t_step = t_step
                    self._time_horizon = self._settling_t_step + 1150
                else:
                    self._starting_urdf_transform = current_urdf_transform

            if t_step == self._settling_t_step + 1:
                self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)

                urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]

                urdf_center = vec3_to_np(urdf_transform.p)

                x_y_theta_dist = self._x_y_theta_dists[self._simulation_num * self._n_envs + env_index]
                
                theta = x_y_theta_dist[2]
                displacement = x_y_theta_dist[3]
                backoff_distance = -x_y_theta_dist[1] - 0.3

                pre_grasp_offset = [x_y_theta_dist[0], backoff_distance, 0.0]

                push_offset = [x_y_theta_dist[0], backoff_distance + displacement, 0.0]

                height = -0.013

                self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=pre_grasp_offset, 
                    from_frame='ee_frame',to_frame='ee_frame'))

                self._grasp_transforms[env_index] = gymapi.Transform(
                    p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.2-height, 0.0]),
                    r=self._pre_grasp_transforms[env_index].r
                )

                self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame'))

                self._post_grasp_transforms[env_index] = gymapi.Transform(
                    p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._push_transforms[env_index].r
                )

                self._tilt_transforms[env_index] = gymapi.Transform(
                    p=self._ee_transforms[env_index].p,
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, -np.pi/2, 0)))

                self._pre_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._ee_transforms[env_index], self._pre_grasp_transforms[env_index], 100)
                self._grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._pre_grasp_transforms[env_index], self._grasp_transforms[env_index], 100)
                self._push_trajectory[env_index] = min_jerk_trajectory_interp(self._grasp_transforms[env_index], self._push_transforms[env_index], 100)
                self._post_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._push_transforms[env_index], self._post_grasp_transforms[env_index], 100)
                self._tilt_trajectory[env_index] = min_jerk_trajectory_interp(self._post_grasp_transforms[env_index], self._tilt_transforms[env_index], 100)
                self._post_release_trajectory[env_index] = min_jerk_trajectory_interp(self._tilt_transforms[env_index], self._ee_transforms[env_index], 100)

            if t_step > self._settling_t_step + 1 and t_step <= self._settling_t_step + 100:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_trajectory[env_index][t_step-(self._settling_t_step + 1)])

            if t_step > self._settling_t_step + 200 and t_step <= self._settling_t_step + 300:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_trajectory[env_index][t_step-(self._settling_t_step + 201)])

            if t_step > self._settling_t_step + 400 and t_step <= self._settling_t_step + 500:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_trajectory[env_index][t_step-(self._settling_t_step + 401)])

            if t_step > self._settling_t_step + 600 and t_step <= self._settling_t_step + 700:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_trajectory[env_index][t_step-(self._settling_t_step + 601)])

            if t_step > self._settling_t_step + 800 and t_step <= self._settling_t_step + 900:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._tilt_trajectory[env_index][t_step-(self._settling_t_step + 801)])
            
            if t_step > self._settling_t_step + 1000 and t_step <= self._settling_t_step + 1100:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_release_trajectory[env_index][t_step-(self._settling_t_step + 1001)])

   
class PickUpURDFTiltedPolicy(Policy):

    def __init__(self, franka, franka_name, urdf, urdf_name, n_envs, x_y_theta_dist_tilts):
        self._franka = franka
        self._franka_name = franka_name
        self._urdf = urdf
        self._urdf_name = urdf_name
        self._n_envs = n_envs
        self._x_y_theta_dist_tilts = x_y_theta_dist_tilts
        self._settling_t_step = 150

        self._simulation_num = 0

        self._time_horizon = self._settling_t_step + 950

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}
        self._tilt_transforms = {}
        self._settling_t_step = 150

        self._pre_grasp_trajectory = {}
        self._grasp_trajectory = {}
        self._post_grasp_trajectory = {}
        self._tilt_trajectory = {}
        self._post_release_trajectory = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def set_x_y_theta_dist_tilts(self, x_y_theta_dist_tilts):
        self._x_y_theta_dist_tilts = x_y_theta_dist_tilts

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]
        urdf_ah = scene.ah_map[env_index][self._urdf_name]

        if t_step == 0:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])
            self._starting_urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]

        else:
            if env_index == 0 and self._settling_t_step == 150:
                current_urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]
                if np.linalg.norm(vec3_to_np(current_urdf_transform.p) - vec3_to_np(self._starting_urdf_transform.p)) < 0.00001:
                    self._settling_t_step = t_step
                    self._time_horizon = self._settling_t_step + 950
                else:
                    self._starting_urdf_transform = current_urdf_transform

            if t_step == self._settling_t_step + 1:
                self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)

                urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]

                urdf_center = vec3_to_np(urdf_transform.p)

                x_y_theta_dist_tilt = self._x_y_theta_dist_tilts[self._simulation_num * self._n_envs + env_index]
                
                theta = x_y_theta_dist_tilt[2]
                displacement = x_y_theta_dist_tilt[3]
                tilt_angle = x_y_theta_dist_tilt[4]
                backoff_distance = -x_y_theta_dist_tilt[1] - 0.1*np.cos(np.abs(tilt_angle)) - 0.2

                pre_push_offset = [x_y_theta_dist_tilt[0], backoff_distance, 0.0]

                push_offset = [x_y_theta_dist_tilt[0], backoff_distance + displacement, 0.0]

                pre_push_height = 0.1*np.sin(np.abs(tilt_angle)) + 0.03

                height = - 0.013

                self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, pre_push_height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist_tilt[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=pre_push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                    from_frame='ee_frame',to_frame='ee_frame'
                ))

                self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_dist_tilt[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame'))

                self._post_grasp_transforms[env_index] = gymapi.Transform(
                    p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._push_transforms[env_index].r
                )

                self._tilt_transforms[env_index] = gymapi.Transform(
                    p=self._ee_transforms[env_index].p,
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, -np.pi/2, 0)))

                self._pre_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._ee_transforms[env_index], self._pre_grasp_transforms[env_index], 100)
                self._grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._pre_grasp_transforms[env_index], self._push_transforms[env_index], 100)
                self._post_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._push_transforms[env_index], self._post_grasp_transforms[env_index], 100)
                self._tilt_trajectory[env_index] = min_jerk_trajectory_interp(self._post_grasp_transforms[env_index], self._tilt_transforms[env_index], 100)
                self._post_release_trajectory[env_index] = min_jerk_trajectory_interp(self._tilt_transforms[env_index], self._ee_transforms[env_index], 100)

            if t_step > self._settling_t_step + 1 and t_step <= self._settling_t_step + 100:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_trajectory[env_index][t_step-(self._settling_t_step + 1)])

            if t_step > self._settling_t_step + 200 and t_step <= self._settling_t_step + 300:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_trajectory[env_index][t_step-(self._settling_t_step + 201)])

            if t_step > self._settling_t_step + 400 and t_step <= self._settling_t_step + 500:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_trajectory[env_index][t_step-(self._settling_t_step + 401)])

            if t_step > self._settling_t_step + 600 and t_step <= self._settling_t_step + 700:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._tilt_trajectory[env_index][t_step-(self._settling_t_step + 601)])
            
            if t_step > self._settling_t_step + 800 and t_step <= self._settling_t_step + 900:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_release_trajectory[env_index][t_step-(self._settling_t_step + 801)])
