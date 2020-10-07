import numpy as np

from carbongym import gymapi
from carbongym_utils.policy import Policy
from carbongym_utils.math_utils import RigidTransform_to_transform, transform_to_RigidTransform, quat_to_rot, np_to_vec3, rpy_to_quat, vec3_to_np, quat_to_np

from autolab_core import RigidTransform

class PickUpURDFPolicy(Policy):

    def __init__(self, franka, franka_name, urdf, urdf_name, n_envs, x_y_thetas):
        self._franka = franka
        self._franka_name = franka_name
        self._urdf = urdf
        self._urdf_name = urdf_name
        self._n_envs = n_envs
        self._x_y_thetas = x_y_thetas
        self._settling_t_step = 150

        self._simulation_num = 0

        self._time_horizon = self._settling_t_step + 450

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._gripper_width = {}
        self._post_grasp_transforms = {}
        self._settling_t_step = 150

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def set_x_y_thetas(self, x_y_thetas):
        self._x_y_thetas = x_y_thetas

    def __call__(self, scene, env_index, t_step, _):
        #if(env_index == 0):
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
                    self._time_horizon = self._settling_t_step + 450
                else:
                    self._starting_urdf_transform = current_urdf_transform

            if t_step == self._settling_t_step + 1:
                self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)

                urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]

                urdf_center = vec3_to_np(urdf_transform.p)

                x_y_theta = self._x_y_thetas[self._simulation_num * self._n_envs + env_index]
                pre_grasp_offset = [x_y_theta[0], 0.0, -x_y_theta[1]]

                self._pre_grasp_transforms[env_index] = gymapi.Transform(
                    #p=block2_transform.p + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, 0.4, 0.0]) + np_to_vec3(pre_grasp_offset),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta[2]))
                    #r=self._ee_transforms[env_index].r
                )

                self._grasp_transforms[env_index] = gymapi.Transform(
                    p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.25, 0.0]),
                    r=self._pre_grasp_transforms[env_index].r
                )

                self._post_grasp_transforms[env_index] = gymapi.Transform(
                    p=self._grasp_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._grasp_transforms[env_index].r
                )

                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

            if t_step == self._settling_t_step + 50:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

            if t_step == self._settling_t_step + 150:
                self._franka.close_grippers(env_ptr, ah)

            if t_step == self._settling_t_step + 200:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_transforms[env_index])

            if t_step == self._settling_t_step + 300:
                self._franka.open_grippers(env_ptr, ah)

            if t_step == self._settling_t_step + 350:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class PickUpURDFSidePolicy(Policy):

    def __init__(self, franka, franka_name, urdf, urdf_name, n_envs, x_y_theta_tilt_dists):
        self._franka = franka
        self._franka_name = franka_name
        self._urdf = urdf
        self._urdf_name = urdf_name
        self._n_envs = n_envs
        self._x_y_theta_tilt_dists = x_y_theta_tilt_dists
        self._settling_t_step = 150

        self._simulation_num = 0

        self._time_horizon = self._settling_t_step + 550

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}
        self._settling_t_step = 150

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

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
                    self._time_horizon = self._settling_t_step + 550
                else:
                    self._starting_urdf_transform = current_urdf_transform

            if t_step == self._settling_t_step + 1:
                self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)

                urdf_transform = self._urdf.get_rb_transforms(env_ptr, urdf_ah)[0]

                urdf_center = vec3_to_np(urdf_transform.p)

                x_y_theta_tilt_dist = self._x_y_theta_tilt_dists[self._simulation_num * self._n_envs + env_index]
            
                theta = x_y_theta_tilt_dist[2]

                if(theta > 0):
                    tilt_angle = x_y_theta_tilt_dist[3]
                    displacement = -x_y_theta_tilt_dist[4]
                    backoff_distance = x_y_theta_tilt_dist[1] + 0.15*np.sin(np.abs(tilt_angle))
                else:
                    tilt_angle = -x_y_theta_tilt_dist[3]
                    displacement = x_y_theta_tilt_dist[4]
                    backoff_distance = -x_y_theta_tilt_dist[1] - 0.15*np.sin(np.abs(tilt_angle))

                pre_grasp_offset = [x_y_theta_tilt_dist[0], backoff_distance, 0.0]

                push_offset = [x_y_theta_tilt_dist[0], backoff_distance + displacement, 0.0]

                height = 0.13*np.cos(np.abs(tilt_angle)) + 0.03

                self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, 0.4, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=pre_grasp_offset, 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                    from_frame='ee_frame',to_frame='ee_frame'
                ))

                self._grasp_transforms[env_index] = gymapi.Transform(
                    p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.4-height, 0.0]),
                    r=self._pre_grasp_transforms[env_index].r
                )

                self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(urdf_center) + np_to_vec3([0.0, height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                    from_frame='ee_frame',to_frame='ee_frame'
                ))

                self._post_grasp_transforms[env_index] = gymapi.Transform(
                    p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.25, 0.0]),
                    r=self._push_transforms[env_index].r
                )

                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

            if t_step == self._settling_t_step + 50:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

            if t_step == self._settling_t_step + 150:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])

            if t_step == self._settling_t_step + 250:
                self._franka.close_grippers(env_ptr, ah)

            if t_step == self._settling_t_step + 300:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_transforms[env_index])

            if t_step == self._settling_t_step + 400:
                self._franka.open_grippers(env_ptr, ah)

            if t_step == self._settling_t_step + 450:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class PickUpBlockOverheadOnlyPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_thetas):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_thetas = x_y_thetas

        self._simulation_num = 0

        self._time_horizon = 500

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._post_grasp_transforms = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def set_x_y_thetas(self, x_y_thetas):
        self._x_y_thetas = x_y_thetas

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
            b = (np.abs(block1_transform.p.x - block3_transform.p.x) * 2 - 0.15) / 2.0
            a = np.abs(block1_transform.p.x - block2_transform.p.x) - b

            block_center = (vec3_to_np(block1_transform.p) + vec3_to_np(block2_transform.p) + vec3_to_np(block3_transform.p)) / 3
        
            if (block1_transform.p.x - block3_transform.p.x) < 0:
                block_center[0] = block2_transform.p.x + 2 * a + b - 0.075
            else:
                block_center[0] = block2_transform.p.x - 2 * a - b + 0.075
            
            x_y_theta = self._x_y_thetas[self._simulation_num * self._n_envs + env_index]
            pre_grasp_offset = [x_y_theta[0], 0.0, -x_y_theta[1]]

            self._pre_grasp_transforms[env_index] = gymapi.Transform(
                #p=block2_transform.p + np_to_vec3([0.0, 0.4, 0.0]) + np_to_vec3(pre_grasp_offset),
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.4, 0.0]) + np_to_vec3(pre_grasp_offset),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta[2]))
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.25, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._grasp_transforms[env_index].p + np_to_vec3([0.0, 0.25, 0.0]),
                r=self._grasp_transforms[env_index].r
            )

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 100:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 200:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 250:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_transforms[env_index])

        if t_step == 350:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 400:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])


class PickUpBlockOverheadPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_z_thetas, x_y_z_rx_ry_rz):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_z_thetas = x_y_z_thetas
        self._x_y_z_rx_ry_rz = x_y_z_rx_ry_rz

        self._simulation_num = 0

        self._time_horizon = 650

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._post_grasp_transforms = {}
        self._random_relative_transforms = {}

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
            b = (np.abs(block1_transform.p.x - block3_transform.p.x) * 2 - 0.15) / 2.0
            a = np.abs(block1_transform.p.x - block2_transform.p.x) - b

            block_center = (vec3_to_np(block1_transform.p) + vec3_to_np(block2_transform.p) + vec3_to_np(block3_transform.p)) / 3
        
            if (block1_transform.p.x - block3_transform.p.x) < 0:
                block_center[0] = block2_transform.p.x + 2 * a + b - 0.075
            else:
                block_center[0] = block2_transform.p.x - 2 * a - b + 0.075
            
            x_y_z_theta = self._x_y_z_thetas[self._simulation_num * self._n_envs + env_index]
            x_y_z_rx_ry_rz = self._x_y_z_rx_ry_rz[self._simulation_num * self._n_envs + env_index]
            pre_grasp_offset = [x_y_z_theta[0], 0.0, -x_y_z_theta[1]]
            random_grasp_trans = [x_y_z_rx_ry_rz[0], x_y_z_rx_ry_rz[2], -x_y_z_rx_ry_rz[1]]

            self._pre_grasp_transforms[env_index] = gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.4, 0.0]) + np_to_vec3(pre_grasp_offset),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_z_theta[3]))
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.25, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._grasp_transforms[env_index].p + np_to_vec3([0.0, x_y_z_theta[2], 0.0]),
                r=self._grasp_transforms[env_index].r
            )

            self._random_relative_transforms[env_index] = gymapi.Transform(
                p=self._post_grasp_transforms[env_index].p + np_to_vec3(random_grasp_trans),
                r=self._post_grasp_transforms[env_index].r * rpy_to_quat((x_y_z_rx_ry_rz[3], x_y_z_rx_ry_rz[5], -x_y_z_rx_ry_rz[4]))
            )

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 100:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 200:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 250:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_transforms[env_index])

        if t_step == 350:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._random_relative_transforms[env_index])

        if t_step == 450:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 550:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class PickUpBlockSideOnlyPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_theta_tilt_dists):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_tilt_dists = x_y_theta_tilt_dists

        self._simulation_num = 0

        self._time_horizon = 600

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}

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
            b = (np.abs(block1_transform.p.x - block3_transform.p.x) * 2 - 0.15) / 2.0
            a = np.abs(block1_transform.p.x - block2_transform.p.x) - b

            block_center = (vec3_to_np(block1_transform.p) + vec3_to_np(block2_transform.p) + vec3_to_np(block3_transform.p)) / 3
        
            if (block1_transform.p.x - block3_transform.p.x) < 0:
                block_center[0] = block2_transform.p.x + 2 * a + b - 0.075
            else:
                block_center[0] = block2_transform.p.x - 2 * a - b + 0.075
            
            x_y_theta_tilt_dist = self._x_y_theta_tilt_dists[self._simulation_num * self._n_envs + env_index]
            
            theta = x_y_theta_tilt_dist[2]

            if(theta > 0):
                tilt_angle = x_y_theta_tilt_dist[3]
                displacement = -x_y_theta_tilt_dist[4]
                backoff_distance = x_y_theta_tilt_dist[1] + 0.15*np.sin(np.abs(tilt_angle))
            else:
                tilt_angle = -x_y_theta_tilt_dist[3]
                displacement = x_y_theta_tilt_dist[4]
                backoff_distance = -x_y_theta_tilt_dist[1] - 0.15*np.sin(np.abs(tilt_angle))

            pre_grasp_offset = [x_y_theta_tilt_dist[0], backoff_distance, 0.0]

            push_offset = [x_y_theta_tilt_dist[0], backoff_distance + displacement, 0.0]

            height = 0.13*np.cos(np.abs(tilt_angle)) + 0.03

            self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.4, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=pre_grasp_offset, 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                from_frame='ee_frame',to_frame='ee_frame'
            ))

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.4-height, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, height, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=push_offset, 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                from_frame='ee_frame',to_frame='ee_frame'
            ))

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.25, 0.0]),
                r=self._push_transforms[env_index].r
            )

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 100:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 200:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])

        if t_step == 300:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 350:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_transforms[env_index])

        if t_step == 450:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 500:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class PickUpBlockSidePolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_z_theta_tilt_dists, x_y_z_rx_ry_rz):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_z_theta_tilt_dists = x_y_z_theta_tilt_dists
        self._x_y_z_rx_ry_rz = x_y_z_rx_ry_rz

        self._simulation_num = 0

        self._time_horizon = 750

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}
        self._random_relative_transforms = {}

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
            b = (np.abs(block1_transform.p.x - block3_transform.p.x) * 2 - 0.15) / 2.0
            a = np.abs(block1_transform.p.x - block2_transform.p.x) - b

            block_center = (vec3_to_np(block1_transform.p) + vec3_to_np(block2_transform.p) + vec3_to_np(block3_transform.p)) / 3
        
            if (block1_transform.p.x - block3_transform.p.x) < 0:
                block_center[0] = block2_transform.p.x + 2 * a + b - 0.075
            else:
                block_center[0] = block2_transform.p.x - 2 * a - b + 0.075
            x_y_z_theta_tilt_dist = self._x_y_z_theta_tilt_dists[self._simulation_num * self._n_envs + env_index]
            x_y_z_rx_ry_rz = self._x_y_z_rx_ry_rz[self._simulation_num * self._n_envs + env_index]
            
            random_grasp_trans = [x_y_z_rx_ry_rz[0], x_y_z_rx_ry_rz[2], -x_y_z_rx_ry_rz[1]]

            theta = x_y_z_theta_tilt_dist[3]

            if(theta > 0):
                tilt_angle = x_y_z_theta_tilt_dist[4]
                displacement = -x_y_z_theta_tilt_dist[5]
                backoff_distance = x_y_z_theta_tilt_dist[1] + 0.15*np.sin(np.abs(tilt_angle))
            else:
                tilt_angle = -x_y_z_theta_tilt_dist[4]
                displacement = x_y_z_theta_tilt_dist[5]
                backoff_distance = -x_y_z_theta_tilt_dist[1] - 0.15*np.sin(np.abs(tilt_angle))

            pre_grasp_offset = [x_y_z_theta_tilt_dist[0], backoff_distance, 0.0]

            push_offset = [x_y_z_theta_tilt_dist[0], backoff_distance + displacement, 0.0]

            height = 0.13*np.cos(np.abs(tilt_angle)) + 0.03

            self._pre_grasp_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.4, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_z_theta_tilt_dist[3]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=pre_grasp_offset, 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                from_frame='ee_frame',to_frame='ee_frame'
            ))

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.4-height, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                p=np_to_vec3(block_center) + np_to_vec3([0.0, height, 0.0]),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_z_theta_tilt_dist[3]))), 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                translation=push_offset, 
                from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                from_frame='ee_frame',to_frame='ee_frame'
            ))

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._push_transforms[env_index].p + np_to_vec3([0.0, x_y_z_theta_tilt_dist[2], 0.0]),
                r=self._push_transforms[env_index].r
            )

            self._random_relative_transforms[env_index] = gymapi.Transform(
                p=self._post_grasp_transforms[env_index].p + np_to_vec3(random_grasp_trans),
                r=self._post_grasp_transforms[env_index].r * rpy_to_quat((x_y_z_rx_ry_rz[3], x_y_z_rx_ry_rz[5], -x_y_z_rx_ry_rz[4]))
            )

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 100:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 200:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])

        if t_step == 300:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 350:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_transforms[env_index])

        if t_step == 450:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._random_relative_transforms[env_index])

        if t_step == 550:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 650:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class FlipBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 250

        self._env_offsets = {}

    def reset(self):
        self._ee_transforms = {}
        self._pre_push_transforms = {}
        self._push_transforms = {}

    def set_next_push(self, env_offsets):
        self._env_offsets = env_offsets

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 0:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 50:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            block_ah = scene.ah_map[env_index][self._block_name]

            block1_idx = self._block.rb_names_map['block1']
            block2_idx = self._block.rb_names_map['block2']
            block3_idx = self._block.rb_names_map['block3']
            block1_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block1_idx]
            block2_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block2_idx]
            block3_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block3_idx]

            self._pre_push_transforms[env_index] = RigidTransform_to_transform(RigidTransform(
                translation=vec3_to_np(block2_transform.p) + [0.0, 0.25, 0.0], rotation=quat_to_rot(self._ee_transforms[env_index].r), from_frame='ee_frame', to_frame='ee_frame')*
                self._env_offsets[env_index]['pre_push_transformation'])

            self._push_transforms[env_index] = RigidTransform_to_transform(RigidTransform(
                translation=vec3_to_np(block2_transform.p) + [0.0, -0.05, 0.0], rotation=quat_to_rot(self._ee_transforms[env_index].r), from_frame='ee_frame', to_frame='ee_frame')*
                self._env_offsets[env_index]['push_transformation'])

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_push_transforms[env_index])

        if t_step == 150:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])