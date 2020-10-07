import numpy as np

from carbongym import gymapi
from carbongym_utils.policy import Policy
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, rpy_to_quat, vec3_to_np, quat_to_np
from min_jerk_interpolation import min_jerk_trajectory_interp, min_jerk_gripper_interp

from autolab_core import RigidTransform


class PickUpThenFlipPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_theta_rots):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_rots = x_y_theta_rots

        self._simulation_num = 0

        self._time_horizon = 750

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._grasp_trajectory = {}
        self._gripper_width = {}
        self._post_grasp_transforms = {}
        self._post_grasp_trajectory = {}
        self._post_release_transforms = {}

    def set_simulation_num(self, simulation_num):
        self._simulation_num = simulation_num

    def set_x_y_theta_rots(self, x_y_theta_rots):
        self._x_y_theta_rots = x_y_theta_rots

    def __call__(self, scene, env_index, t_step, _):
        #if(env_index == 0):
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

            x_y_theta_rot = self._x_y_theta_rots[self._simulation_num * self._n_envs + env_index]
            pre_grasp_offset = [x_y_theta_rot[0], 0.0, -x_y_theta_rot[1]]

            self._pre_grasp_transforms[env_index] = gymapi.Transform(
                #p=block2_transform.p + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_rot[2]))
                #r=self._ee_transforms[env_index].r
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.2, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._post_grasp_transforms[env_index] = gymapi.Transform(
                p=self._grasp_transforms[env_index].p + np_to_vec3([0.0, 0.15, 0.0]),
                r=self._grasp_transforms[env_index].r * rpy_to_quat((0, x_y_theta_rot[3], 0))
            )

            self._post_release_transforms[env_index] = gymapi.Transform(
                p=self._post_grasp_transforms[env_index].p + np_to_vec3([0.0, 0.15, 0.0]),
                r=self._post_grasp_transforms[env_index].r 
            )

            self._grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._pre_grasp_transforms[env_index], self._grasp_transforms[env_index], 100)
            self._post_grasp_trajectory[env_index] = min_jerk_trajectory_interp(self._grasp_transforms[env_index], self._post_grasp_transforms[env_index], 100)

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step >= 150 and t_step < 250:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_trajectory[env_index][t_step-150])

        if t_step == 300:
            self._franka.close_grippers(env_ptr, ah)

        if t_step >= 350 and t_step < 450:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_grasp_trajectory[env_index][t_step-350])

        if t_step == 500:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 550:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_release_transforms[env_index])

        if t_step == 650:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])



class FlipBlockPolicy(Policy):

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

    def set_x_y_z_thetas(self, x_y_z_thetas):
        self._x_y_z_thetas = x_y_z_thetas

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
                #p=block2_transform.p + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                #r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_z_theta[3]))
                r=self._ee_transforms[env_index].r
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.2, 0.0]),
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


class PushBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs, x_y_z_thetas, x_y_z_rx_ry_rz):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_z_thetas = x_y_z_thetas
        self._x_y_z_rx_ry_rz = x_y_z_rx_ry_rz

        self._simulation_num = 0

        self._time_horizon = 250

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

            block2_idx = self._block.rb_names_map['block2']
            block2_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block2_idx]
            
            random_num = np.random.randint(0,1000)

            x_y_z_theta = self._x_y_z_thetas[self._simulation_num * self._n_envs + env_index]
            x_y_z_rx_ry_rz = self._x_y_z_rx_ry_rz[self._simulation_num * self._n_envs + env_index]
            pre_grasp_offset = [x_y_z_theta[0], 0.0, -x_y_z_theta[1]]
            random_grasp_trans = [x_y_z_rx_ry_rz[0], x_y_z_rx_ry_rz[2], -x_y_z_rx_ry_rz[1]]

            self._pre_grasp_transforms[env_index] = gymapi.Transform(
                p=block2_transform.p + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_z_theta[3]))
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.2, 0.0]),
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

class PushBlockPolicy(Policy):

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
            
            self._pre_push_transforms[env_index] = gymapi.Transform(
                p=block2_transform.p + self._env_offsets[env_index]['pre_push_offset'],
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, 0))
            )

            self._push_transforms[env_index] = gymapi.Transform(
                p=block2_transform.p + self._env_offsets[env_index]['push_offset'],
                r=self._pre_push_transforms[env_index].r
            )

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_push_transforms[env_index])

        if t_step == 150:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])