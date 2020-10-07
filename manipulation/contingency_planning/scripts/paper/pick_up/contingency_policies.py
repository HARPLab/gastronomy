import numpy as np

from carbongym import gymapi
from carbongym_utils.policy import Policy
from carbongym_utils.math_utils import *

from min_jerk_interpolation import min_jerk_trajectory_interp

from autolab_core import RigidTransform


class PickUpBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta = np.zeros(3)

        self._time_horizon = 500

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._post_grasp_transforms = {}

    def set_x_y_theta(self, x_y_theta):
        self._x_y_theta = x_y_theta

    def __call__(self, scene, env_index, t_step, _):
        if env_index == 0:
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

                x_y_theta = self._x_y_theta
                pre_grasp_offset = [x_y_theta[0], 0.0, -x_y_theta[1]]

                self._pre_grasp_transforms[env_index] = gymapi.Transform(
                    #p=block2_transform.p + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                    p=np_to_vec3(block_center) + np_to_vec3([0.0, 0.2, 0.0]) + np_to_vec3(pre_grasp_offset),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta[2]))
                    #r=self._ee_transforms[env_index].r
                )

                self._grasp_transforms[env_index] = gymapi.Transform(
                    p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._pre_grasp_transforms[env_index].r
                )

                self._post_grasp_transforms[env_index] = gymapi.Transform(
                    p=self._grasp_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
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


class PickUpBlockOverheadOnlyPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta = np.zeros(3)

        self._time_horizon = 500

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._post_grasp_transforms = {}

    def set_x_y_theta(self, x_y_theta):
        self._x_y_theta = x_y_theta

    def __call__(self, scene, env_index, t_step, _):
        if env_index == 1:
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
                
                x_y_theta = self._x_y_theta
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

class PickUpBlockSideOnlyPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_tilt_dist = np.zeros(5)

        self._time_horizon = 600

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_grasp_transforms = {}

    def set_x_y_theta_tilt_dist(self, x_y_theta_tilt_dist):
        self._x_y_theta_tilt_dist = x_y_theta_tilt_dist

    def __call__(self, scene, env_index, t_step, _):
        if env_index == 1:
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
                
                x_y_theta_tilt_dist = self._x_y_theta_tilt_dist
                
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

class PickUpBlockOnlyFlatPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_dist = np.zeros(4)

        self._time_horizon = 500

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}
        self._push_transforms = {}
        self._post_pick_up_transforms = {}

    def set_x_y_theta_dist(self, x_y_theta_dist):
        self._x_y_theta_dist = x_y_theta_dist

    def __call__(self, scene, env_index, t_step, _):
        if env_index == 2:
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
                
                x_y_theta_dist = self._x_y_theta_dist
                
                theta = x_y_theta_dist[2]
                displacement = x_y_theta_dist[3]
                backoff_distance = -x_y_theta_dist[1] - 0.3

                pre_grasp_offset = [x_y_theta_dist[0], backoff_distance, 0.0]

                push_offset = [x_y_theta_dist[0], backoff_distance + displacement, 0.0]

                height = -0.03

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

                self._post_pick_up_transforms[env_index] = gymapi.Transform(
                    p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._push_transforms[env_index].r
                )

                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

            if t_step == 100:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

            if t_step == 200:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])

            if t_step == 300:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_pick_up_transforms[env_index])

            if t_step == 400:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class PickUpBlockOnlyTiltedPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs
        self._x_y_theta_tilt_dist = np.zeros(5)

        self._simulation_num = 0

        self._time_horizon = 450

    def reset(self):
        self._ee_transforms = {}
        self._pre_push_transforms = {}
        self._push_transforms = {}
        self._post_pick_up_transforms = {}

    def set_x_y_theta_tilt_dist(self, x_y_theta_tilt_dist):
        self._x_y_theta_tilt_dist = x_y_theta_tilt_dist

    def __call__(self, scene, env_index, t_step, _):
        if env_index == 2:
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
                
                x_y_theta_tilt_dist = self._x_y_theta_tilt_dist
                
                theta = x_y_theta_tilt_dist[2]
                tilt_angle = x_y_theta_tilt_dist[3]
                displacement = x_y_theta_tilt_dist[4]
                backoff_distance = -x_y_theta_tilt_dist[1] - 0.1*np.cos(np.abs(tilt_angle)) - 0.2

                pre_push_offset = [x_y_theta_tilt_dist[0], backoff_distance, 0.0]

                push_offset = [x_y_theta_tilt_dist[0], backoff_distance + displacement, 0.0]

                pre_push_height = 0.1*np.sin(np.abs(tilt_angle)) + 0.03

                height = - 0.03

                self._pre_push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(block_center) + np_to_vec3([0.0, pre_push_height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=pre_push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                    from_frame='ee_frame',to_frame='ee_frame'
                ))

                self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(block_center) + np_to_vec3([0.0, height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame'))

                self._post_pick_up_transforms[env_index] = gymapi.Transform(
                    p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.2, 0.0]),
                    r=self._push_transforms[env_index].r
                )

                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_push_transforms[env_index])

            if t_step == 150:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])

            if t_step == 250:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._post_pick_up_transforms[env_index])

            if t_step == 350:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])

class PickUpBlockTiltedWithFlipPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, n_envs):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._n_envs = n_envs

        self._simulation_num = 0

        self._time_horizon = 650

    def reset(self):
        self._ee_transforms = {}
        self._pre_push_transforms = {}
        self._push_transforms = {}
        self._push_interp_trajectories = {}
        self._pick_up_interp_trajectories = {}
        self._post_pick_up_transforms = {}
        self._flip_transforms = {}

    def set_x_y_theta_tilt_dist(self, x_y_theta_tilt_dist):
        self._x_y_theta_tilt_dist = x_y_theta_tilt_dist

    def __call__(self, scene, env_index, t_step, _):
        if env_index == 2:
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
                
                x_y_theta_tilt_dist = self._x_y_theta_tilt_dist
                
                theta = x_y_theta_tilt_dist[2]
                tilt_angle = x_y_theta_tilt_dist[3]
                displacement = x_y_theta_tilt_dist[4]
                backoff_distance = -x_y_theta_tilt_dist[1] - 0.1*np.cos(np.abs(tilt_angle)) - 0.2

                pre_push_offset = [x_y_theta_tilt_dist[0], backoff_distance, 0.0]

                push_offset = [x_y_theta_tilt_dist[0], backoff_distance + displacement, 0.0]

                pre_push_height = 0.1*np.sin(np.abs(tilt_angle)) + 0.03

                height = - 0.03

                self._pre_push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(block_center) + np_to_vec3([0.0, pre_push_height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=pre_push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    rotation=RigidTransform.quaternion_from_axis_angle([tilt_angle, 0, 0]),
                    from_frame='ee_frame',to_frame='ee_frame'
                ))

                self._push_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=np_to_vec3(block_center) + np_to_vec3([0.0, height, 0.0]),
                    r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, x_y_theta_tilt_dist[2]))), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    translation=push_offset, 
                    from_frame='ee_frame',to_frame='ee_frame'))

                self._post_pick_up_transforms[env_index] = gymapi.Transform(
                    p=self._push_transforms[env_index].p + np_to_vec3([0.0, 0.3, 0.0]),
                    r=self._push_transforms[env_index].r
                )

                self._flip_transforms[env_index] = RigidTransform_to_transform(transform_to_RigidTransform(gymapi.Transform(
                    p=self._post_pick_up_transforms[env_index].p,
                    r=self._post_pick_up_transforms[env_index].r), 
                    from_frame='ee_frame',to_frame='ee_frame') * RigidTransform(
                    rotation=RigidTransform.quaternion_from_axis_angle([np.pi/2, 0, 0]), 
                    from_frame='ee_frame',to_frame='ee_frame'))

                self._push_interp_trajectories[env_index] = min_jerk_trajectory_interp(self._pre_push_transforms[env_index], self._push_transforms[env_index], 100)
                self._pick_up_interp_trajectories[env_index] = min_jerk_trajectory_interp(self._push_transforms[env_index], self._post_pick_up_transforms[env_index], 100)

                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_push_transforms[env_index])

            if t_step >= 150 and t_step < 250:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_interp_trajectories[env_index][t_step-150])

            if t_step >= 300 and t_step < 400:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pick_up_interp_trajectories[env_index][t_step-300])

            if t_step == 450:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._flip_transforms[env_index])

            if t_step == 550:
                self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])