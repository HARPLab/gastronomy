import numpy as np

from carbongym import gymapi
from carbongym_utils.policy import Policy
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, quat_to_rot, rpy_to_quat, vec3_to_np, quat_to_np

from autolab_core import RigidTransform


class PickUpBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 700

        self._env_offsets = {}

    def reset(self):
        self._ee_transforms = {}
        self._pre_grasp_transforms = {}
        self._grasp_transforms = {}

    def set_next_grasp(self, env_offsets):
        self._env_offsets = env_offsets

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
            block2_idx = self._block.rb_names_map['block2']
            block3_idx = self._block.rb_names_map['block3']
            block1_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block1_idx]
            block2_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block2_idx]
            block3_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block3_idx]
            
            self._pre_grasp_transforms[env_index] = gymapi.Transform(
                p=block2_transform.p + np_to_vec3([0.0, 0.4, 0.0]) + self._env_offsets[env_index],
                r=self._ee_transforms[env_index].r * rpy_to_quat((0, 0, 0))
            )

            self._grasp_transforms[env_index] = gymapi.Transform(
                p=self._pre_grasp_transforms[env_index].p - np_to_vec3([0.0, 0.25, 0.0]),
                r=self._pre_grasp_transforms[env_index].r
            )

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 100:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 200:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 300:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 400:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 500:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 600:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

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

class FlipBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name):
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 420

        self._env_offsets = {}

    def reset(self):
        self._ee_transforms = {}
        self._pre_push_transforms1 = {}
        self._pre_push_transforms2 = {}
        self._push_transforms = {}
        self._lift_transforms = {}

    def set_next_push(self, env_offsets):
        self._env_offsets = env_offsets

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 0:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._ee_transforms[env_index])
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 50:
            self._ee_transforms[env_index] = self._franka.get_ee_transform(env_ptr, self._franka_name)
            block_ah = scene.ah_map[env_index][self._block_name]

            block1_idx = self._block.rb_names_map['block1']
            block2_idx = self._block.rb_names_map['block2']
            block3_idx = self._block.rb_names_map['block3']
            block1_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block1_idx]
            block2_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block2_idx]
            block3_transform = self._block.get_rb_transforms(env_ptr, block_ah)[block3_idx]

            self._pre_push_transforms1[env_index] = RigidTransform_to_transform(RigidTransform(
                translation=vec3_to_np(block2_transform.p) + [0.0, 0.25, 0.0], rotation=quat_to_rot(self._ee_transforms[env_index].r), from_frame='ee_frame', to_frame='ee_frame')*
                self._env_offsets[env_index]['pre_push_transformation'])

            self._pre_push_transforms2[env_index] = RigidTransform_to_transform(RigidTransform(
                translation=vec3_to_np(block2_transform.p) + [0.0, 0.15, 0.0], rotation=quat_to_rot(self._ee_transforms[env_index].r), from_frame='ee_frame', to_frame='ee_frame')*
                self._env_offsets[env_index]['pre_push_transformation'])

            self._push_transforms[env_index] = RigidTransform_to_transform(RigidTransform(
                translation=vec3_to_np(block2_transform.p) + [0.0, 0.1, 0.0], rotation=quat_to_rot(self._ee_transforms[env_index].r), from_frame='ee_frame', to_frame='ee_frame')*
                self._env_offsets[env_index]['push_transformation'])

            self._lift_transforms[env_index] = RigidTransform_to_transform(RigidTransform(
                translation=vec3_to_np(block2_transform.p) + [0.0, 0.33, 0.0], rotation=quat_to_rot(self._ee_transforms[env_index].r), from_frame='ee_frame', to_frame='ee_frame')*
                self._env_offsets[env_index]['push_transformation'])

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_push_transforms1[env_index])

        if t_step == 150:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_push_transforms2[env_index])

        if t_step == 250:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._push_transforms[env_index])

        if t_step == 300:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 350:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._lift_transforms[env_index])