import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from carbongym_utils.scene import GymScene
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3
from carbongym_utils.draw import draw_transforms

from block_policies import PushBlockPolicy
from get_pan_pose import get_pan_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/push_block.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    
    block = GymURDFAsset(cfg['block']['urdf_path'], scene.gym, scene.sim,
                        asset_options=cfg['block']['asset_options'],
                        shape_props=cfg['block']['shape_props'])
    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors')
    if cfg['pan']['use_pan']:
        pan = GymURDFAsset(cfg['pan']['urdf_path'], scene.gym, scene.sim,
                            asset_options=cfg['pan']['asset_options'],
                            shape_props=cfg['pan']['shape_props'])

    block_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0.42, cfg['table']['dims']['height'] + 0.1, 0.0]
    ))
    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))
    if cfg['pan']['use_pan']:
        pan_pose = get_pan_pose(cfg['pan']['urdf_path'], cfg['table']['dims']['height'])
    
    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose)
    scene.add_asset('block0', block, block_pose, collision_filter=2)
    if cfg['pan']['use_pan']:
        scene.add_asset('pan0', pan, pan_pose)

    policy = PushBlockPolicy(franka, 'franka0', block, 'block0')

    def custom_draws(scene):
        for i, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(i, 'franka0')

            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])

    def reset(env_index, env_ptr):
        franka.set_joints(env_ptr, scene.ah_map[env_index]['franka0'], franka.INIT_JOINTS)

    env_offsets = {}
    for i, env_ptr in enumerate(scene.env_ptrs):
        env_offsets[i] = {}

    while True:        

        for i, env_ptr in enumerate(scene.env_ptrs):
            reset(i, env_ptr)
            block.set_rb_transforms(env_ptr, scene.ah_map[i]['block0'], [block_pose, block_pose, block_pose])
            pre_push_z_offset = (np.random.rand()*2 - 1) * 0.05
            if pre_push_z_offset > 0:
                pre_push_z_offset += 0.05
                push_z_offset = -pre_push_z_offset
            else:
                pre_push_z_offset -= 0.05
                push_z_offset = -pre_push_z_offset
            env_offsets[i]['pre_push_offset'] = np_to_vec3([(np.random.rand()*2 - 1) * 0.05, 0.0, pre_push_z_offset])
            env_offsets[i]['push_offset'] = np_to_vec3([(np.random.rand()*2 - 1) * 0.05, 0.0, push_z_offset])
        policy.reset()
        policy.set_next_push(env_offsets)

        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws)
