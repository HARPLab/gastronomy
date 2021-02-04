import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from carbongym_utils.scene import GymScene
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, transform_to_np
from carbongym_utils.draw import draw_transforms

from cabinet_policies import CabinetOpenPolicy
from random_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/cabinet.yaml')
    parser.add_argument('--urdf_path', '-u', type=str, default='cabinet_right_hinge.urdf')
    parser.add_argument('--inputs', '-i', type=str, default='cabinets/successful_inputs_cabinet_right_hinge.npy')
    parser.add_argument('--num_iterations', '-n', type=int, default=10000)
    parser.add_argument('--data_dir', '-dd', type=str, default='cabinets/')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    
    cabinet = GymURDFAsset(args.urdf_path, scene.gym, scene.sim, 
                        dof_props=cfg['cabinet']['dof_props'],
                        asset_options=cfg['cabinet']['asset_options'],
                        assets_root=cfg['local_assets_path'])
    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors')

    cabinet_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0.8, cfg['table']['dims']['height'] + 0.2, 0.0],
        rotation=RigidTransform.quaternion_from_axis_angle([0, np.pi, 0])
    ))
    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))
    
    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose)
    scene.add_asset('cabinet0', cabinet, cabinet_pose, collision_filter=2)

    def custom_draws(scene):
        for i, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(i, 'franka0')

            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])

    def reset(env_index, env_ptr):
        cabinet.set_joints(env_ptr, scene.ah_map[env_index]['cabinet0'], np.zeros(cabinet.n_dofs))
        franka.set_joints(env_ptr, scene.ah_map[env_index]['franka0'], franka.INIT_JOINTS)

    n_envs = cfg['scene']['n_envs']
    if args.inputs is None:
        num_simulations = int(np.ceil(args.num_iterations / n_envs))

        num_iterations = num_simulations * n_envs

        x_y_zs = get_random_x_y_zs(num_iterations)
    else:
        x_y_zs = np.load(args.inputs)
        
        num_simulations = int(np.floor(x_y_zs.shape[0] / n_envs))

        num_iterations = num_simulations * n_envs
        x_y_zs = x_y_zs[:num_iterations]

    policy = CabinetOpenPolicy(franka, 'franka0', cabinet, 'cabinet0', n_envs, x_y_zs)

    initial_handle_pose = np.zeros((num_iterations,7))
    
    final_handle_pose = np.zeros((num_iterations,7))
    
    def save_handle_pose(scene, t_step, t_sim, simulation_num):

        if t_step == 50:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                door_ah = scene.ah_map[env_idx]['cabinet0']

                handle_idx = cabinet.rb_names_map['door_handle']
                pole_idx = cabinet.rb_names_map['cabinet_door']
                handle_transform = cabinet.get_rb_transforms(env_ptr, door_ah)[handle_idx]
                pole_transform = cabinet.get_rb_transforms(env_ptr, door_ah)[pole_idx]
                initial_handle_pose[simulation_num * n_envs + env_idx] = transform_to_np(handle_transform)
        elif t_step == 699:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                door_ah = scene.ah_map[env_idx]['cabinet0']

                handle_idx = cabinet.rb_names_map['door_handle']
                pole_idx = cabinet.rb_names_map['cabinet_door']
                handle_transform = cabinet.get_rb_transforms(env_ptr, door_ah)[handle_idx]
                pole_transform = cabinet.get_rb_transforms(env_ptr, door_ah)[pole_idx]
                final_handle_pose[simulation_num * n_envs + env_idx] = transform_to_np(handle_transform)
        

    for simulation_num in range(num_simulations):
        print(simulation_num)
        policy.reset()
        for i, env_ptr in enumerate(scene.env_ptrs):
            reset(i, env_ptr)
        policy.set_simulation_num(simulation_num)

        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_handle_pose(scene, t_step, t_sim, simulation_num=simulation_num)))

    save_file_name = args.data_dir+args.urdf_path[:-5]+'.npz'
    np.savez(save_file_name, x_y_zs=x_y_zs, 
                             initial_handle_pose=initial_handle_pose,
                             final_handle_pose=final_handle_pose)