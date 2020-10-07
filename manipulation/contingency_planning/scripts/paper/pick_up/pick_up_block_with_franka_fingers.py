import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from carbongym_utils.scene import GymScene
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, transform_to_np
from carbongym_utils.draw import draw_transforms

from franka_policies import PickUpBlockPolicy
from get_pan_pose import get_pan_pose
from random_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--franka_cfg', '-fc', type=str, default='cfg/paper/franka_fingers.yaml')
    parser.add_argument('--block_cfg_dir', '-bcd', type=str, default='cfg/paper/same_blocks/')
    parser.add_argument('--block_urdf_dir', '-bud', type=str, default='assets/same_blocks/')
    parser.add_argument('--pan_cfg_dir', '-pcd', type=str, default='cfg/paper/pans/')
    parser.add_argument('--pan_urdf_dir', '-pud', type=str, default='assets/pans/')
    parser.add_argument('--inputs', '-i', type=str, default='same_block_data/successful_lift_inputs_franka_fingers.npy')
    parser.add_argument('--block_num', '-b', type=int, default=0)
    parser.add_argument('--pan_num', '-p', type=int, default=0)
    parser.add_argument('--num_iterations', '-n', type=int, default=80000)
    parser.add_argument('--data_dir', '-dd', type=str, default='same_block_data/successful_lift/')
    args = parser.parse_args()
    franka_cfg = YamlConfig(args.franka_cfg)
    block_cfg = YamlConfig(args.block_cfg_dir+'block'+str(args.block_num)+'.yaml')

    use_pan = False
    if args.pan_num > 0 and args.pan_num < 7:
        pan_cfg = YamlConfig(args.pan_cfg_dir+'pan'+str(args.pan_num)+'.yaml')
        use_pan = True

    scene = GymScene(franka_cfg['scene'])
    
    block = GymURDFAsset(block_cfg['block']['urdf_path'], 
                         scene.gym, scene.sim,
                        assets_root=args.block_urdf_dir,
                        asset_options=block_cfg['block']['asset_options'],
                        shape_props=[block_cfg['block']['block1']['shape_props'],
                                     block_cfg['block']['block2']['shape_props'],
                                     block_cfg['block']['block3']['shape_props']])
    table = GymBoxAsset(scene.gym, scene.sim, **franka_cfg['table']['dims'], 
                        shape_props=franka_cfg['table']['shape_props'], 
                        asset_options=franka_cfg['table']['asset_options'])
    franka = GymFranka(franka_cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors')

    if use_pan:
        pan = GymURDFAsset(pan_cfg['pan']['urdf_path'], 
                           scene.gym, scene.sim,
                           assets_root=args.pan_urdf_dir,
                           asset_options=pan_cfg['pan']['asset_options'],
                           shape_props=pan_cfg['pan']['shape_props'])

    table_width = franka_cfg['table']['dims']['width']
    table_height = franka_cfg['table']['dims']['height']

    block_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0.42, table_height + 0.1, 0.0]
    ))
    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[table_width/3, table_height/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, table_height + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))
    if use_pan:
        pan_pose = get_pan_pose(args.pan_num, table_height)
    
    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose)
    scene.add_asset('block0', block, block_pose, collision_filter=2)
    if use_pan:
        scene.add_asset('pan0', pan, pan_pose)

    def custom_draws(scene):
        for i, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(i, 'franka0')

            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])

    def reset(env_index, env_ptr):
        block.set_rb_transforms(env_ptr, scene.ah_map[i]['block0'], [block_pose, block_pose, block_pose])
        franka.set_joints(env_ptr, scene.ah_map[env_index]['franka0'], franka.INIT_JOINTS)

    n_envs = franka_cfg['scene']['n_envs']
    if args.inputs is None:
        num_simulations = int(np.ceil(args.num_iterations / n_envs))

        num_iterations = num_simulations * n_envs

        x_y_z_thetas = get_random_x_y_z_thetas(num_iterations)
    else:
        x_y_z_thetas = np.load(args.inputs)
        num_simulations = int(np.floor(x_y_z_thetas.shape[0] / n_envs))

        num_iterations = num_simulations * n_envs
        x_y_z_thetas = x_y_z_thetas[:num_iterations]

    x_y_z_rx_ry_rz = get_random_x_y_z_rx_ry_rz(num_iterations)

    policy = PickUpBlockPolicy(franka, 'franka0', block, 'block0', n_envs, x_y_z_thetas, x_y_z_rx_ry_rz)

    initial_block_pose = np.zeros((num_iterations,7))

    pre_grasp_contact_forces = np.zeros((num_iterations,3))
    pre_grasp_block_pose = np.zeros((num_iterations,7))

    grasp_contact_forces = np.zeros((num_iterations,3))
    grasp_block_pose = np.zeros((num_iterations,7))
    
    post_grasp_contact_forces = np.zeros((num_iterations,3))
    post_grasp_block_pose = np.zeros((num_iterations,7))
    
    random_relative_movement_contact_forces = np.zeros((num_iterations,3))
    random_relative_movement_block_pose = np.zeros((num_iterations,7))
    
    post_release_block_pose = np.zeros((num_iterations,7))
    
    def save_block_pose(scene, t_step, t_sim, simulation_num):

        if t_step == 49:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                block_ah = scene.ah_map[env_idx]['block0']
                initial_block_pose[simulation_num * n_envs + env_idx] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif t_step == 199:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                franka_ah = scene.ah_map[env_idx]['franka0']
                pre_grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                block_ah = scene.ah_map[env_idx]['block0']
                pre_grasp_block_pose[simulation_num * n_envs + env_idx] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif t_step == 249:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                franka_ah = scene.ah_map[env_idx]['franka0']
                grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                block_ah = scene.ah_map[env_idx]['block0']
                grasp_block_pose[simulation_num * n_envs + env_idx] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif t_step == 349:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                franka_ah = scene.ah_map[env_idx]['franka0']
                post_grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                block_ah = scene.ah_map[env_idx]['block0']
                post_grasp_block_pose[simulation_num * n_envs + env_idx] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif t_step == 449:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                franka_ah = scene.ah_map[env_idx]['franka0']
                random_relative_movement_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                block_ah = scene.ah_map[env_idx]['block0']
                random_relative_movement_block_pose[simulation_num * n_envs + env_idx] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif t_step == 649:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                block_ah = scene.ah_map[env_idx]['block0']
                post_release_block_pose[simulation_num * n_envs + env_idx] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
                

    for simulation_num in range(num_simulations):
        print(simulation_num)
        policy.reset()
        for i, env_ptr in enumerate(scene.env_ptrs):
            reset(i, env_ptr)
        policy.set_simulation_num(simulation_num)

        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_block_pose(scene, t_step, t_sim, simulation_num=simulation_num)))

    save_file_name = args.data_dir+'block'+str(args.block_num)+'_pick_up_block_with_franka_fingers.npz'
    np.savez(save_file_name, x_y_z_thetas=x_y_z_thetas, 
                             x_y_z_rx_ry_rz=x_y_z_rx_ry_rz, 
                             initial_block_pose=initial_block_pose,
                             pre_grasp_contact_forces=pre_grasp_contact_forces,
                             pre_grasp_block_pose=pre_grasp_block_pose,
                             grasp_contact_forces=grasp_contact_forces,
                             grasp_block_pose=grasp_block_pose,
                             post_grasp_contact_forces=post_grasp_contact_forces,
                             post_grasp_block_pose=post_grasp_block_pose,
                             random_relative_movement_contact_forces=random_relative_movement_contact_forces,
                             random_relative_movement_block_pose=random_relative_movement_block_pose,
                             post_release_block_pose=post_release_block_pose)