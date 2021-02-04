import argparse

import numpy as np
import matplotlib.pyplot as plt
from autolab_core import YamlConfig, RigidTransform

from carbongym import gymapi
from carbongym_utils.scene import GymScene
from carbongym_utils.camera import GymCamera
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, transform_to_np
from carbongym_utils.draw import draw_transforms

from tong_policies import PickUpURDFTongsOverheadPolicy
from get_pan_pose import get_pan_pose
from random_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--franka_cfg', '-fc', type=str, default='cfg/paper/franka_tongs.yaml')
    parser.add_argument('--urdf_cfg', '-uc', type=str, default='cfg/paper/urdf.yaml')
    parser.add_argument('--urdf_file_path', '-uf', type=str)
    parser.add_argument('--urdf_dir', '-ud', type=str, default='../dexnet_meshes/')
    parser.add_argument('--pan_cfg_dir', '-pcd', type=str, default='cfg/paper/pans/')
    parser.add_argument('--pan_urdf_dir', '-pud', type=str, default='assets/pans/')
    parser.add_argument('--inputs', '-i', type=str)
    parser.add_argument('--pan_num', '-p', type=int, default=0)
    parser.add_argument('--num_iterations', '-n', type=int, default=500)
    parser.add_argument('--data_dir', '-dd', type=str, default='urdf_data/pick_up/tongs_overhead/')
    args = parser.parse_args()
    franka_cfg = YamlConfig(args.franka_cfg)
    urdf_cfg = YamlConfig(args.urdf_cfg)

    urdf_name = args.urdf_file_path[args.urdf_file_path.rfind('/')+1:-5]

    use_pan = False
    if args.pan_num > 0 and args.pan_num < 7:
        pan_cfg = YamlConfig(args.pan_cfg_dir+'pan'+str(args.pan_num)+'.yaml')
        use_pan = True

    scene = GymScene(franka_cfg['scene'])
    
    urdf = GymURDFAsset(args.urdf_file_path, 
                        scene.gym, scene.sim,
                        shape_props=urdf_cfg['urdf']['shape_props'], 
                        asset_options=urdf_cfg['urdf']['asset_options'],
                        assets_root=args.urdf_dir)
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

    global urdf_pose
    urdf_pose = RigidTransform_to_transform(RigidTransform(
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
    scene.add_asset('urdf0', urdf, urdf_pose)
    if use_pan:
        scene.add_asset('pan0', pan, pan_pose)

    def custom_draws(scene):
        for i, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(i, 'franka0')

            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])

    def reset(env_index, env_ptr, simulation_num):

        global urdf_pose
        urdf.set_rb_transforms(env_ptr, scene.ah_map[env_index]['urdf0'], [urdf_pose])
        franka.set_joints(env_ptr, scene.ah_map[env_index]['franka0'], franka.INIT_JOINTS)

    n_envs = franka_cfg['scene']['n_envs']
    if args.inputs is None:
        num_simulations = int(np.ceil(args.num_iterations / n_envs))

        num_iterations = num_simulations * n_envs

        x_y_thetas = get_random_x_y_thetas(num_iterations)
    else:
        input_data = np.load(args.inputs)

        x_y_thetas = input_data['X']
        results = input_data['Y']

        nonzero_trials = np.nonzero(results)

        if len(nonzero_trials[0]) == 0:
            quit()
        else:
            num_simulations=1
            print(nonzero_trials)
            x_y_thetas = x_y_thetas[nonzero_trials[0],:]
            first_nonzero = int(nonzero_trials[0][0] / 100)

            if first_nonzero == 1:
                urdf_pose = RigidTransform_to_transform(RigidTransform(
                    translation=[0.42, table_height + 0.1, 0.0],
                    rotation=RigidTransform.quaternion_from_axis_angle([np.pi/2, 0, 0])
                ))
            elif first_nonzero == 2:
                urdf_pose = RigidTransform_to_transform(RigidTransform(
                    translation=[0.42, table_height + 0.1, 0.0],
                    rotation=RigidTransform.quaternion_from_axis_angle([0, np.pi/2, 0])
                ))
            elif first_nonzero == 3:
                urdf_pose = RigidTransform_to_transform(RigidTransform(
                    translation=[0.42, table_height + 0.1, 0.0],
                    rotation=RigidTransform.quaternion_from_axis_angle([0, 0, np.pi/2])
                ))
            elif first_nonzero >= 4:
                urdf_pose = RigidTransform_to_transform(RigidTransform(
                    translation=[0.42, table_height + 0.1, 0.0],
                    rotation=RigidTransform.quaternion_from_axis_angle([(np.random.random() * 2 - 1) * np.pi, (np.random.random() * 2 - 1) * np.pi, (np.random.random() * 2 - 1) * np.pi])
                ))
        #num_simulations = int(np.floor(x_y_thetas.shape[0] / n_envs))

        num_iterations = num_simulations * n_envs
        #x_y_thetas = x_y_thetas[:num_iterations]

    cam = GymCamera(scene.gym, scene.sim, cam_props=franka_cfg['camera'])
    scene.add_standalone_camera('cam0', cam, gymapi.Vec3(0.4, 1.5, 0), gymapi.Vec3(0.3999999, 0.5, 0))
    

    policy = PickUpURDFTongsOverheadPolicy(franka, 'franka0', urdf, 'urdf0', n_envs, x_y_thetas)

    initial_urdf_pose = np.zeros((num_iterations,7))

    pre_grasp_contact_forces = np.zeros((num_iterations,3))
    pre_grasp_robot_pose = np.zeros((num_iterations,7))
    desired_pre_grasp_robot_pose = np.zeros((num_iterations,7))
    pre_grasp_urdf_pose = np.zeros((num_iterations,7))

    grasp_contact_forces = np.zeros((num_iterations,3))
    grasp_urdf_pose = np.zeros((num_iterations,7))
    
    post_grasp_contact_forces = np.zeros((num_iterations,3))
    post_grasp_urdf_pose = np.zeros((num_iterations,7))
    
    post_release_urdf_pose = np.zeros((num_iterations,7))
    
    global capture_once
    global starting_urdf_transform
    global settling_t_step
    capture_once = False
    settling_t_step = 150

    def save_urdf_pose(scene, t_step, t_sim, simulation_num):

        global capture_once
        global starting_urdf_transform
        global settling_t_step

        if settling_t_step == 150:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                if env_idx == 0:
                    urdf_ah = scene.ah_map[env_idx]['urdf0']

                    if t_step == 0:
                        starting_urdf_transform = urdf.get_rb_transforms(env_ptr, urdf_ah)[0]
                    else:
                        current_urdf_transform = urdf.get_rb_transforms(env_ptr, urdf_ah)[0]
                        if np.linalg.norm(vec3_to_np(current_urdf_transform.p) - vec3_to_np(starting_urdf_transform.p)) < 0.00001:
                            settling_t_step = t_step
                        else:
                            starting_urdf_transform = current_urdf_transform
        else:
            if t_step == settling_t_step+1:
                if not capture_once:
                    scene.render_cameras()
                    color, depth, seg = cam.frames(scene.ch_map[0]['cam0'], 'cam0')
                    plt.imsave(args.data_dir + urdf_name + '_' + str(simulation_num) + '_color.png', color.data)
                    plt.imsave(args.data_dir + urdf_name + '_' + str(simulation_num) + '_depth.png', depth.data)
                    plt.imsave(args.data_dir + urdf_name + '_' + str(simulation_num) + '_seg.png', seg.data)
                    capture_once = True
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    initial_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 399:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    franka_ah = scene.ah_map[env_idx]['franka0']
                    pre_grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)
                    desired_pre_grasp_robot_pose[simulation_num * n_envs + env_idx] = transform_to_np(franka.get_desired_ee_transform(env_idx, 'franka0'))
                    pre_grasp_robot_pose[simulation_num * n_envs + env_idx] = transform_to_np(franka.get_ee_transform(env_ptr, 'franka0'))
                
                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    pre_grasp_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 450:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    franka_ah = scene.ah_map[env_idx]['franka0']
                    grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    grasp_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 649:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    franka_ah = scene.ah_map[env_idx]['franka0']
                    post_grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    post_grasp_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 849:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    post_release_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])

    for simulation_num in range(num_simulations):
        print(simulation_num)
        policy.reset()
        for i, env_ptr in enumerate(scene.env_ptrs):
            reset(i, env_ptr, simulation_num)
        policy.set_simulation_num(simulation_num)

        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_urdf_pose(scene, t_step, t_sim, simulation_num=simulation_num)))
        capture_once = False
        settling_t_step = 150
        if simulation_num == 0:
            urdf_pose = RigidTransform_to_transform(RigidTransform(
                translation=[0.42, table_height + 0.1, 0.0],
                rotation=RigidTransform.quaternion_from_axis_angle([np.pi/2, 0, 0])
            ))
        elif simulation_num == 1:
            urdf_pose = RigidTransform_to_transform(RigidTransform(
                translation=[0.42, table_height + 0.1, 0.0],
                rotation=RigidTransform.quaternion_from_axis_angle([0, np.pi/2, 0])
            ))
        elif simulation_num == 2:
            urdf_pose = RigidTransform_to_transform(RigidTransform(
                translation=[0.42, table_height + 0.1, 0.0],
                rotation=RigidTransform.quaternion_from_axis_angle([0, 0, np.pi/2])
            ))
        elif simulation_num >= 3:
            urdf_pose = RigidTransform_to_transform(RigidTransform(
                translation=[0.42, table_height + 0.1, 0.0],
                rotation=RigidTransform.quaternion_from_axis_angle([(np.random.random() * 2 - 1) * np.pi, (np.random.random() * 2 - 1) * np.pi, (np.random.random() * 2 - 1) * np.pi])
            ))

    save_file_name = args.data_dir+urdf_name+'_tongs_overhead.npz'
    np.savez(save_file_name, x_y_thetas=x_y_thetas, 
                             initial_urdf_pose=initial_urdf_pose,
                             pre_grasp_contact_forces=pre_grasp_contact_forces,
                             pre_grasp_urdf_pose=pre_grasp_urdf_pose,
                             desired_pre_grasp_robot_pose=desired_pre_grasp_robot_pose,
                             pre_grasp_robot_pose=pre_grasp_robot_pose,
                             grasp_contact_forces=grasp_contact_forces,
                             grasp_urdf_pose=grasp_urdf_pose,
                             post_grasp_contact_forces=post_grasp_contact_forces,
                             post_grasp_urdf_pose=post_grasp_urdf_pose,
                             post_release_urdf_pose=post_release_urdf_pose)
    print('Done')