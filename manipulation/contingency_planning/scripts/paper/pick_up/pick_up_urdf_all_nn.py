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

from franka_policies import PickUpURDFPolicy
from get_pan_pose import get_pan_pose
from random_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--franka_cfg', '-fc', type=str, default='cfg/paper/franka_fingers.yaml')
    parser.add_argument('--urdf_cfg', '-uc', type=str, default='cfg/paper/urdf.yaml')
    parser.add_argument('--urdf_file_path', '-uf', type=str)
    parser.add_argument('--urdf_dir', '-ud', type=str, default='../dexnet_meshes/')
    parser.add_argument('--pan_cfg_dir', '-pcd', type=str, default='cfg/paper/pans/')
    parser.add_argument('--pan_urdf_dir', '-pud', type=str, default='assets/pans/')
    parser.add_argument('--franka_nn', type=str, default='trained_urdf_networks/franka_fingers_convolution_model.h5')
    parser.add_argument('--tong_overhead_nn', type=str, default='trained_urdf_networks/tongs_overhead_convolution_model.h5')
    parser.add_argument('--tong_side_nn', type=str, default='trained_urdf_networks/tongs_side_convolution_model.h5')
    parser.add_argument('--spatula_tilted_nn', type=str, default='trained_urdf_networks/spatula_tilted_with_flip_convolution_model.h5')
    parser.add_argument('--contingency_nn_dir', type=str, default='same_block_data/contingency_data/')
    parser.add_argument('--inputs', '-i', type=str)
    parser.add_argument('--pan_num', '-p', type=int, default=0)
    parser.add_argument('--num_iterations', '-n', type=int, default=1)
    parser.add_argument('--data_dir', '-dd', type=str, default='urdf_data/')
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
        translation=[0.42, table_height + 0.1, 0.0],
        rotation=RigidTransform.quaternion_from_axis_angle([(np.random.random() * 2 - 1) * np.pi, (np.random.random() * 2 - 1) * np.pi, (np.random.random() * 2 - 1) * np.pi])
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
    scene.add_asset('urdf0', urdf, urdf_pose, collision_filter=2)
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

    franka_nn_model = load_model(args.franka_nn)
    tong_overhead_nn_model = load_model(args.tong_overhead_nn)
    tong_side_nn_model = load_model(args.tong_side_nn)
    spatula_tilted_nn_model = load_model(args.spatula_tilted_nn)

    n_envs = franka_cfg['scene']['n_envs']
    
    num_simulations = int(np.ceil(args.num_iterations / n_envs))

    num_iterations = num_simulations * n_envs

    x_y_thetas = get_random_x_y_thetas(num_iterations)

    probabilities = franka_nn_model.predict(x_y_thetas)
    sorted_idx = np.argsort(-probabilities, 0)
    sorted_x_y_thetas = x_y_thetas[sorted_idx].reshape(-1,3)
    probabilities = probabilities[sorted_idx].reshape(-1,1)
    sorted_x_y_thetas = sorted_x_y_thetas[:500,:]
    #probabilities = probabilities[:500] 
    franka_x_y_thetas = np.hstack((np.zeros((500,1)), sorted_x_y_thetas, np.zeros((500,2))))

    x_y_thetas = get_random_x_y_thetas(num_iterations)

    probabilities = tong_overhead_nn_model.predict(x_y_thetas)
    sorted_idx = np.argsort(-probabilities, 0)
    sorted_x_y_thetas = x_y_thetas[sorted_idx].reshape(-1,3)
    probabilities = probabilities[sorted_idx].reshape(-1,1)
    sorted_x_y_thetas = sorted_x_y_thetas[:500,:]
    #probabilities = probabilities[:500] 
    tong_overhead_x_y_thetas = np.hstack((np.ones((500,1)), sorted_x_y_thetas, np.zeros((500,2))))

    x_y_theta_dist_tilts = get_random_x_y_theta_dist_tilts(num_iterations)

    probabilities = tong_side_nn_model.predict(x_y_theta_dist_tilts)
    sorted_idx = np.argsort(-probabilities, 0)
    sorted_x_y_theta_dist_tilts = x_y_theta_dist_tilts[sorted_idx].reshape(-1,5)
    probabilities = probabilities[sorted_idx].reshape(-1,1)
    sorted_x_y_theta_dist_tilts = sorted_x_y_theta_dist_tilts[:500,:]
    #probabilities = probabilities[:500] 
    tong_side_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 2, sorted_x_y_theta_dist_tilts))

    x_y_theta_dists = get_random_x_y_theta_dists(num_iterations)

    probabilities = spatula_flat_nn_model.predict(x_y_theta_dists)
    sorted_idx = np.argsort(-probabilities, 0)
    sorted_x_y_theta_dists = x_y_theta_dists[sorted_idx].reshape(-1,4)
    probabilities = probabilities[sorted_idx].reshape(-1,1)
    sorted_x_y_theta_dists = sorted_x_y_theta_dists[:500,:]
    #probabilities = probabilities[:500] 
    spatula_flat_x_y_theta_dists = np.hstack((np.ones((500,1)) * 3, sorted_x_y_theta_dists, np.zeros((500,1))))

    x_y_theta_dist_tilts = get_random_x_y_theta_dist_tilts(num_iterations)

    probabilities = spatula_tilted_nn_model.predict(x_y_theta_dist_tilts)
    sorted_idx = np.argsort(-probabilities, 0)
    sorted_x_y_theta_dist_tilts = x_y_theta_dist_tilts[sorted_idx].reshape(-1,5)
    probabilities = probabilities[sorted_idx].reshape(-1,1)
    sorted_x_y_theta_dist_tilts = sorted_x_y_theta_dist_tilts[:500,:]
    #probabilities = probabilities[:500] 
    spatula_tilted_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 4, sorted_x_y_theta_dist_tilts))

    sorted_x_y_thetas = np.vstack((franka_x_y_thetas, tong_overhead_x_y_thetas, tong_side_x_y_theta_dist_tilts, spatula_flat_x_y_theta_dists, spatula_tilted_x_y_theta_dist_tilts))
    
    cam = GymCamera(scene.gym, scene.sim, cam_props=franka_cfg['camera'])
    scene.add_standalone_camera('cam0', cam, gymapi.Vec3(0.4, 1.5, 0), gymapi.Vec3(0.3999999, 0.5, 0))
    

    policy = PickUpURDFPolicy(franka, 'franka0', urdf, 'urdf0', n_envs, x_y_thetas)

    initial_urdf_pose = np.zeros((num_iterations,7))

    pre_grasp_contact_forces = np.zeros((num_iterations,3))
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
            elif t_step == settling_t_step + 149:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    franka_ah = scene.ah_map[env_idx]['franka0']
                    pre_grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    pre_grasp_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 199:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    franka_ah = scene.ah_map[env_idx]['franka0']
                    grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    grasp_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 299:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    franka_ah = scene.ah_map[env_idx]['franka0']
                    post_grasp_contact_forces[simulation_num * n_envs + env_idx] = franka.get_ee_ct_forces(env_ptr, franka_ah)

                    urdf_ah = scene.ah_map[env_idx]['urdf0']
                    post_grasp_urdf_pose[simulation_num * n_envs + env_idx] = transform_to_np(urdf.get_rb_transforms(env_ptr, urdf_ah)[0])
            elif t_step == settling_t_step + 349:
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
            

    save_file_name = args.data_dir+urdf_name+'_all_nn.npz'
    np.savez(save_file_name, x_y_thetas=x_y_thetas, 
                             initial_urdf_pose=initial_urdf_pose,
                             pre_grasp_contact_forces=pre_grasp_contact_forces,
                             pre_grasp_urdf_pose=pre_grasp_urdf_pose,
                             grasp_contact_forces=grasp_contact_forces,
                             grasp_urdf_pose=grasp_urdf_pose,
                             post_grasp_contact_forces=post_grasp_contact_forces,
                             post_grasp_urdf_pose=post_grasp_urdf_pose,
                             post_release_urdf_pose=post_release_urdf_pose)