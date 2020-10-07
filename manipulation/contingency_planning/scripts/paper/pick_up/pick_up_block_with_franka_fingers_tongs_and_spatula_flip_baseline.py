import argparse
import glob

import numpy as np
from autolab_core import YamlConfig, RigidTransform
import keras
from keras.models import load_model

from carbongym_utils.scene import GymScene
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, transform_to_np
from carbongym_utils.draw import draw_transforms

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import distance

from contingency_policies import *
from get_pan_pose import get_pan_pose
from random_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--franka_cfg', '-fc', type=str, default='cfg/paper/franka_fingers.yaml')
    parser.add_argument('--franka_spatula_cfg', '-fsc', type=str, default='cfg/paper/franka_spatula.yaml')
    parser.add_argument('--franka_tong_cfg', '-ftc', type=str, default='cfg/paper/franka_tongs.yaml')
    parser.add_argument('--block_cfg_dir', '-bcd', type=str, default='cfg/paper/same_blocks/')
    parser.add_argument('--block_urdf_dir', '-bud', type=str, default='assets/same_blocks/')
    parser.add_argument('--pan_cfg_dir', '-pcd', type=str, default='cfg/paper/pans/')
    parser.add_argument('--pan_urdf_dir', '-pud', type=str, default='assets/pans/')
    parser.add_argument('--franka_nn', type=str, default='same_block_data/franka_fingers_pick_up_only_trained_model.h5')
    parser.add_argument('--tong_overhead_nn', type=str, default='same_block_data/tongs_overhead_pick_up_only_trained_model.h5')
    parser.add_argument('--tong_side_nn', type=str, default='same_block_data/tongs_side_pick_up_only_trained_model.h5')
    parser.add_argument('--spatula_tilted_nn', type=str, default='same_block_data/spatula_tilted_with_flip_pick_up_only_trained_model.h5')
    parser.add_argument('--contingency_nn_dir', type=str, default='same_block_data/contingency_data/')
    parser.add_argument('--block_num', '-b', type=int, default=0)
    parser.add_argument('--pan_num', '-p', type=int, default=0)
    parser.add_argument('--num_iterations', '-n', type=int, default=50000)
    parser.add_argument('--num_simulations', '-s', type=int, default=30)
    parser.add_argument('--data_dir', '-dd', type=str, default='same_block_data/successful_lift/')
    args = parser.parse_args()
    franka_cfg = YamlConfig(args.franka_cfg)
    spatula_franka_cfg = YamlConfig(args.franka_spatula_cfg)
    tong_franka_cfg = YamlConfig(args.franka_tong_cfg)
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
    spatula_franka = GymFranka(spatula_franka_cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors',
                       assets_root='assets', urdf_path=spatula_franka_cfg['franka']['urdf_path'])
    tong_franka = GymFranka(tong_franka_cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors',
                       assets_root='assets', urdf_path=tong_franka_cfg['franka']['urdf_path'])

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
    scene.add_asset('franka0', franka, franka_pose, envs=[0])
    scene.add_asset('franka1', tong_franka, franka_pose, envs=[1])
    scene.add_asset('franka2', spatula_franka, franka_pose, envs=[2])
    scene.add_asset('block0', block, block_pose, collision_filter=2)
    if use_pan:
        scene.add_asset('pan0', pan, pan_pose)

    def custom_draws(scene):
        for i, env_ptr in enumerate(scene.env_ptrs):
            if i == 0:
                ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
                desired_ee_transform = franka.get_desired_ee_transform(i, 'franka0')

                draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])
            elif i == 1:
                ee_transform = tong_franka.get_ee_transform(env_ptr, 'franka1')
                desired_ee_transform = tong_franka.get_desired_ee_transform(i, 'franka1')

                draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])
            elif i == 2:
                ee_transform = spatula_franka.get_ee_transform(env_ptr, 'franka2')
                desired_ee_transform = spatula_franka.get_desired_ee_transform(i, 'franka2')

                draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])

    def reset(env_index, env_ptr):
        if(env_index == 0):
            block.set_rb_transforms(env_ptr, scene.ah_map[i]['block0'], [block_pose, block_pose, block_pose])
            franka.set_joints(env_ptr, scene.ah_map[env_index]['franka0'], franka.INIT_JOINTS)
        elif(env_index == 1):
            block.set_rb_transforms(env_ptr, scene.ah_map[i]['block0'], [block_pose, block_pose, block_pose])
            tong_franka.set_joints(env_ptr, scene.ah_map[env_index]['franka1'], tong_franka.INIT_JOINTS)
        elif(env_index == 2):
            block.set_rb_transforms(env_ptr, scene.ah_map[i]['block0'], [block_pose, block_pose, block_pose])
            spatula_franka.set_joints(env_ptr, scene.ah_map[env_index]['franka2'], spatula_franka.INIT_JOINTS)

    def resample_points(num_samples, probabilities, x_y_thetas):
        probabilities_sum = np.sum(probabilities, 0)
        normalized_probabilities = probabilities / probabilities_sum

        random_samples = np.random.random(num_samples)

        resampled_points = np.zeros((0,x_y_thetas.shape[1]))

        for i in range(normalized_probabilities.shape[0]):
            random_samples -= normalized_probabilities[i]
            nonzero_samples = np.nonzero(random_samples < 0)
            resampled_points = np.vstack((resampled_points,np.repeat(x_y_thetas[i].reshape(1,-1),nonzero_samples[0].shape[0], axis=0)))
            random_samples[nonzero_samples[0]] += 1

        # resampled_points[:,0] += np.random.normal(size=num_samples) * 0.001
        # resampled_points[:,1] += np.random.normal(size=num_samples) * 0.001
        # resampled_points[:,2] += np.random.normal(size=num_samples) * 0.001

        np.random.shuffle(resampled_points)
        return resampled_points


    n_envs = franka_cfg['scene']['n_envs']
    # num_simulations = int(np.ceil(args.num_iterations / n_envs))

    # num_iterations = num_simulations * n_envs
    num_iterations = args.num_iterations

    franka_nn_model = load_model(args.franka_nn)
    tong_overhead_nn_model = load_model(args.tong_overhead_nn)
    tong_side_nn_model = load_model(args.tong_side_nn)
    spatula_tilted_nn_model = load_model(args.spatula_tilted_nn)

    suffices = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_tilted_with_flip']

    num_skills = len(suffices)
    num_successful_inputs = {}
    for suffix in suffices:
        num_successful_inputs[suffix] = 0

    file_paths = glob.glob('same_block_data/pick_up_only/*.npy')

    successful_lift_data = {}

    for i in range(1,31):
        for file_path in file_paths:
            if str(i) in file_path:
                file_data = np.load(file_path)
                for suffix in suffices:
                    if suffix in file_path:
                        if num_successful_inputs[suffix] == 0:
                            successful_lift_data[suffix] = file_data.reshape(-1,1)
                            num_successful_inputs[suffix] = successful_lift_data[suffix].shape[0]
                        else:
                            successful_lift_data[suffix] = np.hstack((successful_lift_data[suffix], file_data.reshape(-1,1)))

    skill_inputs = {}

    for suffix_idx in range(num_skills):
        suffix = suffices[suffix_idx]
        skill_inputs[suffix] = np.load('same_block_data/successful_lift_inputs_pick_up_only_with_' + suffix + '.npy')
        success_indices = np.nonzero(np.sum(successful_lift_data[suffix], 1).reshape(-1))
        successful_lift_data[suffix] = successful_lift_data[suffix][success_indices[0]]
        print(successful_lift_data[suffix].shape)
        if suffix_idx < 2:
            skill_inputs[suffix] = np.hstack((np.ones((successful_lift_data[suffix].shape[0],1)) * suffix_idx, skill_inputs[suffix][success_indices[0]], np.zeros((successful_lift_data[suffix].shape[0],2))))
        else:
            skill_inputs[suffix] = np.hstack((np.ones((successful_lift_data[suffix].shape[0],1)) * suffix_idx, skill_inputs[suffix][success_indices[0]]))

    # Reduce to only first 100 skills
    training_data_successful_lift = np.zeros((0,67))
    training_data_skill_inputs = np.zeros((0,6))
    for suffix_idx in range(num_skills):
        print(successful_lift_data[suffix][:100,:].shape)
        suffix = suffices[suffix_idx]
        training_data_skill_inputs = np.vstack((training_data_skill_inputs,skill_inputs[suffix][:100,:]))
        training_data_successful_lift = np.vstack((training_data_successful_lift,successful_lift_data[suffix][:100,:]))

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

    x_y_theta_dist_tilts = get_random_x_y_theta_dist_tilts(num_iterations)

    probabilities = spatula_tilted_nn_model.predict(x_y_theta_dist_tilts)
    sorted_idx = np.argsort(-probabilities, 0)
    sorted_x_y_theta_dist_tilts = x_y_theta_dist_tilts[sorted_idx].reshape(-1,5)
    probabilities = probabilities[sorted_idx].reshape(-1,1)
    sorted_x_y_theta_dist_tilts = sorted_x_y_theta_dist_tilts[:500,:]
    #probabilities = probabilities[:500] 
    spatula_tilted_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 3, sorted_x_y_theta_dist_tilts))

    sorted_x_y_thetas = np.vstack((franka_x_y_thetas, tong_overhead_x_y_thetas, tong_side_x_y_theta_dist_tilts, spatula_tilted_x_y_theta_dist_tilts))

    # print(np.max(sorted_x_y_thetas, axis=0))
    # print(np.min(sorted_x_y_thetas, axis=0))

    probabilities = np.ones((2000,1))

    resampled_points = resample_points(1000, probabilities, sorted_x_y_thetas)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # Data for three-dimensional scattered points
    # #p = ax.scatter3D(sorted_x_y_thetas[:,0], sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], c=probabilities.flatten());
    # p = ax.scatter3D(resampled_points[:,0], resampled_points[:,1], resampled_points[:,2] + np.random.normal(size=1000) * 0.1, c=ones_probabilities);
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('theta (rad)')
    # fig.colorbar(p, ax=ax)
    # plt.show()

    # print(suffices[int(resampled_points[0,0])])

    # for i in range(4):
    #     density = np.zeros((200,200))
    #     x_ind = 0
    #     skill_resampled_points = np.nonzero(resampled_points[:,0] == i)

    #     for x in np.arange(-0.1,0.1,0.001):
    #         y_ind = 0
    #         for y in np.arange(-0.1,0.1,0.001):
    #             dist = np.sqrt(np.square(resampled_points[skill_resampled_points[0],1] - x) + np.square(resampled_points[skill_resampled_points[0],2] - y))
    #             density[y_ind,x_ind] = np.count_nonzero(dist < 0.005)
    #             y_ind += 1
    #         x_ind += 1

    #     xv, yv = np.meshgrid(np.arange(-0.1,0.1,0.001), np.arange(-0.1,0.1,0.001))

    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')

    #     ax.plot_surface(xv, yv, density,cmap='viridis', edgecolor='none')
    #     if i == int(resampled_points[0,0]):
    #         ax.scatter(resampled_points[0,1], resampled_points[0,2], np.max(np.max(density,1),0), c='red');
    #     ax.set_title('Skill: ' + suffices[i])
    #     ax.set_xlabel('x (m)')
    #     ax.set_ylabel('y (m)')
    #     ax.set_zlabel('Num Samples')
    #     ax.view_init(azim=0, elev=90)
    #     plt.show()

    # current_input = resampled_points[0]

    # skill_id = int(current_input[0])

    # xs = sorted_x_y_thetas[:,:3] - current_input[:3]
    # thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,3] - current_input[3]), np.cos(sorted_x_y_thetas[:,3] - current_input[3]))
    # dists = sorted_x_y_thetas[:,4] - current_input[4]
    # tilts = np.arctan2(np.sin(sorted_x_y_thetas[:,5] - current_input[5]), np.cos(sorted_x_y_thetas[:,5] - current_input[5]))
    # relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

    # new_probabilities = np.ones(probabilities.shape)

    # for suffix_idx in range(num_skills):
    #     if suffix_idx < skill_id:
    #         suffix1 = suffices[suffix_idx]
    #         suffix2 = suffices[skill_id]
    #         key = suffix1 + '_' + suffix2
    #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
    #     elif suffix_idx == skill_id:
    #         key = suffices[suffix_idx]
    #         num_inputs = skill_input_nums[skill_id]
    #     else:
    #         suffix1 = suffices[skill_id]
    #         suffix2 = suffices[suffix_idx]
    #         key = suffix1 + '_' + suffix2
    #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
        
    #     new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs])
        
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     p = ax.scatter3D(sorted_x_y_thetas[500*suffix_idx:500*(suffix_idx+1),1], sorted_x_y_thetas[500*suffix_idx:500*(suffix_idx+1),2], sorted_x_y_thetas[500*suffix_idx:500*(suffix_idx+1),0], c=new_probabilities[500*suffix_idx:500*(suffix_idx+1)].flatten());
    #     ax.set_title('Success Probabilities for Skill: ' + suffices[suffix_idx])
    #     ax.set_xlabel('x (m)')
    #     ax.set_ylabel('y (m)')
    #     ax.set_zlabel('Skill id')
    #     fig.colorbar(p, ax=ax)
    #     ax.view_init(azim=0, elev=90)
    #     plt.show()

    # new_probabilities = np.ones(probabilities.shape)

    # for suffix_idx in range(num_skills):
    #     if suffix_idx < skill_id:
    #         suffix1 = suffices[suffix_idx]
    #         suffix2 = suffices[skill_id]
    #         key = suffix1 + '_' + suffix2
    #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
    #     elif suffix_idx == skill_id:
    #         key = suffices[suffix_idx]
    #         num_inputs = skill_input_nums[skill_id]
    #     else:
    #         suffix1 = suffices[skill_id]
    #         suffix2 = suffices[suffix_idx]
    #         key = suffix1 + '_' + suffix2
    #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
        
    #     new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs])
        
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     p = ax.scatter3D(sorted_x_y_thetas[500*suffix_idx:500*(suffix_idx+1),1], sorted_x_y_thetas[500*suffix_idx:500*(suffix_idx+1),2], sorted_x_y_thetas[500*suffix_idx:500*(suffix_idx+1),0], c=new_probabilities[500*suffix_idx:500*(suffix_idx+1)].flatten());
    #     ax.set_title('Failure Probabilities for Skill: ' + suffices[suffix_idx])
    #     ax.set_xlabel('x (m)')
    #     ax.set_ylabel('y (m)')
    #     ax.set_zlabel('Skill id')
    #     fig.colorbar(p, ax=ax)
    #     ax.view_init(azim=0, elev=90)
    #     plt.show()

    franka_policy = PickUpBlockPolicy(franka, 'franka0', block, 'block0', n_envs)
    tong_overhead_policy = PickUpBlockOverheadOnlyPolicy(tong_franka, 'franka1', block, 'block0', n_envs)
    tong_side_policy = PickUpBlockSideOnlyPolicy(tong_franka, 'franka1', block, 'block0', n_envs)
    spatula_tilted_policy = PickUpBlockTiltedWithFlipPolicy(spatula_franka, 'franka2', block, 'block0', n_envs)

    initial_block_pose = np.zeros((num_iterations,7))
    
    post_pick_up_block_pose = np.zeros((num_iterations,7))
    
    post_release_block_pose = np.zeros((num_iterations,7))
    
    def save_block_pose(scene, t_step, t_sim, simulation_num, skill_id):

        if skill_id == 0:
            env_id = 0
        elif skill_id == 1 or skill_id == 2:
            env_id = 1
        elif skill_id == 3 or skill_id == 4:
            env_id = 2

        if t_step == 49:
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                if env_idx == env_id:
                    block_ah = scene.ah_map[env_idx]['block0']
                    initial_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        
        if skill_id == 0 or skill_id == 1:
            if t_step == 349:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    if env_idx == env_id:
                        block_ah = scene.ah_map[env_idx]['block0']
                        post_pick_up_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
            elif t_step == 499:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    if env_idx == env_id:
                        block_ah = scene.ah_map[env_idx]['block0']
                        post_release_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif skill_id == 2:
            if t_step == 449:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    if env_idx == env_id:
                        block_ah = scene.ah_map[env_idx]['block0']
                        post_pick_up_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
            elif t_step == 599:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    if env_idx == env_id:
                        block_ah = scene.ah_map[env_idx]['block0']
                        post_release_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        elif skill_id == 3:
            if t_step == 449:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    if env_idx == env_id:
                        block_ah = scene.ah_map[env_idx]['block0']
                        post_pick_up_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
            elif t_step == 649:
                for env_idx, env_ptr in enumerate(scene.env_ptrs):
                    if env_idx == env_id:
                        block_ah = scene.ah_map[env_idx]['block0']
                        post_release_block_pose[simulation_num] = transform_to_np(block.get_rb_transforms(env_ptr, block_ah)[1])
        
    block_similarities = np.zeros((1,67))
    block_probabilities = np.ones((1,67))
    block_data_probabilities = {}

    for simulation_num in range(args.num_simulations):
        print(simulation_num)
        franka_policy.reset()
        tong_overhead_policy.reset()
        tong_side_policy.reset()
        spatula_tilted_policy.reset()

        for i, env_ptr in enumerate(scene.env_ptrs):
            reset(i, env_ptr)
        current_input = resampled_points[simulation_num]
        skill_id = int(current_input[0])
        
        if skill_id == 0:
            franka_policy.set_x_y_theta(current_input[1:4])
            scene.run(time_horizon=franka_policy.time_horizon, policy=franka_policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_block_pose(scene, t_step, t_sim, simulation_num=simulation_num, skill_id=skill_id)))

        elif skill_id == 1:
            tong_overhead_policy.set_x_y_theta(current_input[1:4])
            scene.run(time_horizon=tong_overhead_policy.time_horizon, policy=tong_overhead_policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_block_pose(scene, t_step, t_sim, simulation_num=simulation_num, skill_id=skill_id)))

        elif skill_id == 2:
            x_y_theta_tilt_dist = np.hstack((current_input[1:4], current_input[5], current_input[4]))
            tong_side_policy.set_x_y_theta_tilt_dist(x_y_theta_tilt_dist)
            scene.run(time_horizon=tong_side_policy.time_horizon, policy=tong_side_policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_block_pose(scene, t_step, t_sim, simulation_num=simulation_num, skill_id=skill_id)))

        elif skill_id == 3:
            x_y_theta_tilt_dist = np.hstack((current_input[1:4], current_input[5], current_input[4]))
            spatula_tilted_policy.set_x_y_theta_tilt_dist(x_y_theta_tilt_dist)
            scene.run(time_horizon=spatula_tilted_policy.time_horizon, policy=spatula_tilted_policy, custom_draws=custom_draws, cb=(lambda scene, t_step, t_sim: save_block_pose(scene, t_step, t_sim, simulation_num=simulation_num, skill_id=skill_id)))

        if((post_pick_up_block_pose[simulation_num,1] - initial_block_pose[simulation_num,1]) > 0.1):
            print("Successful Lift")

            suffix = suffices[skill_id]
            closest_skill_id = np.argmin(distance.cdist(current_input.reshape(1,-1),training_data_skill_inputs[100*(skill_id):100*(skill_id+1),:]))
            print(closest_skill_id)
            closest_skill_data = training_data_successful_lift[100*(skill_id)+closest_skill_id,:].reshape(1,-1)
            block_similarities += closest_skill_data
            block_probabilities = block_similarities / (simulation_num + 1)
            block_probability_sum = np.sum(block_probabilities)

            

            # xs = sorted_x_y_thetas[:,:3] - current_input[:3]
            # thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,3] - current_input[3]), np.cos(sorted_x_y_thetas[:,3] - current_input[3]))
            # dists = sorted_x_y_thetas[:,4] - current_input[4]
            # tilts = np.arctan2(np.sin(sorted_x_y_thetas[:,5] - current_input[5]), np.cos(sorted_x_y_thetas[:,5] - current_input[5]))
            # relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

            # new_probabilities = np.ones(probabilities.shape)

            # for suffix_idx in range(num_skills):
            #     if suffix_idx < skill_id:
            #         suffix1 = suffices[suffix_idx]
            #         suffix2 = suffices[skill_id]
            #         key = suffix1 + '_' + suffix2
            #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
            #     elif suffix_idx == skill_id:
            #         key = suffices[suffix_idx]
            #         num_inputs = skill_input_nums[skill_id]
            #     else:
            #         suffix1 = suffices[skill_id]
            #         suffix2 = suffices[suffix_idx]
            #         key = suffix1 + '_' + suffix2
            #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
                
            #     new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs])


            #new_probabilities = success_contingency_nn_model.predict(relative_transforms)
            #probabilities = probabilities * 0.7 + 0.3 * new_probabilities            
            #probabilities = probabilities * new_probabilities
            probabilities /= np.max(probabilities)

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # p = ax.scatter3D(sorted_x_y_thetas[:,0], sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], c=new_probabilities[:].flatten());
            # o = ax.scatter3D(resampled_points[simulation_num,0], resampled_points[simulation_num,1], resampled_points[simulation_num,2], c='red');
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('y (m)')
            # ax.set_zlabel('theta (rad)')
            # fig.colorbar(p, ax=ax)
            # plt.show()

            # sorted_idx = np.argsort(-probabilities[:], 0)
            # new_sorted_x_y_thetas = sorted_x_y_thetas[sorted_idx].reshape(-1,6)
            # probabilities[:] = probabilities[sorted_idx].reshape(-1,1)
            # sorted_x_y_thetas[:] = new_sorted_x_y_thetas

            resampled_points = resample_points(1000, probabilities, sorted_x_y_thetas)


            fig = plt.figure()
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(training_data_skill_inputs[:,1], training_data_skill_inputs[:,2], training_data_skill_inputs[:,0], c=individual_block_probabilities[:].flatten());
            o = ax.scatter3D(resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], resampled_points[simulation_num+1,0], c='red');
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Skill id')
            fig.colorbar(p, ax=ax)
            plt.show()
            print(suffices[int(resampled_points[simulation_num+1,0])])

            individual_block_probabilities = np.sum((training_data_successful_lift*block_probabilities), axis=1) / block_probability_sum

            # policy.set_x_y_thetas(resampled_points)

            # print(resampled_points[simulation_num+1])

            # density = np.zeros((200,200))
            # x_ind = 0

            # for x in np.arange(-0.1,0.1,0.001):
            #     y_ind = 0
            #     for y in np.arange(-0.1,0.1,0.001):
            #         dist = np.sqrt(np.square(resampled_points[:,0] - x) + np.square(resampled_points[:,1] - y))
            #         density[y_ind,x_ind] = np.count_nonzero(dist < 0.005)
            #         y_ind += 1
            #     x_ind += 1

            # xv, yv = np.meshgrid(np.arange(-0.1,0.1,0.001), np.arange(-0.1,0.1,0.001))

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')

            # ax.plot_surface(xv, yv, density,cmap='viridis', edgecolor='none')
            # ax.scatter(resampled_points[simulation_num+1,0], resampled_points[simulation_num+1,1], np.max(np.max(density,1),0), c='red');
            # ax.set_title('Surface plot')
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('y (m)')
            # ax.set_zlabel('Num Samples')
            # ax.view_init(azim=0, elev=90)
            # plt.show()
        else:
            # relative_positions = sorted_x_y_thetas[simulation_num:,:2] - sorted_x_y_thetas[simulation_num,:2]
            # relative_thetas = np.arctan2(np.sin(sorted_x_y_thetas[simulation_num:,2] - sorted_x_y_thetas[simulation_num,2]), 
            #                              np.cos(sorted_x_y_thetas[simulation_num:,2] - sorted_x_y_thetas[simulation_num,2]))
            # cur_x_y_theta = sorted_x_y_thetas[simulation_num,:]            
            # cur_x_y_thetas = np.repeat(cur_x_y_theta.reshape(1,-1), relative_thetas.shape[0], axis=0)
            # #relative_transforms = np.hstack((cur_x_y_thetas, relative_positions, relative_thetas.reshape(-1,1)))
            # relative_transforms = np.hstack((relative_positions, relative_thetas.reshape(-1,1)))
            # print(relative_transforms.shape)

            # new_probabilities = contingency_nn_model.predict(relative_transforms)
            # #probabilities[simulation_num+1:] = probabilities[simulation_num+1:] * 0.8 + 0.2 * new_probabilities
            # probabilities[simulation_num:] = probabilities[simulation_num:] * new_probabilities
            # probabilities_sum = np.sum(probabilities[simulation_num:], axis=0)
            # probabilities[simulation_num:] /= probabilities_sum

            # sorted_idx = np.argsort(-probabilities[simulation_num:], 0) + simulation_num
            # new_sorted_x_y_thetas = sorted_x_y_thetas[sorted_idx].reshape(-1,3)
            # probabilities[simulation_num:] = probabilities[sorted_idx].reshape(-1,1)
            # sorted_x_y_thetas[simulation_num:] = new_sorted_x_y_thetas

            # relative_positions = sorted_x_y_thetas[:,:2] - sorted_x_y_thetas[simulation_num,:2]
            # relative_thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,2] - sorted_x_y_thetas[simulation_num,2]), 
            #                              np.cos(sorted_x_y_thetas[:,2] - sorted_x_y_thetas[simulation_num,2]))
            # cur_x_y_theta = sorted_x_y_thetas[simulation_num,:]            
            # cur_x_y_thetas = np.repeat(cur_x_y_theta.reshape(1,-1), relative_thetas.shape[0], axis=0)
            # relative_transforms = np.hstack((cur_x_y_thetas, relative_positions, relative_thetas.reshape(-1,1)))

            # xs = sorted_x_y_thetas[:,:3] - resampled_points[simulation_num,:3]
            # thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,3] - resampled_points[simulation_num,3]), np.cos(sorted_x_y_thetas[:,3] - resampled_points[simulation_num,3]))
            # tilts = np.arctan2(np.sin(sorted_x_y_thetas[:,4] - resampled_points[simulation_num,4]), np.cos(sorted_x_y_thetas[:,4] - resampled_points[simulation_num,4]))
            # dists = sorted_x_y_thetas[:,5] - resampled_points[simulation_num,5]
            # relative_transforms = np.hstack((xs, thetas.reshape(-1,1), tilts.reshape(-1,1), dists.reshape(-1,1)))
            # relative_positions = sorted_x_y_thetas[:,:2] - resampled_points[simulation_num,:2]
            # relative_thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,2] - resampled_points[simulation_num,2]), 
            #                              np.cos(sorted_x_y_thetas[:,2] - resampled_points[simulation_num,2]))
            # cur_x_y_theta = resampled_points[simulation_num,:]            
            # cur_x_y_thetas = np.repeat(cur_x_y_theta.reshape(1,-1), relative_thetas.shape[0], axis=0)
            # relative_transforms = np.hstack((relative_positions, relative_thetas.reshape(-1,1)))

            #new_probabilities = failure_contingency_nn_model.predict(relative_transforms)
            
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # p = ax.scatter3D(sorted_x_y_thetas[:,0], sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], c=new_probabilities[:].flatten());
            # o = ax.scatter3D(resampled_points[simulation_num,0], resampled_points[simulation_num,1], resampled_points[simulation_num,2], c='red');
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('y (m)')
            # ax.set_zlabel('theta (rad)')
            # fig.colorbar(p, ax=ax)
            # plt.show()

            # xs = sorted_x_y_thetas[:,:3] - current_input[:3]
            # thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,3] - current_input[3]), np.cos(sorted_x_y_thetas[:,3] - current_input[3]))
            # dists = sorted_x_y_thetas[:,4] - current_input[4]
            # tilts = np.arctan2(np.sin(sorted_x_y_thetas[:,5] - current_input[5]), np.cos(sorted_x_y_thetas[:,5] - current_input[5]))
            # relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

            # new_probabilities = np.ones(probabilities.shape)

            # for suffix_idx in range(num_skills):
            #     if suffix_idx < skill_id:
            #         suffix1 = suffices[suffix_idx]
            #         suffix2 = suffices[skill_id]
            #         key = suffix1 + '_' + suffix2
            #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
            #     elif suffix_idx == skill_id:
            #         key = suffices[suffix_idx]
            #         num_inputs = skill_input_nums[skill_id]
            #     else:
            #         suffix1 = suffices[skill_id]
            #         suffix2 = suffices[suffix_idx]
            #         key = suffix1 + '_' + suffix2
            #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
                
            #     new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs])

            suffix = suffices[skill_id]
            closest_skill_id = np.argmin(distance.cdist(current_input.reshape(1,-1),training_data_skill_inputs[100*(skill_id):100*(skill_id+1),:]))
            print(closest_skill_id)
            closest_skill_data = training_data_successful_lift[100*(skill_id)+closest_skill_id,:].reshape(1,-1)
            block_similarities = block_similarities + np.logical_not(closest_skill_data)
            block_probabilities = block_similarities / (simulation_num + 1)
            block_probability_sum = np.sum(block_probabilities)

            individual_block_probabilities = np.sum(training_data_successful_lift*block_probabilities) / block_probability_sum

            #probabilities = probabilities * 0.7 + 0.3 * new_probabilities
            #probabilities = probabilities * new_probabilities
            probabilities /= np.max(probabilities)

            # sorted_idx = np.argsort(-probabilities[:], 0)
            # new_sorted_x_y_thetas = sorted_x_y_thetas[sorted_idx].reshape(-1,6)
            # probabilities[:] = probabilities[sorted_idx].reshape(-1,1)
            # sorted_x_y_thetas[:] = new_sorted_x_y_thetas

            resampled_points = resample_points(1000, probabilities, sorted_x_y_thetas)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(training_data_skill_inputs[:,1], training_data_skill_inputs[:,2], training_data_skill_inputs[:,0], c=individual_block_probabilities[:].flatten());
            o = ax.scatter3D(resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], resampled_points[simulation_num+1,0], c='red');
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Skill id')
            fig.colorbar(p, ax=ax)
            plt.show()
            print(suffices[int(resampled_points[simulation_num+1,0])])


            # policy.set_x_y_thetas(resampled_points)

            # print(resampled_points)


            # density = np.zeros((200,200))
            # x_ind = 0

            # for x in np.arange(-0.1,0.1,0.001):
            #     y_ind = 0
            #     for y in np.arange(-0.1,0.1,0.001):
            #         dist = np.sqrt(np.square(resampled_points[:,0] - x) + np.square(resampled_points[:,1] - y))
            #         density[y_ind,x_ind] = np.count_nonzero(dist < 0.005)
            #         y_ind += 1
            #     x_ind += 1

            # xv, yv = np.meshgrid(np.arange(-0.1,0.1,0.001), np.arange(-0.1,0.1,0.001))

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')

            # ax.plot_surface(xv, yv, density,cmap='viridis', edgecolor='none')
            # ax.scatter(resampled_points[simulation_num+1,0], resampled_points[simulation_num+1,1], np.max(np.max(density,1),0), c='red');
            # ax.set_title('Surface plot')
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('y (m)')
            # ax.set_zlabel('Num Samples')
            # ax.view_init(azim=0, elev=90)
            # plt.show()

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # # Data for three-dimensional scattered points
            # p = ax.scatter3D(sorted_x_y_thetas[:,0], sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], c=probabilities[:].flatten());
            # #p = ax.scatter3D(resampled_points[:,0], resampled_points[:,1], resampled_points[:,2], c=ones_probabilities);
            # o = ax.scatter3D(resampled_points[simulation_num+1,0], resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], c='red');
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('y (m)')
            # ax.set_zlabel('theta (rad)')
            # fig.colorbar(p, ax=ax)
            # plt.show()

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # p = ax.scatter3D(sorted_x_y_thetas[:,0], sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], c=probabilities[:].flatten());
            # o = ax.scatter3D(resampled_points[simulation_num+1,0], resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], c='red');
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('y (m)')
            # ax.set_zlabel('theta (rad)')
            # fig.colorbar(p, ax=ax)
            # plt.show()




    # save_file_name = args.data_dir+'block'+str(args.block_num)+'_pick_up_block_with_franka_fingers.npz'
    # np.savez(save_file_name, x_y_thetas=x_y_thetas, 
    #                          x_y_rx_ry_rz=x_y_rx_ry_rz, 
    #                          initial_block_pose=initial_block_pose,
    #                          pre_grasp_contact_forces=pre_grasp_contact_forces,
    #                          pre_grasp_block_pose=pre_grasp_block_pose,
    #                          grasp_contact_forces=grasp_contact_forces,
    #                          grasp_block_pose=grasp_block_pose,
    #                          post_grasp_contact_forces=post_grasp_contact_forces,
    #                          post_grasp_block_pose=post_grasp_block_pose,
    #                          random_relative_movement_contact_forces=random_relative_movement_contact_forces,
    #                          random_relative_movement_block_pose=random_relative_movement_block_pose,
    #                          post_release_block_pose=post_release_block_pose)