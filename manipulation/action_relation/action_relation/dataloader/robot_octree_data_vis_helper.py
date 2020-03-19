import numpy as np

import argparse
from itertools import combinations, permutations
import open3d as o3d
import os

from action_relation.dataloader.robot_octree_data import RobotVoxels
from action_relation.dataloader.robot_octree_data import SceneVoxels
from action_relation.dataloader.robot_octree_data import RobotAllPairVoxels
from action_relation.dataloader.robot_octree_data import create_robot_voxels_from_anchor_pcd_path
from action_relation.utils.open3d_utils import read_point_cloud, visualize


def get_voxels_for_point_cloud_pair(pcd1_path, pcd2_path):
    voxels = create_robot_voxels_from_anchor_pcd_path(
        pcd1_path, pcd2_path, False)
    status = voxels.init_voxel_index()

    voxels_pcd = voxels.convert_full3d_arr_to_open3d()
    if len(voxels_pcd) > 0:
        visualize(list(voxels_pcd.values()))
    # voxels.visualize_full3d()

    return status, voxels


def get_scene_voxels_for_pcd_path_list(obj_pcd_path_list): 
    scene_voxels = SceneVoxels(obj_pcd_path_list)
    status = scene_voxels.init_voxel_index()
    # voxels_pcd = scene_voxels.convert_full3d_arr_to_open3d()
    # visualize([voxels_pcd['scene']])
    scene_voxels.visualize_full3d()

def get_all_pair_voxels_for_pcd_path_list(obj_pcd_path_list):
    all_pair_scene_voxels = RobotAllPairVoxels(obj_pcd_path_list)
    obj_pair_list = list(permutations(range(len(obj_pcd_path_list)), 2))

    for obj_pair in obj_pair_list:
        _, voxels = all_pair_scene_voxels.init_voxels_for_pcd_pair(obj_pair[0], obj_pair[1])
        voxels_pcd_dict = voxels.convert_full3d_arr_to_open3d()
        # visualize([voxels_pcd_dict['anchor'], voxels_pcd_dict['other']])
        # visualize([voxels.anchor_pcd, voxels.other_pcd])
        voxels.visualize_full3d()


def main(args):
    if len(args.pcd_dir) > 0 and os.path.exists(args.pcd_dir):
        obj_pcd_path_list = [os.path.join(args.pcd_dir, p) 
                             for p in os.listdir(args.pcd_dir) 
                             if 'cloud_cluster' in p]
        
        if args.vis_type == 'obj_pair':
            for obj_pair in permutations(obj_pcd_path_list, 2):
                [anchor_path, other_path] = obj_pair
                _, voxels = get_voxels_for_point_cloud_pair(anchor_path, other_path)
        elif args.vis_type == 'all_obj_pair':
            get_all_pair_voxels_for_pcd_path_list(obj_pcd_path_list)
        elif args.vis_type == 'scene':
            get_scene_voxels_for_pcd_path_list(obj_pcd_path_list)
        else:
            raise ValueError(f"Invalid vis type: {args.vis_type}")


    elif len(args.anchor_pcd_path) > 0 and os.path.exists(args.anchor_pcd_path) \
        and len(args.other_pcd_path) > 0 and os.path.exists(args.other_pcd_path):
        anchor_path = args.anchor_pcd_path
        other_path = args.other_pcd_path
        _, voxels = get_voxels_for_point_cloud_pair(anchor_path, other_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test octree loading.")
    parser.add_argument('--anchor_pcd_path', type=str, default='',
                        help='Path to anchor pcd.')
    parser.add_argument('--other_pcd_path', type=str, default='',
                        help='Path to other pcd.')
    parser.add_argument('--pcd_dir', type=str, default='', 
                        help='Path to all pcds.')
    parser.add_argument('--vis_type', type=str, 
                        choices=['obj_pair', 'scene', 'all_obj_pair'], 
                        default='scene', 
                        help='Type of visualization to run.')
    args = parser.parse_args()

    main(args)
