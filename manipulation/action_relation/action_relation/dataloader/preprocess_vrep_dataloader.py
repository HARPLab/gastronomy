import numpy as np
import argparse

import pickle
import json
import os

from action_relation.dataloader.vrep_dataloader import OctreeVoxels

def load_voxel_data(demo_dir):
    '''Load images for all demonstrations.

    demo_dir: str. Path to directory containing the demonstration data.
    '''
    if not os.path.exists(demo_dir):
        print("Dir does not exist: {}".format(demo_dir))
        return demo_idx_to_path_dict

    num_saved = 0
    for root, dirs, files in os.walk(demo_dir, followlinks=True):
        if 'all_img_data.json' in files:
            for i in range(10):
                data_pkl_path = os.path.join(root, '{}_img_data.pkl'.format(i))
                if not os.path.exists(data_pkl_path):
                    break
                with open(data_pkl_path, 'rb') as pkl_f:
                    data = pickle.load(pkl_f)
                
                voxel_data = OctreeVoxels(data['voxels_before'])
                valid_voxels = voxel_data.parse()
                if not valid_voxels:
                    continue
                full_3d = voxel_data.full_3d
                data['voxels_before_full'] = full_3d

                new_data_pkl_path = os.path.join(
                    root, '{}_img_data_full.pkl'.format(i))
                with open(new_data_pkl_path, 'wb') as pkl_f:
                    pickle.dump(data, pkl_f)
                num_saved += 1
                if num_saved % 50 == 0:
                    print("Did save {}".format(num_saved))

def clean_voxel_data(demo_dir):
    num_saved = 0
    for root, dirs, files in os.walk(demo_dir, followlinks=True):
        if 'all_img_data.json' in files:
            for i in range(10):
                data_pkl_path = os.path.join(root, '{}_img_data_full.pkl'.format(i))
                if not os.path.exists(data_pkl_path):
                    break
                with open(data_pkl_path, 'rb') as pkl_f:
                    data = pickle.load(pkl_f)
                
                voxel_data = {
                    'voxels_before': data['voxels_before'],
                    'voxels_before_full': data['voxels_before']
                }
                info_data = data['data']

                new_data_pkl_path = os.path.join(
                    root, '{}_img_data_info.pkl'.format(i))
                with open(new_data_pkl_path, 'wb') as pkl_f:
                    pickle.dump(info_data, pkl_f, protocol=2)
                new_data_pkl_path = os.path.join(
                    root, '{}_img_data_voxels.pkl'.format(i))
                with open(new_data_pkl_path, 'wb') as pkl_f:
                    pickle.dump(voxel_data, pkl_f, protocol=2)

                num_saved += 1
                if num_saved % 50 == 0:
                    print("Did save {}".format(num_saved))

def main(args):
    # load_voxel_data(args.dir)
    clean_voxel_data(args.dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get full 3d voxel data")
    parser.add_argument('--dir', type=str, required=True,
                        help='Dir path containing the data.')
    args = parser.parse_args()
    main(args)
