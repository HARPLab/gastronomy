import numpy as np
import argparse

import os
import pickle
import sys

sys.path.append(os.path.abspath('../../../'))

from sim.sim_vrep.utilities.math_utils import Sat3D, sample_from_edges
from math_utils import create_dense_voxels_from_sparse
from math_utils import get_transformation_matrix_for_vrep_transform_list

def remove_iter_dirs(demo_dir):
    pass

def create_dense_voxels_for_demo_dir(demo_dir):
    print(f"Will process dir: {demo_dir}")
    demo_idx = 0
    for root, dirs, files in os.walk(demo_dir, followlinks=False):
        # Sort the data order so that we do not have randomness associated
        # with it.
        dirs.sort()
        files.sort()

        if 'all_img_data.json' in files:
            # Filter data based on the anchor object.
            all_img_data_json_path = os.path.join(root, 'all_img_data.json')
            all_init_action_idx = [int(f.split('_')[0]) 
                                   for f in files if 'img_data.pkl' in f]

            for i in sorted(all_init_action_idx):
                voxel_pkl_path = os.path.join(root, f'{i}_voxel_data.pkl')
                assert os.path.exists(voxel_pkl_path)
                dense_voxel_pkl_path = os.path.join(root, f'{i}_dense_voxel_data.pkl')

                # assert not os.path.exists(dense_voxel_pkl_path)
                if os.path.exists(dense_voxel_pkl_path):
                    continue

                data_pkl_path = os.path.join(root, '{}_img_data.pkl'.format(i))
                assert os.path.exists(data_pkl_path), \
                    f"Data pickle does not exist: {data_pkl_path}"

                with open(voxel_pkl_path, 'rb') as pkl_f:
                    voxel_data = pickle.load(pkl_f)

                with open(data_pkl_path, 'rb') as data_pkl_f:
                    scene_info = pickle.load(data_pkl_f)
                    scene_info = scene_info['data']
                
                dense_voxel_data = dict()
                dense_voxel_data['voxels_before'] = dict(
                    anchor=create_dense_voxels_from_sparse(
                        voxel_data['voxels_before']['anchor'], 
                        get_transformation_matrix_for_vrep_transform_list(
                            scene_info['before']['anchor_T_matrix'])
                    ),
                    other=create_dense_voxels_from_sparse(
                        voxel_data['voxels_before']['other'], 
                        get_transformation_matrix_for_vrep_transform_list(
                            scene_info['before']['other_T_matrix'])
                    )
                )

                if voxel_data.get('voxels_after') is not None:
                    dense_voxel_data['voxels_after'] = dict(
                        other=create_dense_voxels_from_sparse(
                            voxel_data['voxels_after']['other'], 
                            get_transformation_matrix_for_vrep_transform_list(
                                scene_info['after']['other_T_matrix'])
                        )
                    )

                # Now save it.
                pkl_dense_voxel_path = os.path.join(
                    root, f'{i}_dense_voxel_data.pkl')
                with open(pkl_dense_voxel_path, 'wb') as pkl_f:
                    pickle.dump(dense_voxel_data, pkl_f, protocol=2)

                demo_idx += 1
                if demo_idx % 100 == 0:
                    print("Did process: {}".format(demo_idx))


def main(args):
    for demo_dir in args.dir:
        assert os.path.exists(demo_dir), f"Path does not exist: {demo_dir}"
        create_dense_voxels_for_demo_dir(demo_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dense voxels from sparse')
    parser.add_argument('--dir', action='append', required=True, 
                        help='Path to dir with data.')
    args = parser.parse_args()
    import pprint
    pprint.pprint(args.__dict__)
    main(args)
