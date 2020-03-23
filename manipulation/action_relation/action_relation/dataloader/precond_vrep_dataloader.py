import numpy as np

import csv
import h5py
import os
import ipdb
import pickle
import glob
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import time
import json

from utils import data_utils
from utils import image_utils
from utils.colors import bcolors
from action_relation.dataloader.octree_data import OctreeVoxels, OctreePickleLoader
from action_relation.dataloader.octree_data import CanonicalOctreeVoxels
from action_relation.dataloader.vrep_scene_geometry import VrepSceneGeometry
from action_relation.dataloader.vrep_scene_geometry import get_index_for_action
from action_relation.utils.data_utils import get_dist_matrix_for_data

from action_relation.dataloader.vrep_dataloader import get_img_info_and_voxel_object_for_path

import torch

def load_emb_data_from_h5_path(h5_path, max_data_size=None):
    h5f = h5py.File(h5_path, 'r')
    emb_data = data_utils.recursively_get_dict_from_group(h5f)
    # emb_data keys are stored as str in hdf5 file. Convert them to integers
    # for direct comparison with the file data.
    emb_data_int_key = {}
    for k in emb_data.keys():
        emb_data_int_key[int(k)] = emb_data[k]
    assert len(emb_data) == len(emb_data_int_key), "Emb data size changed."
    emb_data = emb_data_int_key

    data_keys = sorted(emb_data.keys())
    if max_data_size is None:
        max_data_size = len(data_keys)
    else:
        max_data_size = min(max_data_size, len(data_keys))
    
    data_dict = {}
    for i in range(max_data_size):
        key = data_keys[i]
        val = emb_data[key]
        data_dict[i] = {
            'path': str(val['path']),
            'before_img_path': str(val['before_img_path']),
            'precond_label': int(val['precond_label']),
            'emb': val['emb'],
        }
        data_pkl_path = data_dict[i]['path']
        with open(data_pkl_path, 'rb') as pkl_f:
            data = pickle.load(pkl_f)
        vrep_geom = VrepSceneGeometry(data['data'], data_pkl_path,
                                      parse_before_scene_only=True)
        data_dict[i]['vrep_geom'] = vrep_geom
    
    return data_dict

class PrecondVoxelDataloader(object):
    def __init__(self, 
                 config,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1):
        self._config = config
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir
         
        self.max_train_data_size = config.args.max_train_data_size
        self.max_test_data_size = config.args.max_test_data_size
 
        self.train_idx_to_data_dict = dict()
        for train_dir in self.train_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                train_dir, 
                max_data_size=self.max_train_data_size, 
                max_action_idx=1,
                demo_idx=len(self.train_idx_to_data_dict))
            self.train_idx_to_data_dict.update(idx_to_data_dict)

        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))

        self.test_idx_to_data_dict = dict()
        for test_dir in self.test_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                test_dir, 
                max_data_size=self.max_test_data_size, 
                max_action_idx=1,
                demo_idx=len(self.test_idx_to_data_dict))
            self.test_idx_to_data_dict.update(idx_to_data_dict)
            
        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))
        
        self.train_h5_data_dict = None
        self.test_h5_data_dict = None


    def load_emb_data(self, train_h5_path, test_h5_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            if k >= self.max_train_data_size - 1:
                continue
            h5_data = self.train_h5_data_dict[k]
            assert h5_data['path'] == train_idx_data['path']
        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            assert h5_data['path'] == test_idx_data['path']

        
    def load_voxel_data(self, demo_dir, max_data_size=None, max_action_idx=10, 
                        demo_idx=0):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()

            if 'all_img_data.json' in files:

                # Filter data based on the anchor object.
                # all_img_data_json_path = os.path.join(root, 'all_img_data.json')
                # with open(all_img_data_json_path, 'r') as json_f:
                #     pass

                for i in range(0, max_action_idx):
                    data_pkl_path = os.path.join(root, '{}_img_data.pkl'.format(i))
                    if not os.path.exists(data_pkl_path):
                        break
                    with open(data_pkl_path, 'rb') as pkl_f:
                        data = pickle.load(pkl_f)
                    voxel_pkl_path = os.path.join(root, '{}_voxel_data.pkl'.format(i))
                    
                    vrep_geom = VrepSceneGeometry(data['data'], data_pkl_path,
                                                  parse_before_scene_only=True)
                    img_info, voxel_obj = get_img_info_and_voxel_object_for_path(
                        data_pkl_path,
                        voxel_pkl_path,
                        args.voxel_datatype,
                        args.save_full_3d,
                        args.expand_voxel_points,
                        args.add_xy_channels,
                        True
                    )
                    if img_info is None:
                        continue

                    if self.pos_grid is None and args.voxel_datatype == 0:
                        self.pos_grid = torch.Tensor(voxel_obj.create_position_grid())

                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = data_pkl_path
                    demo_idx_to_path_dict[demo_idx]['info'] = img_info
                    demo_idx_to_path_dict[demo_idx]['voxel_obj'] = voxel_obj
                    demo_idx_to_path_dict[demo_idx]['vrep_geom'] = vrep_geom
                    demo_idx_to_path_dict[demo_idx]['before_img_path'] = \
                        os.path.join(root, '{}_before_vision_sensor.png') 
                    demo_idx = demo_idx + 1 

                    if demo_idx % 5000 == 0:
                        print("Did process: {}".format(demo_idx))

        return demo_idx_to_path_dict
    
    def get_h5_data_size(self, train=True):
        if train:
            return len(self.train_h5_data_dict)
        else:
            return len(self.test_h5_data_dict)

    def get_h5_train_data_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_h5_data_dict[idx]
        else:
            data_dict = self.test_h5_data_dict[idx]

        path = data_dict['path']
        emb = data_dict['emb']
        vrep_geom = data_dict['vrep_geom']
        
        other_bb = vrep_geom.get_other_oriented_bounding_box(before=True)
        anchor_bb = vrep_geom.get_anchor_oriented_bounding_box(before=True)
        bb_list = anchor_bb[0].tolist() + anchor_bb[1].tolist() + \
                  other_bb[0].tolist() + other_bb[1].tolist()
        
        precond_label = data_dict['precond_label']

        action = [8.0, 0.0, 0.0]

        # ==== Visualize ====
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x,y,z = voxels[0, ...].nonzero()
        # ax.scatter(x, y, z)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.show()

        data = {
            'emb': emb,
            'path': path,
            'bb_list': bb_list,
            'precond_label': precond_label,
        }
        return data

    def get_data_size(self, train=True):
        if train:
            return len(self.train_idx_to_data_dict)
        else:
            return len(self.test_idx_to_data_dict)

    def get_train_data_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_idx_to_data_dict[idx]
        else:
            data_dict = self.test_idx_to_data_dict[idx]

        path = data_dict['path']
        voxel_obj = data_dict['voxel_obj']
        status, voxels = voxel_obj.parse()
        # if parse returns false voxels should be None
        if status:
            voxels_tensor = torch.Tensor(voxels)

        vrep_geom = data_dict['vrep_geom']
        
        other_bb = vrep_geom.get_other_oriented_bounding_box(before=True)
        anchor_bb = vrep_geom.get_anchor_oriented_bounding_box(before=True)
        bb_list = anchor_bb[0].tolist() + anchor_bb[1].tolist() + \
                  other_bb[0].tolist() + other_bb[1].tolist()
        
        info = data_dict['info']
        if info.get('obj_in_place_after_anchor_remove') is not None:
            precond_label = 1 if info['obj_in_place_after_anchor_remove'] else 0
        else:
            assert info['before'] is not None and info['after'] is not None
            before_pos = info['before']['other_pos']
            after_pos = info['after']['other_pos']
            if after_pos[0] - before_pos[0] > 0.10:
                precond_label = 1
            else:
                precond_label = 0

        action = [8.0, 0.0, 0.0]

        # ==== Visualize ====
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x,y,z = voxels[0, ...].nonzero()
        # ax.scatter(x, y, z)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.show()

        data = {
            'voxels': voxels_tensor,
            'path': path,
            'info': info,
            'bb_list': bb_list,
            'precond_label': precond_label,
            'before_img_path': data_dict['before_img_path'],
        }
        return data
