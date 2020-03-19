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

from itertools import permutations, combinations

from utils import data_utils
from utils import image_utils
from utils.colors import bcolors

from action_relation.dataloader.robot_octree_data import RobotVoxels, SceneVoxels
from action_relation.dataloader.robot_octree_data import RobotAllPairVoxels

import torch
from torchvision.transforms import functional as F

from typing import List, Dict

def _convert_string_key_to_int(d: dict) -> dict:
    dict_with_int_key = {}
    for k in d.keys():
        dict_with_int_key[int(k)] = d[k]
    assert len(d) == len(dict_with_int_key), "Emb data size changed."
    return dict_with_int_key


def plot_voxel_plot(voxel_3d):
    # ==== Visualize ====
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = voxel_3d[0, ...].nonzero()
    # ax.scatter(x - 25, y - 25, z - 25)
    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def load_emb_data_from_h5_path(h5_path, pkl_path, max_data_size=None):
    h5f = h5py.File(h5_path, 'r')
    emb_data = data_utils.recursively_get_dict_from_group(h5f)
    # emb_data keys are stored as str in hdf5 file. Convert them to integers
    # for direct comparison with the file data.
    emb_data = _convert_string_key_to_int(emb_data)

    with open(pkl_path, 'rb') as pkl_f:
        emb_data_pkl = pickle.load(pkl_f)
    emb_data_pkl = _convert_string_key_to_int(emb_data_pkl)
    
    assert sorted(emb_data.keys()) == sorted(emb_data_pkl.keys())

    data_keys = sorted(emb_data.keys())
    if max_data_size is None:
        max_data_size = len(data_keys)
    else:
        max_data_size = min(max_data_size, len(data_keys))
    
    data_dict = {}
    for i in range(max_data_size):
        key = data_keys[i]
        val = emb_data[key]
        pkl_val = emb_data_pkl[key]
        assert pkl_val['precond_label'] == int(val['precond_label'])
        assert len(pkl_val['all_object_pair_path']) == val['emb'].shape[0]
        data_dict[i] = {
            'path': pkl_val['path'],
            'precond_label': int(val['precond_label']),
            'all_object_pair_path': pkl_val['all_object_pair_path'],
            'emb': val['emb'],
        }
    
    return data_dict


def get_all_object_pcd_path_for_data_in_line(scene_path):
    files = os.listdir(scene_path)
    return [os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]


def get_all_object_pcd_path_for_cut_food(scene_path):
    files = os.listdir(scene_path)
    obj_path = [os.path.join(scene_path, p) for p in sorted(files) 
                if 'cloud_cluster' in p]
    knife_path = os.path.join(scene_path, 'knife_object.pcd')
    assert os.path.basename(obj_path[0]) == 'cloud_cluster_0.pcd'
    return [knife_path] + obj_path


def get_all_object_pcd_path_for_box_stacking(scene_path, remove_obj_id):
    files = os.listdir(os.path.join(scene_path, 'object_cloud'))
    remove_cloud_fname = f'cloud_cluster_{remove_obj_id}.pcd'
    if remove_obj_id != -1:
        assert remove_cloud_fname in files, "To remove obj id not found"
    path_list = [os.path.join(scene_path, 'object_cloud', p) for p in sorted(files) 
                 if 'cloud_cluster' in p and p != remove_cloud_fname]
    return path_list


def get_valid_objects_to_remove_for_box_stacking(scene_path, scene_info):
    # For objects that only have one object at the lowest level, we cannot remove
    # that object since if we remove those voxels the scene has no way of knowing
    # where the ground is and the system would be trivially unstable (knowing the ground)
    pos_list, obj_id_list = [], []
    for k, v in scene_info.items():
        if k == 'precond_obj_id' or k == 'test_removal':
            continue
        pos_list.append(scene_info[k]['pos']) 
        obj_id_list.append(int(k))
    pos_arr = np.array(pos_list)
    pos_arr_on_ground = pos_arr[:, 2] > 0.82
    ground_obj_idx = np.where(pos_arr[:, 2] > 0.82)[0]
    assert len(ground_obj_idx) >= 1, "Atleast one object should be on ground"
    if len(ground_obj_idx) > 1:
        return sorted(obj_id_list)
    else:
        assert np.sum(pos_arr_on_ground) == 1
        print(f"Only 1 obj on ground {obj_id_list[ground_obj_idx[0]]}, {scene_path}. Will remove")
        obj_id_list = [obj_id for i, obj_id in enumerate(obj_id_list) 
                       if not pos_arr_on_ground[i]]
        return sorted(obj_id_list)
                    

class RobotSceneObject(object):
    def __init__(self, scene_path, scene_type):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = os.listdir(scene_path)
        assert 'projected_cloud.pcd' in files
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.
        self.cloud_cluster_pcd_path = [
            os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]
        
        self.scene_voxels = SceneVoxels(self.cloud_cluster_pcd_path, scene_type)
        self.scene_voxels.init_voxel_index()

    @classmethod
    def is_dir_with_files_valid(path_to_dir, files):
        return 'projected_cloud.pcd' in files
    
    def get_scene_voxels(self, return_tensor=False):
        _, voxels = self.scene_voxels.parse()
        if return_tensor:
            return torch.FloatTensor(voxels)
        else:
            return voxels

    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path)


class RobotSceneCutFoodObject(object):
    def __init__(self, scene_path, scene_type):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = os.listdir(scene_path)
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.
        self.cloud_cluster_pcd_path = [
            os.path.join(scene_path, p) for p in sorted(files) if 'cloud_cluster' in p]
        self.knife_pcd_path = os.path.join(scene_path, 'knife_object.pcd')
        
        self.scene_voxels = SceneVoxels(
            self.cloud_cluster_pcd_path + [self.knife_pcd_path],
            scene_type)
        self.scene_voxels.init_voxel_index()
    
    @classmethod
    def is_dir_with_files_valid(path_to_dir, files):
        return os.path.basename(path_to_dir) == 'object_cloud' and \
            'knife_object.pcd' in files

    def get_scene_voxels(self, return_tensor=False):
        _, voxels = self.scene_voxels.parse()
        if return_tensor:
            return torch.FloatTensor(voxels)
        else:
            return voxels

    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path) + 1

class RobotSceneBoxStackingObject(object):
    def __init__(self, scene_path, scene_type, precond_label, remove_obj_id):
        self.scene_path = scene_path
        self.scene_type = scene_type
        self.precond_label = precond_label
        self.remove_obj_id = remove_obj_id

        # Now we have multiple objects each with their own cloud cluster
        # in this scene.
        remove_obj_fname = f'cloud_cluster_{remove_obj_id}.pcd'
        object_cloud_dir = os.path.join(scene_path, 'object_cloud')

        files = os.listdir(object_cloud_dir)
        if remove_obj_id != -1:
            assert remove_obj_fname in files, "To remove obj-id not found"
        self.cloud_cluster_pcd_path = [os.path.join(object_cloud_dir, p) 
            for p in sorted(files) if 'cloud_cluster' in p and p != remove_obj_fname]
        
        self.scene_voxels = SceneVoxels(self.cloud_cluster_pcd_path, scene_type)
        self.scene_voxels.init_voxel_index()

    @classmethod
    def is_dir_with_files_valid(path_to_dir, files):
        return os.path.basename(path_to_dir) == 'object_cloud' and \
            'cloud_cluster_0.pcd' in files

    def get_scene_voxels(self, return_tensor=False):
        _, voxels = self.scene_voxels.parse()
        if return_tensor:
            return torch.FloatTensor(voxels)
        else:
            return voxels

    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path)
    
    @property
    def scene_precond_label(self):
        return self.precond_label
    

class RobotSceneMultiBoxStackingObject(object):
    def __init__(self, scene_path, scene_type):
        self.scene_path = scene_path
        self.scene_type = scene_type

        # Read the json file
        info_json_path = os.path.join(scene_path, 'info.json')
        assert os.path.exists(info_json_path), f"JSON file does not exist: {info_json_path}"
        with open(info_json_path, 'r') as json_fp:
            self.scene_info = json.load(json_fp)

        assert self.scene_info.get('precond_obj_id') is not None

    def get_total_number_of_objects(self):
         obj_cloud_path = os.path.join(self.scene_path, 'object_cloud')
         num_objects = len(
             [p for p in os.listdir(obj_cloud_path) if 'cloud_cluster' in p])
         return num_objects

    def create_scenes_by_removing_voxels(self):
        precond_obj_ids = self.scene_info['precond_obj_id']
        scene_obj_list = []
        num_objects = self.get_total_number_of_objects()
        valid_objs = get_valid_objects_to_remove_for_box_stacking(
            self.scene_path, 
            self.scene_info)

        for i, obj_id in enumerate(valid_objs):
            label = 0 if obj_id in precond_obj_ids else 1
            scene_obj_list.append(RobotSceneBoxStackingObject(self.scene_path, 
                                                              self.scene_type, 
                                                              label,
                                                              obj_id))
        scene_obj_list.append(self.create_scene_with_all_voxels())
        return scene_obj_list

    def create_scene_with_all_voxels(self):
        # The entire scene is always stable
        scene_obj = RobotSceneBoxStackingObject(self.scene_path, 
                                                self.scene_type, 
                                                1,
                                                -1)
        return scene_obj
    

class RobotAllPairSceneObject(object):
    def __init__(self, 
                 scene_path, 
                 scene_type, 
                 precond_label=None, 
                 box_stacking_remove_obj_id=None,
                 precond_obj_ids=None,
                 scene_pos_info=None):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = os.listdir(scene_path)
        if scene_type == 'data_in_line':
            assert 'projected_cloud.pcd' in files
        elif scene_type == 'cut_food':
            assert 'knife_object.pcd' in files
        elif scene_type == 'box_stacking':
            assert precond_label is not None
            assert box_stacking_remove_obj_id is not None 
            assert precond_obj_ids is not None
            assert 'info.json' in files
            assert scene_pos_info is not None
            self.precond_label = precond_label
            self.box_stacking_remove_obj_id = box_stacking_remove_obj_id
            self.precond_obj_ids = precond_obj_ids
        else:
            raise ValueError(f"Invalid scene type: {scene_type}")
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.

        # self.cloud_cluster_pcd_path = [
        #     os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]
        if scene_type == 'data_in_line':
            self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_data_in_line(
                scene_path)
        elif scene_type == 'cut_food':
            self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_cut_food(
                scene_path)
        elif scene_type == 'box_stacking':
            self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_box_stacking(
                scene_path, box_stacking_remove_obj_id)
            remove_obj_id_str = str(box_stacking_remove_obj_id)
            if box_stacking_remove_obj_id == -1:
                self.scene_pos_info = {} 
                for k, v in scene_pos_info.items():
                    if k == 'precond_obj_id' or k == 'test_removal':
                        continue
                    else:
                        self.scene_pos_info[int(k)] = v

            elif box_stacking_remove_obj_id >= 0:
                self.scene_pos_info = {}
                for k, v in scene_pos_info.items():
                    if k == 'precond_obj_id' or k == 'test_removal':
                        continue
                    k_int = int(k)
                    if k_int < box_stacking_remove_obj_id:
                        self.scene_pos_info[k_int] = v
                    elif k_int > box_stacking_remove_obj_id:
                        self.scene_pos_info[k_int - 1] = v
                    else:
                        assert k_int == box_stacking_remove_obj_id

            else:
                raise ValueError(f"Invalid value for box removal obj id {box_stacking_remove_obj_id}")
                
            assert len(self.scene_pos_info) == len(self.cloud_cluster_pcd_path)

        else:
            raise ValueError(f"Invalid scene type {scene_type}")
        
        # relation embeddings are not commutative, hence we find N^2 relations
        self.obj_voxels_by_obj_pair_dict = dict()
        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()
        self.obj_pair_list = list(permutations(range(len(self.cloud_cluster_pcd_path)), 2))

        self.robot_all_pair_voxels = RobotAllPairVoxels(self.cloud_cluster_pcd_path)

        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair
            status, robot_voxels = self.robot_all_pair_voxels.init_voxels_for_pcd_pair(
                anchor_idx, other_idx
            )

            self.obj_voxels_by_obj_pair_dict[obj_pair] = robot_voxels
            self.obj_voxels_status_by_obj_pair_dict[obj_pair] = status
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                self.cloud_cluster_pcd_path[anchor_idx],
                self.cloud_cluster_pcd_path[other_idx]
            )

    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path)
    
    @property
    def number_of_object_pairs(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def stable_object_ids(self):
        assert self.scene_type == 'box_stacking', f"Invalid func for scene {self.scene_type}"
        return self.precond_obj_ids
    
    def get_object_center_list_2(self):
        # return self.robot_all_pair_voxels.get_object_center_list()
        center_list = []
        for k in sorted(self.scene_pos_info.keys()):
            assert self.scene_pos_info[k]['orient'] in (0, 1)
            orient_list = [1, 0] if self.scene_pos_info[k]['orient'] == 0 else [0, 1]
            center_list.append(self.scene_pos_info[k]['pos'] + orient_list)
        return center_list

    def get_object_center_list(self):
        obj_info_list = []
        for k in sorted(self.scene_pos_info.keys()):
            orient = self.scene_pos_info[k]['orient']
            assert orient in (0, 1)
            pos = self.scene_pos_info[k]['pos']
            if orient == 0:
                size_by_axes = [0.048, 0.15, 0.03]
            elif orient == 1:
                size_by_axes = [0.15, 0.048, 0.03]
            bound_1 = [pos[0] - size_by_axes[0]/2.0,
                       pos[1] - size_by_axes[1]/2.0,
                       pos[2]]
            bound_2 = [pos[0] + size_by_axes[0]/2.0,
                       pos[1] + size_by_axes[1]/2.0,
                       pos[2] + size_by_axes[2]]
            
            orient_info = [1, 0] if orient == 0 else [0, 1]
            obj_info_list.append(
                pos + orient_info + bound_1 + bound_2
            )
        return obj_info_list
    

    def get_stable_object_label_tensor(self):
        label = torch.zeros((self.number_of_objects)).long()
        for i in self.stable_object_ids:
            label[i] = 1
        return label
    
    def create_position_grid(self):
        for v in self.obj_voxels_by_obj_pair_dict.values():
            return v.create_position_grid()
    
    def get_pair_status_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_voxels_status_by_obj_pair_dict[obj_pair_key]
    
    # ==== Functions that return object point cloud paths ====
    
    def get_object_pair_path_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
    
    def get_all_object_pair_path(self):
        path_list = []
        for obj_pair_key in self.obj_pair_list:
            path = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            path_list.append(path)
        return path_list

    # ==== Functions that return voxel objects ====
    
    def get_object_pair_voxels_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj.parse()[1]
    
    def get_all_object_pair_voxels(self):
        voxel_list, are_obj_far_apart_list = [], []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            _, voxels = voxel_obj.parse()
            if voxels is not None:
                voxel_list.append(torch.FloatTensor(voxels))
                are_obj_far_apart_list.append(0)
            else:
                assert voxel_obj.objects_are_far_apart
                voxels = voxel_obj.get_all_zero_voxels()
                voxel_list.append(torch.FloatTensor(voxels))
                obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
                print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
                are_obj_far_apart_list.append(0)

        return voxel_list, are_obj_far_apart_list
    
    # ==== Functions that return voxel objects ====

    def get_object_pair_voxel_object_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj
    
    def get_all_object_pair_voxel_object(self):
        voxel_obj_list = []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            voxel_obj_list.append(voxel_obj)
        return voxel_obj_list
    
    def get_object_pcd_paths(self):
        return self.cloud_cluster_pcd_path
    

class RobotBoxStackingAllPairSceneObject(object):
    def __init__(self, scene_path, scene_type):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = os.listdir(os.path.join(scene_path, 'object_cloud'))
        self.num_objects = len([p for p in files if 'cloud_cluster' in p])

        # Read the json file
        info_json_path = os.path.join(scene_path, 'info.json')
        assert os.path.exists(info_json_path), f"JSON file does not exist: {info_json_path}"
        with open(info_json_path, 'r') as json_fp:
            self.scene_info = json.load(json_fp)
        
        assert self.scene_info.get('precond_obj_id') is not None
    
    def create_scenes_by_removing_voxels(self):
        precond_obj_ids = self.scene_info['precond_obj_id']
        scene_obj_list = []
        valid_objs = get_valid_objects_to_remove_for_box_stacking(
            self.scene_path, 
            self.scene_info)
        for i, obj_id in enumerate(valid_objs):
            assert self.scene_info.get(str(obj_id)) is not None, f"Missing pos info for obj {i}"
            label = 0 if obj_id in precond_obj_ids else 1
            this_scene_precond_obj_ids = []
            for new_obj_id in precond_obj_ids:
                if new_obj_id < obj_id:
                    # Precond obj id is lower than the current object being removed 
                    # hence this obj_id remains unchanged
                    this_scene_precond_obj_ids.append(new_obj_id)
                elif new_obj_id > obj_id:
                    # Precond obj id is "greater" than the current object being removed 
                    # hence this obj_id remains unchanged
                    this_scene_precond_obj_ids.append(new_obj_id - 1)

            scene_obj_list.append(RobotAllPairSceneObject(
                self.scene_path, 
                self.scene_type, 
                precond_label=label,
                box_stacking_remove_obj_id=i,
                precond_obj_ids=this_scene_precond_obj_ids,
                scene_pos_info=self.scene_info))
        
        scene_obj_list.append(self.create_scene_with_all_voxels())

        return scene_obj_list
    
    def create_scene_with_all_voxels(self):
        # The entire scene is always stable
        all_pair_scene_obj = RobotAllPairSceneObject(
            self.scene_path,
            self.scene_type,
            precond_label=1,
            box_stacking_remove_obj_id=-1,
            precond_obj_ids=self.scene_info['precond_obj_id'],
            scene_pos_info=self.scene_info)
        return all_pair_scene_obj


class AllPairVoxelDataloader(object):
    def __init__(self, 
                 config,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False):

        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking")
        self.scene_type = "box_stacking"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        self.train_idx_to_data_dict = {}
        self.max_train_data_size = 50000
        for train_dir in self.train_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                train_dir, 
                max_data_size=self.max_train_data_size, 
                demo_idx=len(self.train_idx_to_data_dict),
                max_data_from_dir=None)
            if len(self.train_idx_to_data_dict) > 0:
                assert (max(list(self.train_idx_to_data_dict.keys())) < 
                        max(list(idx_to_data_dict.keys())))

            self.train_idx_to_data_dict.update(idx_to_data_dict)
            if len(self.train_idx_to_data_dict) >= self.max_train_data_size:
                break

        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))

        self.test_idx_to_data_dict = {}
        self.max_test_data_size = 50000
        for test_dir in self.test_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                test_dir, 
                max_data_size=self.max_test_data_size, 
                demo_idx=len(self.test_idx_to_data_dict),
                max_data_from_dir=None)
            if len(self.test_idx_to_data_dict) > 0:
                assert (max(list(self.test_idx_to_data_dict.keys())) < 
                        max(list(idx_to_data_dict.keys())))

            self.test_idx_to_data_dict.update(idx_to_data_dict)
            if len(self.test_idx_to_data_dict) >= self.max_test_data_size:
                break

        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        # The following dicts contain two keys ('idx' and 'order')
        self.train_all_pair_sample_order = {}
        self.test_all_pair_sample_order = {}
        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}


    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line' and 'projected_cloud.pcd' not in files:
                continue

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObject(root, self.scene_type)
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                   self.scene_type != 'box_stacking':
                    self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid())
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        if shuffle:
            np.random.shuffle(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        precond_label = 1 if 'true' in path.split('/') else 0
        if precond_label == 0:
            assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        all_obj_pair_voxels, all_obj_pair_far_apart_status = \
            scene_voxel_obj.get_all_object_pair_voxels()
        obj_center_list = scene_voxel_obj.get_object_center_list()
        
        # import pdb; pdb.set_trace()
        # for l in range(len(all_obj_pair_voxels)):
        #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
        # import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'obj_center_list': obj_center_list,
            'precond_label': precond_label,
        }
        if self.scene_type == 'box_stacking':
            stable_obj_label = scene_voxel_obj.get_stable_object_label_tensor()
            data['precond_stable_obj_ids'] = stable_obj_label

        return data
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        return data

    def get_voxels_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        return data

    def get_next_voxel_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        return data
