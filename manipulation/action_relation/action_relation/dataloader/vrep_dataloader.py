import numpy as np

import csv
import h5py
import os
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

import torch
from torchvision.transforms import functional as F

USE_DENSE_VOXELS = True


def get_img_info_and_voxel_object_for_path(data_pkl_path, 
                                           voxel_pkl_path, 
                                           voxel_datatype_to_use,
                                           save_full_3d,
                                           expand_voxel_points,
                                           add_xy_channels,
                                           parse_before_scene_only):
    with open(data_pkl_path, 'rb') as pkl_f:
        data = pickle.load(pkl_f)

    vrep_geom = VrepSceneGeometry(
        data['data'], 
        data_pkl_path, 
        parse_before_scene_only=parse_before_scene_only)

    if voxel_datatype_to_use == 0:
        if data.get('voxels_before') is None:
            voxel_data = OctreePickleLoader(
                voxel_pkl_path, 
                vrep_geom,
                save_full_3d=save_full_3d,
                expand_octree_points=expand_voxel_points)
        else:
            voxel_data = OctreeVoxels(data['voxels_before'])
    else:
        add_size_channels = (add_xy_channels == 1)
        add_xy_channels = True
        voxel_data = CanonicalOctreeVoxels(
            voxel_pkl_path, add_size_channels=add_size_channels)

    if not voxel_data.init_voxel_index():
        voxel_data = None
        return None, None
    
    return data['data'], voxel_data


class SimpleVoxelDataloader(object):
    def __init__(self, config,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False):
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        max_train_data_size = 40000
        curr_dir_max_data = None
        for train_dir in self.train_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                train_dir, 
                max_data_size=max_train_data_size, 
                max_action_idx=8, 
                demo_idx=demo_idx,
                curr_dir_max_data=None)
            if len(self.train_idx_to_data_dict) > 0:
                assert (max(list(self.train_idx_to_data_dict.keys())) < 
                        max(list(idx_to_data_dict.keys())))

            self.train_idx_to_data_dict.update(idx_to_data_dict)
            demo_idx += len(idx_to_data_dict)
            assert demo_idx == len(self.train_idx_to_data_dict)

            if demo_idx >= max_train_data_size:
                break

        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))

        self.test_idx_to_data_dict = {}
        demo_idx, max_test_data_size = 0, 4000
        # curr_dir_max_data = max_test_data_size // len(self.test_dir_list)
        curr_dir_max_data = None
        for test_dir in self.test_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                test_dir, 
                max_data_size=max_test_data_size, 
                max_action_idx=8, 
                demo_idx=demo_idx,
                curr_dir_max_data=curr_dir_max_data)
            if len(self.test_idx_to_data_dict) > 0:
                assert (max(list(self.test_idx_to_data_dict.keys())) < 
                        max(list(idx_to_data_dict.keys())))

            self.test_idx_to_data_dict.update(idx_to_data_dict)
            demo_idx += len(idx_to_data_dict)
            assert demo_idx == len(self.test_idx_to_data_dict)
            if demo_idx >= max_test_data_size:
                break

        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        if config.args.use_contrastive_loss:
            self.contrastive_data_by_action_dict = \
                self.get_contrastive_data(self.train_idx_to_data_dict)

        self.train_sample_order = {}
        self.test_sample_order = {}

        self.use_relative_and_absolute_actions = True
        # Predict relative voxel displacement than absolute pose changes which 
        # are obviously impossible given that we are using voxels.
        self.use_voxel_displacement = True
    
    def filter_data_dict_into_separate_actions(self, idx_to_data_dict):
        '''Filter data dict into separate actions.
        
        Return: Dict. Dict with action_idx as key and list of data indexes as
            value.
        '''
        data_key_by_action_dict = {}
        for k, data in idx_to_data_dict.items():
            action_relative_or_absolute = data['info'].get('action_relative_or_absolute', 'relative')
            action = data['info']['action']

            if action_relative_or_absolute == 'relative':
                vrep_geom = data['vrep_geom']
                action_info_dict = vrep_geom.get_and_save_info_for_relative_action()
                action_idx = action_info_dict['action_idx']
            elif  action_relative_or_absolute == 'absolute':
                action_idx = get_index_for_action(action)
            else:
                raise ValueError("Invalid value for action_relative_or_absolute")

            if data_key_by_action_dict.get(action_idx) is None:
                data_key_by_action_dict[action_idx] = []
            data_key_by_action_dict[action_idx].append(k)

        return data_key_by_action_dict

    def get_contrastive_data(self, idx_to_data_dict):
        top_k, bottom_k = 100, 100
        data_key_by_action_dict = self.filter_data_dict_into_separate_actions(
            self.train_idx_to_data_dict)
        
        contrastive_data_by_action_dict = {}

        total_idx = len(idx_to_data_dict)
        for action_idx, data_keys in data_key_by_action_dict.items():
            top_k_idxs_arr = np.zeros((total_idx, top_k), dtype=np.int32)
            top_k_idxs_info_list = [[] for _ in range(total_idx)]
            bottom_k_idxs_arr = np.zeros((total_idx, bottom_k), dtype=np.int32)
            bottom_k_idxs_info_list = [[] for _ in range(total_idx)]

            data_list = [self.train_idx_to_data_dict[i] for i in data_keys]
            inter_scene_dist_dict = get_dist_matrix_for_data(
                data_list,
                save_inter_scene_dist_dir=None,
                top_K=top_k,
                bottom_K=bottom_k,
            )
            for scene_idx, scene_dist_data in inter_scene_dist_dict['idx'].items():
                # scene_idx is the idx from 0 to len(data_keys)
                actual_scene_idx = data_keys[scene_idx]
                for i, top_data in enumerate(scene_dist_data['top']):
                    dist, other_scene_idx, other_scene_path = top_data
                    actual_other_scene_idx = data_keys[other_scene_idx]
                    top_k_idxs_arr[actual_scene_idx][i] = actual_other_scene_idx
                    top_k_idxs_info_list[i].append(
                        (dist, other_scene_idx, other_scene_path, 
                         actual_scene_idx, actual_other_scene_idx)
                    )
                for i, bottom_data in enumerate(scene_dist_data['bottom']):
                    dist, other_scene_idx, other_scene_path = bottom_data
                    actual_other_scene_idx = data_keys[other_scene_idx]
                    bottom_k_idxs_arr[actual_scene_idx][i] = actual_other_scene_idx
                    bottom_k_idxs_info_list[i].append(
                        (dist, other_scene_idx, other_scene_path, 
                         actual_scene_idx, actual_other_scene_idx)
                    )

            contrastive_data_by_action_dict[action_idx] = dict(
                top_k_idxs_arr=top_k_idxs_arr,
                top_k_idxs_info_list=top_k_idxs_info_list,
                bottom_k_idxs_arr=bottom_k_idxs_arr, 
                bottom_k_idxs_info_list=bottom_k_idxs_info_list, 
            )
        return contrastive_data_by_action_dict

    def get_img_info_and_voxel_object_for_path(self, data_pkl_path, voxel_pkl_path):
        args = self._config.args
        data, voxel_obj = get_img_info_and_voxel_object_for_path(
            data_pkl_path, 
            voxel_pkl_path, 
            args.voxel_datatype_to_use,
            args.save_full_3d,
            args.expand_voxel_points,
            args.add_xy_channels)
        
        if self.pos_grid is None and self.voxel_datatype_to_use == 0:
            self.pos_grid = torch.Tensor(voxel_obj.create_position_grid())
        return data['data'], voxel_obj


    def load_voxel_data(self, demo_dir, max_data_size=None, max_action_idx=10,
                        demo_idx=0, curr_dir_max_data=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        curr_dir_max_data: Int. Max amount of data to load from this dir.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        curr_dir_data_count = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if curr_dir_max_data is not None and curr_dir_data_count > curr_dir_max_data:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()

            if 'all_img_data.json' in files:

                # Filter data based on the anchor object.
                all_img_data_json_path = os.path.join(root, 'all_img_data.json')
                '''
                with open(all_img_data_json_path, 'r') as json_f:
                    all_img_data = json.load(json_f)
                    anchor_args = all_img_data['anchor_args']
                    other_args = all_img_data['other_args']
                    if anchor_args['obj_type'] != 0:
                        continue
                '''
                all_init_action_idx = [int(f.split('_')[0]) 
                                       for f in files if 'img_data.pkl' in f]

                # for i in range(0, max_action_idx, 1):
                for i in sorted(all_init_action_idx):
                    data_pkl_path = os.path.join(root, '{}_img_data.pkl'.format(i))
                    if not os.path.exists(data_pkl_path):
                        break
                    with open(data_pkl_path, 'rb') as pkl_f:
                        data = pickle.load(pkl_f)
                    
                    vrep_geom = VrepSceneGeometry(data['data'], data_pkl_path)
                    if self.load_contact_data:
                        # Load contact data
                        contact_pkl_path = os.path.join(root, '{}_contact_data.pkl'.format(i))
                    else:
                        contact_pkl_path = None

                    if USE_DENSE_VOXELS:
                        voxel_pkl_path = os.path.join(root, '{}_dense_voxel_data.pkl'.format(i))
                    else:
                        voxel_pkl_path = os.path.join(root, '{}_voxel_data.pkl'.format(i))

                    if self.voxel_datatype_to_use == 0:
                        if data.get('voxels_before') is None:
                            voxel_data = OctreePickleLoader(
                                voxel_pkl_path, 
                                vrep_geom,
                                save_full_3d=args.save_full_3d,
                                expand_octree_points=args.expand_voxel_points,
                                contact_pkl_path=contact_pkl_path,
                                convert_to_dense_voxels=False)
                        else:
                            voxel_data = OctreeVoxels(data['voxels_before'])
                    else:
                        add_size_channels = (args.add_xy_channels == 1)
                        add_xy_channels = True
                        voxel_data = CanonicalOctreeVoxels(
                            voxel_pkl_path, add_size_channels=add_size_channels)


                    if not voxel_data.init_voxel_index():
                        voxel_data = None
                        continue
                        
                    if self.load_contact_data:
                        # Load contact data
                        contact_pkl_path = os.path.join(
                            root, '{}_contact_data.pkl'.format(i))
                        assert os.path.exists(contact_pkl_path), "Contact data does not exist"
                        with open(contact_pkl_path, 'rb') as contact_f:
                            contact_data = pickle.load(contact_f)
                    else:
                        contact_data = None
                    
                    # relative_action = \
                    #     (data['data']['action_relative_or_absolute'] == 'relative')
                    relative_action = True
                    if relative_action:
                        action_info = vrep_geom.get_and_save_info_for_relative_action()
                        if action_info is None:
                            # print(f"Dropping data {data_pkl_path}")
                            continue

                    # if vrep_geom.check_if_same_other_obj_z_axis_before_after():
                        # continue
                    # else:
                        # print(data_pkl_path)
                    
                    if self.pos_grid is None and \
                       self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(
                            voxel_data.create_position_grid())
                        '''
                        import ipdb; ipdb.set_trace()
                        xy = self.pos_grid[0, :, :, 0].cpu().numpy()
                        plt.imshow(np.array(xy + np.min(xy)).astype(np.int32))
                        plt.show()
                        plt.colorbar()
                        xy = self.pos_grid[1, :, :, 0].cpu().numpy()
                        plt.imshow(np.array(xy + np.min(xy)).astype(np.int32))
                        plt.show()
                        plt.colorbar()
                        import ipdb; ipdb.set_trace()
                        '''


                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = data_pkl_path
                    demo_idx_to_path_dict[demo_idx]['info'] = data['data']
                    demo_idx_to_path_dict[demo_idx]['voxel_obj'] = voxel_data
                    demo_idx_to_path_dict[demo_idx]['vrep_geom'] = vrep_geom
                    demo_idx_to_path_dict[demo_idx]['before_img_path'] = \
                        os.path.join(root, '{}_before_vision_sensor.png') 
                    if self.load_contact_data and contact_data is not None:
                        demo_idx_to_path_dict[demo_idx]['contact_data'] = contact_data
                    demo_idx = demo_idx + 1 
                    curr_dir_data_count = curr_dir_data_count + 1

                    # import ipdb; ipdb.set_trace() 
                    if demo_idx % 1000 == 0:
                        print("Did process: {}".format(demo_idx))
        print(f"Did process {curr_dir_data_count} from dir: {demo_dir}")
        return demo_idx_to_path_dict

    def reset_batch_sampler(self, train=True):
        if train:
            order = sorted(self.train_idx_to_data_dict.keys())
            np.random.shuffle(order)
            self.train_sample_order = { 'order': order, 'idx': 0, }
        else:
            self.test_sample_order = {
                'order': list(sorted(self.test_idx_to_data_dict.keys())),
                'idx': 0,
            }

    def get_data_size(self, train=True):
        if train:
            return len(self.train_idx_to_data_dict)
        else:
            return len(self.test_idx_to_data_dict)

    def get_train_idx_for_batch(self, batch_size, train=True):
        sample_order = self.train_sample_order if train else self.test_sample_order
        idx = sample_order['idx']
        batch_demo_idx = sample_order['order'][idx:idx+batch_size]
        sample_order['idx'] = idx + batch_size

        return batch_demo_idx

    @staticmethod
    def get_classes_for_delta_pose(delta_pose):
        p = abs(delta_pose)
        if p < 0.005:
            return 0
        elif p < 0.05:
            return 1
        elif p < 0.2:
            return 2
        elif p < 0.28:
            return 3
        elif p < 0.5:
            return 4
        else:
            raise ValueError("invalid class")
        return p
    
    @staticmethod
    def get_classes_for_delta_orient_5(deltabatch_obj_delta_pose_list_orient):
        d = delta_orient
        if d > -0.5 and d <= -0.01:
            klass = 0
        elif d > -0.01 and d <= -0.001:
            klass = 1
        elif d > -0.001 and d <= 0.001:
            klass = 2
        elif d > 0.001 and d <= 0.01:
            klass = 3
        elif d > 0.01 and d < 0.5:
            klass = 4
        else:
            raise ValueError("Invalid change in orient: {}".format(d))
        return klass

    @staticmethod
    def get_classes_for_delta_orient(delta_orient):
        d = delta_orient
        if d > -0.5 and d <= -0.005:
            klass = 0
        elif d > -0.005 and d <= 0.005:
            klass = 1
        elif d > 0.005 and d < 0.5:
            klass = 2
        else:
            raise ValueError("Invalid change in orient: {}".format(d))
        return klass

    def get_voxel_obj_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_idx_to_data_dict[idx]
        else:
            data_dict = self.test_idx_to_data_dict[idx]
        return data_dict['voxel_obj']

    def get_contrastive_data_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_idx_to_data_dict[idx]
        else:
            data_dict = self.test_idx_to_data_dict[idx]
        
        action = data_dict['info']['action']
        action_idx = get_index_for_action(action)

        top_k_idxs_arr = self.contrastive_data_by_action_dict[
            action_idx]['top_k_idxs_arr'][idx]
        bottom_k_idxs_arr = self.contrastive_data_by_action_dict[
            action_idx]['bottom_k_idxs_arr'][idx]

        sim = top_k_idxs_arr[np.random.randint(len(top_k_idxs_arr))]
        diff = bottom_k_idxs_arr[np.random.randint(len(bottom_k_idxs_arr))]

        # print("Selecting triplet: {}, {}, {}".format(idx, sim, diff))

        # TODO: Verify that sim and diff have large distances ?
        
        anchor_data = self.get_train_data_at_idx(idx, train=train)
        sim_data = self.get_train_data_at_idx(sim, train=train)
        diff_data = self.get_train_data_at_idx(diff, train=train)

        return anchor_data, sim_data, diff_data

    def get_train_data_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_idx_to_data_dict[idx]
        else:
            data_dict = self.test_idx_to_data_dict[idx]

        path = data_dict['path']

        if self._config.args.octree_0_multi_thread:
            voxels_tensor = None
        else:
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
        # action = torch.Tensor(info['action'])
        # before_pose = torch.Tensor(info['before']['other_pos'] +
        #                            info['before']['other_q'])
        # after_pose = torch.Tensor(info['after']['other_pos'] +
        #                           info['after']['other_q'])
        # delta_pose = after_pose - before_pose
        if self.use_relative_and_absolute_actions:
            action_relative_or_absolute = info.get('action_relative_or_absolute', 'relative')
            action = []
            if action_relative_or_absolute == 'relative':
                action.append(0)
                action_info = vrep_geom.get_action_data_for_relative_action(
                    use_true_action_idx=True
                )
                action += action_info[1]
                action_idx = action_info[0]
            elif action_relative_or_absolute == 'absolute':
                raise ValueError("Not using absolute loss right now")
                action.append(1)
                action += info['action']
                action_idx = get_index_for_action(action)
            else:
                raise ValueError(f"Invalid type {action_relative_or_absolute}")
        else:
            action = info['action']
            action_idx = get_index_for_action(action)

        before_pose = info['before']['other_pos'] + info['before']['other_q']
        after_pose = info['after']['other_pos'] + info['after']['other_q']

        # Get the delta pose from the action
        if self.use_voxel_displacement:
            delta_position = vrep_geom.get_voxel_displacement().tolist()
            delta_orient = [after_pose[i]-before_pose[i] for i in range(3, 6)]
            delta_pose = delta_position + delta_orient
        else:
            delta_pose = [after_pose[i]-before_pose[i] for i in range(len(before_pose))]
            # Use the the delta pose directly 
            delta_pose_contrastive = [i for i in delta_pose]
        
        # Get the delta pose for contrastive loss
        if self.use_relative_and_absolute_actions:
            action_relative_or_absolute = info.get('action_relative_or_absolute', 'relative')
            if action_relative_or_absolute == 'relative':
                # Get delta pose for contrastive loss
                real_by_max_disp = vrep_geom.get_real_by_max_disp_ratio().tolist()
                delta_pose_contrastive = real_by_max_disp + delta_orient
            elif action_relative_or_absolute == 'absolute':
                delta_pose_contrastive = [after_pose[i]-before_pose[i] 
                                          for i in range(len(before_pose))]
            else:
                raise ValueError("Invalid value for action type")
        else:
            delta_pose_contrastive = [after_pose[i]-before_pose[i] 
                                      for i in range(len(before_pose))]


        # delta_classes = [SimpleVoxelDataloader.get_classes_for_delta_pose(
        #                  delta_pose[i]) for i in range(2)]
        delta_classes = [0, 0, 0]

        # delta_orient_classes = [
        #     SimpleVoxelDataloader.get_classes_for_delta_orient(
        #         delta_pose[i]) for i in range(3, 6)]
        delta_orient_classes = [0, 0, 0]

        ft_sensor_mean = info.get('ft_sensor_mean')

        if info.get('contact') is not None and info['contact']['before_contact'] is not None:
            contact_data = np.array(info['contact']['before_contact'])
            if contact_data.shape[0] == 1:
                contact_mu = contact_data[0, 2:2+3]
                contact_cov = np.eye(3)
            else:
                contact_mu = np.mean(contact_data[:, 2:2+3], axis=0)
                epsilon_cov = 1e-2
                contact_cov = np.cov(contact_data[:, 2:2+3].T) + epsilon_cov*np.eye(3)
            contact_dict = {'mu': contact_mu, 
                            'cov': contact_cov.reshape(-1), 
                            'mask': 1}
        else:
            contact_data, contact_dict = None, {'mask': 0}

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
            'before_pose': before_pose,
            'after_pose': after_pose,
            'delta_pose': delta_pose,
            'delta_pose_contrastive': delta_pose_contrastive,
            'delta_classes': delta_classes,
            'delta_orient_classes': delta_orient_classes,
            'bb_list': bb_list,
            'action': action,
            'action_idx': action_idx,
            'ft_sensor': ft_sensor_mean,
            'contact_data': contact_data,
            'contact_dict': contact_dict,
            'before_img_path': data_dict['before_img_path'],
        }
        return data
