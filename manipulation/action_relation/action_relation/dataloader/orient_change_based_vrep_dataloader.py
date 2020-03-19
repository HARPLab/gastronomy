import numpy as np

import copy
import os

from utils.colors import bcolors
from action_relation.dataloader.vrep_dataloader import SimpleVoxelDataloader


class OrientChangeBasedVrepDataloader(SimpleVoxelDataloader):
    def __init__(self, config,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False):
        super(OrientChangeBasedVrepDataloader, self).__init__(
            config,
            train_dir_list=train_dir_list,
            test_dir_list=test_dir_list,
            voxel_datatype_to_use=voxel_datatype_to_use,
            load_contact_data=load_contact_data,
        )

        self.th_x, self.th_y, self.th_z = 0.001, 0.001, 0.001
    
        assert len(self.train_idx_to_data_dict) > 0
        self.train_orient_change_to_demo_idx_dict = self.filter_loaded_voxel_data(
            self.train_idx_to_data_dict, self.th_x, self.th_y, self.th_z)

        assert len(self.test_idx_to_data_dict) > 0
        self.test_orient_change_to_demo_idx_dict = self.filter_loaded_voxel_data(
            self.test_idx_to_data_dict, self.th_x, self.th_y, self.th_z)
        
        self.train_sample_order = {}
        self.test_sample_order = {}

    def filter_loaded_voxel_data(self, demo_idx_to_data_dict, th_x, th_y, th_z):
        orient_change_to_demo_idx_dict = {
            'gt_th_x': [],
            'gt_th_y': [],
            'gt_th_z': [],
            'lt_th': []
        }

        for demo_idx, data_dict in demo_idx_to_data_dict.items():
            demo_info = data_dict['info']
            before_pose = demo_info['before']['other_q']
            after_pose = demo_info['after']['other_q']
            delta_pose = [after_pose[i]-before_pose[i] 
                          for i in range(len(before_pose))]
            is_gt_th = False
            if abs(delta_pose[0]) > th_x:
                orient_change_to_demo_idx_dict['gt_th_x'].append(demo_idx)
                is_gt_th = True

            if abs(delta_pose[1]) > th_y:
                orient_change_to_demo_idx_dict['gt_th_y'].append(demo_idx)
                is_gt_th = True

            if abs(delta_pose[2]) > th_z:
                orient_change_to_demo_idx_dict['gt_th_z'].append(demo_idx)
                is_gt_th = True
            
            if not is_gt_th:
                orient_change_to_demo_idx_dict['lt_th'].append(demo_idx)
            
        return orient_change_to_demo_idx_dict
    
    def get_data_size(self, train=True):
        # Just use all the data for testing
        if not train:
            return len(self.test_idx_to_data_dict)

        orient_change_dict = self.train_orient_change_to_demo_idx_dict if train \
            else self.test_orient_change_to_demo_idx_dict

        gt_th_data = 0 
        for ax in ['x', 'y', 'z']:
            gt_th_data += len(orient_change_dict[f'gt_th_{ax}'])
        lt_th_data = len(orient_change_dict['lt_th'])

        return min(gt_th_data*3, gt_th_data + lt_th_data)
    
    def reset_batch_sampler(self, train=True):
        if train:
            for k, v in self.train_orient_change_to_demo_idx_dict.items():
                order = copy.deepcopy(v)
                np.random.shuffle(order)
                self.train_sample_order[k] = { 'order': order, 'idx': 0 }
        else:
            self.test_sample_order = {
                'order': list(sorted(self.test_idx_to_data_dict.keys())),
                'idx': 0
            }

    def get_train_idx_for_batch(self, batch_size, train=True):
        if not train:
            idx = self.test_sample_order['idx']
            data_order = self.test_sample_order['order'][idx:idx+batch_size]
            # Do not mod this so that if we have an error we can cath it
            self.test_sample_order['idx'] = idx+batch_size
            return data_order
        else:
            sample_dict = self.train_sample_order
            sample_less_than_th_prob = 0.25
            npu = np.random.uniform
            batch_demo_idx = []
            for i in range(batch_size):
                if npu(0, 1) < sample_less_than_th_prob:
                    key = 'lt_th'
                else:
                    ax = ['x', 'y', 'z']
                    key = f'gt_th_{ax[np.random.randint(0, 3)]}'

                idx = sample_dict[key]['idx']
                demo_idx = sample_dict[key]['order'][idx]
                sample_dict[key]['idx'] = (idx+1) % len(sample_dict[key]['order'])
                if sample_dict[key]['idx'] == 0:
                    print(f"Did reset data while sampling {key}")
                
                batch_demo_idx.append(demo_idx)
        return batch_demo_idx
    