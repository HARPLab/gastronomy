import numpy as np
import argparse
import pickle
import h5py
import sys
import os
import pprint
import ipdb
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import torch.optim as optim
import time
import json
import copy
from tqdm import tqdm

sys.path.append(os.getcwd())

from robot_learning.logger.tensorboardx_logger import TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network, to_numpy
from utils.colors import bcolors
from utils.data_utils import str2bool
from utils.data_utils import recursively_save_dict_contents_to_group
from utils.data_utils import convert_list_of_array_to_dict_of_array_for_hdf5
from utils.image_utils import get_image_tensor_mask_for_bb

from action_relation.dataloader.vrep_dataloader import SimpleVoxelDataloader
from action_relation.model.voxel_model import VoxelModel
from action_relation.model.unscaled_voxel_model import UnscaledVoxelModel
from action_relation.model.model_utils import ScaledMSELoss
from action_relation.model.losses import TripletLoss, get_contrastive_loss
from action_relation.utils.data_utils import get_euclid_dist_matrix_for_data

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer


def fn_voxel_parse(voxel_obj):
    status, a = voxel_obj.parse()
    return a

from multiprocessing import Pool


class VoxelRelationTrainer(BaseVAETrainer):
    def __init__(self, config):
        super(VoxelRelationTrainer, self).__init__(config)
        self.hidden_dim = 64
        args = config.args

        if args.octree_0_multi_thread:
            self.voxel_pool = Pool(args.batch_size)

        self.dataloader = SimpleVoxelDataloader(
            config,
            voxel_datatype_to_use=args.voxel_datatype)

        args = config.args
        if args.voxel_datatype == 0:
            self.model = UnscaledVoxelModel(
                args.z_dim, 6, args, n_classes=2*args.classif_num_classes,
                use_spatial_softmax=args.use_spatial_softmax,
                )
        elif args.voxel_datatype == 1:
            self.model = VoxelModel(args.z_dim, 6, args,
                                    n_classes=2*args.classif_num_classes)
        else:
            raise ValueError("Invalid voxel datatype: {}".format(
                args.voxel_datatype))

        # self.loss = nn.BCELoss()
        if args.loss_type == 'regr':
            if args.scaled_mse_loss:
                self.loss = ScaledMSELoss(0.0001)
                self.pose_pred_loss = ScaledMSELoss(0.0001)
            else:
                self.loss = nn.MSELoss()
                self.pose_pred_loss = nn.MSELoss()
        elif args.loss_type == 'classif':
            self.pose_pred_loss = nn.CrossEntropyLoss()

        self.inv_model_loss = nn.MSELoss()

        self.contrastive_margin = args.contrastive_margin
        self.triplet_loss = TripletLoss(self.contrastive_margin)

        self.opt = optim.Adam(self.model.parameters(), lr=config.args.lr)

    def save_checkpoint(self, epoch):
        cp_filepath = self.model_checkpoint_filename(epoch)
        torch.save({'model': self.model.state_dict()}, cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        checkpoint_models = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint_models['model'])

    def log_model_to_tensorboard(self, train_step_count):
        '''Log weights and gradients of network to Tensorboard.'''
        model_l2_norm, model_grad_l2_norm = \
            get_weight_norm_for_network(self.model)
        # if args.model_type == 'vae':
        #     self.logger.summary_writer.add_histogram(
        #             'histogram/mu_linear_layer_weights',
        #             self.model.mu_linear.weight,
        #             train_step_count)
        #     self.logger.summary_writer.add_histogram(
        #             'histogram/logvar_linear_layer_weights',
        #             self.model.logvar_linear.weight,
        #             train_step_count)
        self.logger.summary_writer.add_scalar(
                'weight/model',
                model_l2_norm,
                train_step_count)
        self.logger.summary_writer.add_scalar(
                'grad/model',
                model_grad_l2_norm,
                train_step_count)

    def process_contrastive_raw_batch_data(self, batch_data):
        '''Process raw batch data and collect relevant objects in a dict.'''
        proc_batch_dict = {
            # Save input image
            'batch_voxel_list': [],
            # Save object pose before action.
            'batch_obj_before_pose_list': [],
            # Save object pose after action.
            'batch_obj_after_pose_list': [],
            # Save object change pose
            'batch_obj_delta_pose_list': [],
            # Save action info.
            'batch_action_list': [],
            # Save object bounding boxes.
            'batch_bb_list': [],
        }
        args = self.config.args
        x_dict = {
            0: proc_batch_dict,
            1: copy.deepcopy(proc_batch_dict),
            2: copy.deepcopy(proc_batch_dict),
        }

        for b, contrastive_data in enumerate(batch_data):
            for i, data in enumerate(contrastive_data):
                x_dict[i]['batch_obj_before_pose_list'].append(data['before_pose'])
                x_dict[i]['batch_obj_after_pose_list'].append(data['after_pose'])
                x_dict[i]['batch_obj_delta_pose_list'].append(data['delta_pose'])
                x_dict[i]['batch_action_list'].append(data['action'])
                x_dict[i]['batch_bb_list'].append(data['bb_list'])

                # import ipdb; ipdb.set_trace()

                if len(data['voxels'].size()) == 3:
                    voxels = data['voxels'].unsqueeze(0)
                elif len(data['voxels'].size()) == 4:
                    voxels = data['voxels']
                else:
                    raise ValueError("Invalid voxels size: {}".format(
                        data['voxels'].size()))

                if args.add_xy_channels == 0:
                    pass
                elif args.add_xy_channels == 1:
                    if args.voxel_datatype == 0:
                        pos_grid_tensor = self.dataloader.pos_grid
                        voxels = torch.cat([voxels, pos_grid_tensor], dim=0)
                else:
                    raise ValueError("Invalid add_xy_channels: {}".format(
                        args.add_xy_channels))

                x_dict[i]['batch_voxel_list'].append(voxels)

        return x_dict

    def process_raw_batch_data(self, batch_data):
        '''Process raw batch data and collect relevant objects in a dict.'''
        proc_batch_dict = {
            # Save input image
            'batch_voxel_list': [],
            # Save object pose before action.
            'batch_obj_before_pose_list': [],
            # Save object pose after action.
            'batch_obj_after_pose_list': [],
            # Save object change pose
            'batch_obj_delta_pose_list': [],
            # Save object delta pose class.
            'batch_obj_delta_pose_class_list': [],
            # Save action info.
            'batch_action_list': [],
            # Save action index info.
            'batch_action_index_list': [],
            # Save object bounding boxes.
            'batch_bb_list': [],
        }
        args = self.config.args
        x_dict = proc_batch_dict

        for b, data in enumerate(batch_data):
            x_dict['batch_obj_before_pose_list'].append(data['before_pose'])
            x_dict['batch_obj_after_pose_list'].append(data['after_pose'])
            x_dict['batch_obj_delta_pose_list'].append(data['delta_pose'])
            x_dict['batch_action_list'].append(data['action'])
            x_dict['batch_action_index_list'].append(data['action_idx'])
            x_dict['batch_obj_delta_pose_class_list'].append(data['delta_classes'])
            x_dict['batch_bb_list'].append(data['bb_list'])

            # import ipdb; ipdb.set_trace()

            if len(data['voxels'].size()) == 3:
                voxels = data['voxels'].unsqueeze(0)
            elif len(data['voxels'].size()) == 4:
                voxels = data['voxels']
            else:
                raise ValueError("Invalid voxels size: {}".format(
                    data['voxels'].size()))

            if args.add_xy_channels == 0:
                pass
            elif args.add_xy_channels == 1:
                if args.voxel_datatype == 0:
                    pos_grid_tensor = self.dataloader.pos_grid
                    voxels = torch.cat([voxels, pos_grid_tensor], dim=0)
            else:
                raise ValueError("Invalid add_xy_channels: {}".format(
                    args.add_xy_channels))

            x_dict['batch_voxel_list'].append(voxels)

        return x_dict

    def collate_batch_data_to_tensors(self, proc_batch_dict):
        '''Collate processed batch into tensors.'''
        # Now collate the batch data together
        x_tensor_dict = {}
        x_dict = proc_batch_dict
        device = self.config.get_device()
        args = self.config.args

        x_tensor_dict['batch_voxel'] = torch.stack(x_dict['batch_voxel_list']).to(device)
        x_tensor_dict['batch_obj_before_pose_list'] = torch.Tensor(
            x_dict['batch_obj_before_pose_list']).to(device)
        x_tensor_dict['batch_obj_delta_pose_list'] = torch.Tensor(
            x_dict['batch_obj_delta_pose_list']).to(device)
        x_tensor_dict['batch_action_list'] = torch.Tensor(
            x_dict['batch_action_list']).to(device)
        x_tensor_dict['batch_action_index_list'] = torch.LongTensor(
            x_dict['batch_action_index_list']).to(device)
        x_tensor_dict['batch_bb_list'] = torch.Tensor(
            x_dict['batch_bb_list']).to(device)
        if x_dict.get('batch_obj_delta_pose_class_list') is not None:
            x_tensor_dict['batch_obj_delta_pose_class_list'] = torch.LongTensor(
                x_dict['batch_obj_delta_pose_class_list']).to(device)

        return x_tensor_dict

    def run_model_on_contrastive_batch(self, x_tensor_dict, batch_size,
                                       train=True, save_preds=True):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        anchor_emb = self.model.forward_image(x_tensor_dict[0]['batch_voxel'])
        sim_img_emb = self.model.forward_image(x_tensor_dict[1]['batch_voxel'])
        diff_img_emb = self.model.forward_image(x_tensor_dict[2]['batch_voxel'])

        triplet_loss = args.weight_contrastive_loss * \
            self.triplet_loss(anchor_emb, sim_img_emb, diff_img_emb)

        total_loss = triplet_loss

        if train:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

        batch_result_dict = {
            'triplet_loss': triplet_loss.item(),
        }

        return batch_result_dict
    
    def get_image_emb(self, x_tensor_dict):
        '''Get image embedding for dict with voxel tensors.
        
        Return: Tensor of image embeddings.
        '''
        voxel_data = x_tensor_dict['batch_voxel']
        with torch.no_grad():
            img_emb = self.model.forward_image(voxel_data)
        return img_emb

    def run_model_on_batch_use_contrastive_loss(self, 
                                                x_tensor_dict,
                                                batch_size,
                                                train=False,
                                                save_preds=False):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        voxel_data = x_tensor_dict['batch_voxel']
        img_emb = self.model.forward_image(voxel_data)

        loss = args.weight_contrastive_loss * get_contrastive_loss(
            img_emb,
            x_tensor_dict['batch_obj_delta_pose_list'][:, :3],
            x_tensor_dict['batch_action_index_list'],
            args.contrastive_margin,
            args.contrastive_gt_pose_margin,
        )
        total_loss = loss

        if train:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

        batch_result_dict = {
            'triplet_loss': loss.item(),
            'total_loss': total_loss.item(),
        }
        batch_result_dict['img_emb'] = img_emb
        batch_result_dict['img_action_emb'] = img_emb

        return batch_result_dict


    def run_model_on_batch(self,
                           x_tensor_dict,
                           batch_size,
                           contrastive_x_tensor_dict=None,
                           train=False,
                           save_preds=False):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        voxel_data = x_tensor_dict['batch_voxel']
        img_emb = self.model.forward_image(voxel_data)
        # TODO: Add hinge loss
        # img_emb = img_emb.squeeze()

        if args.use_bb_in_input:
            img_emb_with_action = torch.cat([
                img_emb,
                x_tensor_dict['batch_bb_list'],
                x_tensor_dict['batch_obj_before_pose_list'],
                x_tensor_dict['batch_action_list']], dim=1)
        else:
            img_emb_with_action = torch.cat([
                img_emb,
                x_tensor_dict['batch_action_list']], dim=1)

        if args.use_contrastive_loss and train:
            assert contrastive_x_tensor_dict is not None, "Contrastive data is None"
            sim_img_emb = self.model.forward_image(
                contrastive_x_tensor_dict[1]['batch_voxel']
            )
            diff_img_emb = self.model.forward_image(
                contrastive_x_tensor_dict[2]['batch_voxel']
            )

            triplet_loss = args.weight_contrastive_loss * \
                self.triplet_loss(img_emb, sim_img_emb, diff_img_emb)
        else:
            triplet_loss = 0

        img_action_emb = self.model.forward_image_with_action(img_emb_with_action)

        pred_delta_pose = self.model.forward_predict_delta_pose(img_action_emb)
        # pose_pred_loss = args.weight_bb * self.pose_pred_loss(
        #     pred_delta_pose,  x_tensor_dict['batch_obj_delta_pose_list'])

        if args.loss_type == 'regr':
            position_pred_loss = args.weight_pos * self.pose_pred_loss(
                pred_delta_pose[:, :3],
                x_tensor_dict['batch_obj_delta_pose_list'][:, :3]
            )
            angle_pred_loss = args.weight_angle * self.pose_pred_loss(
                pred_delta_pose[:, 3:],
                x_tensor_dict['batch_obj_delta_pose_list'][:, 3:]
            )
        elif args.loss_type == 'classif':
            n_classes = args.classif_num_classes

            true_label_x = x_tensor_dict['batch_obj_delta_pose_class_list'][:, 0]
            true_label_y = x_tensor_dict['batch_obj_delta_pose_class_list'][:, 1]
            position_pred_loss_x = args.weight_pos * self.pose_pred_loss(
                pred_delta_pose[:, :n_classes], true_label_x)
            position_pred_loss_y = args.weight_pos * self.pose_pred_loss(
                pred_delta_pose[:, n_classes:], true_label_y)
            position_pred_loss = position_pred_loss_x + position_pred_loss_y

            angle_pred_loss = 0

        else:
            raise ValueError("Invalid loss type: {}".format(args.loss_type))

        # img_action_with_delta_pose = torch.cat(
        #    [img_action_emb, x_tensor_dict['batch_other_bb_pred_list']], dim=1)
        # pred_img_emb = self.model.forward_predict_original_img_emb(
        #    img_action_with_delta_pose)
        #inv_model_loss = args.weight_inv_model * self.inv_model_loss(
        #   pred_img_emb, img_emb)
        inv_model_loss = 0.

        # total_loss = pose_pred_loss + inv_model_loss
        total_loss = position_pred_loss + angle_pred_loss + triplet_loss

        if train:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

        batch_result_dict['img_emb'] = img_emb
        batch_result_dict['img_action_emb'] = img_action_emb
        batch_result_dict['pred_delta_pose'] = pred_delta_pose
        batch_result_dict['pose_pred_loss'] = position_pred_loss + angle_pred_loss
        batch_result_dict['position_pred_loss'] = position_pred_loss
        if args.loss_type == 'classif':
            batch_result_dict['position_pred_loss_x'] = position_pred_loss_x
            batch_result_dict['position_pred_loss_y'] = position_pred_loss_y

            # Save conf matrix.
            _, pred_x = torch.max(pred_delta_pose[:, :n_classes], dim=1)
            _, pred_y = torch.max(pred_delta_pose[:, n_classes:], dim=1)
            conf_x = np.zeros((n_classes, n_classes), dtype=np.int32)
            conf_y = np.zeros((n_classes, n_classes), dtype=np.int32)
            for b in range(true_label_x.size(0)):
                conf_x[true_label_x[b].item(), pred_x[b].item()] += 1
                conf_y[true_label_y[b].item(), pred_y[b].item()] += 1
            batch_result_dict['conf_x'] = conf_x
            batch_result_dict['conf_y'] = conf_y


        batch_result_dict['angle_pred_loss'] = angle_pred_loss
        batch_result_dict['inv_model_loss'] = inv_model_loss
        batch_result_dict['triplet_loss'] = triplet_loss
        batch_result_dict['total_loss'] = total_loss

        if not train and save_preds:
            batch_result_dict['pos_gt'] = \
                to_numpy(x_tensor_dict['batch_obj_delta_pose_list'][:, :3])
            batch_result_dict['pos_class_gt'] = \
                to_numpy(x_tensor_dict['batch_obj_delta_pose_class_list'])
            if args.loss_type == 'regr':
                batch_result_dict['pos_pred'] = to_numpy(pred_delta_pose[:, :3])
            else:
                batch_result_dict['pos_pred'] = to_numpy(pred_delta_pose)

        return batch_result_dict

    def print_train_update_to_console(self, current_epoch, num_epochs,
                                      curr_batch, num_batches, train_step_count,
                                      batch_result_dict, train=True):
        args = self.config.args
        e, batch_idx = current_epoch, curr_batch
        total_loss = batch_result_dict['total_loss']

        color_fn = bcolors.okblue if train else bcolors.c_yellow
        if train_step_count % args.print_freq_iters == 0:
            print(color_fn(
                "[{}/{}], \t Batch: [{}/{}], \t    total_loss: {:.4f}".format(
                    e, num_epochs, batch_idx, num_batches, total_loss)))


    def plot_train_update_to_tensorboard(self, x_dict, x_tensor_dict,
        batch_result_dict, step_count, log_prefix='',
        plot_images=True, plot_loss=True):
        args = self.config.args

        if plot_images:
            # self.logger.summary_writer.add_images(
            #         log_prefix + '/image/input_image',
            #         x_tensor_dict['batch_img'].clone().cpu()[:,:3,:,:],
            #         step_count)
            pass

        if plot_loss:
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/total_loss',
                batch_result_dict['total_loss'],
                step_count,
            )

            if batch_result_dict.get('avg_loss') is not None:
                self.logger.summary_writer.add_scalar(
                    log_prefix+'loss/avg_loss',
                    batch_result_dict['avg_loss'],
                    step_count,
                )

            if args.use_contrastive_loss:
                self.logger.summary_writer.add_scalar(
                    log_prefix+'loss/triplet_loss',
                    batch_result_dict['triplet_loss'],
                    step_count,
                )

    def get_online_contrastive_samples(self, batch_online_contrastive_data,
                                       contrastive_data_log_idx,
                                       sample_hard_examples_only=True,
                                       sample_all_examples=False,
                                       squared=True):
        assert ((sample_hard_examples_only and not sample_all_examples) or
                (not sample_hard_examples_only and sample_all_examples))

        num_data = len(batch_online_contrastive_data['info'])
        all_emb = np.vstack(batch_online_contrastive_data['img_emb'])
        emb_dot_prod = np.dot(all_emb, all_emb.T)

        sq_norm = np.diagonal(emb_dot_prod)

        emb_dist = sq_norm[None, :] - 2*emb_dot_prod + sq_norm[:, None]
        emb_dist = np.maximum(emb_dist, 0.0)

        if not squared:
            emb_dist = np.sqrt(emb_dist)
        # Now find the tronline_contrastive_gt_pose_marginiplets in here.
        action_idx_arr = online_contrastive_gt_pose_marginnp.array([
            d['action_idxonline_contrastive_gt_pose_margin'] for d in batch_online_contrastive_data['info']
        ], dtype=np.int32)
        gt_pos_arr = np.array([
            d['delta_pose'] for d in batch_online_contrastive_data['info']
        ])

        # Take m = [2, 1, 2, 3] and run the below code.
        # (m * (m[None, :] == m[:, None]))[:, None, :] == m[:, None]  
        # Add 1 to remove any 0's 
        idx_arr = (action_idx_arr + 1)  
        action_label_mask = idx_arr * (idx_arr[None, :] == idx_arr[:, None])
        # label_mask (i,j,k) is True iff idx_arr[i] == idx_arr[j] == idx_arr[k]
        action_label_mask = action_label_mask[:, None, :] == idx_arr[:, None]

        gt_pos_dist = get_euclid_dist_matrix_for_data(gt_pos_arr[:, :3])

        # Get masked distances
        # masked_gt_pos_dist = gt_pos_dist * sim_action_labels
        # masked_emb_dist = emb_dist * sim_action_labels

        gt_dist_diff = gt_pos_dist[:, :, None] - gt_pos_dist[:, None, :]
        gt_dist_diff = gt_dist_diff * action_label_mask
        gt_margin = self.config.args.contrastive_gt_pose_margin
        gt_triplet_idxs = np.where(-gt_dist_diff > gt_margin)

        emb_margin = self.contrastive_margin
        emb_dist_diff = emb_dist[:, :, None] - emb_dist[:, None, :]
        emb_dist_diff = emb_dist_diff * action_label_mask
        emb_triplet_idxs = np.where(
            (emb_dist_diff > 1e-4) & (-emb_dist_diff < emb_margin))

        # Take an & over gt_dist_diff and emb_dist_diff and use those indexes.
        if sample_hard_examples_only:
            valid_triplet_idxs = np.where((-gt_dist_diff > gt_margin) &
                                        ((-emb_dist_diff < emb_margin) &
                                        (emb_dist_diff > 1e-4)))
        elif sample_all_examples:
            valid_triplet_idxs = np.where((-gt_dist_diff > gt_margin))
        else:
            raise ValueError("Unclear which triplets to sample.")
            

        # Now get the actual samples?
        num_triplets = valid_triplet_idxs[0].shape[0]
        if num_triplets > 0:
            valid_triplet_idxs_arr = np.vstack(valid_triplet_idxs).transpose()
        else:
            return None

        self.log_sampled_contrastive_examples(
            contrastive_data_log_idx,   
            batch_online_contrastive_data,
            valid_triplet_idxs_arr,
            action_idx_arr,
            gt_pos_arr,
            gt_pos_dist,
            all_emb,
            emb_dist)


        return valid_triplet_idxs_arr

    def log_sampled_contrastive_examples(self, 
                                         contrastive_data_log_idx,   
                                         batch_online_contrastive_data,
                                         triplet_idxs,
                                         action_idx_arr,
                                         gt_pose_arr,
                                         gt_pose_matrix,
                                         pred_emb_arr,
                                         pred_emb_dist_matrix):
        '''Save some of the sampled contrasive examples.'''
        all_action_idxs = np.unique(action_idx_arr)
        data_by_action_idx = {}

        def _get_data_contrastive_batch(idx):
            info_dict = batch_online_contrastive_data['info'][idx]


        for action in all_action_idxs:
            action_mask = action_idx_arr == action
            action_mask_idx = np.where(action_mask)
    
        max_log_samples = 512
        max_triplets = triplet_idxs.shape[0]
        for i in range(max_log_samples):
            trip_i = np.random.randint(max_triplets)
            # triplet is length 3 array with (anchor, same, diff)
            triplet = triplet_idxs[trip_i]
            anchor_idx, same_idx, diff_idx = triplet[0], triplet[1], triplet[2]
            anchor_data = batch_online_contrastive_data['info'][anchor_idx]
            same_data = batch_online_contrastive_data['info'][same_idx]
            diff_data = batch_online_contrastive_data['info'][diff_idx]

            triplet_data = [anchor_data, same_data, diff_data]
            
            triplet_data_dict = {
                'path': [t['path'] for t in triplet_data],
                'delta_pose': [t['delta_pose'] for t in triplet_data],
                'action': [t['action'] for t in triplet_data],
                'action_idx': [t['action_idx'] for t in triplet_data],
            }
            assert anchor_data['action_idx'] == same_data['action_idx']
            assert anchor_data['action_idx'] == diff_data['action_idx']

            D = gt_pose_matrix
            a, b, c = anchor_idx, same_idx, diff_idx
            triplet_data_dict['gt_pose_dist'] = \
                [D[a, a], D[a, b], D[a, c], 
                 D[b, a], D[b, b], D[b, c],
                 D[c, a], D[c, b], D[c, c]]

            assert (D[a, c] - D[a, b]) > self.config.args.contrastive_gt_pose_margin

            D = pred_emb_dist_matrix
            triplet_data_dict['emb_dist'] = \
                [D[a, a], D[a, b], D[a, c], 
                 D[b, a], D[b, b], D[b, c],
                 D[c, a], D[c, b], D[c, c]]
                
            if data_by_action_idx.get(anchor_data['action_idx']) is None:
                data_by_action_idx[anchor_data['action_idx']] = []

            data_by_action_idx[anchor_data['action_idx']].append(
                triplet_data_dict)
        
        # Now save this dict
        data_pkl_path = os.path.join(args.result_dir, 'contrastive_data', 
                                     '{}_data.pkl'.format(contrastive_data_log_idx))
        if not os.path.exists(os.path.dirname(data_pkl_path)):
            os.makedirs(os.path.dirname(data_pkl_path))
        
        with open(data_pkl_path, 'wb') as data_f:
            pickle.dump(data_by_action_idx, data_f, protocol=2)
            print("Did save sampled contrastive data: {}".format(data_pkl_path))

        return data_by_action_idx

    def estimate_triplet_loss(self, train, contrastive_step_count, 
                              max_data_size=None):
        '''Estimate the total triplet loss on part of the data.
        '''
        args = self.config.args
        dataloader = self.dataloader
        data_size = dataloader.get_data_size(train)
        if max_data_size is not None:
            data_size = min(data_size, max_data_size)
        iter_order = np.arange(data_size)
        batch_size = 32
        num_batches = data_size // batch_size
        data_idx = 0
      
        all_train_data = {'info': [], 'img_emb': []}
        print(bcolors.c_green("==== Will validate on subset of train data ===="))
        for batch_idx in tqdm(range(num_batches)):
            # Get raw data from the dataloader.
            batch_data = []
            batch_get_start_time = time.time()

            while len(batch_data) < batch_size and data_idx < len(iter_order):
                data = dataloader.get_train_data_at_idx(
                    iter_order[data_idx], train=train)
                batch_data.append(data)
                # Save data that we think will be useful for online
                # contrastive loss.
                online_contrastive_data = {
                    'data_idx': iter_order[data_idx],
                    'train': train,
                    'path': data['path'],
                    'info': data['info'],
                    'before_pose': data['before_pose'],
                    'after_pose': data['after_pose'],
                    'delta_pose': data['delta_pose'],
                    'action': data['action'],
                    'action_idx': data['action_idx'],
                }
                all_train_data['info'].append(online_contrastive_data)
                data_idx = data_idx + 1
            batch_get_end_time = time.time()
            # Process raw batch data
            proc_data_start_time = time.time()
            x_dict = self.process_raw_batch_data(batch_data)
            # Now collate the batch data together
            x_tensor_dict = self.collate_batch_data_to_tensors(x_dict)
            img_emb = self.get_image_emb(x_tensor_dict)
            all_train_data['img_emb'].append(to_numpy(img_emb))
        
        contrastive_samples = self.get_online_contrastive_samples(
            all_train_data, 0,
            sample_hard_examples_only=False,
            sample_all_examples=True,
        )
        assert contrastive_samples is not None
        print("Will get triplet loss on data: {}".format(
            contrastive_samples.shape[0]))
        batch_result_dict = self.train_online_triplet_loss(
            all_train_data,
            contrastive_samples,
            contrastive_step_count,
            num_iters=1,
            batch_size=args.online_contrastive_batch_size,
            train=False,
            log_prefix='/test/',
            permute_data=False,
        )
        self.logger.summary_writer.add_scalar(
            '/test/triplet_loss/sample_count',
            contrastive_samples.shape[0],
            contrastive_step_count, 
        )
        print(bcolors.c_green(
            "Num contrastive samples: {}, total_loss: {:.4f}".format(
                contrastive_samples.shape[0],
                batch_result_dict['total_loss_sum'],
        )))

    def train_online_triplet_loss(self,
                                  batch_online_contrastive_data,
                                  triplet_idxs,
                                  step_count,
                                  num_iters=1,
                                  batch_size=8,
                                  train=True,
                                  permute_data=True, 
                                  log_prefix='/train/'):
        args = self.config.args
        dataloader = self.dataloader

        batch_result_dict = {}
        for iter_idx in range(num_iters):
            data_size = triplet_idxs.shape[0]
            # get number of batches.
            num_batches = (data_size // batch_size)
            if data_size % batch_size != 0:
                num_batches += 1 
            
            data_order = np.arange(data_size)
            if permute_data:
                data_order = np.random.permutation(data_order)
            data_idx = 0
            total_loss = 0
            print_freq, save_freq = max(num_batches // 10, 1), max(num_batches // 5, 1)
            
            for batch_num in range(num_batches):
                # Get contrastive data from the dataloader
                batch_data = []
                batch_get_start_time = time.time()
                for b in range(batch_size):
                    triplet = triplet_idxs[data_order[data_idx]]
                    data_anchor = dataloader.get_train_data_at_idx(triplet[0])
                    data_sim = dataloader.get_train_data_at_idx(triplet[1])
                    data_diff = dataloader.get_train_data_at_idx(triplet[2])

                    batch_data.append((data_anchor, data_sim, data_diff))
                    data_idx = data_idx + 1

                    if data_idx >= data_size:
                        break

                x_dict = self.process_contrastive_raw_batch_data(batch_data)
                x_tensor_dict = {}
                for key, contrastive_x_dict in x_dict.items():
                    x_tensor_dict[key] = self.collate_batch_data_to_tensors(
                        contrastive_x_dict)

                model_run_start_time = time.time()
                batch_result_dict = self.run_model_on_batch_use_contrastive_loss(
                    x_tensor_dict,
                    batch_size,
                    train=True,
                    save_preds=True)
                model_run_end_time = time.time()
                total_loss += batch_result_dict['total_loss']

                if train:
                    if batch_num % print_freq == 0:
                        print(bcolors.c_red(
                            "batch: [{}/{}] data_size: {} batch_size: {}"
                              "\t triplet loss: {:.4f}".format(
                            batch_num, num_batches, data_size, batch_size,
                            batch_result_dict['triplet_loss'])))

                    if batch_num % save_freq == 0:
                        self.logger.summary_writer.add_scalar(
                            log_prefix+'triplet_loss/loss',
                            batch_result_dict['triplet_loss'],
                            step_count
                        )
                        step_count += 1
                else:
                    if batch_num % print_freq == 0:
                        print(bcolors.c_yellow(
                            "batch: [{}/{}] data_size: {} batch_size: {}"
                              "\t test triplet loss: {:.4f}".format(
                            batch_num, num_batches, data_size, batch_size,
                            batch_result_dict['triplet_loss'])))
        
        self.logger.summary_writer.add_scalar(
            log_prefix+'triplet_loss/total_loss_sum', total_loss, step_count)
        self.logger.summary_writer.add_scalar(
            log_prefix+'triplet_loss/total_loss_avg', total_loss/num_batches, step_count)
        if train:
            print(bcolors.c_red(
                "Train Triplet loss iter done \t total_loss,"
                "sum: {:.4f}, avg: {:.4f}".format(
                    total_loss, total_loss/num_batches)))
        else:
            print(bcolors.c_yellow(
                "Test Triplet loss iter done \t total_loss, "
                "sum: {:.4f}, avg: {:.4f}".format(
                    total_loss, total_loss/num_batches)))

        batch_result_dict['step_count'] = step_count
        batch_result_dict['total_loss_sum'] = total_loss
        batch_result_dict['total_loss_avg'] = total_loss/num_batches

        return batch_result_dict


    def train(self, train=True, viz_images=False, save_embedding=True,
              log_prefix=''):
        print("Begin training")
        args = self.config.args
        log_freq_iters = args.log_freq_iters if train else 10
        dataloader = self.dataloader
        device = self.config.get_device()
        train_data_size = dataloader.get_data_size(train)
        train_data_idx_list = list(range(0, train_data_size))

        online_contrastive_freq = 1

        # Reset log counter
        train_step_count, test_step_count = 0, 0
        contrastive_step_count, contrastive_data_log_idx = 0, 0
        self.model.to(device)
        # Switch off the norm layers in ResnetEncoder

        result_dict = {
            'emb': {
                'img_emb': [],
                'img_action_emb': [],
            },
            'data_info': {
                'path': [],
                'info': [],
                'action': [],
            },
            'output': {
                'pos_gt': [],
                'pos_pred': [],
            },
            'conf': {
                'train_x': [],
                'test_x': [],
                'train_y': [],
                'test_y': [],
            }
        }
        num_epochs = args.num_epochs if train else 1
        loss_list = []

        for e in range(num_epochs):
            if train:
                iter_order = np.random.permutation(train_data_idx_list)
            else:
                iter_order = np.arange(train_data_size)

            batch_size = args.batch_size if train else 32
            num_batches = train_data_size // batch_size
            data_idx = 0

            for batch_idx in range(num_batches):
                # Get raw data from the dataloader.
                batch_data = []

                # for b in range(batch_size):
                batch_get_start_time = time.time()

                while len(batch_data) < batch_size and data_idx < len(iter_order):
                    actual_data_idx = iter_order[data_idx]
                    data = dataloader.get_train_data_at_idx(actual_data_idx, train=train)
                    batch_data.append(data)
                    data_idx = data_idx + 1

                batch_get_end_time = time.time()
                # print("Data time: {:.4f}".format(
                    # batch_get_end_time - batch_get_start_time))

                # Process raw batch data
                proc_data_start_time = time.time()
                x_dict = self.process_raw_batch_data(batch_data)
                # Now collate the batch data together
                x_tensor_dict = self.collate_batch_data_to_tensors(x_dict)
                proc_data_end_time = time.time()

                run_batch_start_time = time.time()
                batch_result_dict = self.run_model_on_batch_use_contrastive_loss(
                    x_tensor_dict,
                    batch_size,
                    train=train,
                    save_preds=True)
                run_batch_end_time = time.time()

                if not train and save_embedding:
                    for b in range(batch_size):
                        for k in ['path', 'info', 'action']:
                            result_dict['data_info'][k].append(batch_data[b][k])

                        result_dict['emb']['img_emb'].append(
                            batch_result_dict['img_emb'][b].detach().cpu().numpy())

                self.print_train_update_to_console(
                    e, num_epochs, batch_idx, num_batches,
                    train_step_count, batch_result_dict)

                plot_images = viz_images and train \
                    and train_step_count %  log_freq_iters == 0
                plot_loss = train \
                    and train_step_count % args.print_freq_iters == 0

                if plot_loss:
                    batch_result_dict['avg_loss'] = np.mean(loss_list)
                    loss_list = []
                else:
                    loss_list.append(batch_result_dict['total_loss'])

                if train:
                    self.plot_train_update_to_tensorboard(
                        x_dict, x_tensor_dict, batch_result_dict,
                        train_step_count,
                        plot_loss=plot_loss,
                        plot_images=plot_images,
                    )

                if train:
                    if train_step_count % log_freq_iters == 0:
                        self.log_model_to_tensorboard(train_step_count)

                    # Save trained models
                    if train_step_count % args.save_freq_iters == 0:
                        self.save_checkpoint(train_step_count)

                    # Run current model on val/test data.
                    if train_step_count % args.test_freq_iters == 0:
                        # Remove old stuff from memory
                        x_dict = None
                        x_tensor_dict = None
                        batch_result_dict = None
                        torch.cuda.empty_cache()

                        num_batch_test, test_batch_size = 8, 128
                        test_data_size = self.dataloader.get_data_size(
                                train=False)
                        test_iter_order = np.arange(test_data_size)
                        test_data_idx = 0
                        total_test_loss = 0
                        self.model.eval()

                        print(bcolors.c_red("==== Test begin ==== "))
                        for test_e in range(num_batch_test):
                            batch_data = []

                            b = 0
                            while len(batch_data) < batch_size and \
                                test_data_idx < len(test_iter_order):
                                data = dataloader.get_train_data_at_idx(
                                    test_iter_order[test_data_idx], train=False)
                                batch_data.append(data)
                                test_data_idx = test_data_idx + 1
                                b = b + 1

                            # Process raw batch data
                            x_dict = self.process_raw_batch_data(batch_data)
                            batch_data = None

                            # Now collate the batch data together
                            x_tensor_dict = self.collate_batch_data_to_tensors(
                                x_dict)
                            with torch.no_grad():
                                batch_result_dict = self.run_model_on_batch_use_contrastive_loss(
                                    x_tensor_dict, test_batch_size, train=False)
                                total_test_loss += batch_result_dict['total_loss']

                            self.print_train_update_to_console(
                                e, num_epochs, test_e, num_batch_test,
                                train_step_count, batch_result_dict, 
                                train=False)

                            plot_images = test_e == 0
                            self.plot_train_update_to_tensorboard(
                                x_dict, x_tensor_dict, batch_result_dict,
                                test_step_count,
                                plot_loss=True,
                                plot_images=plot_images,
                                log_prefix='/test/'
                            )

                            test_step_count += 1

                        print(bcolors.c_red("==== Test end ==== "))

                        # Plot the total loss on the entire dataset. Hopefull,
                        # this would decrease over time.
                        self.logger.summary_writer.add_scalar(
                            '/test/all_batch_loss/loss_avg',
                            total_test_loss / max(num_batch_test, 1),
                            test_step_count)
                        self.logger.summary_writer.add_scalar(
                            '/test/all_batch_loss/loss',
                            total_test_loss,
                            test_step_count)

                    x_dict = None
                    x_tensor_dict = None
                    batch_result_dict = None
                    torch.cuda.empty_cache()
                    self.model.train()

                train_step_count += 1
                torch.cuda.empty_cache()

        return result_dict

def update_test_args_with_train_args(test_args, train_args):
    assert train_args.z_dim == test_args.z_dim
    assert train_args.add_xy_channels == test_args.add_xy_channels

def main(args):
    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    # Load the args from previously saved checkpoint
    if len(args.checkpoint_path) > 0:
        config_pkl_path = os.path.join(args.result_dir, 'config.pkl')
        with open(config_pkl_path, 'rb') as config_f:
            old_args = pickle.load(config_f)
            print(bcolors.c_red("Did load config: {}".format(config_pkl_path)))
        # Now update the current args with the old args for the train model
        update_test_args_with_train_args(args, old_args)

    config = BaseVAEConfig(args, dtype=dtype)
    create_log_dirs(config)

    trainer = VoxelRelationTrainer(config)

    if len(args.checkpoint_path) > 0:
        trainer.load_checkpoint(args.checkpoint_path)
        result_dict = trainer.train(train=False,
                                    viz_images=False,
                                    save_embedding=True)
        test_result_dir = os.path.join(
            os.path.dirname(args.checkpoint_path), '{}_result_{}'.format(
                args.cp_prefix, os.path.basename(args.checkpoint_path)[:-4]))
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        emb_h5_path = os.path.join(test_result_dir, 'result_emb.h5')
        emb_h5_f = h5py.File(emb_h5_path, 'w')
        result_h5_dict = {'emb': result_dict['emb'],
                          'output': result_dict['output']}
        recursively_save_dict_contents_to_group(emb_h5_f, '/', result_h5_dict)
        emb_h5_f.flush()
        emb_h5_f.close()
        print("Did save emb: {}".format(emb_h5_path))

        pkl_path = os.path.join(test_result_dir, 'result_info.pkl')
        with open(pkl_path, 'wb') as pkl_f:
            result_pkl_dict = {'data_info': result_dict['data_info']}
            pickle.dump(result_pkl_dict, pkl_f, protocol=2)
            print("Did save test info: {}".format(pkl_path))
    else:
        config_pkl_path = os.path.join(args.result_dir, 'config.pkl')
        config_json_path = os.path.join(args.result_dir, 'config.json')
        with open(config_pkl_path, 'wb') as config_f:
            pickle.dump((args), config_f, protocol=2)
            print(bcolors.c_red("Did save config: {}".format(config_pkl_path)))
        with open(config_json_path, 'w') as config_json_f:
            config_json_f.write(json.dumps(args.__dict__))

        # trainer.get_data_stats_for_classif()
        trainer.train(viz_images=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train for edge classification')
    parser.add_argument('--model_type', type=str, default='vae_recons',
                        choices=['vae',
                                 'sg2im',
                                 'visual_rel_sg2im',
                                 'multi_object_visual_rel_sg2im'],
                        help='Model type to use.')
    add_common_args_to_parser(parser,
                              cuda=True,
                              result_dir=True,
                              checkpoint_path=True,
                              num_epochs=True,
                              batch_size=True,
                              lr=True,
                              save_freq_iters=True,
                              log_freq_iters=True,
                              print_freq_iters=True,
                              test_freq_iters=True)

    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to hdf5 file.')
    parser.add_argument('--test_dir', type=str, default='',
                        help='Path to hdf5 file.')
    parser.add_argument('--cp_prefix', type=str, default='',
                        help='Prefix to be used to save embeddings.')

    parser.add_argument('--z_dim', type=int, default=128,
                        help='Embedding size to extract from image.')

    # Loss weights
    parser.add_argument('--scaled_mse_loss', type=str2bool, default=False,
                        help='Use scaled MSE loss in contrast to regular MSE.')
    parser.add_argument('--classif_num_classes', type=int, default=5,
                        help='Number of classes for classification.')
    parser.add_argument('--loss_type', type=str, default='regr',
                        choices=['classif', 'regr'],
                        help='Loss type to use')
    parser.add_argument('--weight_pos', type=float, default=1.0,
                        help='Weight for pos pred loss.')
    parser.add_argument('--weight_angle', type=float, default=1.0,
                        help='Weight for orientation change pred loss.')
    parser.add_argument('--weight_inv_model', type=float, default=1.0,
                        help='Weight for inverse model loss.')
    parser.add_argument('--weight_contrastive_loss', type=float, default=1.0,
                        help='Weight to use for contrastive loss.')

    parser.add_argument('--add_xy_channels', type=int, default=0,
                        choices=[0, 1],
                        help='0: no xy append, 1: xy append '
                             '2: xy centered on bb')
    parser.add_argument('--use_bb_in_input', type=int, default=1, choices=[0,1],
                        help='Use bb in input')
    # 0: sparse voxels that is the scene size is fixed and we have the voxels in there.
    # 1: dense voxels, such that the given scene is rescaled to fit the max size.
    parser.add_argument('--voxel_datatype', type=int, default=0,
                         choices=[0, 1],
                         help='Voxel datatype to use.')
    parser.add_argument('--octree_0_multi_thread', type=str2bool, default=0,
                         help='Use multithreading in octree 0.')
    parser.add_argument('--use_spatial_softmax', type=str2bool, default=False,
                         help='Use spatial softmax.')
    parser.add_argument('--save_full_3d', type=str2bool, default=False,
                        help='Save 3d voxel representation in memory.')
    parser.add_argument('--expand_voxel_points', type=str2bool, default=False,
                        help='Expand voxel points to internal points of obj.')

    parser.add_argument('--use_contrastive_loss', type=str2bool, default=False,
                        help='Use contrastive loss during training.')
    parser.add_argument('--use_online_contrastive_loss', type=str2bool, default=True,
                        help='Use online contrastive loss during training.')
    parser.add_argument('--contrastive_margin', type=float, default=1.0,
                        help='Margin to use in triplet loss.')
    parser.add_argument('--online_contrastive_batch_size', type=int, default=32,
                        help='Batch size for online contrastive training.')
    parser.add_argument('--online_contrastive_test_freq', type=int, default=10,
                        help='Frequency to estimate contrative loss on test set.')
    parser.add_argument('--contrastive_gt_pose_margin', default=1.0, type=float,
                        help='GT pose margin for online contrastive loss')

    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=100)

    main(args)
