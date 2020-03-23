import numpy as np
import open3d

import argparse
import pickle
import h5py
import sys
import os
import pprint
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import torch.optim as optim
import time
import json
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append(os.getcwd())

from robot_learning.logger.tensorboardx_logger import TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network, to_numpy
from utils.colors import bcolors
from utils.data_utils import str2bool
from utils.data_utils import recursively_save_dict_contents_to_group
from utils.data_utils import convert_list_of_array_to_dict_of_array_for_hdf5
from utils.image_utils import get_image_tensor_mask_for_bb

from action_relation.dataloader.real_robot_dataloader import AllPairVoxelDataloader 
from action_relation.model.losses import WeightedBCELoss
from action_relation.model.unscaled_voxel_model import PrecondEmbeddingModel
from action_relation.model.unscaled_voxel_model import UnscaledVoxelModel, UnscaledPrecondVoxelModel
from action_relation.model.unfactored_scene_voxel_model import UnfactoredSceneEmbeddingModel
from action_relation.model.unfactored_scene_voxel_model import UnfactoredSceneClassifModel
from action_relation.model.unfactored_scene_voxel_model import Resnet18EmbModel
from action_relation.model.voxel_model import VoxelModel
from action_relation.model.factored_scene_voxel_model import FactoredPairwiseEmbeddingModel
from action_relation.model.factored_scene_voxel_model import FactoredPairwiseAttentionEmbeddingModel
from action_relation.model.factored_scene_voxel_model import FactoredPairwiseEmbeddingWithIdentityLabelModel
from action_relation.trainer.train_voxels_online_contrastive import create_model_with_checkpoint
from action_relation.trainer.train_voxels_online_contrastive import create_voxel_trainer_with_checkpoint
from action_relation.utils.data_utils import get_euclid_dist_matrix_for_data

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer


ALL_OBJ_PAIRS_GNN = 'all_object_pairs_gnn'
ALL_OBJ_PAIRS_GNN_RAW_INFO = 'all_object_pairs_gnn_raw_obj_info'


def save_emb_data_to_h5(result_dir, result_dict):
    emb_h5_path = os.path.join(result_dir, 'train_result_emb.h5')
    emb_h5_f = h5py.File(emb_h5_path, 'w')

    # Create a new dictionary so that each scene emb which can have different
    # number of objects as compared to other scenes will be stored separately.
    result_h5_dict = {'emb': {}}
    for k, v in result_dict['emb'].items():
        if k != 'train_img_emb' and k != 'test_img_emb':
            result_h5_dict['emb'][k] = v
        else:
            assert type(v) is list
            result_h5_dict['emb'][k] = dict()
            for scene_i, scene_emb in enumerate(v):
                result_h5_dict['emb'][k][f'{scene_i:05d}'] = np.copy(scene_emb)

    result_h5_dict = {'emb': result_h5_dict['emb'] }
    recursively_save_dict_contents_to_group(emb_h5_f, '/', result_h5_dict)
    emb_h5_f.flush()
    emb_h5_f.close()
    pkl_path = os.path.join(result_dir, 'train_result_info.pkl')
    with open(pkl_path, 'wb') as pkl_f:
        pkl_output_dict = {'output': result_dict['output']}
        pickle.dump(pkl_output_dict, pkl_f, protocol=2)

    print(bcolors.c_blue("Did save emb data: {}".format(emb_h5_path)))


class MultiObjectVoxelPrecondTrainerE2E(BaseVAETrainer):
    def __init__(self, config):
        super(MultiObjectVoxelPrecondTrainerE2E, self).__init__(config)

        args = config.args

        # there are a couple of methods in AllPairVoxelDataloader that should be
        # implemented by any dataloader that needs to be used for multi-object
        # precond learning.
        self.dataloader = AllPairVoxelDataloader(
            config,
            voxel_datatype_to_use=args.voxel_datatype,
            load_all_object_pair_voxels=('all_object_pairs' in args.train_type),
            load_scene_voxels=(args.train_type == 'unfactored_scene' or 
                               args.train_type == 'unfactored_scene_resnet18'),
            )

        args = config.args
        
        # TODO: Use the arguments saved in the emb_checkpoint_dir to create 

        if args.train_type == 'all_object_pairs':
            self.emb_model = create_model_with_checkpoint(
                'simple_model',
                args.emb_checkpoint_path,
            )

            self.classif_model = PrecondEmbeddingModel(args.z_dim, args)
        elif args.train_type == 'all_object_pairs_g_f_ij':
            self.emb_model = create_model_with_checkpoint(
                'small_simple_model',
                args.emb_checkpoint_path,
            )

            self.classif_model = FactoredPairwiseEmbeddingModel(args.z_dim, args)
        elif args.train_type == 'all_object_pairs_g_f_ij_attn':
            self.emb_model = create_model_with_checkpoint(
                'small_simple_model',
                args.emb_checkpoint_path,
            )

            self.classif_model = FactoredPairwiseAttentionEmbeddingModel(args.z_dim, args)
        elif args.train_type == 'all_object_pairs_g_f_ij_cut_food':
            self.emb_model = create_model_with_checkpoint(
                'small_simple_model',
                args.emb_checkpoint_path,
            )
            self.classif_model = FactoredPairwiseEmbeddingWithIdentityLabelModel(
                args.z_dim, args)

        elif args.train_type == 'all_object_pairs_gnn':
            from action_relation.model.factored_scene_gnn_model import GNNNodeAndSceneClassifier
            self.emb_model = create_model_with_checkpoint(
                'small_simple_model',
                args.emb_checkpoint_path,
            )
            self.classif_model = GNNNodeAndSceneClassifier(args.z_dim, args)
            self.stable_obj_pred_loss = nn.CrossEntropyLoss()
        elif args.train_type == 'all_object_pairs_gnn_raw_obj_info':
            from action_relation.model.factored_scene_gnn_model import GNNVolumeBasedNodeAndSceneClassifier
            self.emb_model = create_model_with_checkpoint(
                'small_simple_model',
                args.emb_checkpoint_path,
            )
            self.classif_model = GNNVolumeBasedNodeAndSceneClassifier(args.z_dim, args)
            self.stable_obj_pred_loss = nn.CrossEntropyLoss()

        elif args.train_type == 'unfactored_scene':
            self.emb_model = UnfactoredSceneEmbeddingModel(args.z_dim, args)
            self.classif_model = UnfactoredSceneClassifModel(args.z_dim, args)
            assert args.emb_lr > 1e-5, "Invalid emb lr"
        elif args.train_type == 'unfactored_scene_resnet18':
            self.emb_model = Resnet18EmbModel(args.z_dim, args)
            self.classif_model = UnfactoredSceneClassifModel(args.z_dim, args)
            assert args.emb_lr > 1e-5, "Invalid emb lr"
        else:
            raise ValueError(f"Invalid train type: {args.train_type}")


        self.opt_emb = optim.Adam(self.emb_model.parameters(), lr=args.emb_lr)
        self.opt_classif = optim.Adam(self.classif_model.parameters(), lr=args.lr)
        classif_lr_lambda = lambda epoch: 0.995**epoch
        self.classif_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.opt_classif, classif_lr_lambda)
        emb_lr_lambda = lambda epoch: 0.995**epoch
        self.emb_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.opt_emb, emb_lr_lambda)

        if args.use_dynamic_bce_loss:
            self.precond_loss = WeightedBCELoss(
                pos_weight_is_dynamic=True, weight=torch.Tensor([1.1, 0.9]))
        else:
            self.precond_loss = nn.BCELoss()

    def get_model_list(self):
        return [self.emb_model, self.classif_model]

    def get_state_dict(self):
        return {
            'emb_model': self.emb_model.state_dict(),
            'classif_model': self.classif_model.state_dict(),
        }

    def set_model_device(self, device=torch.device("cpu")):
        model_list = self.get_model_list()
        for m in model_list:
            m.to(device)
    
    def set_model_to_train(self):
        model_list = self.get_model_list()
        for m in model_list:
            m.train()

    def set_model_to_eval(self):
        model_list = self.get_model_list()
        for m in model_list:
            m.eval()

    def save_checkpoint(self, epoch):
        cp_filepath = self.model_checkpoint_filename(epoch)
        torch.save(self.get_state_dict(), cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        cp_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.emb_model.load_state_dict(cp_models['emb_model'])
        self.classif_model.load_state_dict(cp_models['classif_model'])

    def log_model_to_tensorboard(self, train_step_count):
        '''Log weights and gradients of network to Tensorboard.'''
        for name, m in zip(['emb', 'classif'], [(self.emb_model, self.opt_emb), 
                                                (self.classif_model, self.opt_classif)]):
            model, opt = m[0], m[1] 
            model_l2_norm, model_grad_l2_norm = \
                get_weight_norm_for_network(model)
            self.logger.summary_writer.add_scalar(
                '/model/{}/weight'.format(name),
                model_l2_norm,
                train_step_count)
            self.logger.summary_writer.add_scalar(
                '/model/{}/grad'.format(name),
                model_grad_l2_norm,
                train_step_count)
            self.logger.summary_writer.add_scalar(
                '/model/{}/lr'.format(name),
                opt.param_groups[0]['lr'],
                train_step_count)

    def print_train_update_to_console(self, current_epoch, num_epochs,
                                      curr_batch, num_batches, train_step_count,
                                      batch_result_dict, train=True):
        args = self.config.args
        e, batch_idx = current_epoch, curr_batch
        total_loss = batch_result_dict['total_loss']

        color_fn = bcolors.okblue if train else bcolors.c_yellow

        if train_step_count % args.print_freq_iters == 0:
            if args.train_type == ALL_OBJ_PAIRS_GNN:
                print(color_fn(
                    "[{}/{}], \t Batch: [{}/{}], \t    total_loss: {:.4f} "
                    " \t precond: {:.4f},  stable_obj: {:.4f} ".format(
                        e, num_epochs, batch_idx, num_batches,
                        batch_result_dict['total_loss'], 
                        batch_result_dict['precond_loss'],
                        batch_result_dict['stable_obj_loss'])))
            else:
                print(color_fn(
                    "[{}/{}], \t Batch: [{}/{}], \t    total_loss: {:.4f}".format(
                        e, num_epochs, batch_idx, num_batches,
                        batch_result_dict['total_loss'])))

    def plot_train_update_to_tensorboard(self, x_dict, x_tensor_dict,
                                         batch_result_dict, step_count,
                                         log_prefix='', plot_images=True,
                                         plot_loss=True):
        args = self.config.args

        if plot_images:
            pass

        if len(log_prefix) == 0:
            for opt_key_val_pair in (('model_emb/lr', self.opt_emb),
                                     ('model_classif/lr', self.opt_classif)):
                self.logger.summary_writer.add_scalar(
                    opt_key_val_pair[0],
                    opt_key_val_pair[1].param_groups[0]['lr'],
                    step_count
                )

        if plot_loss:
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/total_loss',
                batch_result_dict['total_loss'],
                step_count
            )
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/precond_loss',
                batch_result_dict['precond_loss'],
                step_count
            )
            if args.train_type == ALL_OBJ_PAIRS_GNN:
                self.logger.summary_writer.add_scalar(
                    log_prefix+'loss/stable_obj_loss',
                    batch_result_dict['stable_obj_loss'],
                    step_count
                )


    def process_raw_voxels(self, voxel_data):
        args = self.config.args
        if len(voxel_data.size()) == 3:
            voxels = voxel_data.squeeze(0)
        elif len(voxel_data.size()) == 4:
            voxels = voxel_data
        else:
            raise ValueError("Invalid voxels size: {}".format(
                voxel_data.size()))

        if args.add_xy_channels == 0:
            pass
        elif args.add_xy_channels == 1:
            if args.voxel_datatype == 0:
                pos_grid_tensor = self.dataloader.pos_grid
                voxels = torch.cat([voxels, pos_grid_tensor], dim=0)
            # import pdb; pdb.set_trace()
        else:
            raise ValueError("Invalid add_xy_channels: {}".format(
                args.add_xy_channels))
        return voxels

    def process_raw_batch_data(self, batch_data):
        '''Process raw batch data and collect relevant objects in a dict.'''
        proc_batch_dict = {
            # Save scene index starting from 0
            'batch_scene_index_list': [],
            # Save input image
            'batch_voxel_list': [],
            # Save info for object pairs which are far apart?
            'batch_obj_pair_far_apart_list': [],
            # Save info for object positions
            'batch_obj_center_list': [],
            # Save precond output
            'batch_precond_label_list': [],
            # Save stable object ids
            'batch_precond_stable_obj_id_list': [],
            # Save scene_path
            'batch_scene_path_list': [],
            # Save all object pair path in a scene
            'batch_scene_all_object_pair_path_list': [],
        }

        args = self.config.args
        x_dict = proc_batch_dict

        for b, data in enumerate(batch_data):
            if 'all_object_pairs' in args.train_type:
                for voxel_idx, voxels in enumerate(data['all_object_pair_voxels']):
                    proc_voxels = self.process_raw_voxels(voxels)
                    x_dict['batch_voxel_list'].append(proc_voxels)
                    x_dict['batch_scene_index_list'].append(b)
                    x_dict['batch_obj_pair_far_apart_list'].append(
                        data['all_object_pair_far_apart_status'][voxel_idx]
                    )


                x_dict['batch_scene_all_object_pair_path_list'].append(
                    data['all_object_pair_path'])
                x_dict['batch_obj_center_list'].append(torch.Tensor(data['obj_center_list']))

            elif args.train_type == 'unfactored_scene' or \
                 args.train_type == 'unfactored_scene_resnet18':
                voxels = data['scene_voxels']
                proc_voxels = self.process_raw_voxels(voxels)
                x_dict['batch_voxel_list'].append(proc_voxels)
            else:
                raise ValueError(f"Invalid train type {args.train_type}")
        
            if args.train_type == 'all_object_pairs_gnn' or \
               args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
                x_dict['batch_precond_stable_obj_id_list'].append(
                    data['precond_stable_obj_ids'])

            x_dict['batch_precond_label_list'].append(data['precond_label'])
            x_dict['batch_scene_path_list'].append(data['scene_path'])

        return x_dict

    def collate_batch_data_to_tensors(self, proc_batch_dict):
        '''Collate processed batch into tensors.'''
        # Now collate the batch data together
        x_tensor_dict = {}
        x_dict = proc_batch_dict
        device = self.config.get_device()
        args = self.config.args

        if len(x_dict['batch_voxel_list']) > 0:
            x_tensor_dict['batch_voxel'] = torch.stack(
                x_dict['batch_voxel_list']).to(device)
            x_tensor_dict['batch_obj_pair_far_apart_list'] = torch.LongTensor(
                x_dict['batch_obj_pair_far_apart_list']).to(device)

        x_tensor_dict['batch_precond_label_list'] = torch.FloatTensor(
            x_dict['batch_precond_label_list']).to(device)
        x_tensor_dict['batch_scene_index_list'] = torch.LongTensor(
            x_dict['batch_scene_index_list']).to(device)
        
        if args.train_type == 'all_object_pairs_gnn' or \
           args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
            # This will be a list of tensors since each scene can have a variable
            # number of objects.
            x_tensor_dict['batch_precond_stable_obj_id_list'] = []
            for t in x_dict['batch_precond_stable_obj_id_list']:
                t = t.to(device)
                x_tensor_dict['batch_precond_stable_obj_id_list'].append(t)
        
        if x_dict.get('batch_obj_center_list') is not None:
            x_tensor_dict['batch_obj_center_list'] = x_dict['batch_obj_center_list']

        return x_tensor_dict

    def run_model_on_batch(self,
                           x_tensor_dict,
                           batch_size,
                           train=False,
                           save_preds=False,
                           save_emb=False):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        voxel_data = x_tensor_dict['batch_voxel']
        
        # Now join the img_emb into a list for each scene.
        if  'all_object_pairs' in args.train_type:
            # Get the embeddings
            if args.emb_lr <= 1e-6:
                with torch.no_grad():
                    img_emb = self.emb_model.forward_image(voxel_data)
            else:
                img_emb = self.emb_model.forward_image(voxel_data)

            if args.train_type == 'all_object_pairs_gnn':
                inp_emb = self.classif_model.forward_all_emb_with_backbone(img_emb)
            elif args.train_type ==  ALL_OBJ_PAIRS_GNN_RAW_INFO:
                inp_emb = img_emb
            else:
                inp_emb = img_emb

            scene_emb_list, scene_obj_pair_far_apart_list = [[]], [[]]
            scene_obj_label_list = [[]]
            for b in range(inp_emb.size(0) - 1):
                scene_emb_list[-1].append(inp_emb[b])
                scene_obj_pair_far_apart_list[-1].append(
                    x_tensor_dict['batch_obj_pair_far_apart_list'][b])

                if args.train_type == 'all_object_pairs_g_f_ij_cut_food':
                    if len(scene_obj_label_list[-1]) == 0:
                        scene_obj_label_list[-1].append(0)
                    elif len(scene_obj_label_list[-1]) == 1:
                        scene_obj_label_list[-1].append(1)
                    elif len(scene_obj_label_list[-1]) > 1:
                        scene_obj_label_list[-1].append(2)
                    else:
                        raise ValueError("Invalid")

                if x_tensor_dict['batch_scene_index_list'][b] != x_tensor_dict['batch_scene_index_list'][b+1]:
                    scene_emb_list.append([])
                    scene_obj_pair_far_apart_list.append([])
                    scene_obj_label_list.append([])

            # Add the last embedding correctly.
            scene_emb_list[-1].append(inp_emb[-1])
            scene_obj_pair_far_apart_list[-1].append(
                x_tensor_dict['batch_obj_pair_far_apart_list'][-1])
            if args.train_type == 'all_object_pairs_g_f_ij_cut_food':
                if len(scene_obj_label_list[-1]) == 0:
                    scene_obj_label_list[-1].append(0)
                elif len(scene_obj_label_list[-1]) == 1:
                    scene_obj_label_list[-1].append(1)
                elif len(scene_obj_label_list[-1]) > 1:
                    scene_obj_label_list[-1].append(2)
                else:
                    raise ValueError("Invalid")
            
            if args.train_type == 'all_object_pairs_g_f_ij':
                pred_precond = self.classif_model.forward_scene_emb_predict_precond(
                    scene_emb_list, scene_obj_pair_far_apart_list)
            elif args.train_type == 'all_object_pairs_g_f_ij_cut_food':
                pred_precond = self.classif_model.forward_scene_emb_predict_precond(
                    scene_emb_list, scene_obj_pair_far_apart_list, 
                    scene_obj_label_list=scene_obj_label_list)
            elif args.train_type == 'all_object_pairs_g_f_ij_attn':
                pred_precond = self.classif_model.forward_scene_emb_predict_precond(
                    scene_emb_list, scene_obj_pair_far_apart_list)
            elif args.train_type == 'all_object_pairs_gnn':
                obj_center_tensor_list = x_tensor_dict['batch_obj_center_list']
                pred_precond, pred_stable_obj_list = self.classif_model.forward_scene_emb_predict_precond(
                    scene_emb_list, scene_obj_pair_far_apart_list, 
                    obj_center_tensor_list=obj_center_tensor_list)
                pred_precond = torch.clamp(pred_precond, min=1e-10)
            elif args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
                obj_center_tensor_list = x_tensor_dict['batch_obj_center_list']
                pred_precond, pred_stable_obj_list = self.classif_model.forward_scene_emb_predict_precond(
                    scene_emb_list, scene_obj_pair_far_apart_list, 
                    obj_info_tensor_list=obj_center_tensor_list)
                pred_precond = torch.clamp(pred_precond, min=1e-10)
            else:
                raise ValueError(f"Invalid train type: {args.train_type}")

        elif args.train_type == 'unfactored_scene' or \
             args.train_type == 'unfactored_scene_resnet18':
            img_emb = self.emb_model.forward_image(voxel_data)
            pred_precond = self.classif_model.forward_scene_emb_predict_precond(
                img_emb)
            pred_precond = torch.clamp(pred_precond, min=1e-8)
        else:
             raise ValueError(f"Invalid train type {args.train_type}")
        
        gt_precond_label = x_tensor_dict['batch_precond_label_list']

        use_stable_obj_loss = False
        if args.train_type == 'all_object_pairs_gnn' or \
           args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
            precond_loss = args.weight_precond_loss * self.precond_loss(
                pred_precond.squeeze(), gt_precond_label)
            total_stable_obj_precond_loss = 0.0
            for scene_i, pred_table_obj_logits in enumerate(pred_stable_obj_list):
                stable_obj_gt = x_tensor_dict['batch_precond_stable_obj_id_list'][scene_i]
                stable_loss_i = self.stable_obj_pred_loss(pred_table_obj_logits, stable_obj_gt)
                total_stable_obj_precond_loss += stable_loss_i
            if use_stable_obj_loss:
                mean_stable_obj_precond_loss = total_stable_obj_precond_loss/len(pred_stable_obj_list)
                total_loss = precond_loss + mean_stable_obj_precond_loss
            else:
                mean_stable_obj_precond_loss = 0.0
                total_loss = precond_loss

        else:
            precond_loss = args.weight_precond_loss * self.precond_loss(
                pred_precond.squeeze(), gt_precond_label)

            total_loss = precond_loss

        if train and not args.save_embedding_only:
            self.opt_emb.zero_grad()
            self.opt_classif.zero_grad()
            total_loss.backward()
            if args.emb_lr >= 1e-5:
                if 'all_object_pairs' in args.train_type:
                    # raise ValueError("Not frozen")
                    print("Not frozen")
                self.opt_emb.step()
            self.opt_classif.step()
        
        batch_result_dict['pred_precond'] = pred_precond
        batch_result_dict['total_loss'] = total_loss.item()
        batch_result_dict['precond_loss'] = precond_loss.item()
        if args.train_type == 'all_object_pairs_gnn' or \
           args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
            if use_stable_obj_loss:
                batch_result_dict['stable_obj_loss'] = mean_stable_obj_precond_loss.item()
            else:
                assert abs(mean_stable_obj_precond_loss) < 0.0001
                batch_result_dict['stable_obj_loss'] = mean_stable_obj_precond_loss

        conf, gt_label_arr, pred_label_arr = self.get_conf_matrix_for_preds(
            pred_precond, gt_precond_label)
        batch_result_dict['conf'] = conf

        if save_preds:
            batch_result_dict['gt_label'] = gt_label_arr
            batch_result_dict['pred_label'] = pred_label_arr
        if save_emb:
            if args.train_type == 'all_object_pairs':
                batch_result_dict['scene_img_emb'] = scene_emb_list
            elif args.train_type == 'all_object_pairs_g_f_ij':
                batch_result_dict['scene_img_emb'] = scene_emb_list
            elif args.train_type == 'all_object_pairs_g_f_ij_cut_food':
                batch_result_dict['scene_img_emb'] = scene_emb_list
            elif args.train_type == 'all_object_pairs_g_f_ij_attn':
                batch_result_dict['scene_img_emb'] = scene_emb_list
            elif args.train_type == 'all_object_pairs_gnn':
                batch_result_dict['scene_img_emb'] = scene_emb_list
            elif args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
                batch_result_dict['scene_img_emb'] = scene_emb_list
            elif args.train_type == 'unfactored_scene':
                batch_result_dict['scene_img_emb'] = img_emb 
            elif args.train_type == 'unfactored_scene_resnet18':
                batch_result_dict['scene_img_emb'] = img_emb 
            else:
                raise ValueError("Invalid train type")

        return batch_result_dict
    
    def get_conf_matrix_for_preds(self, preds, gt_labels):
        conf = np.zeros((2, 2), dtype=np.int32)
        pred_label_arr = to_numpy(preds > 0.5).astype(np.int32).reshape(-1)
        # use np.around to make sure we get 0 and 1 correctly e.g. 0.999 
        # should be 1  not 0
        gt_label_arr = np.around(to_numpy(gt_labels)).astype(np.int32)
        for b in range(gt_labels.size(0)):
            conf[gt_label_arr[b], pred_label_arr[b]] += 1
        return conf, gt_label_arr, pred_label_arr
    
    def get_embeddings_for_pretrained_model_with_multiple_objects(
        self, 
        checkpoint_path,
        use_train_data=True):
        '''Get embedding for scenes where there only exist multiple objects in 
        the scene.
        '''
        args = self.config.args
        emb_trainer = create_voxel_trainer_with_checkpoint(
            checkpoint_path, cuda=args.cuda)
        emb_result_dict = dict(h5= dict(), pkl=dict())

        device = self.config.get_device()
        emb_trainer.model.to(device)

        # Reset scene
        dataloader = self.dataloader
        dataloader.reset_scene_batch_sampler(train=use_train_data, shuffle=False)

        scene_batch_size = 1
        train_data_size = dataloader.number_of_scene_data(use_train_data)
        num_batch_scenes = train_data_size // scene_batch_size 
        if train_data_size % scene_batch_size > 0: 
            num_batch_scenes += 1

        data_idx, emb_result_idx = 0, 0
        print_freq = num_batch_scenes // 5
        if print_freq == 0:
            print_freq = 1

        for batch_idx in range(num_batch_scenes):
            batch_data = []
            voxel_data_list = []
            batch_get_start_time = time.time()

            while len(batch_data) < scene_batch_size and data_idx < train_data_size:
                data = dataloader.get_next_all_object_pairs_for_scene(use_train_data)
                voxel_data_list += data['all_object_pair_voxels']
                batch_data.append(data)
                data_idx += 1

            batch_get_end_time = time.time()

            voxel_data = torch.Tensor(voxel_data_list).to(device)
            voxel_emb = emb_trainer.get_embedding_for_data(voxel_data)
            voxel_emb_arr = to_numpy(voxel_emb)

            result_data_idx = 0
            for b, data in enumerate(batch_data):
                assert emb_result_dict['h5'].get(str(emb_result_idx)) is None
                emb_result_dict['h5'][str(emb_result_idx)] = {
                    'emb': voxel_emb_arr,
                    'precond_label': data['precond_label'],
                }
                emb_result_dict['pkl'][str(emb_result_idx)] = {
                    'path': data['scene_path'],
                    'all_object_pair_path': data['all_object_pair_path'],
                    'precond_label': data['precond_label'],
                }
                emb_result_idx += 1
                result_data_idx += 1

            if batch_idx % print_freq == 0:
                print("Got emb for {}/{}".format(batch_idx, num_batch_scenes))
        
        return emb_result_dict
    
    def get_embeddings_for_pretrained_model(self, 
                                            checkpoint_path, 
                                            use_train_data=True):
        '''Get embedding for scenes where there only exist a pair of obejcts.
        '''

        args = self.config.args
        emb_trainer = create_voxel_trainer_with_checkpoint(
            checkpoint_path, cuda=args.cuda)
        
        emb_result_dict = dict(h5=dict(), pkl=dict())

        dataloader = self.dataloader
        device = self.config.get_device()
        train_data_size = dataloader.get_data_size(use_train_data)
        train_data_idx_list = list(range(0, train_data_size))
        emb_trainer.model.to(device)
        iter_order = np.arange(train_data_size)

        batch_size = args.batch_size
        num_batches = train_data_size // batch_size
        if train_data_size % batch_size != 0:
            num_batches += 1
        data_idx, emb_result_idx = 0, 0
        print_freq = num_batches // 10
        if print_freq == 0:
            print_freq = 1

        for batch_idx in range(num_batches): # Get raw data from the dataloader.
            batch_data = []

            # for b in range(batch_size):
            batch_get_start_time = time.time()

            while len(batch_data) < batch_size and data_idx < len(iter_order):
                actual_data_idx = iter_order[data_idx]
                data = dataloader.get_train_data_at_idx(actual_data_idx, 
                                                        actual_data_idx)
                batch_data.append(data)
                data_idx = data_idx + 1

            batch_get_end_time = time.time()
            voxel_data = torch.stack([d['voxels'] for d in batch_data]).to(device)
            voxel_emb = emb_trainer.get_embedding_for_data(voxel_data)
            voxel_emb_arr = to_numpy(voxel_emb)

            # import ipdb; ipdb.set_trace()
            for b, data in enumerate(batch_data):
                assert emb_result_dict['h5'].get(str(emb_result_idx)) is None
                emb_result_dict['h5'][str(emb_result_idx)] = {
                    'emb': voxel_emb_arr[b],
                    'precond_label': data['precond_label'],
                }
                emb_result_dict['pkl'][str(emb_result_idx)] = {
                    'path': data['path'],
                    'before_img_path': data['before_img_path'],
                    'precond_label': data['precond_label'],
                }
                emb_result_idx += 1

            if batch_idx % print_freq == 0:
                print("Got emb for {}/{}".format(batch_idx, num_batches))
        
        return emb_result_dict
    
    def did_end_train_epoch(self):
        self.classif_lr_scheduler.step()
        self.emb_lr_scheduler.step()
    
    def get_next_data_from_dataloader(self, dataloader, train):
        args = self.config.args
        data = None
        if args.train_type == 'all_object_pairs':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij_cut_food':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij_attn':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_gnn':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'unfactored_scene':
            data = dataloader.get_next_voxel_for_scene(train=train)
        elif args.train_type == 'unfactored_scene_resnet18':
            data = dataloader.get_next_voxel_for_scene(train=train)
        else:
            raise ValueError(f"Invalid train type: {args.train_type}")
        return data
   
    def train(self, train=True, viz_images=False, save_embedding=True, log_prefix=''):
        print("Begin training")
        args = self.config.args
        log_freq_iters = args.log_freq_iters if train else 10
        dataloader = self.dataloader
        device = self.config.get_device()

        train_data_size = dataloader.number_of_scene_data(train)

        # Reset log counter 
        train_step_count, test_step_count = 0, 0
        self.set_model_device(device)

        result_dict = {
            'data_info': {
                'path': [],
                'info': [],
            },
            # 'emb' Key is saved in hdf5 files, hence add keys here that will
            #  be numpy arrays.
            'emb': {
                'train_img_emb': [],
                'test_img_emb': [],
                'train_gt': [],
                'train_pred': [],
                'test_gt': [],
                'test_pred': [],
            },
            'output': {
                'gt': [],
                'pred': [],
                'test_gt': [],
                'test_pred': [],
                'best_test_gt': [],
                'best_test_pred': [],
                'test_f1_score': [],
                'test_wt_f1_score': [],
                'test_conf': [],
                'scene_path': [],
                'scene_all_object_pair_path': [],
                'test_scene_path': [],
                'test_scene_all_object_pair_path': [],
                'best_test_scene_path': [],
                'best_test_scene_all_object_pair_path': [],
            },
            'conf': {
                'train': [],
                'test': [],
            }
        }
        num_epochs = args.num_epochs if train else 1
        if args.save_embedding_only:
            num_epochs = 1

        for e in range(num_epochs):
            dataloader.reset_scene_batch_sampler(train=train, shuffle=train)

            batch_size = args.batch_size if train else 32
            num_batches = train_data_size // batch_size
            if train_data_size % batch_size != 0:
                num_batches += 1

            data_idx = 0

            n_classes = args.classif_num_classes
            result_dict['conf']['train'].append(
                np.zeros((n_classes, n_classes), dtype=np.int32))
            for k in ['gt', 'pred', 'scene_path', 'scene_all_object_pair_path', 
                      'test_scene_path', 'test_scene_all_object_pair_path']:
                result_dict['output'][k] = []
            for k in ['train_img_emb', 'train_gt', 'train_pred']:
                result_dict['emb'][k] = []

            for batch_idx in range(num_batches):
                # Get raw data from the dataloader.
                batch_data = []
                batch_get_start_time = time.time()

                while len(batch_data) < batch_size and data_idx < train_data_size:
                    data = self.get_next_data_from_dataloader(dataloader, train)
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
                batch_result_dict = self.run_model_on_batch(
                    x_tensor_dict,
                    batch_size,
                    train=train,
                    save_preds=True,
                    save_emb=True)
                run_batch_end_time = time.time()

                # print("Batch get: {:4f}   \t  proc data: {:.4f}  \t  run: {:.4f}".format(
                    # batch_get_end_time - batch_get_start_time,
                    # proc_data_end_time - proc_data_start_time,
                    # run_batch_end_time - run_batch_start_time
                # ))
                if args.loss_type == 'classif':
                    result_dict['conf']['train'][-1] += batch_result_dict['conf']
                
                result_dict['output']['gt'].append(batch_result_dict['gt_label'])
                result_dict['output']['pred'].append(batch_result_dict['pred_label'])
                for b in range(len(batch_data)):
                    result_dict['emb']['train_gt'].append(batch_result_dict['gt_label'][b])
                    result_dict['emb']['train_pred'].append(batch_result_dict['pred_label'][b])

                if save_embedding:
                    result_dict['output']['scene_path'] += x_dict['batch_scene_path_list']
                    result_dict['output']['scene_all_object_pair_path'] += \
                        x_dict['batch_scene_all_object_pair_path_list']

                    for b in range(len(batch_data)):
                        if args.train_type == 'all_object_pairs':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(torch.stack(batch_result_dict['scene_img_emb'][b])))
                        elif args.train_type == 'all_object_pairs_g_f_ij':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(torch.stack(batch_result_dict['scene_img_emb'][b])))
                        elif args.train_type == 'all_object_pairs_g_f_ij_attn':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(torch.stack(batch_result_dict['scene_img_emb'][b])))
                        elif args.train_type == 'all_object_pairs_g_f_ij_cut_food':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(torch.stack(batch_result_dict['scene_img_emb'][b])))
                        elif args.train_type == 'all_object_pairs_gnn':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(torch.stack(batch_result_dict['scene_img_emb'][b])))
                        elif args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(torch.stack(batch_result_dict['scene_img_emb'][b])))
                        elif args.train_type == 'unfactored_scene':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(batch_result_dict['scene_img_emb'][b]))
                        elif args.train_type == 'unfactored_scene_resnet18':
                            result_dict['emb']['train_img_emb'].append(
                                to_numpy(batch_result_dict['scene_img_emb'][b]))
                        else:
                            raise ValueError(f'Invalid train type {args.train_type}')

                self.print_train_update_to_console(
                    e, num_epochs, batch_idx, num_batches,
                    train_step_count, batch_result_dict)
                
                plot_images = viz_images and train and train_step_count %  log_freq_iters == 0
                plot_loss = train and train_step_count % args.print_freq_iters == 0

                if train and not args.save_embedding_only:
                    self.plot_train_update_to_tensorboard(
                        x_dict, x_tensor_dict, batch_result_dict,
                        train_step_count,
                        plot_loss=plot_loss,
                        plot_images=plot_images,
                    )

                if train and not args.save_embedding_only:
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
                        for k in ['test_img_emb', 'test_gt', 'test_pred']:
                            result_dict['emb'][k] = []

                        result_dict['output']['test_gt'] = []
                        result_dict['output']['test_pred'] = []

                        test_data_size = dataloader.number_of_scene_data(train=False)
                        test_batch_size = args.batch_size
                        num_batch_test = test_data_size // test_batch_size
                        if test_data_size % test_batch_size != 0:
                            num_batch_test += 1

                        # Do NOT sort the test data.
                        dataloader.reset_scene_batch_sampler(train=False, shuffle=False)

                        test_data_idx, total_test_loss = 0, 0

                        all_gt_label_list, all_pred_label_list = [], []

                        self.set_model_to_eval()

                        result_dict['conf']['test'].append(
                            np.zeros((n_classes, n_classes), dtype=np.int32))

                        print(bcolors.c_yellow("==== Test begin ==== "))
                        for test_e in range(num_batch_test):
                            batch_data = []

                            while (len(batch_data) < test_batch_size and test_data_idx < test_data_size):
                                data = self.get_next_data_from_dataloader(dataloader, False)
                                batch_data.append(data)
                                test_data_idx = test_data_idx + 1

                            # Process raw batch data
                            x_dict = self.process_raw_batch_data(batch_data)
                            # Now collate the batch data together
                            x_tensor_dict = self.collate_batch_data_to_tensors(x_dict)
                            with torch.no_grad():
                                batch_result_dict = self.run_model_on_batch(
                                    x_tensor_dict, test_batch_size, train=False, save_preds=True, 
                                    save_emb=True)
                                total_test_loss += batch_result_dict['total_loss']
                                all_gt_label_list.append(batch_result_dict['gt_label'])
                                all_pred_label_list.append(batch_result_dict['pred_label'])
                                # for b in range(len(batch_data)):
                                #     result_dict['emb']['test_img_emb'].append(
                                #         to_numpy(batch_result_dict['img_emb'][b]))
                                result_dict['output']['test_scene_path'] += x_dict['batch_scene_path_list']
                                result_dict['output']['test_scene_all_object_pair_path'] += \
                                    x_dict['batch_scene_all_object_pair_path_list']

                            result_dict['conf']['test'][-1] += batch_result_dict['conf']

                            self.print_train_update_to_console(
                                e, num_epochs, test_e, num_batch_test,
                                train_step_count, batch_result_dict, train=False)

                            plot_images = test_e == 0
                            plot_loss = True
                            self.plot_train_update_to_tensorboard(
                                x_dict, x_tensor_dict, batch_result_dict,
                                test_step_count,
                                plot_loss=plot_loss,
                                plot_images=plot_images,
                                log_prefix='/test/'
                            )

                            test_step_count += 1

                        # Calculate metrics
                        gt_label = np.concatenate(all_gt_label_list)
                        pred_label = np.concatenate(all_pred_label_list)
                        normal_f1 = f1_score(gt_label, pred_label)
                        wt_f1 = f1_score(gt_label, pred_label, average='weighted')
                        self.logger.summary_writer.add_scalar(
                            '/metrics/test/normal_f1', normal_f1, test_step_count)
                        self.logger.summary_writer.add_scalar(
                            '/metrics/test/wt_f1', wt_f1, test_step_count)
                        result_dict['output']['test_f1_score'].append(normal_f1)
                        result_dict['output']['test_wt_f1_score'].append(wt_f1)
                        result_dict['output']['test_conf'].append(
                            result_dict['conf']['test'][-1])
                        result_dict['emb']['test_gt'] = np.copy(gt_label)
                        result_dict['emb']['test_pred'] = np.copy(pred_label)

                        result_dict['output']['test_gt'] = gt_label
                        result_dict['output']['test_pred'] = pred_label
                        if len(result_dict['output']['test_wt_f1_score']) == 1 or \
                            wt_f1 > np.max(result_dict['output']['test_wt_f1_score'][:-1]):
                            result_dict['output']['best_test_gt'] = gt_label
                            result_dict['output']['best_test_pred'] = pred_label
                            result_dict['output']['best_test_scene_path'] = \
                                result_dict['output']['test_scene_path']
                            result_dict['output']['best_test_scene_all_object_pair_path'] = \
                                result_dict['output']['test_scene_all_object_pair_path']


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

                        print(bcolors.c_yellow(
                            "Test:  \t          F1: {:.4f}\n"
                            "       \t       Wt-F1: {:.4f}\n"
                            "       \t        conf:\n{}".format(
                            normal_f1, wt_f1,
                            np.array_str(result_dict['conf']['test'][-1],
                                         precision=0))))
                        print(bcolors.c_yellow(
                            ' ==== Test Epoch conf end ===='))

                    x_dict = None
                    x_tensor_dict = None
                    batch_result_dict = None
                    torch.cuda.empty_cache()
                    if train:
                        self.set_model_to_train()

                train_step_count += 1
                torch.cuda.empty_cache()

            self.did_end_train_epoch()

            for k in ['gt', 'pred']:
                result_dict['output'][k] = np.concatenate(
                    result_dict['output'][k]).astype(np.int32)
            
            normal_f1 = f1_score(result_dict['output']['gt'], 
                                 result_dict['output']['pred'])
            wt_f1 = f1_score(result_dict['output']['gt'], 
                             result_dict['output']['pred'], 
                             average='weighted')
            self.logger.summary_writer.add_scalar(
                '/metrics/train/normal_f1', normal_f1, train_step_count)
            self.logger.summary_writer.add_scalar(
                '/metrics/train/wt_f1', wt_f1, train_step_count)

            if args.loss_type == 'classif':
                print(bcolors.c_red(
                    "Train:  \t          F1: {:.4f}\n"
                    "        \t       Wt-F1: {:.4f}\n"
                    "        \t        conf:\n{}".format(
                    normal_f1, wt_f1,
                    np.array_str(result_dict['conf']['train'][-1],
                                    precision=0))))
                # Find min wt f1
                if len(result_dict['output']['test_wt_f1_score']) > 0:
                    max_f1_idx = np.argmax(result_dict['output']['test_wt_f1_score'])
                    print(bcolors.c_cyan(
                        "Max test wt f1:\n"
                        "               \t    F1: {:.4f}\n"
                        "               \t    Wt-F1: {:.4f}\n"
                        "               \t    conf:\n{}".format(
                        result_dict['output']['test_f1_score'][max_f1_idx],
                        result_dict['output']['test_wt_f1_score'][max_f1_idx],
                        np.array_str(result_dict['conf']['test'][max_f1_idx],
                                     precision=0))))

            save_emb_data_to_h5(args.result_dir, result_dict)
            print(' ==== Epoch done ====')

        for k in ['train_gt', 'train_pred']:
            result_dict['emb'][k] = np.array(result_dict['emb'][k])

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

    trainer = MultiObjectVoxelPrecondTrainerE2E(config)

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

        pkl_path = os.path.join(test_result_dir, 'result_info.pkl')
        with open(pkl_path, 'wb') as pkl_f:
            result_pkl_dict = {
                'data_info': result_dict['data_info'],
                'output': result_dict['output'],
                'conf': result_dict['conf']}
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
        result_dict = trainer.train(viz_images=True)
        if args.save_embedding_only:
            if not os.path.exists(args.emb_save_path):
                # Create all intermediate dirs if required
                os.makedirs(args.emb_save_path)
            save_emb_data_to_h5(args.emb_save_path, result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train for precond classification directly from images.')
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

    parser.add_argument('--train_dir', required=True, action='append',
                        help='Path to hdf5 file.')
    parser.add_argument('--test_dir', required=True, action='append',
                        help='Path to hdf5 file.')
    parser.add_argument('--train_type', type=str, default='all_pairs',
                        choices=[
                            'all_object_pairs', 
                            'unfactored_scene',
                            'unfactored_scene_resnet18',
                            'all_object_pairs_g_f_ij',
                            'all_object_pairs_g_f_ij_cut_food',
                            'all_object_pairs_g_f_ij_attn',
                            'all_object_pairs_gnn',
                            'all_object_pairs_gnn_raw_obj_info',
                            ],
                        help='Training type to follow.')
    parser.add_argument('--emb_lr', required=True, type=float, default=0.0,
                        help='Learning rate to use for embeddings.')

    parser.add_argument('--cp_prefix', type=str, default='',
                        help='Prefix to be used to save embeddings.')
    parser.add_argument('--max_train_data_size', type=int, default=10000,
                        help='Max train data size.')
    parser.add_argument('--max_test_data_size', type=int, default=10000,
                        help='Max test data size.')

    parser.add_argument('--z_dim', type=int, default=128,
                        help='Embedding size to extract from image.')

    # Loss weights
    parser.add_argument('--loss_type', type=str, default='classif',
                        choices=['classif'], help='Loss type to use')
    parser.add_argument('--weight_precond_loss', type=float, default=1.0,
                        help='Weight for precond pred loss.')
    parser.add_argument('--classif_num_classes', type=int, default=2,
                        help='Number of classes for classification.')
    parser.add_argument('--use_dynamic_bce_loss', type=str2bool, default=False,
                        help='Use dynamic BCE loss.')

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
    parser.add_argument('--use_spatial_softmax', type=str2bool, default=False,
                         help='Use spatial softmax.')
    parser.add_argument('--save_full_3d', type=str2bool, default=False,
                        help='Save 3d voxel representation in memory.')
    parser.add_argument('--expand_voxel_points', type=str2bool, default=False,
                        help='Expand voxel points to internal points of obj.')

    # Get Embeddings for data
    parser.add_argument('--save_embedding_only', type=str2bool, default=False,
                        help='Do not train precond model, just save the embedding for train data.')
    parser.add_argument('--emb_checkpoint_path', type=str, default='', 
                        help='Checkpoint path for embedding model.')
    parser.add_argument('--emb_save_path', type=str, default='', 
                        help='Path to save embeddings.')

    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=120)

    main(args)