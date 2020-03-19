import numpy as np
import open3d

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
from sklearn.metrics import f1_score

sys.path.append(os.getcwd())

from robot_learning.logger.tensorboardx_logger import TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network, to_numpy
from utils.colors import bcolors
from utils.data_utils import str2bool
from utils.data_utils import recursively_save_dict_contents_to_group
from utils.data_utils import convert_list_of_array_to_dict_of_array_for_hdf5
from utils.image_utils import get_image_tensor_mask_for_bb

from action_relation.dataloader.precond_vrep_dataloader import PrecondVoxelDataloader
from action_relation.dataloader.real_robot_dataloader import AllPairVoxelDataloader 
from action_relation.model.losses import TripletLoss, get_contrastive_loss
from action_relation.model.model_utils import ScaledMSELoss
from action_relation.model.unscaled_voxel_model import PrecondEmbeddingModel
from action_relation.model.unscaled_voxel_model import UnscaledVoxelModel, UnscaledPrecondVoxelModel
from action_relation.model.voxel_model import VoxelModel
from action_relation.trainer.train_voxels_online_contrastive import create_voxel_trainer_with_checkpoint
from action_relation.utils.data_utils import get_euclid_dist_matrix_for_data

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer


def save_emb_data_to_h5(result_dir, result_dict):
    emb_h5_path = os.path.join(result_dir, 'train_result_emb.h5')
    emb_h5_f = h5py.File(emb_h5_path, 'w')
    result_h5_dict = {'emb': result_dict['emb'],
                      'output': result_dict['output']}
    recursively_save_dict_contents_to_group(emb_h5_f, '/', result_h5_dict)
    emb_h5_f.flush()
    emb_h5_f.close()
    print(bcolors.c_blue("Did save emb data: {}".format(emb_h5_path)))


class VoxelPrecondTrainer(BaseVAETrainer):
    def __init__(self, config):
        super(VoxelPrecondTrainer, self).__init__(config)

        args = config.args

        if args.is_multi_obj_data:
            self.dataloader = AllPairVoxelDataloader(
                config,
                voxel_datatype_to_use=args.voxel_datatype
                )
        else:
            self.dataloader = PrecondVoxelDataloader(
                config,
                voxel_datatype_to_use=args.voxel_datatype)

        args = config.args
        if args.use_embeddings:
            self.model = PrecondEmbeddingModel(args.z_dim, args)
        else:
            if args.voxel_datatype == 0:
                self.model = UnscaledPrecondVoxelModel(
                    args.z_dim, args, n_classes=2,
                    use_spatial_softmax=args.use_spatial_softmax,
                    )
            elif args.voxel_datatype == 1:
                self.model = VoxelModel(args.z_dim, 6, args,
                                        n_classes=2*args.classif_num_classes)
            else:
                raise ValueError("Invalid voxel datatype: {}".format(
                    args.voxel_datatype))
        self.precond_loss = nn.BCELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=config.args.lr)
        lr_lambda = lambda epoch: 0.95**epoch
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def get_model_list(self):
        return [self.model]

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
        torch.save({'model': self.model.state_dict()}, cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        checkpoint_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint_models['model'])

    def log_model_to_tensorboard(self, train_step_count):
        '''Log weights and gradients of network to Tensorboard.'''
        model_l2_norm, model_grad_l2_norm = \
            get_weight_norm_for_network(self.model)
        self.logger.summary_writer.add_scalar(
            '/model/weight',
            model_l2_norm,
            train_step_count)
        self.logger.summary_writer.add_scalar(
            '/model/grad',
            model_grad_l2_norm,
            train_step_count)
        self.logger.summary_writer.add_scalar(
            '/model/lr',
            self.opt.param_groups[0]['lr'],
            train_step_count
        )

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
                    e, num_epochs, batch_idx, num_batches, total_loss,
                    batch_result_dict['total_loss'])))

    def plot_train_update_to_tensorboard(self, x_dict, x_tensor_dict,
                                         batch_result_dict, step_count,
                                         log_prefix='', plot_images=True,
                                         plot_loss=True):
        args = self.config.args

        if plot_images:
            pass

        if len(log_prefix) == 0:
            self.logger.summary_writer.add_scalar(
                'model/lr',
                self.opt.param_groups[0]['lr'],
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
        else:
            raise ValueError("Invalid add_xy_channels: {}".format(
                args.add_xy_channels))
        return voxels

    def process_raw_batch_data(self, batch_data):
        '''Process raw batch data and collect relevant objects in a dict.'''
        proc_batch_dict = {
            # Save input image
            'batch_voxel_list': [],
            # Save input emb
            'batch_inp_emb_list': [],
            # Save object bounding boxes.
            'batch_bb_list': [],
            # Save precond output
            'batch_precond_label_list': [],
        }
        args = self.config.args
        x_dict = proc_batch_dict

        for b, data in enumerate(batch_data):
            x_dict['batch_bb_list'].append(data['bb_list'])
            x_dict['batch_precond_label_list'].append(data['precond_label'])

            if data.get('voxels') is not None:
                voxels = self.process_raw_voxels(data['voxels'])
                x_dict['batch_voxel_list'].append(voxels)
            if data.get('emb') is not None:
                x_dict['batch_inp_emb_list'].append(data['emb'])

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
        if len(x_dict['batch_inp_emb_list']) > 0:
            x_tensor_dict['batch_inp_emb'] = torch.Tensor(
                x_dict['batch_inp_emb_list']).to(device)
        x_tensor_dict['batch_bb_list'] = torch.Tensor(
            x_dict['batch_bb_list']).to(device)
        x_tensor_dict['batch_precond_label_list'] = torch.FloatTensor(
            x_dict['batch_precond_label_list']).to(device)

        return x_tensor_dict
    
    def run_emb_model_on_batch(self,
                               x_tensor_dict,
                               batch_size,
                               train=False,
                               save_preds=False):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        inp_data = x_tensor_dict['batch_inp_emb']
        gt_precond_label = x_tensor_dict['batch_precond_label_list']
        pred_precond = self.model.forward_predict_precond(inp_data)
        precond_loss = args.weight_precond_loss * self.precond_loss(
            pred_precond.squeeze(), gt_precond_label)

        total_loss = precond_loss

        if train:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
        
        batch_result_dict['pred_precond'] = pred_precond
        batch_result_dict['total_loss'] = total_loss.item()
        batch_result_dict['precond_loss'] = precond_loss.item()

        conf, gt_label_arr, pred_label_arr = self.get_conf_matrix_for_preds(
            pred_precond, gt_precond_label)
        batch_result_dict['conf'] = conf

        if save_preds:
            batch_result_dict['gt_label'] = gt_label_arr
            batch_result_dict['pred_label'] = pred_label_arr

        return batch_result_dict

    def run_model_on_batch(self,
                           x_tensor_dict,
                           batch_size,
                           train=False,
                           save_preds=False):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        voxel_data = x_tensor_dict['batch_voxel']
        gt_precond_label = x_tensor_dict['batch_precond_label_list']
        pred_precond = self.model.forward_predict_precond(voxel_data)
        precond_loss = args.weight_precond_loss * self.precond_loss(
            pred_precond.squeeze(), gt_precond_label)

        total_loss = precond_loss

        if train:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
        
        batch_result_dict['pred_precond'] = pred_precond
        batch_result_dict['total_loss'] = total_loss.item()
        batch_result_dict['precond_loss'] = precond_loss.item()

        conf, gt_label_arr, pred_label_arr = self.get_conf_matrix_for_preds(
            pred_precond, gt_precond_label)
        batch_result_dict['conf'] = conf

        if save_preds:
            batch_result_dict['gt_label'] = gt_label_arr
            batch_result_dict['pred_label'] = pred_label_arr

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
        self.lr_scheduler.step()
   
    def train(self, train=True, viz_images=False, save_embedding=True,
              use_emb_data=False, log_prefix=''):
        print("Begin training")
        args = self.config.args
        log_freq_iters = args.log_freq_iters if train else 10
        dataloader = self.dataloader
        device = self.config.get_device()
        if use_emb_data:
            train_data_size = dataloader.get_h5_data_size(train)
        else:
            train_data_size = dataloader.get_data_size(train)
        train_data_idx_list = list(range(0, train_data_size))

        # Reset log counter 
        train_step_count, test_step_count = 0, 0
        self.set_model_device(device)

        result_dict = {
            'data_info': {
                'path': [],
                'info': [],
            },
            'emb': {
                'train_img_emb': [],
                'test_img_emb': [],
                'train_gt': [],
                'train_pred': [],
                'test_gt': [],
                'tese_pred': [],
            },
            'output': {
                'gt': [],
                'pred': [],
                'test_f1_score': [],
                'test_wt_f1_score': [],
                'test_conf': [],
            },
            'conf': {
                'train': [],
                'test': [],
            }
        }
        num_epochs = args.num_epochs if train else 1

        for e in range(num_epochs):
            if train:
                iter_order = np.random.permutation(train_data_idx_list)
            else:
                iter_order = np.arange(train_data_size)

            batch_size = args.batch_size if train else 32
            num_batches = train_data_size // batch_size
            data_idx = 0

            n_classes = args.classif_num_classes
            result_dict['conf']['train'].append(
                np.zeros((n_classes, n_classes), dtype=np.int32))
            for k in ['gt', 'pred']:
                result_dict['output'][k] = []
            for k in ['train_img_emb', 'train_gt', 'train_pred']:
                result_dict['emb'][k] = []

            for batch_idx in range(num_batches):
                # Get raw data from the dataloader.
                batch_data = []
                # for b in range(batch_size):
                batch_get_start_time = time.time()

                while len(batch_data) < batch_size and data_idx < len(iter_order):
                    actual_data_idx = iter_order[data_idx]
                    if use_emb_data:
                        data = dataloader.get_h5_train_data_at_idx(
                            actual_data_idx, train=train)
                    else:
                        data = dataloader.get_train_data_at_idx(
                            actual_data_idx, train=train)
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
                model_fn = self.run_emb_model_on_batch \
                    if use_emb_data else self.run_model_on_batch
                batch_result_dict = model_fn(
                    x_tensor_dict,
                    batch_size,
                    train=train,
                    save_preds=True)
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
                    result_dict['emb']['train_img_emb'].append(
                        to_numpy(batch_result_dict['img_emb'][b]))
                    result_dict['emb']['train_gt'].append(
                        batch_result_dict['gt_label'][b])
                    result_dict['emb']['train_pred'].append(
                        batch_result_dict['pred_label'][b])

                self.print_train_update_to_console(
                    e, num_epochs, batch_idx, num_batches,
                    train_step_count, batch_result_dict)
                
                plot_images = viz_images and train \
                    and train_step_count %  log_freq_iters == 0
                plot_loss = train \
                    and train_step_count % args.print_freq_iters == 0

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
                        for k in ['test_img_emb', 'test_gt', 'test_pred']:
                            result_dict['emb'][k] = []

                        test_batch_size = args.batch_size
                        if use_emb_data:
                            test_data_size = self.dataloader.get_h5_data_size(
                                train=False)
                        else:
                            test_data_size = self.dataloader.get_data_size(
                                train=False)
                        num_batch_test = test_data_size // test_batch_size
                        if test_data_size % test_batch_size != 0:
                            num_batch_test += 1
                        # Do NOT sort the test data.
                        test_iter_order = np.arange(test_data_size)
                        test_data_idx, total_test_loss = 0, 0

                        all_gt_label_list, all_pred_label_list = [], []

                        self.set_model_to_eval()

                        result_dict['conf']['test'].append(
                            np.zeros((n_classes, n_classes), dtype=np.int32))

                        print(bcolors.c_yellow("==== Test begin ==== "))
                        for test_e in range(num_batch_test):
                            batch_data = []

                            while (len(batch_data) < test_batch_size and 
                                   test_data_idx < len(test_iter_order)):
                                if use_emb_data:
                                    data = dataloader.get_h5_train_data_at_idx(
                                        test_iter_order[test_data_idx], train=False)
                                else:
                                    data = dataloader.get_train_data_at_idx(
                                        test_iter_order[test_data_idx], train=False)
                                batch_data.append(data)
                                test_data_idx = test_data_idx + 1

                            # Process raw batch data
                            x_dict = self.process_raw_batch_data(batch_data)
                            # Now collate the batch data together
                            x_tensor_dict = self.collate_batch_data_to_tensors(
                                x_dict)
                            with torch.no_grad():
                                model_fn = self.run_emb_model_on_batch \
                                    if use_emb_data else self.run_model_on_batch
                                batch_result_dict = model_fn(
                                    x_tensor_dict, test_batch_size, train=False,
                                    save_preds=True)
                                total_test_loss += batch_result_dict['total_loss']
                                all_gt_label_list.append(batch_result_dict['gt_label'])
                                all_pred_label_list.append(batch_result_dict['pred_label'])
                                for b in range(len(batch_data)):
                                    result_dict['emb']['test_img_emb'].append(
                                        to_numpy(batch_result_dict['img_emb'][b]))

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

    trainer = VoxelPrecondTrainer(config)

    if args.use_embeddings:
        # Let's get the trained embeddings
        assert len(args.emb_checkpoint_path) > 0
        train_emb_h5_path = os.path.join(args.emb_save_path, 'train_emb_data.h5')
        train_emb_pkl_path = os.path.join(args.emb_save_path, 'train_emb_data_info.pkl')
        test_emb_h5_path = os.path.join(args.emb_save_path, 'test_emb_data.h5')
        test_emb_pkl_path = os.path.join(args.emb_save_path, 'test_emb_data_info.pkl')

        if not os.path.exists(os.path.dirname(train_emb_h5_path)):
            os.makedirs(os.path.dirname(train_emb_h5_path))

        if not os.path.exists(train_emb_h5_path) or not os.path.exists(train_emb_pkl_path):
            if args.is_multi_obj_data:
                train_emb_data = trainer.get_embeddings_for_pretrained_model_with_multiple_objects(
                    args.emb_checkpoint_path,
                    use_train_data=True
                )
            else:
                train_emb_data = trainer.get_embeddings_for_pretrained_model(
                    args.emb_checkpoint_path,
                    use_train_data=True
                )

            h5f = h5py.File(train_emb_h5_path, 'w')
            recursively_save_dict_contents_to_group(h5f, '/', train_emb_data['h5'])
            h5f.flush()
            h5f.close()
            with open(train_emb_pkl_path, 'wb') as train_emb_pkl_f:
                pickle.dump(train_emb_data['pkl'], train_emb_pkl_f, protocol=2)
                print(f"Did save train emb pkl data: {train_emb_pkl_path}")

        if not os.path.exists(test_emb_h5_path) or not os.path.exists(test_emb_pkl_path):
            if args.is_multi_obj_data:
                test_emb_data = trainer.get_embeddings_for_pretrained_model_with_multiple_objects(
                    args.emb_checkpoint_path,
                    use_train_data=False
                )
            else:
                test_emb_data = trainer.get_embeddings_for_pretrained_model(
                    args.emb_checkpoint_path,
                    use_train_data=False
                )
            h5f = h5py.File(test_emb_h5_path, 'w')
            recursively_save_dict_contents_to_group(h5f, '/', test_emb_data['h5'])
            h5f.flush()
            h5f.close()

            with open(test_emb_pkl_path, 'wb') as test_emb_pkl_f:
                pickle.dump(test_emb_data['pkl'], test_emb_pkl_f, protocol=2)
                print(f"Did save train emb pkl data: {test_emb_pkl_path}")

        trainer.dataloader.load_emb_data(train_emb_h5_path, 
                                         train_emb_pkl_path,
                                         test_emb_h5_path,
                                         test_emb_pkl_path)
        
        import ipdb; ipdb.set_trace()

    if len(args.checkpoint_path) > 0:
        trainer.load_checkpoint(args.checkpoint_path)
        result_dict = trainer.train(train=False,
                                    viz_images=False,
                                    save_embedding=True,
                                    use_emb_data=args.use_embeddings)
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
        result_dict = trainer.train(viz_images=True, 
                                    use_emb_data=args.use_embeddings)
        emb_h5_path = os.path.join(args.result_dir, 'train_result_emb.h5')
        emb_h5_f = h5py.File(emb_h5_path, 'w')
        result_h5_dict = {'emb': result_dict['emb'],
                          'output': result_dict['output']}
        recursively_save_dict_contents_to_group(emb_h5_f, '/', result_h5_dict)
        emb_h5_f.flush()
        emb_h5_f.close()
        print("Did save emb: {}".format(emb_h5_path))
 


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
    parser.add_argument('--is_multi_obj_data', type=str2bool, default=False,
                        help='Each scene has multiple objects.'
                             'Can be real or simulated.') 

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
    parser.add_argument('--use_embeddings', type=str2bool, default=False,
                        help='Use embeddings to learn precond model.')
    parser.add_argument('--emb_checkpoint_path', type=str, default='', 
                        help='Checkpoint path for embedding model.')
    parser.add_argument('--emb_save_path', type=str, default='', 
                        help='Path to save embeddings.')

    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=120)

    main(args)