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

sys.path.append(os.getcwd())

from robot_learning.logger.tensorboardx_logger import TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network
from utils.colors import bcolors
from utils.data_utils import str2bool
from utils.data_utils import recursively_save_dict_contents_to_group
from utils.data_utils import convert_list_of_array_to_dict_of_array_for_hdf5
from utils.image_utils import get_image_tensor_mask_for_bb

from action_relation.dataloader.vrep_dataloader import SimpleVoxelDataloader
from action_relation.model.voxel_model import VoxelModel
from action_relation.model.unscaled_voxel_model import UnscaledVoxelModel
from action_relation.model.model_utils import ScaledMSELoss

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer


def to_numpy(a_tensor, copy=True, detach=True):
    if detach:
        if copy:
            return a_tensor.clone().detach().cpu().numpy()
        else:
            return a_tensor.detach().cpu().numpy()
    else:
        if copy:
            return a_tensor.clone().cpu().numpy()
        else:
            return a_tensor.cpu().numpy()

def fn_voxel_parse(voxel_obj):
    status, a = voxel_obj.parse()
    return a

from multiprocessing import Pool


class VoxelRelationTrainer(BaseVAETrainer):
    def __init__(self, config):
        super(VoxelRelationTrainer, self).__init__(config)
        self.hidden_dim = 64
        args = config.args

        self.voxel_pool = Pool(args.batch_size)

        self.dataloader = SimpleVoxelDataloader(
            config,
            voxel_datatype_to_use=args.voxel_datatype)

        args = config.args
        if args.voxel_datatype == 0:
            self.model = UnscaledVoxelModel(
                args.z_dim, 6, args, n_classes=2*args.classif_num_classes)
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
        x_tensor_dict['batch_bb_list'] = torch.Tensor(
            x_dict['batch_bb_list']).to(device)
        x_tensor_dict['batch_obj_delta_pose_class_list'] = torch.LongTensor(
            x_dict['batch_obj_delta_pose_class_list']).to(device)

        return x_tensor_dict

    def run_model_on_batch(self,
                           x_tensor_dict,
                           batch_size,
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
        total_loss = position_pred_loss + angle_pred_loss

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
                                      batch_result_dict):
        args = self.config.args
        e, batch_idx = current_epoch, curr_batch
        total_loss = batch_result_dict['total_loss']

        if train_step_count % args.print_freq_iters == 0:
            if args.loss_type == 'regr':
                print(bcolors.okblue(
                        "[{}/{}], \t Batch: [{}/{}], \t    total_loss: {:.4f}, "
                        "\t  bb: {:.4f}, \t   position: {:.3f}, angle: {:.3f}".format(
                            e, num_epochs, batch_idx, num_batches, total_loss,
                            batch_result_dict['pose_pred_loss'],
                            batch_result_dict['position_pred_loss'],
                            batch_result_dict['angle_pred_loss'])))
            elif args.loss_type == 'classif':
                print(bcolors.okblue(
                        "[{}/{}], \t Batch: [{}/{}], \t    total_loss: {:.4f}, "
                        "\t  bb: {:.4f}, \t   pos_x: {:.4f}, pos_y: {:.4f}".format(
                            e, num_epochs, batch_idx, num_batches, total_loss,
                            batch_result_dict['pose_pred_loss'],
                            batch_result_dict['position_pred_loss_x'],
                            batch_result_dict['position_pred_loss_y'])))


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
                log_prefix+'loss/pose_pred_loss', 
                batch_result_dict['pose_pred_loss'],
                step_count
            )
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/position_pred_loss',
                batch_result_dict['position_pred_loss'],
                step_count
            )
            if args.loss_type == 'classif':
                for k in ['position_pred_loss_x', 'position_pred_loss_y']:
                    self.logger.summary_writer.add_scalar(
                        log_prefix+'loss/{}'.format(k),
                        batch_result_dict[k],
                        step_count
                    )

            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/angle_pred_loss',
                batch_result_dict['angle_pred_loss'],
                step_count
            )
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/inv_model_loss',
                batch_result_dict['inv_model_loss'],
                step_count,
            )
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/total_loss',
                batch_result_dict['total_loss'],
                step_count,
            )

    def get_data_stats_for_classif(self, train=True):
        print("Get stats for classification")
        args = self.config.args
        dataloader = self.dataloader
        train_data_size = dataloader.get_data_size(train)
        all_classes = []
        for e in range(1):
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
                    data = dataloader.get_train_data_at_idx(
                        iter_order[data_idx], train=train)
                    if data['voxels'] is not None:
                        batch_data.append(data)
                    all_classes.append(data['delta_classes'])
                    data_idx = data_idx + 1
                batch_get_end_time = time.time()
                print("Data time: {:.4f}".format(
                    batch_get_end_time - batch_get_start_time))


        all_classes_arr = np.array(all_classes, dtype=np.int32)
        data_count_dict = {}
        for col in range(all_classes_arr.shape[1]):
            classes = np.unique(all_classes_arr[:, col])
            data_count_dict[col] = {}
            for k in classes:
                data_count_dict[k] = np.sum(all_classes_arr[:, col] == k)
            print("Col: {}".format(col))
            pprint.pprint(data_count_dict)
    

    def train(self, train=True, viz_images=False, save_embedding=True,
              log_prefix=''):
        print("Begin training")
        args = self.config.args
        log_freq_iters = args.log_freq_iters if train else 10
        dataloader = self.dataloader
        device = self.config.get_device()
        train_data_size = dataloader.get_data_size(train)
        train_data_idx_list = list(range(0, train_data_size))

        # Reset log counter
        train_step_count, test_step_count = 0, 0
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

        for e in range(num_epochs):
            if train: 
                iter_order = np.random.permutation(train_data_idx_list)
            else:
                iter_order = np.arange(train_data_size)

            batch_size = args.batch_size if train else 32
            num_batches = train_data_size // batch_size
            data_idx = 0
            print(num_batches)

            if args.loss_type == 'classif':
                n_classes = args.classif_num_classes
                for conf_key in ['train_x', 'train_y']:
                    result_dict['conf'][conf_key].append(
                        np.zeros((n_classes, n_classes), dtype=np.int32))

            for batch_idx in range(num_batches):
                # Get raw data from the dataloader.
                batch_data = []
                # for b in range(batch_size):
                batch_get_start_time = time.time()

                if args.octree_0_multi_thread:
                    voxel_obj_list = [dataloader.get_voxel_obj_at_idx(
                        iter_order[data_idx + b]) for b in range(batch_size)
                            if data_idx + b < len(iter_order)]
                    voxel_tensor_list = self.voxel_pool.map(
                            fn_voxel_parse, voxel_obj_list)
                    voxel_tensor_list = [torch.Tensor(x) for x in voxel_tensor_list]

                b = 0
                while len(batch_data) < batch_size and data_idx < len(iter_order):
                    data = dataloader.get_train_data_at_idx(
                        iter_order[data_idx], train=train)
                    data['voxels'] = voxel_tensor_list[b]
                    batch_data.append(data)
                    data_idx = data_idx + 1
                    b = b + 1

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
                    x_tensor_dict, batch_size, train=train, save_preds=True)
                run_batch_end_time = time.time()

                # print("Batch get: {:4f}   \t  proc data: {:.4f}  \t  run: {:.4f}".format(
                #     batch_get_end_time - batch_get_start_time,
                #     proc_data_end_time - proc_data_start_time,
                #     run_batch_end_time - run_batch_start_time
                # ))
                if args.loss_type == 'classif':
                    result_dict['conf']['train_x'][-1] += batch_result_dict['conf_x']
                    result_dict['conf']['train_y'][-1] += batch_result_dict['conf_y']

                
                if not train and save_embedding:
                    for b in range(batch_size):
                        for k in ['path', 'info', 'action']:
                            result_dict['data_info'][k].append(batch_data[b][k])
                        
                        result_dict['emb']['img_emb'].append(
                            batch_result_dict['img_emb'][b].detach().cpu().numpy())
                        result_dict['emb']['img_action_emb'].append(
                            batch_result_dict['img_action_emb'][b].detach().cpu().numpy())

                        if batch_result_dict.get('pos_gt') is not None:
                            result_dict['output']['pos_gt'].append(
                                batch_result_dict['pos_gt'][b])
                            result_dict['output']['pos_pred'].append(
                                batch_result_dict['pos_pred'][b])
                            
                            # diff = batch_result_dict['pos_gt'][b] - \
                                    # batch_result_dict['pos_pred'][b]
                            # if np.sum(diff*diff) > 0.001:
                                # print("path: {}, error: {}".format(
                                    # batch_data[b]['path'], diff))

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

                        num_batch_test, test_batch_size = 40, 16
                        test_data_size = self.dataloader.get_data_size(
                                train=False)
                        test_iter_order = np.random.permutation(
                                np.arange(test_data_size))
                        test_data_idx = 0
                        self.model.eval()

                        n_classes = args.classif_num_classes
                        for conf_key in ['test_x', 'test_y']:
                            result_dict['conf'][conf_key].append(
                                np.zeros((n_classes, n_classes), dtype=np.int32))

                        for test_e in range(num_batch_test):
                            batch_data = []

                            if args.octree_0_multi_thread:
                                voxel_obj_list = [dataloader.get_voxel_obj_at_idx(
                                    test_iter_order[test_data_idx + b]) 
                                    for b in range(test_batch_size)
                                    if test_data_idx + b < len(test_iter_order)]

                                voxel_tensor_list = self.voxel_pool.map(
                                        fn_voxel_parse, voxel_obj_list)
                                voxel_tensor_list = [torch.Tensor(x) for x in voxel_tensor_list]

                            b = 0
                            while len(batch_data) < batch_size and \
                                test_data_idx < len(test_iter_order):
                                data = dataloader.get_train_data_at_idx(
                                    test_iter_order[test_data_idx], train=False)

                                if args.octree_0_multi_thread:
                                    data['voxels'] = voxel_tensor_list[b]
                                if data['voxels'] is not None:
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
                                batch_result_dict = self.run_model_on_batch(
                                    x_tensor_dict, test_batch_size, train=False)

                            if args.loss_type == 'classif':
                                result_dict['conf']['test_x'][-1] += \
                                    batch_result_dict['conf_x']
                                result_dict['conf']['test_y'][-1] += \
                                    batch_result_dict['conf_y']

                            print(bcolors.c_red("==== Test begin ==== "))
                            self.print_train_update_to_console(
                                e, num_epochs, test_e, num_batch_test,
                                train_step_count, batch_result_dict)
                            print(bcolors.c_red("==== Test end ==== "))

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

                        if args.loss_type == 'classif':
                            print(' ==== Test Epoch conf start ====')
                            print('conf x: \n{}'.format(
                                np.array_str(result_dict['conf']['test_x'][-1], 
                                             precision=0)))
                            print('conf y: \n{}'.format(
                                np.array_str(result_dict['conf']['test_y'][-1], 
                                             precision=0)))
                            print(' ==== Test Epoch conf end ====')

                    x_dict = None
                    x_tensor_dict = None
                    batch_result_dict = None
                    torch.cuda.empty_cache()
                    self.model.eval()

                train_step_count += 1
                torch.cuda.empty_cache()

            if args.loss_type == 'classif':
                print('conf x: \n{}'.format(
                    np.array_str(result_dict['conf']['train_x'][-1], 
                                 precision=0)))
                print('conf y: \n{}'.format(
                    np.array_str(result_dict['conf']['train_y'][-1], 
                                 precision=0)))
                print(' ==== Epoch done ====')

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

    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=100)

    main(args)
