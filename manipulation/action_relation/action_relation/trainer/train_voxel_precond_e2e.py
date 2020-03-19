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
from action_relation.model.losses import TripletLoss, get_contrastive_loss
from action_relation.model.model_utils import ScaledMSELoss
from action_relation.model.unscaled_voxel_model import PrecondEmbeddingModel
from action_relation.model.unscaled_voxel_model import UnscaledVoxelModel, UnscaledPrecondVoxelModel
from action_relation.model.voxel_model import VoxelModel
from action_relation.trainer.train_voxel_precond import VoxelPrecondTrainer
from action_relation.trainer.train_voxels_online_contrastive import create_model_with_checkpoint
from action_relation.trainer.train_voxels_online_contrastive import create_voxel_trainer_with_checkpoint
from action_relation.utils.data_utils import get_euclid_dist_matrix_for_data
from action_relation.trainer.train_voxel_precond import save_emb_data_to_h5

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer


class VoxelE2EPrecondTrainer(VoxelPrecondTrainer):
    def __init__(self, config):
        super(VoxelE2EPrecondTrainer, self).__init__(config)

        self.model = None
        self.emb_model = create_model_with_checkpoint(
            'simple_model',
            args.emb_checkpoint_path,
        )
        self.classif_model = PrecondEmbeddingModel(args.z_dim, args)

        self.opt_emb = optim.Adam(self.emb_model.parameters(), lr=0.0)
        self.opt_classif = optim.Adam(self.classif_model.parameters(), lr=args.lr)
        classif_lr_lambda = lambda epoch: 0.90**epoch
        self.classif_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.opt_classif, classif_lr_lambda)
        emb_lr_lambda = lambda epoch: 0.99**epoch
        self.emb_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.opt_emb, emb_lr_lambda)

    def did_end_train_epoch(self):
        self.classif_lr_scheduler.step()
        self.emb_lr_scheduler.step()

    def get_model_list(self):
        return [self.emb_model, self.classif_model]

    def get_state_dict(self):
        return {
            'emb_model': self.emb_model.state_dict(),
            'classif_model': self.classif_model.state_dict(),
        }

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
            raise ValueError("Invalid voxels size: {}".format(voxel_data.size()))

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
        # model = args.model
        model = 'simple_model'
        if model == 'simple_model':
            emb_data = self.emb_model.forward_image(voxel_data, relu_on_emb=True)
        else:
            emb_data = self.emb_model.forward_image(voxel_data)
        # If we do not want to train end-to-end we should detach emb_data?
        pred_precond = self.classif_model.forward_predict_precond(emb_data)
        precond_loss = args.weight_precond_loss * self.precond_loss(
            pred_precond.squeeze(), gt_precond_label)

        total_loss = precond_loss

        if train:
            self.opt_emb.zero_grad()
            self.opt_classif.zero_grad()
            total_loss.backward()
            self.opt_emb.step()
            self.opt_classif.step()
        
        batch_result_dict['pred_precond'] = pred_precond
        batch_result_dict['total_loss'] = total_loss.item()
        batch_result_dict['precond_loss'] = precond_loss.item()

        conf, gt_label_arr, pred_label_arr = self.get_conf_matrix_for_preds(
            pred_precond, gt_precond_label) 

        batch_result_dict['conf'] = conf
        batch_result_dict['img_emb']  = emb_data

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

    trainer = VoxelE2EPrecondTrainer(config)

    if len(args.checkpoint_path) > 0:
        trainer.load_checkpoint(args.checkpoint_path)
        result_dict = trainer.train(train=False,
                                    viz_images=False,
                                    save_embedding=True,
                                    use_emb_data=False)
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
        result_dict = trainer.train(viz_images=True, use_emb_data=False)
        save_emb_data_to_h5(args.result_dir, result_dict)


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

    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to hdf5 file.')
    parser.add_argument('--test_dir', type=str, default='',
                        help='Path to hdf5 file.')
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

    np.random.seed(0)
    torch.manual_seed(0)

    main(args)
