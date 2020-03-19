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

sys.path.append(os.getcwd())

from robot_learning.logger.tensorboardx_logger import TensorboardXLogger
from utils.torch_utils import get_weight_norm_for_network
from utils.colors import bcolors
from utils.data_utils import recursively_save_dict_contents_to_group
from utils.data_utils import convert_list_of_array_to_dict_of_array_for_hdf5
from utils.image_utils import get_image_tensor_mask_for_bb

from action_relation.dataloader.pygame_dataloader import SimpleImageDataloader
from action_relation.model.simple_model import SimpleModel

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer

class RelationTrainer(BaseVAETrainer):
    def __init__(self, config):
        super(RelationTrainer, self).__init__(config)
        self.hidden_dim = 64
        args = config.args

        self.dataloader = SimpleImageDataloader(config)

        args = config.args
        self.model = SimpleModel(args.z_dim, 2, args)

        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()
        self.bb_pred_loss = nn.MSELoss()
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

    def get_mean_bb_pred_loss(self, all_bb_tensor, pred_bb, mask_idx):
        mask = torch.ones(all_bb_tensor.size()).type(torch.ByteTensor)
        mask[mask_idx, :] = 0
        gt_tensor = all_bb_tensor.masked_select(mask).view(-1, mask.size(1))

        if type(pred_bb) is np.ndarray:
            pred_bb = torch.Tensor(pred_bb).to(gt_tensor.device)
        assert type(pred_bb) is torch.Tensor

        assert gt_tensor.size() == pred_bb.size(), \
            "GT and pred bb size do not match."
        loss = self.bb_pred_loss(pred_bb, gt_tensor).cpu().numpy()
        loss_per_object = torch.mean((pred_bb - gt_tensor)**2, dim=1)
        loss_per_object = loss_per_object.cpu().numpy()
        assert len(loss_per_object.shape) == 1, \
            "Loss per object should be a 1-d array."
        return float(loss), loss_per_object

    def process_raw_batch_data(self, batch_data):
        '''Process raw batch data and collect relevant objects in a dict.'''
        proc_batch_dict = {
            # Save input image
            'batch_img_list': [],
            # Save input bb info for bb_delta prediction
            'batch_obj_bb_list': [],
            # Save bb_info to be predicted.
            'batch_other_bb_pred_list': [],
            # Save action info.
            'batch_action_list': [],
        }
        args = self.config.args
        x_dict = proc_batch_dict

        for b, data in enumerate(batch_data):
            bb_info = data['bb_info']
            bb_before = bb_info['anchor_bb_before'] + bb_info['other_bb_before']
            bb_delta = bb_info['other_bb_delta']
            x_dict['batch_obj_bb_list'].append(bb_before)
            x_dict['batch_other_bb_pred_list'].append(bb_delta)

            if args.add_xy_channels == 0:
                img = data['img']
            elif args.add_xy_channels == 1:
                xv, yv = torch.meshgrid([torch.arange(0,256), 
                                         torch.arange(0,256)])
                img = torch.cat([data['img'],
                                 xv.unsqueeze(0).float(), 
                                 yv.unsqueeze(0).float()], dim=0)
            elif args.add_xy_channels == 2:
                bb_other = bb_info['other_bb_before']
                x, y = int(bb_other[0]), int(bb_other[1])

                xrb, yrb = torch.meshgrid([torch.arange(0, 256-y),
                                           torch.arange(0, 256-x)])
                xrt, yrt = torch.meshgrid([torch.arange(-y, 0),
                                           torch.arange(0, 256-x)])
                xlt, ylt = torch.meshgrid([torch.arange(-y, 0),
                                           torch.arange(-x, 0)])
                xlb, ylb = torch.meshgrid([torch.arange(0, 256-y),
                                           torch.arange(-x, 0)])
                xv = torch.cat([
                    torch.cat([xlt, xrt], dim=1),
                    torch.cat([xlb, xrb], dim=1)], dim=0)
                yv = torch.cat([
                    torch.cat([ylt, yrt], dim=1),
                    torch.cat([ylb, yrb], dim=1)], dim=0)
                # xy_list.append(
                #     np.stack([xv.unsqueeze(0), yv.unsqueeze(0)], axis=0))
                img = torch.cat([data['img'], 
                                 xv.unsqueeze(0).float(),
                                 yv.unsqueeze(0).float()], dim=0)

            x_dict['batch_img_list'].append(img)

            action_data = [data['action_info']['action_type'],
                           data['action_info']['force']]
            x_dict['batch_action_list'].append(action_data)

        return x_dict

    def collate_batch_data_to_tensors(self, proc_batch_dict):
        '''Collate processed batch into tensors.'''
        # Now collate the batch data together
        x_tensor_dict = {}
        x_dict = proc_batch_dict
        device = self.config.get_device()
        args = self.config.args

        x_tensor_dict['batch_img'] = torch.stack(x_dict['batch_img_list']).to(device)
        x_tensor_dict['batch_obj_bb_list'] = torch.Tensor(
            x_dict['batch_obj_bb_list']).to(device) / 256.0
        x_tensor_dict['batch_action_list'] = torch.Tensor(
            x_dict['batch_action_list']).to(device)
        x_tensor_dict['batch_other_bb_pred_list'] = torch.Tensor(
            x_dict['batch_other_bb_pred_list']).to(device) / 256.0

        return x_tensor_dict

    def run_model_on_batch(self,
                           x_tensor_dict,
                           batch_size,
                           train=False):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        img_emb = self.model.forward_image(x_tensor_dict['batch_img'])
        # TODO: Add hinge loss
        img_emb = img_emb.squeeze()

        if args.use_bb_in_input:
            img_emb_with_action = torch.cat([
                img_emb, 
                x_tensor_dict['batch_obj_bb_list'],
                x_tensor_dict['batch_action_list']], dim=1)
        else:
            img_emb_with_action = torch.cat([
                img_emb, 
                x_tensor_dict['batch_action_list']], dim=1)
        
        img_action_emb = self.model.forward_image_with_action(img_emb_with_action)

        pred_delta_bb = self.model.forward_predict_delta_pose(img_action_emb)
        bb_pred_loss = args.weight_bb * self.bb_pred_loss(
            pred_delta_bb,  x_tensor_dict['batch_other_bb_pred_list'])
        bb_per_pred_loss = args.weight_bb * \
            torch.pow((pred_delta_bb - x_tensor_dict['batch_other_bb_pred_list']), 2)
        
        img_action_with_delta_pose = torch.cat(
            [img_action_emb, x_tensor_dict['batch_other_bb_pred_list']], dim=1)
        pred_img_emb = self.model.forward_predict_original_img_emb(
            img_action_with_delta_pose)
        inv_model_loss = args.weight_inv_model * self.inv_model_loss(
            pred_img_emb, img_emb)

        # total_loss = bb_pred_loss + inv_model_loss
        total_loss = bb_pred_loss

        if train:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

        batch_result_dict['img_emb'] = img_emb
        batch_result_dict['img_action_emb'] = img_action_emb
        batch_result_dict['pred_delta_bb'] = pred_delta_bb
        batch_result_dict['bb_pred_loss'] = bb_pred_loss
        batch_result_dict['inv_model_loss'] = inv_model_loss
        batch_result_dict['total_loss'] = total_loss
        batch_result_dict['bb_pred_loss_per_item'] = bb_per_pred_loss.detach()

        return batch_result_dict

    def print_train_update_to_console(self, current_epoch, num_epochs,
                                      curr_batch, num_batches, train_step_count,
                                      batch_result_dict):
        args = self.config.args
        e, batch_idx = current_epoch, curr_batch
        total_loss = batch_result_dict['total_loss']

        if train_step_count % args.print_freq_iters == 0:
            print(bcolors.okblue(
                    "[{}/{}], \t Batch: [{}/{}], \t    total_loss: {:.4f}, "
                    "\t  bb: {:.4f}, \t   inv_model: {:.4f}".format(
                        e, num_epochs, batch_idx, num_batches, total_loss,
                        batch_result_dict['bb_pred_loss'],
                        batch_result_dict['inv_model_loss'])))

    def plot_train_update_to_tensorboard(self, 
                                         x_dict,
                                         x_tensor_dict,
                                         batch_result_dict,
                                         batch_data,
                                         step_count,
                                         log_prefix='',
                                         plot_images=True,
                                         plot_loss=True):
        args = self.config.args

        if plot_images:
            self.logger.summary_writer.add_images(
                log_prefix + '/image/input_image',
                x_tensor_dict['batch_img'].clone().cpu()[:,:3,:,:],
                step_count)

        if plot_loss:
            self.logger.summary_writer.add_scalar(
                log_prefix+'loss/bb_pred_loss', 
                batch_result_dict['bb_pred_loss'],
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
        
            scene_type_list = [x['img_info']['scene_type'] for x in batch_data]
            scene_type_arr = np.array(scene_type_list, dtype=np.int32)
            bb_loss_per_item = batch_result_dict['bb_pred_loss_per_item'].cpu().numpy()
            for scene_type in set(scene_type_list):
                bb_loss_scene = bb_loss_per_item[[scene_type_arr == scene_type]]
                self.logger.summary_writer.add_scalar(
                    log_prefix+'loss/scene_{}/total_loss'.format(scene_type),
                    np.mean(bb_loss_scene),
                    step_count,
                )

    

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
                'img_path': [],
                'img_info': [],
                'action_info': [],
            }
        }
        num_epochs = args.num_epochs if train else 1
        for e in range(num_epochs):
            if train:
                iter_order = np.random.permutation(train_data_idx_list)
            else:
                iter_order = np.arange(train_data_size)

            batch_size = args.batch_size if train else 2
            num_batches = train_data_size // batch_size
            data_idx = 0
            for batch_idx in range(num_batches):
                # Get raw data from the dataloader.
                batch_data = []
                for b in range(batch_size):
                    data = dataloader.get_train_data_at_idx(
                        iter_order[data_idx], train=train)
                    batch_data.append(data)
                    data_idx = data_idx + 1

                # Process raw batch data
                x_dict = self.process_raw_batch_data(batch_data)

                # Now collate the batch data together
                x_tensor_dict = self.collate_batch_data_to_tensors(x_dict)
                batch_result_dict = self.run_model_on_batch(
                    x_tensor_dict, batch_size, train=train)
                
                if not train and save_embedding:
                    for b in range(batch_size):
                        for k in ['img_path', 'img_info', 'action_info']:
                            result_dict['data_info'][k].append(batch_data[b][k])
                        
                        result_dict['emb']['img_emb'].append(
                            batch_result_dict['img_emb'][b].detach().cpu().numpy())
                        result_dict['emb']['img_action_emb'].append(
                            batch_result_dict['img_action_emb'][b].detach().cpu().numpy())

                self.print_train_update_to_console(
                    e, num_epochs, batch_idx, num_batches,
                    train_step_count, batch_result_dict)

                plot_images = viz_images and train \
                    and train_step_count %  log_freq_iters == 0
                plot_loss = train \
                    and train_step_count % args.print_freq_iters == 0

                if train:
                    self.plot_train_update_to_tensorboard(
                        x_dict, 
                        x_tensor_dict,
                        batch_result_dict,
                        batch_data,
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

                        num_batch_test, test_batch_size = 3, 7
                        test_data_size = self.dataloader.get_data_size(
                                train=False)
                        test_iter_order = np.random.permutation(
                                np.arange(test_data_size))
                        test_data_idx = 0
                        self.model.eval()
                        for test_e in range(num_batch_test):
                            batch_data = []
                            for b in range(test_batch_size):
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
                                batch_result_dict = self.run_model_on_batch(
                                    x_tensor_dict, test_batch_size, train=False)

                            print(bcolors.c_red("==== Test begin ==== "))
                            self.print_train_update_to_console(
                                e, num_epochs, test_e, num_batch_test,
                                train_step_count, batch_result_dict)
                            print(bcolors.c_red("==== Test end ==== "))

                            plot_images = test_e == 0
                            plot_loss = True
                            self.plot_train_update_to_tensorboard(
                                x_dict, x_tensor_dict, batch_result_dict,
                                batch_data,
                                test_step_count,
                                plot_loss=plot_loss,
                                plot_images=plot_images,
                                log_prefix='/test/'
                            )

                            test_step_count += 1
                            batch_data = None

                    x_dict = None
                    x_tensor_dict = None
                    batch_result_dict = None
                    torch.cuda.empty_cache()
                    self.model.eval()

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

    trainer = RelationTrainer(config)

    if len(args.checkpoint_path) > 0:
        trainer.load_checkpoint(args.checkpoint_path)
        result_dict = trainer.train(train=False,
                                    viz_images=False,
                                    save_embedding=True)
        test_result_dir = os.path.join(
            os.path.dirname(args.checkpoint_path),
            'result_{}'.format(os.path.basename(args.checkpoint_path)[:-4]))
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        emb_h5_path = os.path.join(test_result_dir, 'result_emb.h5')
        emb_h5_f = h5py.File(emb_h5_path, 'w')
        result_h5_dict = {'emb': result_dict['emb']}
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
        with open(config_pkl_path, 'wb') as config_f:
            pickle.dump((args), config_f, protocol=2)
            print(bcolors.c_red("Did save config: {}".format(config_pkl_path)))
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

    parser.add_argument('--z_dim', type=int, default=128,
                        help='Embedding size to extract from image.')

    # Loss weights
    parser.add_argument('--weight_bb', type=float, default=1.0,
                        help='Weight for BB pred loss.')
    parser.add_argument('--weight_inv_model', type=float, default=1.0,
                        help='Weight for inverse model loss.')

    parser.add_argument('--add_xy_channels', type=int, default=0,
                        choices=[0, 1, 2],
                        help='0: no xy append, 1: xy append '
                             '2: xy centered on bb')
    parser.add_argument('--use_bb_in_input', type=int, default=1, choices=[0,1],
                        help='Use bb in input')

    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=100)

    main(args)
