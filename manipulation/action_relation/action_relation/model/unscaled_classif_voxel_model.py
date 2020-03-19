import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import numpy as np

from vae.models.crn import RefinementNetwork
from vae.models.layers import build_mlp
from vae.models.pix2pix_models import ResnetGenerator_Encoder
from utils.torch_utils import Spatial3DSoftmax

from action_relation.model.simple_model import SpatialSoftmaxImageEncoder
from action_relation.model.resnet import ResNet, resnet18, resnet10
from action_relation.model.unscaled_voxel_model import CML_2


class UnscaledClassifVoxelModel(nn.Module):
    def __init__(self, img_emb_size, action_size, args, use_bb_in_input=True,
                 use_spatial_softmax=True):
        super(UnscaledClassifVoxelModel, self).__init__()
        self.img_emb_size = img_emb_size
        self.action_size = action_size
        self.use_spatial_softmax = use_spatial_softmax
        self.args = args

        if args.add_xy_channels == 0:
            input_nc = 3
        elif args.add_xy_channels == 1:
            input_nc = 6
        else:
            raise ValueError("Invalid add_xy_channels")
        
        self.voxels_to_rel_emb_model = CML_2(input_nc)
        if use_spatial_softmax:
            z_ss_size = 96
            self.spatial_softmax_encoder = Spatial3DSoftmaxImageEncoder(
                input_nc, z_ss_size
            )
        else:
            z_ss_size = 0

        # 6 for other oject pose (6D) and 12 for both anchor and other bb.
        bb_size = 6 + 12 if args.use_bb_in_input else 0
        self.img_action_model = build_mlp(
            [img_emb_size + bb_size + action_size + z_ss_size, 
             img_emb_size//2, img_emb_size//2, img_emb_size//4],
            activation='relu',
        )

        if args.pos_loss_type == 'regr':
            output_size = 3
            self.delta_pose_model = build_mlp(
                [img_emb_size//4, 64, output_size],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            n_classes = args.pos_classif_num_classes
            assert n_classes > 0
            self.delta_pos_x = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_pos_y = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_pos_z = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
        
        if args.orient_loss_type == 'regr':
            output_size = 3
            self.delta_orient_model = build_mlp(
                [img_emb_size//4, 64, output_size],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            n_classes = args.orient_classif_num_classes
            assert n_classes > 0
            self.delta_orient_x = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_orient_y = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_orient_z = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )

    def forward_image(self, img, relu_on_emb=False):
        batch_size = img.size(0)
        z = self.voxels_to_rel_emb_model(img, relu_on_emb=relu_on_emb)
        if self.use_spatial_softmax:
            z_ss = self.spatial_softmax_encoder(img)
            return torch.cat([z, z_ss], dim=1)
        else:
            return z
    
    def forward_image_with_action(self, img_with_action):
        img_action_z = self.img_action_model(img_with_action)
        return img_action_z
    
    def forward_predict_delta_pose(self, img_action_z):
        args = self.args
        delta_pose_dict = {}
        if args.pos_loss_type == 'regr':
            delta_pos = self.delta_pose_model(img_action_z)
            delta_pose_dict['delta_pos'] = delta_pos
        else:
            delta_pos_x = self.delta_pos_x(img_action_z)
            delta_pos_y = self.delta_pos_y(img_action_z)
            delta_pos_z = self.delta_pos_z(img_action_z)
            delta_pose_dict['delta_pos_x'] = delta_pos_x
            delta_pose_dict['delta_pos_y'] = delta_pos_y
            delta_pose_dict['delta_pos_z'] = delta_pos_z

        if args.orient_loss_type == 'regr':
            delta_orient = self.delta_orient_model(img_action_z)
            delta_pose_dict['delta_orient'] = delta_orient
        else:
            delta_orient_x = self.delta_orient_x(img_action_z)
            delta_orient_y = self.delta_orient_y(img_action_z)
            delta_orient_z = self.delta_orient_z(img_action_z)
            delta_pose_dict['delta_orient_x'] = delta_orient_x
            delta_pose_dict['delta_orient_y'] = delta_orient_y
            delta_pose_dict['delta_orient_z'] = delta_orient_z
        
        return delta_pose_dict
    
    def forward_predict_original_img_emb(self, z):
        z_original = self.inv_model(z)
        return z_original


class UnscaledClassifResNetVoxelModel(nn.Module):
    def __init__(self, resnet_klass, img_emb_size, action_size, args, 
                 use_bb_in_input=True, use_spatial_softmax=True):
        super(UnscaledClassifResNetVoxelModel, self).__init__()
        self.img_emb_size = img_emb_size
        self.action_size = action_size
        self.use_spatial_softmax = use_spatial_softmax
        self.args = args

        if args.add_xy_channels == 0:
            input_nc = 3
        elif args.add_xy_channels == 1:
            input_nc = 6
        else:
            raise ValueError("Invalid add_xy_channels")

        self.resnet = resnet_klass(emb_size=img_emb_size, input_channels=input_nc)

        z_ss_size = 0
        bb_size = 6 + 12 if args.use_bb_in_input else 0
        self.img_action_model = build_mlp(
            [img_emb_size + bb_size + action_size + z_ss_size, 
             img_emb_size//2, img_emb_size//4],
            activation='relu',
            final_nonlinearity=True,
        )

        if args.pos_loss_type == 'regr':
            output_size = 3
            self.delta_pose_model = build_mlp(
                [img_emb_size//4, 64, output_size],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            n_classes = args.pos_classif_num_classes
            assert n_classes > 0
            self.delta_pos_x = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_pos_y = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_pos_z = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
        
        if args.orient_loss_type == 'regr':
            output_size = 3
            self.delta_orient_model = build_mlp(
                [img_emb_size//4, 64, output_size],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            n_classes = args.orient_classif_num_classes
            assert n_classes > 0
            self.delta_orient_x = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_orient_y = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )
            self.delta_orient_z = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )

        
    def forward_image(self, img):
        batch_size = img.size(0)
        z = self.resnet(img)
        if self.use_spatial_softmax:
            z_ss = self.spatial_softmax_encoder(img)
            return torch.cat([z, z_ss], dim=1)
        else:
            return z

    def forward_image_with_action(self, img_with_action):
        img_action_z = self.img_action_model(img_with_action)
        return img_action_z
    
    def forward_predict_delta_pose(self, img_action_z):
        args = self.args
        delta_pose_dict = {}
        if args.pos_loss_type == 'regr':
            delta_pos = self.delta_pose_model(img_action_z)
            delta_pose_dict['delta_pos'] = delta_pos
        else:
            delta_pos_x = self.delta_pos_x(img_action_z)
            delta_pos_y = self.delta_pos_y(img_action_z)
            delta_pos_z = self.delta_pos_z(img_action_z)
            delta_pose_dict['delta_pos_x'] = delta_pos_x
            delta_pose_dict['delta_pos_y'] = delta_pos_y
            delta_pose_dict['delta_pos_z'] = delta_pos_z

        if args.orient_loss_type == 'regr':
            delta_orient = self.delta_orient_model(img_action_z)
            delta_pose_dict['delta_orient'] = delta_orient
        else:
            delta_orient_x = self.delta_orient_x(img_action_z)
            delta_orient_y = self.delta_orient_y(img_action_z)
            delta_orient_z = self.delta_orient_z(img_action_z)
            delta_pose_dict['delta_orient_x'] = delta_orient_x
            delta_pose_dict['delta_orient_y'] = delta_orient_y
            delta_pose_dict['delta_orient_z'] = delta_orient_z
        
        return delta_pose_dict


def get_unscaled_classif_resnet10(z_dim, action_size, args, **kwargs):
    return UnscaledClassifResNetVoxelModel(
        resnet10, z_dim, action_size, args, **kwargs)

def get_unscaled_classif_resnet18(z_dim, action_size, args, **kwargs):
    return UnscaledClassifResNetVoxelModel(
        resnet18, z_dim, action_size, args, **kwargs)
