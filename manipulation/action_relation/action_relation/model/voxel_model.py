import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae.models.crn import RefinementNetwork
from vae.models.layers import build_mlp
from vae.models.pix2pix_models import ResnetGenerator_Encoder
from utils.torch_utils import Spatial3DSoftmax

from action_relation.model.simple_model import SpatialSoftmaxImageEncoder

class SimpleConvModel(nn.Module):
    def __init__(self, input_nc):
        super(SimpleConvModel, self).__init__()
        self.input_nc = input_nc
        c = input_nc
        self.conv3d_1 = nn.Conv3d( c, 64, (5, 5, 3), stride=(2, 2, 2), padding=1)
        self.conv3d_2 = nn.Conv3d(64, 64, (5, 5, 3), stride=(2, 2, 2), padding=1)
        self.conv3d_3 = nn.Conv3d(64, 32, 3, stride=(1, 1, 1), padding=0)
        self.linear = nn.Linear(1600, 512)

    def forward(self, x):
        x = F.relu(self.conv3d_1(x))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self, input_nc):
        super(CML, self).__init__()
        self.input_nc = input_nc
        c = input_nc
        self.conv3d_1 = nn.Conv3d( c, 64, 5, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(64, 64, 5, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(64, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_5 = nn.Conv3d(64, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_6 = nn.Conv3d(64, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        x = F.relu(self.conv3d_1(x))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))
        x = F.relu(self.conv3d_4(x))
        x = F.relu(self.conv3d_5(x))
        x = F.relu(self.conv3d_6(x))
        x = x.view(x.size(0), -1)
        return x

class Spatial3DSoftmaxImageEncoder(nn.Module):
    def __init__(self, inp_channels, out_size, inp_size=64):
        super(Spatial3DSoftmaxImageEncoder, self).__init__()
        self.inp_size = inp_size
        self.final_c = out_size // 3
        self.encoder = nn.Sequential(
            nn.Conv3d(inp_channels, 64, 5, stride=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 5, stride=1),
            nn.ReLU(),
        )
        # self.spatial_softmax = Spatial3DSoftmax(17, 17, 9, self.final_c, 
                                                # temperature=1)
        self.spatial_softmax = Spatial3DSoftmax(24, 24, 8, self.final_c, 
                                                temperature=1)
    
    def forward(self, img):
        x = self.encoder(img)
        output = self.spatial_softmax(x)
        return output

class VoxelModel(nn.Module):
    def __init__(self, img_emb_size, action_size, args, use_bb_in_input=True,
                 n_classes=0):
        super(VoxelModel, self).__init__()
        self.img_emb_size = img_emb_size
        self.action_size = action_size
        self.args = args

        if args.add_xy_channels == 0:
            input_nc = 3
        elif args.add_xy_channels == 1:
            input_nc = 6
        else:
            raise ValueError("Invalid add_xy_channels")
        
        # self.voxels_to_rel_emb_model = CML(input_nc)
        self.voxels_to_rel_emb_model = SimpleConvModel(input_nc)
        z_ss_size = 48
        self.spatial_softmax_encoder = Spatial3DSoftmaxImageEncoder(
            input_nc, z_ss_size
        )

        # 6 for other oject pose (6D) and 12 for both anchor and other bb.
        bb_size = 6 + 12 if args.use_bb_in_input else 0
        self.img_action_model = build_mlp(
            [img_emb_size + bb_size + action_size + z_ss_size, 
             512, 512, img_emb_size//2],
            activation='relu',
        )

        use_regression = (args.loss_type == 'regr')
        if use_regression:
            self.delta_pose_model = build_mlp(
                [img_emb_size//2, 128, 6],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            assert n_classes > 0, "Invalid number of classes."
            print("Using classif model with {} classes".format(n_classes))
            self.delta_pose_model = build_mlp(
                [img_emb_size//2, 128, n_classes],
                activation='relu',
                final_nonlinearity=False
            )

        # self.inv_model = build_mlp(
        #     [img_emb_size + 4, 256, 256, img_emb_size+z_ss_size]
        # )
    

    def forward_image(self, img):
        batch_size = img.size(0)
        z = self.voxels_to_rel_emb_model(img)
        z_ss = self.spatial_softmax_encoder(img)
        return torch.cat([z, z_ss], dim=1)
        # return z.view(batch_size, -1)
    
    def forward_image_with_action(self, img_with_action):
        img_action_z = self.img_action_model(img_with_action)
        return img_action_z
    
    def forward_predict_delta_pose(self, img_action_z):
        delta_pose = self.delta_pose_model(img_action_z)
        return delta_pose
    
    def forward_predict_original_img_emb(self, z):
        z_original = self.inv_model(z)
        return z_original
