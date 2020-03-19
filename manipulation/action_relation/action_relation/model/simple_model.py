import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae.models.crn import RefinementNetwork
from vae.models.layers import build_mlp
from vae.models.pix2pix_models import ResnetGenerator_Encoder

from utils.torch_utils import SpatialSoftmax

class ConvImageEncoder(nn.Module):
    def __init__(self, inp_channels, out_size, inp_size=64):
        super(ConvImageEncoder, self).__init__()
        self.inp_size = inp_size
        if inp_size == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(inp_channels, 10, 5, 1),
                nn.ReLU(),
                nn.Conv2d(10, 20, 5, 2),
                nn.ReLU(),
                nn.Conv2d(20, 10, 5, 2),
                nn.ReLU())
        elif inp_size == 128:
            pass
        elif inp_size == 256:
            self.encoder = nn.Sequential(
                nn.Conv2d(inp_channels, 128, 5, 2),
                nn.ReLU(),
                nn.Conv2d(128, 256, 5, 2),
                nn.ReLU(),
                nn.Conv2d(256, 256, 5, 2),
                nn.ReLU(),
                nn.Conv2d(256, 256, 5, 2),
                nn.ReLU(),
                nn.Conv2d(256, 128, 5, 2),
                nn.ReLU())
        else:
            raise ValueError("Invalid input size to Conv encoder: {}".format(
                inp_size))
        self.linear1 = nn.Linear(3200, out_size)

    def forward(self, inp):
        batch_size = inp.size(0)
        out = self.encoder(inp)
        out = self.linear1(out.view(batch_size, -1))
        return out

class SpatialSoftmaxImageEncoder(nn.Module):
    def __init__(self, inp_channels, out_size, inp_size=64):
        super(SpatialSoftmaxImageEncoder, self).__init__()
        self.inp_size = inp_size
        self.final_c = out_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(inp_channels, 64, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, self.final_c, 5, stride=1),
            nn.ReLU(),
        )
        self.spatial_softmax = SpatialSoftmax(117, 117, self.final_c, 
                                              temperature=1)
    
    def forward(self, img):
        x = self.encoder(img)
        output = self.spatial_softmax(x)
        return output


class SimpleModel(nn.Module):
    def __init__(self, img_emb_size, action_size, args, use_bb_in_input=True):
        super(SimpleModel, self).__init__()
        self.img_emb_size = img_emb_size
        self.action_size = action_size
        self.args = args

        if args.add_xy_channels == 0:
            input_nc = 3
        elif args.add_xy_channels in (1, 2):
            input_nc = 5
        
        self.img_to_rel_emb_model = ConvImageEncoder(
            input_nc,
            img_emb_size,
            inp_size=256
        )
        z_ss_size = 128
        self.spatial_softmax_encoder = SpatialSoftmaxImageEncoder(
            input_nc, z_ss_size
        )

        # self.img_to_rel_emb_model = ResnetGenerator_Encoder(
        #     input_nc=input_nc,
        #     output_nc=3,
        #     n_blocks=1,
        #     ngf=2,
        #     n_downsampling=8,
        #     final_channels=img_emb_size//2)

        bb_size = 8 if args.use_bb_in_input else 0
        self.img_action_model = build_mlp(
            [img_emb_size + bb_size + action_size + z_ss_size, 
             256, 256, img_emb_size],
            activation='relu',
            final_nonlinearity=True,
        )

        self.delta_pose_model = build_mlp(
            [img_emb_size, 128, 4],
            activation='relu',
            final_nonlinearity=False,
        )

        self.inv_model = build_mlp(
            [img_emb_size + 4, 256, 256, img_emb_size+z_ss_size]
        )
    
    def forward_image(self, img):
        z = self.img_to_rel_emb_model(img)
        z_ss = self.spatial_softmax_encoder(img)
        return torch.cat([z, z_ss], dim=1)
    
    def forward_image_with_action(self, img_with_action):
        img_action_z = self.img_action_model(img_with_action)
        return img_action_z
    
    def forward_predict_delta_pose(self, img_action_z):
        delta_pose = self.delta_pose_model(img_action_z)
        return delta_pose
    
    def forward_predict_original_img_emb(self, z):
        z_original = self.inv_model(z)
        return z_original
