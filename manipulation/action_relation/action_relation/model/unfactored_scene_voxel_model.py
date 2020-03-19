import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae.models.crn import RefinementNetwork
from vae.models.layers import build_mlp
from vae.models.pix2pix_models import ResnetGenerator_Encoder
from utils.torch_utils import Spatial3DSoftmax

from action_relation.model.simple_model import SpatialSoftmaxImageEncoder
from action_relation.model.resnet import resnet18_classif, resnet10_classif
from action_relation.model.resnet import resnet18, resnet10
from action_relation.model.resnet import resnet18_entire_scene, resnet10


class UnfactoredSceneCML_1(nn.Module):
    '''Input size is and the output size is 
    
    Works on inputs of size (B, 3, 50, 50), where B is the batch size.
    '''
    def __init__(self, input_nc, final_emb_size, out_emb=None):
        super(UnfactoredSceneCML_1, self).__init__()
        self.input_nc = input_nc
        self.out_emb = out_emb

        assert final_emb_size <= 256

        c = input_nc
        self.conv_model = nn.Sequential(
            nn.Conv3d( c, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, 5, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 16, 3, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 8, 3, stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        self.linear_model = nn.Sequential(
            nn.Linear(1440, 512),
            nn.ReLU(),
            nn.Linear(512, final_emb_size),
        )

    def forward(self, inp):
        batch_size = inp.size(0)
        x = self.conv_model(inp)
        x = x.view(batch_size, -1)

        out = self.linear_model(x)
        
        if self.out_emb is not None:
            out = self.out_emb(out)

        return out


class UnfactoredSceneEmbeddingModel(nn.Module):
    def __init__(self, out_emb_size, args):
        super(UnfactoredSceneEmbeddingModel, self).__init__()
        self.out_emb_size = out_emb_size
        self.args = args

        input_nc = 2
        self.model = UnfactoredSceneCML_1(input_nc, out_emb_size)

    def forward_image(self, inp):
        out = self.model(inp)
        return out
    

class UnfactoredSceneClassifModel(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(UnfactoredSceneClassifModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args

        input_nc = 2
        if inp_emb_size <= 512 and inp_emb_size > 128:
            self.model = nn.Sequential(
                nn.Linear(inp_emb_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        elif inp_emb_size <= 128:
            assert inp_emb_size >= 64
            self.model = nn.Sequential(
                nn.Linear(inp_emb_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        elif inp_emb_size > 512:
            self.model = nn.Sequential(
                nn.Linear(inp_emb_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )


    def forward_scene_emb_predict_precond(self, inp):
        out = self.model(inp)
        sigmoid_out = F.sigmoid(out)
        return sigmoid_out


def get_unscaled_resnet10(**kwargs):
    return resnet10_classif(**kwargs)


def get_unscaled_resnet18(**kwargs):
    return resnet18_classif(**kwargs)


class Resnet18EmbModel(nn.Module):
    def __init__(self, out_emb_size, args):
        super(Resnet18EmbModel, self).__init__()
        self.out_emb_size = out_emb_size
        self.args = args

        input_nc = 2
        self.model = resnet18_entire_scene(emb_size=out_emb_size, input_channels=input_nc)

    def forward_image(self, inp):
        out = self.model(inp)
        return out