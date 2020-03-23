import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae.models.crn import RefinementNetwork
from vae.models.layers import build_mlp
from vae.models.pix2pix_models import ResnetGenerator_Encoder
from utils.torch_utils import Spatial3DSoftmax

from action_relation.model.simple_model import SpatialSoftmaxImageEncoder
from action_relation.model.resnet import ResNet, resnet18, resnet10

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

class CML_2(nn.Module):
    '''Input size is and the output size is '''
    def __init__(self, input_nc):
        super(CML_2, self).__init__()
        self.input_nc = input_nc
        c = input_nc
        self.conv3d_1 = nn.Conv3d( c, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(64, 64, 5, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, 32, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(32, 16, 3, stride=(1, 1, 1), padding=(1, 1, 1))
        # self.linear = nn.Linear(2880, 512)

        # Works on inputs of size (B, 3, 50, 50), where B is the batch size.
        self.linear = nn.Linear(3456, 512)

    def forward(self, inp, relu_on_emb=False):
        # assert not relu_on_emb 
        x = F.relu(self.conv3d_1(inp))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))
        x = F.relu(self.conv3d_4(x))
        x = x.view(x.size(0), -1)
        # if relu_on_emb:
        #     x = F.relu(self.linear(x))
        # else:
        #     x = self.linear(x)
        x = self.linear(x)
        return x


class CML_3(nn.Module):
    '''Input size is and the output size is 
    
    Works on inputs of size (B, 3, 50, 50), where B is the batch size.
    '''
    def __init__(self, input_nc):
        super(CML_3, self).__init__()
        self.input_nc = input_nc
        c = input_nc
        self.conv3d_1 = nn.Conv3d( c, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(64, 64, 5, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, 32, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(32, 16, 3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.linear = nn.Linear(3456, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, inp, relu_on_emb=False):
        # assert not relu_on_emb 
        x = F.relu(self.conv3d_1(inp))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))
        x = F.relu(self.conv3d_4(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CML_4(nn.Module):
    '''Input size is and the output size is 
    
    Works on inputs of size (B, 3, 50, 50), where B is the batch size.
    '''
    def __init__(self, input_nc, final_emb_size, out_emb=None):
        super(CML_4, self).__init__()
        self.input_nc = input_nc
        self.out_emb = out_emb

        c = input_nc
        self.conv3d_1 = nn.Conv3d( c, 64, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(64, 64, 5, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, 32, 3, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(32, 16, 3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.linear = nn.Linear(3456, 512)
        self.fc1 = nn.Linear(512, final_emb_size)

    def forward(self, inp, relu_on_emb=False):
        # assert not relu_on_emb 
        x = F.relu(self.conv3d_1(inp))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))
        x = F.relu(self.conv3d_4(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear(x))
        
        if self.out_emb is None:
            out = self.fc1(x)
        else:
            out = self.out_emb(self.fc1(x))

        return out


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
        self.spatial_softmax = Spatial3DSoftmax(52, 52, 52, self.final_c, 
                                                temperature=1)
    
    def forward(self, img):
        x = self.encoder(img)
        output = self.spatial_softmax(x)
        return output

class BoundingBoxOnlyModel(nn.Module):
    def __init__(self, img_emb_size, action_size, args, use_bb_in_input=True,
                 n_classes=0, use_spatial_softmax=True):
        super(BoundingBoxOnlyModel, self).__init__()
        # 6 for other oject pose (6D) and 12 for both anchor and other bb.
        bb_size = 6 + 12 if args.use_bb_in_input else 0
        assert bb_size > 0
        self.bb_action_model = build_mlp(
            [bb_size + action_size, 
             512, 256, 64],
            activation='relu',
        )

        use_regression = (args.loss_type == 'regr')
        if use_regression:
            self.delta_pose_model = build_mlp(
                [64, 6],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            assert n_classes > 0, "Invalid number of classes."
            print("Using classif model with {} classes".format(n_classes))
            self.delta_pose_model = build_mlp(
                [64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )

    def forward_image_with_action(self, img_with_action):
        out = self.bb_action_model(img_with_action)
        return out
    
    def forward_predict_delta_pose(self, img_action_z):
        delta_pose = self.delta_pose_model(img_action_z)
        return delta_pose


class UnscaledVoxelModel(nn.Module):
    def __init__(self, img_emb_size, action_size, args, use_bb_in_input=True,
                 n_classes=0, use_spatial_softmax=True):
        super(UnscaledVoxelModel, self).__init__()
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

        if img_emb_size <= 36:
            self.voxels_to_rel_emb_model = CML_3(input_nc)
        else:
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

        use_regression = (args.loss_type == 'regr')
        if use_regression:
            output_size = 6
            self.delta_pose_model = build_mlp(
                [img_emb_size//4, 64, output_size],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            assert n_classes > 0, "Invalid number of classes."
            print("Using classif model with {} classes".format(n_classes))
            self.delta_pose_model = build_mlp(
                [img_emb_size//4, 64, n_classes],
                activation='relu',
                final_nonlinearity=False
            )

        # self.inv_model = build_mlp(
        #     [img_emb_size + 4, 256, 256, img_emb_size+z_ss_size]
        # )
    

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
        delta_pose = self.delta_pose_model(img_action_z)
        return delta_pose
    
    def forward_predict_original_img_emb(self, z):
        z_original = self.inv_model(z)
        return z_original


class SmallEmbUnscaledVoxelModel(nn.Module):
    def __init__(self, img_emb_size, action_size, args, use_bb_in_input=True,
                 n_classes=0, use_spatial_softmax=True):
        super(SmallEmbUnscaledVoxelModel, self).__init__()
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

        # Why do we need an extra +3 ?
        # self.voxels_to_rel_emb_model = CML_4(input_nc, img_emb_size+3, out_emb=None)
        self.voxels_to_rel_emb_model = CML_4(input_nc, img_emb_size, out_emb=None)

        if use_spatial_softmax:
            z_ss_size = 96
            self.spatial_softmax_encoder = Spatial3DSoftmaxImageEncoder(
                input_nc, z_ss_size
            )
        else:
            z_ss_size = 0

        # 6 for other oject pose (6D) and 12 for both anchor and other bb.
        bb_size = 6 + 12 if args.use_bb_in_input else 0
        delta_pose_model_inp_size = img_emb_size \
                                        + bb_size \
                                        + action_size \
                                        + z_ss_size

        use_regression = (args.loss_type == 'regr')
        delta_pose_model_out_size = 6 if use_regression else n_classes
        if (delta_pose_model_inp_size // 4) >= 32:
            delta_pose_model_layers = [delta_pose_model_inp_size, 
                                       delta_pose_model_inp_size//2, 
                                       delta_pose_model_inp_size//4, 
                                       delta_pose_model_out_size]
        else:
            delta_pose_model_layers = [delta_pose_model_inp_size, 
                                       delta_pose_model_inp_size//2, 
                                       delta_pose_model_out_size]
        if use_regression:
            output_size = 6
            self.delta_pose_model = build_mlp(
                delta_pose_model_layers,
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            assert n_classes > 0, "Invalid number of classes."
            print("Using classif model with {} classes".format(n_classes))
            self.delta_pose_model = build_mlp(
                delta_pose_model_layers,
                activation='relu',
                final_nonlinearity=False
            )

        # self.inv_model = build_mlp(
        #     [img_emb_size + 4, 256, 256, img_emb_size+z_ss_size]
        # )
    

    def forward_image(self, img, relu_on_emb=False):
        batch_size = img.size(0)
        z = self.voxels_to_rel_emb_model(img, relu_on_emb=relu_on_emb)
        if self.use_spatial_softmax:
            z_ss = self.spatial_softmax_encoder(img)
            return torch.cat([z, z_ss], dim=1)
        else:
            return z
    
    def forward_image_with_action(self, img_with_action):
        # DO NOTHING.
        return img_with_action
    
    def forward_predict_delta_pose(self, img_action_z):
        delta_pose = self.delta_pose_model(img_action_z)
        return delta_pose
    
    def forward_predict_original_img_emb(self, z):
        z_original = self.inv_model(z)
        return z_original


class UnscaledResNetVoxelModel(nn.Module):
    def __init__(self, resnet_klass, img_emb_size, action_size, args, 
                 use_bb_in_input=True, n_classes=0, use_spatial_softmax=True):
        super(UnscaledResNetVoxelModel, self).__init__()
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

        use_regression = (args.loss_type == 'regr')
        if use_regression:
            output_size = 6
            self.delta_pose_model = build_mlp(
                [img_emb_size//4, output_size],
                activation='relu',
                final_nonlinearity=False,
            )
            print("Using regression model.")
        else:
            assert n_classes > 0, "Invalid number of classes."
            print("Using classif model with {} classes".format(n_classes))
            self.delta_pose_model = build_mlp(
                [img_emb_size//4, n_classes],
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
        delta_pose = self.delta_pose_model(img_action_z)
        return delta_pose
    


class UnscaledPrecondVoxelModel(nn.Module):
    def __init__(self, img_emb_size, args, use_bb_in_input=True,
                 n_classes=0, use_spatial_softmax=True):
        super(UnscaledPrecondVoxelModel, self).__init__()
        self.img_emb_size = img_emb_size
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
        bb_size = 12 if args.use_bb_in_input else 0
        self.img_emb_pred_model = build_mlp(
            [img_emb_size + bb_size + z_ss_size, 
             256, 64, 1],
            activation='relu',
            final_nonlinearity=False
        )

    def forward_predict_precond(self, inp, bb_input=None):
        batch_size = inp.size(0)
        z = self.voxels_to_rel_emb_model(inp)
        if self.use_spatial_softmax:
            z_ss = self.spatial_softmax_encoder(inp)
            z = torch.cat([z, z_ss], dim=1)

        if bb_input is not None: 
            z = torch.cat([z, bb_input], dim=1)
        
        out = F.sigmoid(self.img_emb_pred_model(z))
        return out


class PrecondEmbeddingModel(nn.Module):
    def __init__(self, inp_emb_size, args, use_bb_in_input=False):
        super(PrecondEmbeddingModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args

        bb_size = 12 if args.use_bb_in_input else 0
        self.model = nn.Sequential(
            nn.Linear(inp_emb_size*6 + bb_size, 1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_predict_precond(self, inp, bb_input=None):
        batch_size = inp.size(0)
        if bb_input is not None: 
            inp = torch.cat([inp, bb_input], dim=1)
        out = self.model(inp)
        sigmoid_out = F.sigmoid(out)
        return sigmoid_out
    
    def forward_scene_emb_predict_precond(self, scene_emb_list, bb_input=None):
        inp_list = []
        for scene_emb in scene_emb_list:
            inp_scene = torch.stack(scene_emb).view(-1)
            inp_list.append(inp_scene)

        inp = torch.stack(inp_list)
        batch_size = inp.size(0)
        if bb_input is not None: 
            inp = torch.cat([inp, bb_input], dim=1)
        out = self.model(inp)
        sigmoid_out = F.sigmoid(out)
        return sigmoid_out


def get_unscaled_resnet10(z_dim, action_size, args, **kwargs):
    return UnscaledResNetVoxelModel(
        resnet10, z_dim, action_size, args, **kwargs)


def get_unscaled_resnet18(z_dim, action_size, args, **kwargs):
    return UnscaledResNetVoxelModel(
        resnet18, z_dim, action_size, args, **kwargs)
