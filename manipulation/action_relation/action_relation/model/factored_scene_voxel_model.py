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


class FactoredPairwiseEmbeddingModel(nn.Module):
    '''
    This model implements g(\sum_{i, j}[f(phi_i, phi_j)])
    '''
    def __init__(self, inp_emb_size, args, use_bb_in_input=False):
        super(FactoredPairwiseEmbeddingModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args

        bb_size = 12 if args.use_bb_in_input else 0

        self.f = nn.Sequential(
            nn.Linear(inp_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.g = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )


    def forward_predict_precond(self, inp, bb_input=None):
        raise ValueError("Not implemented.")
    

    def forward_scene_emb_predict_precond(self, scene_emb_list, 
                                          scene_obj_pair_far_apart_list,
                                          bb_input=None):
        out_list = []
        device = scene_emb_list[0][0].device
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            all_pair_scene_emb_tensor = torch.stack(all_pair_scene_emb)
            all_pair_far_apart_tensor = torch.stack(
                scene_obj_pair_far_apart_list[scene_i]).to(device)
            assert all_pair_scene_emb_tensor.size(0) == all_pair_far_apart_tensor.size(0)
            f_ij = self.f(all_pair_scene_emb_tensor)
            weights = (1 - all_pair_far_apart_tensor).float()
            f_ij_sum = torch.sum(f_ij * weights.unsqueeze(1), axis=0, keepdim=True)

            out = self.g(f_ij_sum)
            sigmoid_out = torch.sigmoid(out)
            out_list.append(sigmoid_out)
        
        output = torch.stack(out_list).squeeze(2)
        return output


class FactoredPairwiseEmbeddingWithIdentityLabelModel(nn.Module):
    '''
    This model implements g(\sum_{i, j}[f(phi_i, phi_j)])
    '''
    def __init__(self, inp_emb_size, args, use_bb_in_input=False):
        super(FactoredPairwiseEmbeddingWithIdentityLabelModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args

        bb_size = 12 if args.use_bb_in_input else 0
        label_id_size = 2

        self.f = nn.Sequential(
            nn.Linear(inp_emb_size + label_id_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.g = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward_predict_precond(self, inp, bb_input=None):
        raise ValueError("Not implemented.")
    
    def get_obj_pair_number_ids(self, num_obj):
        obj_id_list = []
        for i in range(num_obj):
            if i == 0 or i == 1:
                obj_id_list.append(i + 1)
            else:
                obj_id_list.append(3)
        from itertools import permutations
        obj_pair = list(permutations(obj_id_list, 2))
        return torch.Tensor(obj_pair)

    def forward_scene_emb_predict_precond(self, scene_emb_list, 
                                          scene_obj_pair_far_apart_list,
                                          bb_input=None,
                                          scene_obj_label_list=None):
        out_list = []
        device = scene_emb_list[0][0].device
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            all_pair_scene_emb_tensor = torch.stack(all_pair_scene_emb)
            all_pair_far_apart_tensor = torch.stack(
                scene_obj_pair_far_apart_list[scene_i]).to(device)
            assert all_pair_scene_emb_tensor.size(0) == all_pair_far_apart_tensor.size(0)

            num_objs = (1 + np.sqrt(1 + 4 * len(all_pair_scene_emb))) / 2
            num_objs_int = int((1 + np.sqrt(1 + 4 * len(all_pair_scene_emb))) // 2)
            assert np.around(num_objs - 0.1) == num_objs_int
            assert np.around(num_objs + 0.1) == num_objs_int

            # Code for 0-1 ids
            # obj_pair_ids = torch.zeros((len(all_pair_scene_emb)), 2)
            # obj_pair_ids[:, 1] = 1
            # obj_pair_ids[0, 0] = 1
            # obj_pair_ids[0, 1] = 0
            # obj_pair_ids[num_objs_int-1, 0] = 1
            # obj_pair_ids[num_objs_int-1, 1] = 0
            # obj_pair_ids = obj_pair_ids.to(device)

            obj_pair_ids = self.get_obj_pair_number_ids(num_objs_int).to(device)

            inp_emb_tensor = torch.cat([obj_pair_ids, all_pair_scene_emb_tensor], dim=1)
            f_ij = self.f(inp_emb_tensor)
            weights = (1 - all_pair_far_apart_tensor).float()
            f_ij_sum = torch.sum(f_ij * weights.unsqueeze(1), axis=0, keepdim=True)

            out = self.g(f_ij_sum)
            sigmoid_out = torch.sigmoid(out)
            out_list.append(sigmoid_out)
        
        output = torch.stack(out_list).squeeze(2)
        return output


class FactoredPairwiseEmbeddingWithIdentityLabelModel_2(nn.Module):
    '''
    This model implements g(\sum_{i, j}[f(phi_i, phi_j)])
    '''
    def __init__(self, inp_emb_size, args, use_bb_in_input=False):
        super(FactoredPairwiseEmbeddingWithIdentityLabelModel_2, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args

        bb_size = 12 if args.use_bb_in_input else 0

        self.f1 = nn.Sequential(
            nn.Linear(inp_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.f2 = nn.Sequential(
            nn.Linear(inp_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )

        self.g = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward_predict_precond(self, inp, bb_input=None):
        raise ValueError("Not implemented.")
    

    def forward_scene_emb_predict_precond(self, scene_emb_list, 
                                          scene_obj_pair_far_apart_list,
                                          bb_input=None,
                                          scene_obj_label_list=None):
        out_list = []
        device = scene_emb_list[0][0].device
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            num_objs = (1 + np.sqrt(1 + 4 * len(all_pair_scene_emb))) / 2
            num_objs_int = int((1 + np.sqrt(1 + 4 * len(all_pair_scene_emb))) // 2)
            assert np.around(num_objs - 0.1) == num_objs_int
            assert np.around(num_objs + 0.1) == num_objs_int

            f1_objs_list = [all_pair_scene_emb[0], all_pair_scene_emb[num_objs_int-1]]
            f2_objs_list = [emb_i for i, emb_i in enumerate(all_pair_scene_emb)
                            if i != 0 and i != num_objs_int-1]
            far_apart_list = scene_obj_pair_far_apart_list[scene_i]
            f1_far_apart_list = [0, 0]
            f2_far_apart_list = [far_apart_list[i] for i in range(len(far_apart_list))
                                 if i != 0 and i != num_objs_int-1]
            
            f1_objs_tensor = torch.stack(f1_objs_list)
            f1_ij = self.f1(f1_objs_tensor)
            f1_ij_sum = torch.sum(f1_ij, axis=0, keepdim=True)

            if len(f2_objs_list) > 0:
                f2_objs_tensor = torch.stack(f2_objs_list)
                f2_far_apart_tensor = torch.stack(f2_far_apart_list).to(device)
                f2_ij = self.f2(f2_objs_tensor)
                f2_weights = (1 - f2_far_apart_tensor).float()
                f2_ij_sum = torch.sum(f2_ij * f2_weights.unsqueeze(1), axis=0, keepdim=True)
            else:
                f2_ij_sum = torch.zeros(f1_ij_sum.size()).to(device)
            
            f_ij_sum = torch.cat([f1_ij_sum, f2_ij_sum], dim=1)
            out = self.g(f_ij_sum)
            sigmoid_out = torch.sigmoid(out)
            out_list.append(sigmoid_out)
        
        output = torch.stack(out_list).squeeze(2)
        return output


class FactoredPairwiseAttentionEmbeddingModel(nn.Module):
    def __init__(self, inp_emb_size, args, use_bb_in_input=False):
        super(FactoredPairwiseAttentionEmbeddingModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args

        bb_size = 12 if args.use_bb_in_input else 0

        self.backbone = nn.Sequential(
            nn.Linear(inp_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.f = nn.Sequential(
            nn.Linear(64, 16),
            nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.g = nn.Sequential(
            nn.Linear(16, 1),
        )


    def forward_predict_precond(self, inp, bb_input=None):
        raise ValueError("Not implemented.")
    

    def forward_scene_emb_predict_precond(self, scene_emb_list, 
                                          scene_obj_pair_far_apart_list,
                                          bb_input=None):
        out_list = []
        device = scene_emb_list[0][0].device
        for scene_i, all_pair_scene_emb in enumerate(scene_emb_list):
            all_pair_scene_emb_tensor = torch.stack(all_pair_scene_emb)
            all_pair_far_apart_tensor = torch.stack(
                scene_obj_pair_far_apart_list[scene_i]).to(device)
            assert all_pair_scene_emb_tensor.size(0) == all_pair_far_apart_tensor.size(0)
            bb_out = self.backbone(all_pair_scene_emb_tensor)

            f_ij = self.f(bb_out) * self.a(bb_out) 
            weights = (1 - all_pair_far_apart_tensor).float()
            f_ij_sum = torch.mean(f_ij * weights.unsqueeze(1), axis=0, keepdim=True)

            out = self.g(f_ij_sum)
            sigmoid_out = torch.sigmoid(out)
            out_list.append(sigmoid_out)
        
        output = torch.stack(out_list).squeeze(2)
        return output