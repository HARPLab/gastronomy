import torch
import torch.nn as nn

import sys
sys.path.append("..")
import ipdb
st = ipdb.set_trace
import hyperparams as hyp
import archs.encoder3D as encoder3D
from utils_basic import l2_normalize

class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        self.net = encoder3D.Net3D_NOBN(in_channel=4, pred_dim=hyp.feat_dim).cuda()

    def forward(self, feat, summ_writer, mask=None,prefix=""):
        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat.shape)
        summ_writer.summ_feat(f'feat/{prefix}feat0_input', feat)
    
        feat = self.net(feat)
        feat = l2_normalize(feat, dim=1)
        summ_writer.summ_feat(f'feat/{prefix}feat3_out', feat)

        return feat,  total_loss


