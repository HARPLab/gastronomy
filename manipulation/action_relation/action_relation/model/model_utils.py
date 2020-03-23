import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb


class ScaledMSELoss(nn.Module):
    '''Scale MSE loss using the norm of ground truth values.
    
    Use min_error_norm for cases where the gt ~ 0.
    '''
    def __init__(self, min_error_norm):
        super(ScaledMSELoss, self).__init__()
        self.min_error_norm = min_error_norm

        self.non_reduced_loss = None
        self.use_pred_in_den = True 
    
    def forward(self, input, target):
        sq_error = torch.sum(torch.pow(input - target, 2), keepdim=True, dim=1)

        target_norm = torch.norm(target, dim=1, keepdim=True)
        target_norm = torch.clamp(target_norm, self.min_error_norm)


        if self.use_pred_in_den:
            input_norm = torch.norm(input, dim=1, keepdim=True)
            input_norm = torch.clamp(input_norm.detach(), self.min_error_norm)
            self.non_reduced_loss = sq_error/(target_norm + input_norm)
        else:
            self.non_reduced_loss = sq_error/target_norm

        assert self.non_reduced_loss.max().item() <= 1e4, "Very large loss"

        loss = torch.mean(self.non_reduced_loss)
        return loss
