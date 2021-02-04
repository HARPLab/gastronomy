import torch
import numpy as np

def NLL_Loss(dist, target, mode=None):
    """
    Negative log likelihood loss, where the predictions are distributions
    Inputs:
        target (Tensor): Target/ground truth value. Should be shape
            batch size x 1
        dist (torch.distribution): The output distribution of the
            network. Should have a log_prob() method.
    """
    #TODO should this be divided by the batch size?
    return -dist.log_prob(target).sum() / dist.batch_shape[0]

def weighted_binary_cross_entropy_with_logits(logits, targets, pos_weight=1.0, weight=None, reduction='mean'):
    """
    Inputs:
        logits: the prediction logits of size [N,C], N samples and C Classes. Output of network (No sigmoid)
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
        For other args see: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    Ref:
     - https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    if not (targets.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))
    if pos_weight is None:
        pos_weight = 1.0

    max_val = (-logits).clamp(min=0)
    log_weight = 1 + (pos_weight - 1) * targets
    # NOTE the below can be simplified: https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1 - targets) * logits + log_weight * (((-max_val).exp() + (-logits - max_val).exp()).log() + max_val) 

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError

class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, pos_weight=None, weight=None, PosWeightIsDynamic=False, WeightIsDynamic=False, reduction='mean'):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        Ref:
         - https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
        Notes:
         - another possible way to do this: https://discuss.pytorch.org/t/dealing-with-imbalanced-datasets-in-pytorch/22596/4
        """
        super().__init__()
        self.register_buffer('weight', weight) #NOTE: idk if these buffers are necessary, since they might be getting overwritten in forward
        self.register_buffer('pos_weight', pos_weight)
        self.reduction = reduction
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, logits, target, epsilon=1e-5):
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts + epsilon)

        if self.weight is not None:
            return weighted_binary_cross_entropy_with_logits(logits,
                                                             targets=target,
                                                             pos_weight=self.pos_weight,
                                                             weight=self.weight,
                                                             reduction=self.reduction)
        else:
            return weighted_binary_cross_entropy_with_logits(logits, 
                                                             targets=target,
                                                             pos_weight=self.pos_weight,
                                                             weight=None,
                                                             reduction=self.reduction)