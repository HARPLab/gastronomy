import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dist_matrix_for_feat_tensor(feat_tensor, squared=True):
    feat_dot_prod = torch.matmul(feat_tensor, feat_tensor.T)
    sq_norm = torch.diagonal(feat_dot_prod)
    feat_dist = sq_norm[None, :] - 2*feat_dot_prod + sq_norm[:, None]
    feat_dist = torch.clamp(feat_dist, min=1e-10)
    if not squared:
        feat_dist = feat_dist.pow(0.5)
    return feat_dist

# Get labels for triplet mining
def get_orient_contrastive_loss(emb_tensor, gt_pose_tensor, label_tensor, 
                                emb_margin, gt_sim_margin, gt_diff_margin):
    idx_tensor = label_tensor + 1
    label_mask = idx_tensor * (idx_tensor[None, :] == idx_tensor[:, None]).long()
    # label_mask (i,j,k) is True iff idx_tensor[i] == idx_tensor[j] == idx_tensor[k]
    label_mask = label_mask[:, None, :] == idx_tensor[:, None]
    device = emb_tensor.device

    assert gt_pose_tensor.size(1) == 3
    n = gt_pose_tensor.size(0)
    pose_sim_all = torch.zeros((n, n)).to(device)
    for i in range(3):
        pose_i = gt_pose_tensor[:, i]
        pose_sim_i = torch.abs(pose_i[:, None] - pose_i[None, :]) < gt_sim_margin
        pose_sim_all += pose_sim_i.float()
    # poses are similar only if they are similar on all axis.
    final_pose_sim = torch.abs(pose_sim_all - 3.0) < 1e-4
    # Since this is a symmetric matrix triangulate it
    final_pose_sim = torch.triu(final_pose_sim.float())
    # Make the diagonals 0.
    final_pose_sim = final_pose_sim * (1 - torch.eye(n).to(device))
    
    # Poses are different if they are differnt by a given magnitude in any of the axes
    pose_diff_all = torch.zeros((n, n)).to(device)
    for i in range(3):
        pose_i = gt_pose_tensor[:, i]
        pose_diff_i = torch.abs(pose_i[:, None] - pose_i[None, :]) > gt_diff_margin
        pose_diff_all += pose_diff_i.float()
    # (N, N) array, 1 if (i, j) are diff.
    # final_pose_diff= np.abs(pose_diff_all - 1.0) < 1e-4
    final_pose_diff = pose_diff_all > 0.9
    # Since this is a symmetric matrix triangulate it
    final_pose_diff = torch.triu(final_pose_diff.float())
    # Make the diagonals 0.
    final_pose_diff = final_pose_diff * (1 - torch.eye(n).to(device))
    
    # if (i, j) are sim and (j, k) are diff then (i, j, k) would have value 2
    gt_pose_dist_val = final_pose_sim[:, :, None] + final_pose_diff[:, None, :]
    gt_pose_dist_mask = torch.abs(gt_pose_dist_val - 2.0) < 1e-4

    total_mask = (label_mask * gt_pose_dist_mask).float()
    all_loss_count = total_mask.sum().item()
    if all_loss_count == 0:
        return 0.0

    emb_dist = get_dist_matrix_for_feat_tensor(emb_tensor)    
    emb_dist_diff = emb_dist[:, :, None] - emb_dist[:, None, :]
    # emb_dist_diff = (emb_dist_diff * total_mask.float())

    all_loss = F.relu(emb_dist_diff + emb_margin) * total_mask
    # This will not include data points that satisfy the triplet criterion.
    #  all_loss_count = max(1, torch.sum(all_loss > 1e-6).item())
    total_loss_sum = all_loss.sum()

    loss = total_loss_sum / all_loss_count

    return loss
    

def get_orient_contrastive_loss_2(emb_tensor, gt_pose_tensor, label_tensor, 
                                emb_margin, gt_pose_margin, squared=True):
    emb_dist = get_dist_matrix_for_feat_tensor(emb_tensor)    

    idx_tensor = label_tensor + 1
    label_mask = idx_tensor * (idx_tensor[None, :] == idx_tensor[:, None]).long()
    # label_mask (i,j,k) is True iff idx_tensor[i] == idx_tensor[j] == idx_tensor[k]
    label_mask = label_mask[:, None, :] ==  idx_tensor[:, None]

    gt_pose_dist = None
    assert gt_pose_tensor.size(1) == 3
    n = gt_pose_tensor.size(0)
    pose_sim_all = torch.zeros((n, n)).to(emb_tensor.device)
    for i in range(3):
        pose_i = gt_pose_tensor[:, i]
        pose_sim_i = torch.abs(pose_i[:, None] - pose_i[None, :]) < gt_pose_margin
        pose_sim_all += pose_sim_i.float()
    # poses are similar only if they are similar on all axis.
    final_pose_sim = torch.abs(pose_sim_all - 3.0) < 1e-4

    # Since this is a symmetric matrix triangulate it
    final_pose_sim = torch.triu(final_pose_sim)
    # Make the diagonals 0.
    final_pose_sim = final_pose_sim * (1 - torch.eye(n, n).to(emb_tensor.device))

    gt_pose_dist_val = final_pose_sim[:, :, None] - final_pose_sim[:, None, :]
    gt_pose_dist_mask = torch.abs(gt_pose_dist_val - 1.0) < 1e-4

    total_mask = (label_mask * gt_pose_dist_mask).float()
    all_loss_count = total_mask.sum().item()
    if all_loss_count == 0:
        return 0.0

    emb_dist_diff = emb_dist[:, :, None] - emb_dist[:, None, :]
    # emb_dist_diff = (emb_dist_diff * total_mask.float())

    all_loss = F.relu(emb_dist_diff + emb_margin) * total_mask
    # This will not include data points that satisfy the triplet criterion.
    #  all_loss_count = max(1, torch.sum(all_loss > 1e-6).item())
    total_loss_sum = all_loss.sum()

    loss = total_loss_sum / all_loss_count

    return loss


def get_contrastive_loss(emb_tensor, gt_pose_tensor, label_tensor, 
                         emb_margin, gt_pose_margin, squared=True):
    emb_dist = get_dist_matrix_for_feat_tensor(emb_tensor)    

    idx_tensor = label_tensor + 1
    label_mask = idx_tensor * (idx_tensor[None, :] == idx_tensor[:, None]).long()
    # label_mask (i,j,k) is True iff idx_tensor[i] == idx_tensor[j] == idx_tensor[k]
    label_mask = label_mask[:, None, :] ==  idx_tensor[:, None]

    gt_pose_dist = get_dist_matrix_for_feat_tensor(gt_pose_tensor)
    gt_pose_dist_diff = gt_pose_dist[:, :, None] - gt_pose_dist[:, None, :]
    gt_pose_dist_diff = gt_pose_dist_diff * label_mask.float()
    # gt_pose_dist_diff + gt_pose_margin has to be < 0
    gt_pose_dist_mask = gt_pose_dist_diff < -gt_pose_margin
    total_mask = (label_mask * gt_pose_dist_mask).float()
    all_loss_count = total_mask.sum().item()
    if all_loss_count == 0:
        return 0.0

    emb_dist_diff = emb_dist[:, :, None] - emb_dist[:, None, :]
    # emb_dist_diff = (emb_dist_diff * total_mask.float())

    all_loss = F.relu(emb_dist_diff + emb_margin) * total_mask
    # This will not include data points that satisfy the triplet criterion.
    #  all_loss_count = max(1, torch.sum(all_loss > 1e-6).item())
    total_loss_sum = all_loss.sum()

    loss = total_loss_sum / all_loss_count

    return loss



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, 
                                  size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for positive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        weight_tensor = torch.zeros(loss.size()).to(loss.device)
        weight_tensor[targets == 0] = weight[0]
        weight_tensor[targets == 1] = weight[1]
        loss = loss * weight_tensor

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()
    

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1, weight=None, pos_weight_is_dynamic=False, 
                size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            pos_weight_is_dynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', torch.Tensor(pos_weight))
        self.size_average = size_average
        self.reduce = reduce
        self.pos_weight_is_dynamic = pos_weight_is_dynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.pos_weight_is_dynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)