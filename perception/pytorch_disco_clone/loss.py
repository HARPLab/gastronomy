import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hyp
import ipdb
st = ipdb.set_trace


class CustomLoss(nn.Module):
    def __init__(self, device, config, num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        # self.alpha = config['alpha']
        # self.beta = config['beta']
        self.alpha = hyp.pixor_alpha
        self.beta = hyp.pixor_beta

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * x_t.log()

        return loss.mean()

    def cross_entropy(self, x, y):
        weight = y.clone()
        weight = weight*99
        weight += 1
        # st()
        return F.binary_cross_entropy(input=x, target=y, reduction='elementwise_mean',weight=weight)


    def forward(self, preds, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size = targets.size(0)
        if targets.shape[1] == 9: # 3D case
            cls_targets, loc_targets = targets.split([1, 8], dim=1)
        elif targets.shape[1] == 7: # 2D case
            cls_targets, loc_targets = targets.split([1, 6], dim=1)
        else:
            print("Some error in dimensions. Exitting")
            exit(0)

        if preds.size(1) == 9:
            cls_preds, loc_preds = preds.split([1, 8], dim=1)
        elif preds.size(1) == 29:
            cls_preds, loc_preds, _ = preds.split([1, 8, 20], dim=1)
        elif preds.size(1) == 7:
            cls_preds, loc_preds = preds.split([1, 6], dim=1)
        ################################################################
        if hyp.use_pixor_focal_loss:
            cls_loss = self.focal_loss(cls_preds, cls_targets)
        else:
            cls_loss = self.cross_entropy(cls_preds, cls_targets) * self.alpha
        cls = cls_loss.item()
        ################################################################
        # reg_loss = SmoothL1Loss(loc_preds, loc_targets)
        ################################################################
        pos_pixels = cls_targets.sum()
        # st()
        if pos_pixels > 0:
            # print("Number of pos value: ", pos_pixels)
            loc_loss = F.smooth_l1_loss(cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
            loc = loc_loss.item()
            loss = loc_loss + cls_loss
        else:
            loc = 0.0
            loss = cls_loss
        
        # print("pos_pixels, cls and loc loss: ", pos_pixels, cls_loss, loc_loss)
        return loss, cls, loc


def test():
    loss = CustomLoss()
    pred = torch.sigmoid(torch.randn(1, 2, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    loss = loss(pred, label)
    print(loss)


if __name__ == "__main__":
    test()
