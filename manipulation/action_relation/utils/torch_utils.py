import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_entropy(std):
  var = std.pow(2)
  entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
  return entropy.sum(1)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
  params = []
  for param in model.parameters():
    params.append(param.data.view(-1))
  flat_params = torch.cat(params)
  return flat_params


def set_flat_params_to(model, flat_params):
  prev_ind = 0
  for param in model.parameters():
    flat_size = int(np.prod(list(param.size())))
    param.data.copy_(
        flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
    prev_ind += flat_size

def get_flat_grad_from(net, grad_grad=False):
  grads = []
  for param in net.parameters():
    if grad_grad:
      grads.append(param.grad.grad.view(-1))
    else:
      grads.append(param.grad.view(-1))

  flat_grad = torch.cat(grads)
  return flat_grad

def clip_grads(model, clip_val=5.0):
  #average_abs_norm = 0
  #max_abs_norm = 0
  #count = 0.0
  for p in model.parameters():
    if p.grad is not None:
      #count += 1.0
      #average_abs_norm += p.grad.data.abs().mean()
      #if p.grad.data.abs().max() > max_abs_norm:
      #    max_abs_norm = p.grad.data.abs().max()
      p.grad.data = p.grad.data.clamp(-clip_val, clip_val)

  #average_abs_norm /= count

  #return average_abs_norm, max_abs_norm

def get_norm_scalar_for_layer(layer, norm_p):
  return float(layer.norm(p=norm_p).cpu().detach().numpy())

def get_norm_inf_scalar_for_layer(layer):
  data = layer.data.cpu().numpy()
  return np.max(np.abs(data))

def get_weight_norm_for_network(model):
  wt_l2_norm, grad_l2_norm = -10000.0, -10000.0
  for param in model.parameters():
    wt_l2_norm = np.maximum(wt_l2_norm,
                            get_norm_scalar_for_layer(param, 2))
    if param.grad is not None:
      grad_l2_norm = np.maximum(
          grad_l2_norm, get_norm_scalar_for_layer(param.grad, 2))

  return wt_l2_norm, grad_l2_norm

def add_scalars_to_summary_writer(summary_writer, 
                                  tags_prefix,
                                  tags_dict,
                                  step_count):
    '''Add dictionary of scalars to summary writer.'''
    '''
    for tag_key, tag_value in tags_dict.items():
        tag = tags_prefix + '/' + tag_key
        summary_writer.add_scalar(
                tag,
                tag_value,
                step_count)
    '''
    summary_writer.add_scalars(tags_prefix, tags_dict, step_count)

# Copied from Pytorch (0.5 unstable)
def clip_grad_value(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
    parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
        single Tensor that will have gradients normalized
    clip_value (float or int): maximum allowed value of the gradients
        The gradients are clipped in the range [-clip_value, clip_value]
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


def to_numpy(a_tensor, copy=True, detach=True):
    '''Convert torch tensor to numpy.'''
    if detach:
        if copy:
            return a_tensor.clone().detach().cpu().numpy()
        else:
            return a_tensor.detach().cpu().numpy()
    else:
        if copy:
            return a_tensor.clone().cpu().numpy()
        else:
            return a_tensor.cpu().numpy()

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, 
                 data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format 
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1,
                               keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1,
                               keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class Spatial3DSoftmax(torch.nn.Module):
    def __init__(self, height, width, depth, channel, temperature=None, 
                 data_format='NCHW'):
        super(Spatial3DSoftmax, self).__init__()
        self.data_format = data_format 
        self.height = height
        self.width = width
        self.depth = depth
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y, pos_z = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width),
                np.linspace(-1., 1., self.depth),
                )
        size = self.height * self.width * self.depth
        pos_x = torch.from_numpy(pos_x.reshape(size)).float()
        pos_y = torch.from_numpy(pos_y.reshape(size)).float()
        pos_z = torch.from_numpy(pos_z.reshape(size)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        batch_size = feature.size(0)
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width*self.depth)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1,
                               keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1,
                               keepdim=True)
        expected_z = torch.sum(self.pos_z*softmax_attention, dim=1,
                               keepdim=True)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xyz.view(batch_size, -1)

        return feature_keypoints
