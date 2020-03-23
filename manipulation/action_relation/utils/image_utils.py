import torch
import torchvision
import numpy as np

import cv2
import json
import os
import sys

from PIL import Image
from scipy.ndimage.interpolation import zoom
import scipy.ndimage

import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def load_image_at_path(img_path, transforms=None, desired_shape=None):
    img = Image.open(img_path).convert("RGB")

    if desired_shape is not None:
        # img = resize(img, desired_shape)
        img = img.resize(desired_shape)
     
    if transforms is not None:
        img = transforms(img, None)
    return img


# TODO(Mohit): This code has been used from the pytroch maskrcnn repository, we
# have to use this since this is being used implicitly in training our own
# object detector.
def pad_tensors_to_match_model(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    # TODO Ideally, just remove this and let me model handle arbitrary
    # input sizes
    if size_divisible > 0:
        import math

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = tensors[0].new(*batch_shape).zero_()
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    image_sizes = [im.shape[-2:] for im in tensors]
    return batched_imgs, image_sizes


def is_valid_image_path(path):
    return path.endswith('jpg') or path.endswith('jpeg') or \
            path.endswith('png') or path.endswith('bmp')


def zoom_into_image_with_bounding_box(img, bounding_box, padding=0):
    _, img_height, img_width = img.shape
    x_start, x_end = int(bounding_box[0]), int(bounding_box[2])
    y_start, y_end = int(bounding_box[1]), int(bounding_box[3])
    x_start = max(x_start - padding, 0)
    x_end = min(x_end + padding, img_width)
    y_start = max(y_start - padding, 0)
    y_end = min(y_end + padding, img_height)
    # return img[:, x_start:x_end, y_start:y_end]
    return img[:, y_start:y_end, x_start:x_end]


def zoom_into_image_with_bounding_box_maintain_aspect_ratio(
        img, bounding_box, padding=0):
    _, img_height, img_width = img.shape
    x_start, x_end = int(bounding_box[0]), int(bounding_box[2])
    y_start, y_end = int(bounding_box[1]), int(bounding_box[3])

    x_start = max(x_start - padding, 0)
    x_end = min(x_end + padding, img_width)
    y_start = max(y_start - padding, 0)
    y_end = min(y_end + padding, img_height)
    new_img_height, new_img_width = y_end - y_start, x_end - x_start

    if new_img_width < new_img_height:
        diff = new_img_height - new_img_width
        x_left = max(0, x_start - diff // 2) 
        x_right = min(img_width, x_end + (diff - (x_start - x_left)))
        new_img = img[:, y_start:y_end, x_left:x_right]
    elif new_img_height < new_img_width:
        diff = new_img_width - new_img_height
        y_up = max(0, y_start - diff // 2) 
        y_down = min(img_height , y_end + (diff - (y_start - y_up)))
        new_img = img[:, y_up:y_down, x_start:x_end]
    else:
        new_img = img[:, y_start:y_end, x_start:x_end]
    return new_img


def visualize_image_tensor(img_tensor, is_batch=False, is_bgr=False):
    if is_batch:
        img_tensor = img_tensor[0]
    rgb_channel = 0 if img_tensor.shape[0] == 3 else 2
    # convert to numpy
    img_tensor_copy = img_tensor.clone()
    img_arr = img_tensor_copy.cpu().numpy()

    if rgb_channel == 0:
        img_arr = img_arr.transpose(1, 2, 0)
    if is_bgr:
        img_arr = img_arr[:, :, ::-1]
    print("img range min: {:.2f}, max: {:.2f}".format(
        np.min(img_arr), np.max(img_arr)))
    plt.imshow(img_arr)
    plt.show()


def resize_image_using_zoom(img, max_size=256):
    [Y, X] = img.shape[:2]

    # resize
    max_dim = max([Y, X])
    zoom_factor = 1. * max_size / max_dim
    img = zoom(img, [zoom_factor, zoom_factor, 1])
    return img


def resize_image(img, size=(256,256)):
    return_img = cv2.resize(img, size)
    return return_img


def get_image_tensor_mask_for_bb(h, w, c, bb_xyxy, fmt='chw'):
    x0, y0 = bb_xyxy[0], bb_xyxy[1]
    x1, y1 = bb_xyxy[2], bb_xyxy[3]
    t = None
    if fmt == 'chw':
        if c > 0:
            t = torch.zeros((c, h, w))
            t[:, y0:y1, x0:x1] = 1
        else:
            t = torch.zeros((h, w))
            t[y0:y1, x0:x1] = 1
    elif fmt == 'hwc':
        if c > 0:
            t = torch.zeros((h, w, c))
            t[y0:y1, x0:x1, :] = 1
        else:
            t = torch.zeros((h, w))
            t[y0:y1, x0:x1] = 1
    else:
        raise ValueError("Invalid fmt: {}".format(fmt))
    return t


def zoom_array(inArray, finalShape, sameSum=False,
               zoomFunction=scipy.ndimage.zoom, **zoomKwargs):
    """

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.

    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?

    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.

    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    Parameters
    ----------

    inArray: n-dimensional numpy array (1D also works)
    finalShape: resulting shape of an array
    sameSum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    zoomFunction: by default, scipy.ndimage.zoom. You can plug your own.
    zoomKwargs:  a dict of options to pass to zoomFunction.
    """
    # import ipdb; ipdb.set_trace()
    inArray = np.asarray(inArray, dtype=np.double)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    # rescaled = zoomFunction(inArray, zoomMultipliers, **zoomKwargs)
    rescaled = zoomFunction(inArray, zoomMultipliers, 
                            mode='constant', cval=0, order=0)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)

    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled, zoomMultipliers
