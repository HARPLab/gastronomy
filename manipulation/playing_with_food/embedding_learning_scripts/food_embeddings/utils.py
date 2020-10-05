import re
import os
import glob
import shutil
import pickle
import cv2
import numpy as np
import torch

def save_checkpoint(state, is_best, filepath=None, filename=None):
    """
    Save a state dict of the model to resume training later
    """
    if filepath is None:
        filepath = '.'
    if filename is None:
        filename = 'checkpoint.pth.tar'
    torch.save(state, f'{filepath}/{filename}')
    if is_best:
        shutil.copyfile(f'{filepath}/{filename}', f'{filepath}/model_best.pth.tar')

def numerical_sort(value):
    """
    Key function for sorting files numerically
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def letterbox_image(img, in_dim):
    '''
    resize image to square with unchanged aspect ratio using padding
    
    Inputs:
        img: image to resize
        in_dim: dimension to resize to
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = in_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((in_dim[1], in_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    #convert BGR to RGBef letterbox_image(img, in_dim):
    return canvas[:,:,::-1]

def sub_dirs(directory):
    """
    Return all of the immediate subdirectories in directory as a list of strings,
    sorted numerically and including hidden directories. Gives relative paths
    """
    dir_list = []
    for files in sorted(os.listdir(directory), key=numerical_sort):
        if os.path.isdir(os.path.join(directory, files)):
            dir_list.append(os.path.join(directory, files))
    return dir_list

def sub_files(directory, file_suffix):
    """
    Return all the immediate files/directories in directory that end
    in file_suffix as a numerical sorted list. Give relative paths
    """
    file_list = []
    for filename in sorted(glob.glob('{}/*{}'.format(directory, file_suffix)), key=numerical_sort):
        file_list.append(filename)
    return file_list

def pad_and_stack(list_of_arrays, pad_mode='constant'):
    """
    Takes a list of arrays, pads each dim of the array until all of the arrays are
    the same size and stacks them together on a new axis (axis=0).
    This doesn't work if the arrays in the list have different number of dims
    NOTE: this just addds the pading to the end of each dimension
    Inputs:
        list_of_arrays(list): a list of numpy arrays
        pad_mode(str): 'constant' for zero padding, 'edge' for padding with the edge values
            see https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

    """
    output = []
    max_dims = [array.shape for array in list_of_arrays]
    max_dims = np.stack(max_dims, axis=0)
    max_dims = tuple(np.max(max_dims, axis=0))
    for array in list_of_arrays:
        if array.shape == max_dims:
            output.append(array)
        else:
            pad_width = np.abs(np.array(max_dims) - np.array(array.shape))
            pad_width = [(0, pad) for pad in pad_width] # add padding to end of array
            temp_out = np.pad(array, pad_width=pad_width, mode=pad_mode)
            # helpful ref on np.pad: https://stackoverflow.com/a/46115998
            output.append(temp_out)
    output = np.stack(output, axis=0)

    return output

def is_empty(any_structure, verbose=False):
    if any_structure:
        if verbose:
            print('Structure is not empty.')
        return False
    else:
        if verbose:
            print('Structure is empty.')
        return True

def get_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

