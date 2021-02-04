import numpy as np
import re
import os
import cv2
import glob
import matplotlib.pyplot as plt
import torch
import itertools
import quaternion
from collections import Mapping, Container, Counter
from sys import getsizeof
from matplotlib.lines import Line2D
from multiprocessing import Pool

def make_mask(size, area, background, mask_value, dtype=None):
    """
    Creates a mask of specified size, with the background 

    Inputs:
        size - (tuple): shape of the mask
        area - (np.array of size Nx4): locations to fill mask in,
               format rows as [xmin, ymin, xmax, ymax],
               N is the number of rectanglular areas to fill
        background: value to set the base mask to
        mask_value: value to set the masking area to
    """
    if area.ndim == 1:
        area = np.expand_dims(area, axis = 0)
    mask = np.ones(size, dtype=dtype)
    mask = mask*background
    for i in range(area.shape[0]):
        mask[area[i,1]:area[i,3], area[i,0]:area[i,2]] = mask_value

    return mask

def get_mask(array, axis=2, mask_value=0):
    """
    Returns a boolean mask of the given array. False when all values along
    axis are equal to the mask_value, True elsewhere
    """
    mask = (array != mask_value).all(axis=axis)

    return mask

def many_processes(function, num_processes, args=None):
    """
    Function uses multiprocessing library to run multiple processes
    #TODO not working with my code right now, using too much memory per process
    Inputs:
        function - target function to use multiple processes for
        num_processes - (int): number of parallel processes to run
        args - arguments to pass to the function
    Outputs:
        outputs - the outputs of the processes, 
                  the number of outputs is equal to num_processes
    """
    p = Pool(processes=num_processes)
    output = p.map(function, args)
    p.close()

    return output[0]

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value , dict1[key]]
 
    return dict3

def mins_max(center, height, width, dtype=int):
    """
    Returns the x and y minimum values of a rectangular area
    Inputs: 
        center - (np.array, 2x1): give (y,x) or (h,w) coordinates of center
        height - (int): height of the rectangle
        width - (int): width of the rectangle
    Outputs:
        out: np.array of type dtype and form [xmin, ymin, xmax, ymax]
    """
    center = center.astype(dtype)
    y1 = center[0] - height//2
    y2 = y1 + height
    x1 = center[1] - width//2
    x2 = x1 + width
    out = np.array([x1, y1, x2, y2], dtype=dtype)

    return out

def numerical_sort(value):
    """
    Key function for sorting files numerically
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def num2yx(input_num, num_rows, num_cols, order='C'):
    """
    Given input index of flattened array, return the (y, x) coordinates of a matrix
    Inputs:
        input_num - (int): the input number
        num_rows - (int): total number of rows in matrix
        num_cols - (int): total number of columns in matrix
        order - (string): 'C' for row-major order
    Outputs:
        y - (int): the row coordinate of input
        x - (int): the column coordinate of input
    """
    assert input_num <= num_rows*num_cols
    # not as computationally efficient
    # temp = np.zeros((num_rows, num_cols))
    # temp[input_num] = 1
    # y, x = np.nonzero(temp)
    x = input_num % num_cols
    y = input_num // num_cols

    return int(y), int(x)        

def yx2num(y_x, num_rows, num_cols, order='C'):
    """
    Given input, return the output index of a flattened array of size num_rows x num_cols
    
    Inputs:
        y_x - (tuple of ints): the input (y,x) coordinates
        num_rows - (int): total number of rows in matrix
        num_cols - (int): total number of columns in matrix
        order - (string): 'C' for row-major order
    Outputs:
        output - (int): the index of the flattened array of given size
    """
    assert y_x[0] <= num_rows and y_x[1] <= num_cols
    temp = np.zeros((num_rows, num_cols))
    temp[y_x] = 1
    temp = temp.flatten(order=order)
    output = np.nonzero(temp)[0]

    return int(output)

def ordered_combos(sequence, length):
    """
    Returns all the ordered combinations of a sequence of numbers

    Inputs:
        sequence: ordered sequence
        length: length of the possible combinations 
    Outputs:
        combos - (np.array): all possible ordered combinations,
                 size N x length, where N is the number of combinations
    """
    sequence.sort()
    diff = len(sequence) - length
    assert diff >= 0

    if diff == 0:
        combos = np.array(sequence, dtype=int).reshape(1, length)
        return combos

    combos = np.zeros((diff+1, length))
    for i in range(diff + 1):
        combos[i] = sequence[i:i+length]

    return np.array(combos, dtype=int)

def plot_box(images, labels, shapes, seq_length, subplot_shape=(1,1), title=None):
    """
    Generate image of the predictions the network made
    Takes RGB images as HxWxD
    Inputs:
        images - (torch.tensor): either a single image or sequence of images
                 should be formatted as (seq_length, H, W, D)
        labels - (torch.tensor): correspomding labels for image
                 formatted as (seq_length, 2) where a single label is (y,x)
        shapes - (torch.tensor): corresponding object shape for the label
                 same format as labels but the single shape is (height, width)
        seq_length - (int): length of sequence, 1 if one image
        subplot_shape - (tuple): format to plot subplot as, same convention as matplotlib
        title - (string): the title of the figure
    Outputs:
        fig - matplotlib figure, need to use plt.show() to see figure
    """
    fig = plt.figure()
    counter = 0
    while counter < seq_length:
        if seq_length < 2:
            label = labels
            image = images
            obj_shape = shapes
        else:
            label = labels[counter]
            image = images[counter]
            obj_shape = shapes[counter]
        plt.subplot(subplot_shape[0], subplot_shape[1], counter+1)
        height, width = obj_shape[0], obj_shape[1]
        x1, y1, _, _ = mins_max((label[0], label[1]), height, width)
        image_utils.plt_imshow_tensor(image.cpu())
        box = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', fill=False)
        plt.gca().add_patch(box)

        if title is not None:
            plt.title(f"{title}")

        counter += 1

    return fig

def rand_location(area, num_samples):
    """
    Gives random y,x placement locations
    Inputs:
        area - x,y range to sample from, formatted as [x1,y1,x2,y2]
        num_samples - (int): number of samples to generate
    Outputs:
        y - (np.array): array of size num_samples giving the y locations
        x - (np.array): array of size num_samples giving the x locations
    """
    x_min = area[0] 
    y_min = area[1] 
    x_max = area[2] 
    y_max = area[3]
    x = np.arange(x_min, x_max+1) 
    y = np.arange(y_min, y_max+1)
    rand_x = np.random.randint(0, x.size, size=num_samples)
    rand_y = np.random.randint(0, y.size, size=num_samples)

    return y[rand_y], x[rand_x]

def sub_dirs(directory):
    """
    return all of the immediate subdirectories in directory as a list of strings, sorted numerically
    """
    dir_list = []
    for files in sorted(os.listdir(directory), key=numerical_sort):
        if os.path.isdir(os.path.join(directory, files)):
            dir_list.append(os.path.join(directory, files))
    return dir_list

def sub_files(directory, file_suffix, prefix=None):
    """
    return all the immediate files in directory that end in file_suffix as numerical sorted list
    """
    file_list = []
    if prefix is not None:
        for filename in sorted(glob.glob('{}/{}*{}'.format(directory, prefix, file_suffix)), key=numerical_sort):
            file_list.append(filename)
    else:
        for filename in sorted(glob.glob('{}/*{}'.format(directory, file_suffix)), key=numerical_sort):
            file_list.append(filename)
    return file_list

def save_as_img(array, save_path, file_type='png'):
    """
    Inputs:
        array - (H, W, D): assuming RGB image as a numpy array
        save_path - (string): full file path to save image to (including name)
        file_type - (string): file type to save image as, default is png
    """
    array = array/255
    plt.imsave(f'{save_path}.{file_type}', array)

def plot_prediction(image, prediction, width, height):
    """
    Plots the location of the prediction

    Inputs:
        prediction (np.array): 1-D array with 2 elements, (x,y),
            which is the center coordinates of the prediction
        width (int): width of the object to be placed, in pixels
        height (int): height of the object to be placed, in pixels
    """
    corner = (prediction[1]-height/2, prediction[0]-width/2)
    box = plt.Rectangle(corner, width, height, linewidth=1,
        edgecolor='r', fill=False)
    plt.figure(0)
    img = image.copy()
    img_rgb = img[:,:,::-1] #convert BGR to RGB
    plt.imshow(img_rgb)
    plt.gca().add_patch(box)

    plt.show()

    return

def remove_first_last(array1, array2, amount=1):
    """
    remove first element of array1 and last of array2 along axis
    for the sequences, want the labels to be for the next image not the current
    INputs:
        array1 - (np.array): array to remove first element
        array2 - (np.array): array to remove last element
        amount - (int): amount to remove from the beginning and end
    """
    if array1.ndim  == 2:
        out1 = array1[amount:, :]
    elif array1.ndim == 3:
        out1 = array1[:, amount:, :]
    elif array1.ndim == 4:
        out1 = array1[amount:, :, :, :]
    else:
        out1 = array1[:, amount:, :, :, :]
    
    if array2.ndim == 2:
        out2 = array2[:-amount]
    elif array2.ndim == 3:
        out2 = array2[:, :-amount, :]
    elif array2.ndim == 4:
        out2 = array2[:-amount, :, :, :]
    else:
        out2 = array2[:, :-amount, :, :, :]

    return out1, out2

def get_accuracy(predicted, ground_truth, accept_thresh=None):
    #TODO move this to accuracy.py file
    """
    calculate the percentage of times the predictions were correct
    
    predicted - the predicted output of the neural network
    ground_truth - the value to compare predicted values to 
    accept_thresh - acceptance threshold, any values less than the
                    difference between the threshold and ground
                    truth are considered successes
    """
    #NOTE might want to use IOU instead of this

    with torch.no_grad():
        if accept_thresh is None:
            num_correct = torch.eq(predicted.int(), ground_truth)

        else:
            ge = torch.ge(predicted, (ground_truth - accept_thresh))
            le = torch.le(predicted, (ground_truth + accept_thresh))
            num_correct = torch.mul(ge, le)

        accuracy = torch.sum(num_correct) / (np.prod(num_correct.shape))

    return accuracy

def plot_grad_flow(named_parameters):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    NOTE: This slows process speed, so only use for debugging
    Ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    fig = plt.figure()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            temp_xlabel = n[7:][:-7] # only keep the layer name
            layers.append(temp_xlabel)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    return fig

def conv2d_out_dim(in_dim, kernel_size, padding=1, stride=1, dilation=1):
    """Function to calculate the output size of a dimension after being passed through torch.nn.Conv2d"""
    out_dim = ((in_dim + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
    return out_dim

def maxpool2d_out_dim(in_dim, kernel_size, padding=1, stride=1, dilation=1):
    """Function to calculate the output size of a dimension after being passed through torch.nn.MaxPool2d"""
    out_dim = ((in_dim + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
    return out_dim

    #TODO make a util function to calculate the output size of a layer given a input dim
    #ie get the input size of a linear layer by giving input h or w
 
def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
 
    :param o: the object
    :param ids:
    :return:
    Ref:
     - https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str) or isinstance(0, unicode):
        return r
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r

def rpy_to_quat(rpy):
    return quaternion.from_euler_angles(rpy)

def quat_to_rpy(q):
    """
    Convert a quaternion numpy array to RPY in radians. Assuming q is formatted as "x,y,z,w"
    """
    q = quaternion.quaternion(q[3], q[0], q[1], q[2])
    return quaternion.as_euler_angles(q)

def s_curve(x, A=1, w=1, phi=0):
    """
    Generate the y values of a sine curve with given array of x values
    
    Inputs:
        x (np.array): array of x values to get the y values for
        A (float): amplitude of the sine wave 
        w (float): angular frequency of the sine wave
        phi (float): phase shift of the curve
    """
    return A * np.sin(w*x + phi)

def elipse_curve(theta, x_radius=1, y_radius=1, center=[0,0]):
    """
    Generate x and y coordinates for given theta and radii in polar coordinates. 
    
    Inputs:
        theta (np.array): array of theta values to get cartesian coordinates for
        x (float): radius of the elipse along its x-axis
        y (float): radius of the elipse along its y-axis
        center (list): the (x,y) coordinates of the center of the elipse 
    """
    x = x_radius * np.cos(theta) + center[0]
    y = y_radius * np.sin(theta) + center[1]
    return x, y

def multimode(array, axis=1, num_modes=2, decimals=3, trans=False):
    """
    Get the first N modes in array along the given axis. If more than one mode
    same count, the first nodes will be returned until num_modes are returned.
    NOTE: This has only been tested on 2-D arrays
    Inputs:
        array (np.array): array to find modes for.
        axis (int): axis to get modes along.
        num_modes (int): number of modes to return per axis index.
        trans (bool): whether to transpose the outputs
    Outputs:
        modes (np.array): ((dim_of_given_axis) x num_modes) array of the mode values
        count (np.array): ((dim_of_given_axis) x num_modes) array of the count of 
            each value in the modes array
    Ref:
        - https://stackoverflow.com/questions/14793516/how-to-produce-multiple-modes-in-python
    """
    modes = []
    count = []
    for i in range(array.shape[axis]):
        # get slice of array 
        sliced = list(slicer(np.round(array,decimals=decimals),i,axis)) # for counting later
        a = sliced.copy()
        temp_modes = []
        while True:
            # group most_common output by frequency
            freqs = itertools.groupby(Counter(a).most_common(), lambda x:x[1])
            # pick off the first group (highest frequency)
            temp_modes.extend([val for val,count in next(freqs)[1]])
            if len(temp_modes) < num_modes:
                # remove the values that are already in the list of modes
                a = [value for value in a if value not in temp_modes] #TODO there is probably a faster way to do this
            else:
                break
        modes.append(temp_modes[:num_modes])
        temp_count = [sliced.count(value) for value in temp_modes[:num_modes]]
        count.append(temp_count)
    if trans:
        return np.array(modes).T, np.array(count).T
    else:
        return np.array(modes), np.array(count)

def slicer(array, idx, axis):
    """
    Get all the values at a index on a specific axis.
    e.g. slicer(np.array([[0,1][2,3]]), 1, 0) returns np.array([2,3])
    Inputs:
        array (np.array): Array to slice
        idx (int): index of the row, column, etc. to get
        axis (int): axis to get slice along
    Ref:
        - https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
    """
    sl = [slice(None)] * array.ndim
    sl[axis] = idx
    return array[tuple(sl)]

