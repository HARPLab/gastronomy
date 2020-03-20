import numpy as np
import re
import os
import cv2
import glob
import matplotlib.pyplot as plt
import torch
import itertools

from multiprocessing import Pool
from PIL import Image, ImageDraw

def crop_image(img, mode='Rectangle', fill_color=(0,0,0)):
    """
    Crops the blank space out of an image (ie. where the pixels are zero)
    
    Inputs:
        img - (RGB image, np.array): input image to crop
        mode - (string): shape to crop to, default is a rectangle.
                other option include: circle
        fill_color - (tuple): RGB color to fill blank space with

    Outputs:
        cropped_image - (RGB image, np.array): cropped image
        npAlpha - (numpy array, HxW): the alpha array for the cropped image

    References:
        https://stackoverflow.com/questions/51486297/cropping-an-image-in-a-circular-way-using-python
        https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
        https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    """
    # convert to B&W image and crop image into rectangle
    bw_img = img.max(axis=2)
    non_empty_columns = np.where(bw_img.max(axis=0)>0)[0]
    non_empty_rows = np.where(bw_img.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    crop = img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    
    if mode == 'Circle':
        # Convert to 'Image' obj and make alpha layer w/ circle
        pil_img = Image.fromarray(np.uint8(crop))
        h, w = pil_img.size
        alpha = Image.new('L', pil_img.size, 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice([0,0,h,w],0,360,fill=255)
        
        # add alpha layer to RGB and convert back to 'Image' obj
        npAlpha = np.array(alpha)
        npImage = np.dstack((crop,npAlpha))
        npImage = Image.fromarray(np.uint8(npImage))
        
        # replace empty areas w/ fill_color and return image
        cropped_image = Image.new('RGB', npImage.size, fill_color)
        cropped_image.paste(npImage, mask=npImage.split()[3])
    
    return np.array(cropped_image), npAlpha

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

def plt_imshow_tensor(img, one_channel=False):
    """
    convert input to cpu, HWD, RGB, and unnormalize first
    """
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    npimg = npimg / 255.0 # change to values between 1 and 0
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

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

def paste_image(im1, im2, location, alpha1=0, alpha2=1, alpha_mask=None):
    """
    Paste image 2 onto image 1 at location

    Inputs:
        im1 - (H, W, D): background image, assuming RGB image as a numpy array
        im2 - (H, W, D): foreground image, assuming RGB image as a numpy array
        location - (H, W): location to place center of im2 onto im1
        alpha1 - (float): value between 0 and 1 to weigh image1 pixels
        alpha2 - (float): value between 0 and 1 to weigh image2 pixels
                          (alpha values from weighted sum)
        alpha_mask - (np.array, HxW): alpha layer for im2

    Outputs:
        image - (RGB image, np.array): output RGB image
    """
    # get dimensions
    h, w = im2.shape[:2]
    loc = mins_max(location, h, w, dtype=int)
    # crop im2 to same size as im1 and add alpha channel
    im2_crop = np.zeros(im1.shape)
    alpha_crop = np.zeros(im1.shape[:2])
    im2_crop[loc[1]:loc[3], loc[0]:loc[2], :] = im2
    alpha_crop[loc[1]:loc[3], loc[0]:loc[2]] = alpha_mask
    im2_crop = np.dstack((im2_crop, alpha_crop))
    # Convert to 'Image' objects
    im2_crop = Image.fromarray(np.uint8(im2_crop))
    image = Image.fromarray(np.uint8(im1))
    # paste image
    image.paste(im2_crop, mask=im2_crop.split()[3])

    return np.array(image)

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
        plt_imshow_tensor(image.cpu())
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

def sub_files(directory, file_suffix):
    """
    return all the immediate files in directory that end in file_suffix as list
    """
    file_list = []
    for filename in sorted(glob.glob('{}/{}'.format(directory, file_suffix)), key=numerical_sort):
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
    elif array2.ndim == 4:
        out2 = array2[amount:, :, :, :]
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