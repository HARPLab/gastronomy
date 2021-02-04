import numpy as np
import cv2
import glob
import torch
import quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy import ndimage, stats
from PIL import Image, ImageDraw

from autolab_core import Point
import perception
import rnns.utils as utils

#TODO move some of the function in utils.py over here, you'll need to rename some
# of them or combine them with these functions

#################### Functions for image processing ####################

def get_images_in_dir(directory, image_types=['.png','.jpg','jpeg'], file_prefix='*',
                      gray=False, resize=True, crop=None, resolution=224):
    """
    Gather all of the images in a directory and return as a list
    Args:
        directory (str): directory containing images
        image_types (list): list of strings of all the image type extensions to look for
        file_prefix (str): look for files that have this in their names
        gray (bool): set to true to read images as grayscale, RGB otherwise
        resize (bool): whether to reshape images with constant aspect ratio and padding
        crop (list): list of ints containing the pixel locations to crop to as 
            [xmin, ymin, xmax, ymax], leave as None to not crop
        resolution (int): the resolution to reshape image to, uses resizes with constant
            aspect ratio and zero padding to a (resolution x resolution x channel) image
    """
    if gray:
        img_type = 0
    else:
        img_type = 1 # rgb
    images = []
    for image_type in image_types:
        for filename in sorted(glob.glob(f'{directory}/{file_prefix}{image_type}'), key=utils.numerical_sort):
            img = cv2.imread(filename, img_type)
            if crop is not None:
                img = img[crop[1]:crop[3], crop[0]:crop[2]]

            if resize:
                #reshape images
                img = letterbox_image(img, resolution)
            else:
                # convert BGR to RGB
                img = img[:,:,::-1]
            images.append(img)

    return images

#################### Functions for image transforms ####################

def deproject_pixel(pixel, depth, intrinsics, extrinsics, frame):
    """
    Deproject a pixel in image coordinates to 3D space coordinates
    INPUT IS (W,H) NOT (H,W)
    Inputs:
        pixel (np.array): the camera image coordinates as (w,h) aka (x,y)
        depth (float): the depth value of the pixel in meters
        intrinsics (perception.CameraIntrinsics): intrinsics of the camera
        extrinsics (autolab_core.RigidTransform): extrinsics of the camera
            i.e. the transform from camera frame to world frame
        frame (str): name of the frame the pixel is in
    Outputs:
        point (np.array): array of the deprojected point in world frame
    """
    # Deproject pixel and transform to world frame
    pixel = Point(pixel.reshape(2), frame)
    point = intrinsics.deproject_pixel(depth, pixel)
    point = extrinsics.apply(point)

    return point.vector

def project_point(point, intrinsics, extrinsics, frame='world'):
    """
    Project a point in 3D space onto the image plane
    OUTPUT IS (W,H) NOT (H,W)
    Inputs:
        point (np.array): the world coordinates of the point to tranform (x,y,z)
        intrinsics (perception.CameraIntrinsics): intrinsics of the camera
        extrinsics (autolab_core.RigidTransform): extrinsics of the camera.
            i.e. the transform from camera frame to world frame
        frame (str): string of the name of the coordinate frame point is in
    Outputs:
        pixel (np.array): (x,y) aka (w,h) array of the projected pixel in image frame
    """
    point = Point(point, 'world')
    point = extrinsics.inverse().apply(point)
    pixel = intrinsics.project(point)

    return pixel.vector

def rotate_image(image, rotation, pivot=None, crop_size=[64,64],
                 r_tf=[np.pi/2, np.pi/2, np.pi/2], clip=None, viz=False):
    """
    Rotate an image about a specific pixel and crop the image around that point
    Inputs:
        image (np.array): an image of shape [height, width, channels], can also be [h,w]
        rotation (np.array): rotation as a quaternion (w,x,y,z) (get from states array)
        pivot(np.array or list): list of integer values containing the pixel coordinates
            of the point of rotation as (h,w) aka (y,x). This value will be the center
            of the final output crop as well. Default is center of the image.
        crop_size (np.array or list): list of integer values of the size of the output crop
        r_tf (np.array or list): additional rotation to apply to rotation as rpy
            e.g (np.pi/2, np.pi/2, np.pi/2) can be applied for ee rotation
            to obj rotation
        clip (float or string): set the maximum pixel value as this value. If 'mode' is
            given than the mode of the final cropped image will be used.
            NOTE: This might cause issues 
        viz (bool): set to True to visualize the cropped and rotated output
    Ref:
        - https://stackoverflow.com/a/25459080
        - possibly helpful: https://stackoverflow.com/a/3570834
    """
    # get y axis rotation and convert to degrees
    rotation = quaternion.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    if r_tf is not None:
        r_tf = quaternion.from_euler_angles(r_tf[0], r_tf[1], r_tf[2])
        rotation *= r_tf 
    rotation = quaternion.as_euler_angles(rotation)
    rot = rotation[1] * (180/np.pi) 

    # pad the image so the pivot point is at center of image
    padX = [image.shape[1] - pivot[1], pivot[1]]
    padY = [image.shape[0] - pivot[0], pivot[0]]
    # for black & white or depth images
    if image.ndim == 2:
        imgP = np.pad(image, [padY, padX], 'constant')
    # for rgb images
    elif image.ndim == 3 and image.shape[-1] == 3:
        imgP = np.pad(image, [padY, padX, [0, 0]], 'constant')
    else:
        raise ValueError('Invalid image dimension the number of channels should be 1 or 3')
    
    # rotate image and crop so the pivot point is at the same pixel location
    imgR = ndimage.rotate(imgP, rot, reshape=False)
    crop = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    
    output = crop_image(crop, crop_size=crop_size, crop_center=pivot)

    if clip is not None:
        if clip is 'mode':
            clip = stats.mode(np.round(output, decimals=5), axis=None)[0]
        output = np.clip(output, a_max=clip)

    if viz:
        plt.imshow(output)
        plt.show()

    return output

#################### Functions for editing images ####################

def crop_image(image, crop_size, crop_center=None, corners=False):
    """
    Crop a patch out of an image, crops at the center by default

    image (np.array): the image to crop of shape [height, width, channels], can also be [h,w]
    crop_size (list): list of integer values of the height and width values of the
        output crop (h,w). If corners=True then crop_size=[xmin, ymin, xmax, ymax].
    crop_center(list): list of integer values of the center point of the crop (h,w)
    corners (bool): set to True is you want to crop the image by giving the min and
        max corners of the cropping area. crop_size=[xmin, ymin, xmax, ymax].
    """
    if corners:
        return image[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
    else:
        if crop_center is None:
            crop_center = [image.shape[0]//2, image.shape[1]//2]
        h = crop_center[0]
        w = crop_center[1]
        h_crop = crop_size[0]//2
        w_crop = crop_size[1]//2
    
        return image[h-h_crop:h+h_crop, w-w_crop:w+w_crop]

def crop_alpha_image(img, mode='Rectangle', fill_color=(0,0,0)):
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
    if mode == 'Circle':
        # convert to B&W image and crop image into rectangle
        bw_img = img.max(axis=2)
        non_empty_columns = np.where(bw_img.max(axis=0)>0)[0]
        non_empty_rows = np.where(bw_img.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        crop = img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

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

def rgb_to_rgba_mask(img, mask_value=0, axis=2, alpha_scale=1):
        """
        Take a rgb image and add an alpha layer as the 4th channel
        Wherever the image has "mask_value" in each value along "axis" the alpha
        layer will equal zero there. Everywhere else will be alpha_scale
        
        Args:
            img (np.array): rgb image of shape HxWxD. Values should range from 0 to 255
            mask_value (int): the value to look for when setting the alpha layer values to zero
            axis (int): which axis to look for mask_value along
            alpha_scale (float): the value to scale the alpha layer by. Must be <= 1.0
        """
        assert alpha_scale <= 1.0
        mask = utils.get_mask(img, axis=2, mask_value=0).astype(np.uint8)
        mask = (mask * 255) * alpha_scale
        rgba = np.dstack((img, mask)).astype(np.uint8)
        # rgba = Image.fromarray(rgba)

        return rgba

def paste_image(im1, im2, location, alpha_mask=None):
    """
    Paste image 2 onto image 1 at location

    Inputs:
    #TODO update these descriptions
        im1 - (H, W, D): background image, assuming RGB image as a numpy array
        im2 - (H, W, D): foreground image, assuming RGB image as a numpy array
        location - (H, W): location to place center of im2 onto im1 in pixel space
        alpha_mask - (np.array, HxW): alpha layer for im2

    Outputs:
        image - (RGB image, np.array): output RGB image
    """
    assert im1.ndim == 3 and im1.ndim == im2.ndim
    assert np.max(im2) != 1 and np.max(im1) != 1 # make sure pixel values are 0-255
    if alpha_mask is None:
        assert im2.shape[2] == 4
        alpha_mask = im2[:,:,-1]
        im2 = im2[:,:,:-1]
    else:
        assert alpha_mask.shape == im2.shape[:2]
    # get dimensions
    h, w = im2.shape[:2]
    #TODO replace below with resize_and_pad
    loc = utils.mins_max(location, h, w, dtype=int)
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

def letterbox_image(img, out_dim, fill_value=0, input_img_type='BGR', alpha_thresh=120):
    '''
    resize image with unchanged aspect ratio using padding
    
    Inputs:
        img (np.array): image to resize, should be RGB image of shape HxWxD
        out_dim (tuple): tuple of int of the dimension to resize to. as (w,h)
        fill_value (int): value to use for padding
        input_img_type(str): the type of image the input image is.
            'RGB', 'RGBA', 'BGR' or 'BGRA'
        alpha_thresh (int): threshold value to use for trying to fix the values
            in the alpha layer that are close to zero after resizing.
            This is in terms of pixel intensity of the RGB layers so [0,3*255]
    Outputs:
        canvas (np.array): resized image as RGB or RGBA
    Ref:
     - 
    '''
    if input_img_type == 'RGB':
        pass
        channels = 3
    elif input_img_type == 'BGR':
        img = img[:,:,::-1]
        channels = 3
    elif input_img_type == 'RGBA':
        r, g, b, a = cv2.split(img)
        img = cv2.merge((r,g,b,a))
        channels = 4 
    elif input_img_type == 'BGRA':
        b, g, r, a = cv2.split(img)
        img = cv2.merge((r,g,b,a))
        channels = 4
    else:
        raise ValueError(f'{input_img_type} is not valid!')
    
    # resize the image so aspect ratio is not changed
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = out_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_LINEAR)

    if 'A' in input_img_type:
        # try to fix artifacts created by interpolating alpha layer
        indexes = np.nonzero(np.sum(resized_image[:,:,:-1],axis=-1)>125)
        resized_image[:,:, -1][indexes] = 255

    # pad empty space so image is the desired shape
    canvas = np.full((h, w, channels), fill_value)
    canvas[(h-new_h)//2:(h-new_h)//2+new_h,(w-new_w)//2:(w-new_w)//2+new_w, :] = resized_image

    return canvas 

def resize_and_pad(img, img_size, resize, img_center=None, pad_value=0, img_type='RGB'):
    """
    Resize an image with a constant aspect ratio, but keep the image a certain size by padding it.
    img_size should be larger than resize. Same as letterbox image, but adds extra padding
    #TODO haven't tested on RGB or gray scale images yet
    Args:
        img (np.array): the image to resize
        img_size (list): the desired final size of the image after padding as [H,W]
        resize (list): the size to resize the image to as [H,W]
        img_center (list): where to place the resized image in the final padded image array.
            This should be the pixel location of where to put the center of the resized 
            image as [h,w]. It is the center of the img_size by default
        pad_value (int): the value to pad the image with. Should be between 0 and 255
        img_type (str):  the type of image used for input. Can be 'RGB', 'RGBA', 'BGR' or 'BGRA'
            see letterbox_image
    """
    assert len(img_size) == 2 and len(resize) == 2

    if len(img.shape) == 3:
        img_size = img_size + [3]

    if img_center is None:
        img_center = [img_size[0]//2, img_size[1]//2]
    else:
        assert len(img_center) == 2
    
    # Resize the image w/ constant aspect ratio
    resized_img = letterbox_image(img, (resize[1],resize[0]), fill_value=pad_value, input_img_type=img_type)
    # Pad the image
    corners = utils.mins_max(img_center, resize[0], resize[1], dtype=int)
    output = np.full(img_size, pad_value)
    if len(img.shape) == 3:
        output[corners[1]:corners[3], corners[0]:corners[2], :] = resized_img
    else:
        output[corners[1]:corners[3], corners[0]:corners[2]] = resized_img

    return output

def random_resize(image, mean_size=15, std_size=2, pad_value=0, img_type='RGB'):
    """
    Randomly resize the given image by sampling from a gaussian distribution.
    This keeps the image's aspect ratio constant and pads the image
    Args:
        image (np.array): A numpy array of the image to resize. Should be a RGB image of size HxWxD
        mean_size (int): mean size value to use
        std_size (int): std size value to use
        img_type (str):  the type of image used for input.
            Can be 'RGB', 'RGBA', 'BGR' or 'BGRA'. See letterbox_image
    """
    #TODO might want to add random rotations here too.
    img_size = np.random.normal(loc=mean_size, scale=std_size, size=2).astype(int)
    output = letterbox_image(image, img_size, fill_value=pad_value, input_img_type=img_type)
    return output

#################### Functions for figures ####################

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

def plot_placement_pred(image, prediction, std, label, intrinsics, extrinsics,
                        image_mu, image_std, label_mu, label_std,
                        initial_obj_pose=np.array([0.2, 0.56, -0.275]),
                        num_samples=1, convert=True, viz=False, title=None,
                        binary_classifier=False, binary_result=None,
                        n_stds=10):
    """
    Plots the location of the prediction of the placement shift network
    Putting here, instead of utils so we dont need to keep importing autolab

    Inputs:
        image (torch.tensor): input image to the network (i.e. input1)
        prediction (torch.tensor): the output of the network (the mean if it's a distribution)
        std (torch.tensor): standard deviation of the distribution, same shape as prediction
            (Only for the distributions, leave as None for binary classification)
        label (torch.tensor): the ground_truth for the prediction
        intrinsics (perception.CameraIntrinsics): intrinsics of the camera used
        extrinsics (autolab_core.RigidTransform): the extrinsics of the camera
        convert (bool): set to True if the inputs are directly from the network
            False if already converted to cpu and numpy
        image_mu (torch.tensor): mean image array used to normalize images
        image_std (torch.tensor): standard deviation image array used to normalize images
        label_mu (torch.tensor): mean label array used to normalize labels
        label_std (torch.tensor): standard deviation label array used to normalize labels
        initial_obj_pose (np.array): the known initial pose of the object
        num_samples (int): THIS ONLY WORKS WITH ONE RIGHT NOW
        convert (bool): set to True to convert to CPU and numpy
        viz (bool): set to True to view graph
        binary_classifier (bool): set to True if the model is making binary classifications
        binary_result (torch.tensor): binary classification label of whether the sample
            was close or not
        n_stds (int): the number of standard deviations to show in the plot for distributions
    Ref:
     - https://stackoverflow.com/a/11423554
     - https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
     - https://matplotlib.org/examples/axes_grid/demo_parasite_axes2.html
     - https://stackoverflow.com/questions/57216993/define-aspect-ratio-when-using-twinx-in-new-version-of-matplotlib
    """
    # convert to numpy/cpu and return a sample to plot
    img, pred, std, lab, idx = convert_and_sample(image, prediction, std, label, num_samples)
    img = (img * image_std[:,:,0]) + image_mu[:,:,0]
    lab = (lab * label_std) + label_mu
    lab = np.concatenate((np.array([lab.flatten()[0]]), np.array([0]),np.array([lab.flatten()[1]])))
    if not binary_classifier:
        pred = (pred * label_std) + label_mu
        std = std * label_std
        pred = np.concatenate((np.array([pred.flatten()[0]]),np.array([0]),np.array([pred.flatten()[1]]))) #Only works for one sample
        std = np.concatenate((np.array([std.flatten()[0]]),np.array([0]),np.array([std.flatten()[1]])))
    else:
        #NOTE this probably isn't the most efficient way
        prediction = pred.copy().astype(float)
        pred = np.zeros(3) #place holder

    # find position of prediction and ground truth wrt reference point in pixel space
    ref_pred_pixel = project_point((initial_obj_pose - pred), intrinsics, extrinsics)
    ref_label_pixel = project_point((initial_obj_pose - lab), intrinsics, extrinsics)
    ref_pixel = project_point(initial_obj_pose, intrinsics, extrinsics)
    # get the difference values in pixel space
    pred_diff = ref_pred_pixel - ref_pixel
    label_diff = ref_label_pixel - ref_pixel
    img_center = np.array([img.shape[0]//2, img.shape[1]//2]) #NOTE only works for 1 sample
    # Apply to center of image (ie. release-centric)
    pred_center = np.clip(img_center + pred_diff, [0,0], img.shape)
    label_center = np.clip(img_center + label_diff, [0,0], img.shape)
    # Get the range of the image in meters
    min_pixel = np.array([ref_pixel[1], ref_pixel[0] - img.shape[0]//2])
    depth1 = img[0, img.shape[1]//2]
    min_position = deproject_pixel(min_pixel, depth1, intrinsics, extrinsics, intrinsics.frame)
    max_pixel = np.array([ref_pixel[1], ref_pixel[0] + img.shape[0]//2])
    depth2 = img[img.shape[0]-1, img.shape[1]//2]
    max_position = deproject_pixel(max_pixel, depth2, intrinsics, extrinsics, intrinsics.frame)
    diff = np.abs(max_position - min_position)[0]

    # Make the figure
    fig, ax1 = plt.subplots()
    plt.imshow(img)
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    # Add a second y-axis with the real world dimensions
    ax2 = ax1.twinx()
    ax2_color = 'tab:blue'
    ax2.set_ylabel('Meters', color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    y2_range = np.round(np.linspace(-diff/2, diff/2, num=len(ax1.get_yticks())), decimals=4)
    ax2.set_yticks(ax1.get_xticks())
    ax2.set_yticklabels(y2_range)
    fig.tight_layout()

    if title is not None:
        plt.title(title)
    else:
        plt.title("Network Predictions")

    if not binary_classifier:
        pred_shape = patches.Circle(pred_center, radius=3, linewidth=1, edgecolor='k', facecolor='r')
        plt.gca().add_patch(pred_shape)
        label_shape = patches.Circle(label_center, radius=1, linewidth=1, edgecolor='k', facecolor='b')
        plt.gca().add_patch(label_shape)
        for i in range(n_stds):
            # pred_shape = patches.Circle(pred_center, radius=i*10*std[0], linewidth=2, edgecolor='k', facecolor='none')
            pred_shape = patches.Ellipse(pred_center, width=i*std[1], height=i*std[0], edgecolor='k', facecolor='r')
            plt.gca().add_patch(pred_shape)
        plt.legend([Line2D(range(1), range(1), color='white', linewidth=0, marker='o', markersize=8, markerfacecolor='r'),
                    Line2D(range(1), range(1), color='white', linewidth=0, marker='o', markersize=8, markerfacecolor='b')], 
                    ['prediction', 'ground_truth'],
                    handletextpad=-0.3
        )
    else:
        binary_result = binary_result[idx].cpu().numpy().astype(float)
        label_shape = patches.Circle(label_center, radius=1, linewidth=1, edgecolor='k', facecolor='b')
        plt.gca().add_patch(label_shape)
        plt.legend([Line2D(range(1), range(1), color='white', linewidth=0, marker='o', markersize=8, markerfacecolor='white'),
                    Line2D(range(1), range(1), color='white', linewidth=0, marker='o', markersize=8, markerfacecolor='b')], 
                    [f'prediction={prediction}', f'ground_truth={binary_result}'],
                    handletextpad=-0.3
        )

    # make sure the figure is square and the aspect ratio doesn't get changed
    #TODO the aspect ratio is still a little bit off, try https://stackoverflow.com/questions/54545758/create-equal-aspect-square-plot-with-multiple-axes-when-data-limits-are-differ/54555334#54555334
    squarify(fig)
    if viz:
        plt.show()
    
    return fig
    
def squarify(fig):
    """https://stackoverflow.com/a/57228358"""
    w, h = fig.get_size_inches()
    if w > h:
        t = fig.subplotpars.top
        b = fig.subplotpars.bottom
        axs = h*(t-b)
        l = (1.-axs/w)/2
        fig.subplots_adjust(left=l, right=1-l)
    else:
        t = fig.subplotpars.right
        b = fig.subplotpars.left
        axs = w*(t-b)
        l = (1.-axs/h)/2
        fig.subplots_adjust(bottom=l, top=1-l)

def convert_and_sample(image, prediction, std, label, num_samples=1):
    """
    Convert the inputs so they can be used to makes plots
    Only works for B&W images
    """
    idxs = np.random.randint(low=0, high=label.shape[0], size=num_samples)
    image = image[idxs,0,:,:].squeeze() # only get the image of scene
    prediction = prediction[idxs]
    label = label[idxs]
    
    img = image.cpu().numpy()
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    if std is not None:
        std = std[idxs]
        std = std.cpu().numpy()

    return img, prediction, std, label, idxs

def get_yolo_label():
    raise NotImplementedError
