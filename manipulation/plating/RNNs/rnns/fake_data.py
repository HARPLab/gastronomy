import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import tqdm
from PIL import Image, ImageDraw

import rnns.utils as utils
import rnns.image_utils as image_utils
import rnns.preprocess as preprocess

# def make_train_data(img_reso, num_samples, backgrounds, foregrounds, foreground_ids,
#                     seq_length=1, location=None, masks= None, label_type = None,
#                     patterns=None):


def collect_ref_images(image_dir, resize=False, resolution=(224,224)):
    """
    Gather the reference images that will be used to make the augmented data set.
    More specifically, a group of foreground images that will be pasted onto a
    group of background images to make the data set.
    
    Args:
        image_dir (str): The path to the directory containing the reference images.
            The directory should contain a directory named "foreground_images" and
            another named "background_images". The foreground image directory should
            have seperate folders for each object type where the folders' names are the
            labels to use for the corresponding images inside. The background images
            directory should contain images of the background images to use.
        resize (bool): Whether to resize the image
        resolution (tuple)- desired image resolution to crop images to, using fixed
            aspect ratio and padding.
    Outputs:
        foreground_imgs (dict): each key's value is a list of foreground images.
            The keys are the string of the object type.
        background_imgs (list): list of containing the background images as numpy arrays
    """
    background_imgs = image_utils.get_images_in_dir(directory=f'{image_dir}/background_images',
                                                    resize=resize, resolution=resolution)
    object_dirs = utils.sub_dirs(f'{image_dir}/foreground_images')
    foreground_imgs = {}
    for object_dir in object_dirs:
        label = os.path.split(object_dir)[-1]
        foreground_imgs[label] = image_utils.get_images_in_dir(directory=f'{object_dir}',
                                                               resize=False)
    return foreground_imgs, background_imgs


class PlacingPatterns:
    def __init__(self, backgrounds, foregrounds, work_area, image_shape, alpha_masks=None,
                 image_type='RGBA'):
        """
        Inputs:

            backgrounds (np.array): RGB images of the background to paste foreground
                images onto. Should be shape (num_backgrounds, H, W, D).
            foregrounds (dict): the keys are the object types and the values are 
                lists of RGB images of the corresponding foreground object to place.
            work_area (array-like): 1D array or list of ints, giving the area where
                the objects can be placed. Format is [xmin, ymin, xmax, ymax].
            image_shape (array-like): 1D array, list or tuple or the desired output image
                resolution as ints. Should be of the format [H,W]
            alpha_masks (dict): the alpha layers for foreground objects, same format as
                foregrounds argument. Leave as none if the foreground images already
                have the alpha layer as their 4th channel.
            image_type (str): The type of image the foregrounds are. Can be 'RGB', 'RGBA',
                'BGR' or 'BGRA'. The backgrounds are assumed to be RGB images.
        """
        if alpha_masks is not None:
            assert 'A' not in image_type
        self.backgrounds = backgrounds
        self.num_backgrounds = backgrounds.shape[0]
        self.foregrounds = foregrounds
        self.foreground_keys = list(foregrounds.keys())
        num_classes = len(self.foreground_keys)
        self.class_indexes = list(range(num_classes))
        self.image_shape = tuple(image_shape) + (3,)
        self.work_area = work_area
        self.masks = alpha_masks
        self.image_type = image_type

    def place_pattern(self, pattern_idx, start, sequence_length, **kwargs):
        switcher = {
            0: self.l2r,
            1: self.r2l,
            2: self.forward_slash,
            3: self.back_slash,
            4: self.letter_c,
            5: self.letter_u,
            6: self.letter_s,
            7: self.letter_w,
            8: self.u2d,
            9: self.d2u
        }
        func = switcher.get(pattern_idx, "whoops")
        return func(start, sequence_length, **kwargs)

    def l2r(self, start, sequence_length, xmean=20, xstd=3, ymean=0, ystd=2):
        """
        returns the object centers for a left to right placing pattern
        NOTE: default parameter values are for a sequence_length of around 8 to 10
        Inputs:

        Outputs:
            sequence - (np.array): (N x 2 -> sequence_length x (h,w)) 
        """
        sequence = np.zeros((sequence_length, 2))
        position = start
        for i in range(sequence_length):
            sequence[i, 1] = position[1] + np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] + np.random.normal(ymean, ystd, 1)
            position = sequence[i,:]
        return sequence

    def r2l(self, start, sequence_length, xmean=20, xstd=3, ymean=0, ystd=2):
        """
        returns the object centers for a right to left placing pattern
        NOTE: default parameter values are for a sequence_length of around 8 to 10
        """
        sequence = np.zeros((sequence_length, 2))
        position = start
        for i in range(sequence_length):
            sequence[i, 1] = position[1] - np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] - np.random.normal(ymean, ystd, 1)
            position = sequence[i,:] 
        return sequence

    def forward_slash(self, start):
        raise NotImplementedError

    def back_slash(self, start):
        raise NotImplementedError

    def letter_c(self, start, sequence_length, x_radius=60, y_radius=60, dtheta=18, mean=[0,0,0], std=[2,2,5]):
        """
        Generate y and x coordinates for a C pattern. This creates a elipse that starts at 270 degrees and goes (-).
        NOTE: default parameter values are for a sequence_length of around 11
        Inputs:
            start (array-like): the (y,x) coordinates of the starting point in pixel space
            sequence_length (int): the length of the sequence or the number of points to generate
            x_radius (int): radius of the elipse along the x-axis, in pixel space
            y_radius (int): radius of the elipse along the y-axis, in pixel space
            d_theta (int): the increment to use for the angle between points in the
                sequence. Should be in degrees.
            mean (list): (y,x,r) mean of the distribution used to add noise to pattern
            std (list): (y,x,r) std of the distribution used to add noise to pattern
        Outputs:
            sequence (np.array): (sequence_length x 2) array of the (y,x) coordinates of the pattern
        """
        x_radius = x_radius + np.random.normal(mean[2], std[2], 1).astype(int)
        y_radius = y_radius + np.random.normal(mean[2], std[2], 1).astype(int)

        theta = np.arange(270 - dtheta, 269-((sequence_length-1)*dtheta), -dtheta) * (np.pi/180)
        sequence = np.zeros((sequence_length, 2))
        sequence[0, :] = start 
        
        y_center = start[0] + y_radius
        x_center = start[1]
        x, y = utils.elipse_curve(theta, x_radius, y_radius, [x_center, y_center])
        
        sequence[1:,0] = y.astype(int) + np.random.normal(mean[0], std[0], sequence_length-1).astype(int)
        sequence[1:,1] = x.astype(int) + np.random.normal(mean[1], std[1], sequence_length-1).astype(int)
        
        return sequence

    def letter_u(self, start, sequence_length, x_radius=60, y_radius=60, dtheta=18, mean=[0,0,0], std=[2,2,5]):
        """
        Generate y and x coordinates for a U pattern. This creates a elipse that starts at 180 degrees to (-).
        NOTE: default parameter values are for a sequence_length of around 11
        Inputs:
            start (array-like): the (y,x) coordinates of the starting point in pixel space
            sequence_length (int): the length of the sequence or the number of points to generate
            x_radius (int): radius of the elipse along the x-axis, in pixel space
            y_radius (int): radius of the elipse along the y-axis, in pixel space
            d_theta (int): the increment to use for the angle between points in the
                sequence. Should be in degrees.
            mean (list): (y,x,r) mean of the distribution used to add noise to pattern
            std (list): (y,x,r) std of the distribution used to add noise to pattern
        Outputs:
            sequence (np.array): (sequence_length x 2) array of the (y,x) coordinates of the pattern
        """
        x_radius = x_radius + np.random.normal(mean[2], std[2], 1).astype(int)
        y_radius = y_radius + np.random.normal(mean[2], std[2], 1).astype(int)

        theta = np.arange(180 - dtheta, 179-((sequence_length-1)*dtheta), -dtheta) * (np.pi/180)
        sequence = np.zeros((sequence_length, 2))
        sequence[0, :] = start 
        
        y_center = start[0]
        x_center = start[1] + x_radius
        x, y = utils.elipse_curve(theta, x_radius, y_radius, [x_center, y_center])
        
        sequence[1:,0] = y.astype(int) + np.random.normal(mean[0], std[0], sequence_length-1).astype(int)
        sequence[1:,1] = x.astype(int) + np.random.normal(mean[1], std[1], sequence_length-1).astype(int)
        
        return sequence

    def letter_s(self, start, sequence_length, dy=9, A=30, w=np.pi/5,
                phi=3*np.pi/8, mean=[0,0,0], std=[2,2,3]):
        """
        Generate y and x coordinates for a S pattern. Uses a sine curve. Sine function takes radians.
        NOTE: these parameters are for seq_length around 14
        TODO: change the y's so they aren't linear, this might make the S look better
        
        Inputs:
            start (array-like): the (y,x) coordinates of the starting point in pixel space
            sequence_length (int): the length of the sequence or the number of points to generate
            d_y (int): the increment to use for the distance between the y values of the
                points in the sequence. This is in pixel coordinates.
            A (int): the amplitude to use for the sine curve.
            w (float): the angular frequency of the sine curve.
            phi (float): the phase shift of the sine curve. Changing this will make this
                no longer an S.
            mean (list): (y,x, A) mean of the distribution used to add noise to pattern
            std (list): (y,x, A) std of the distribution used to add noise to pattern
        Outputs:
            sequence (np.array): (sequence_length x 2) array of the (y,x) coordinates of the pattern
        """
        A = A + np.random.normal(mean[2], std[2], 1).astype(int)
        y = np.arange(0, (sequence_length*dy), dy)
        x = utils.s_curve(y, A=A, w=w, phi=phi)
        sequence = np.zeros((sequence_length, 2))
        sequence[:,0] = y.astype(int) + start[0] + np.random.normal(mean[0], std[0], sequence_length).astype(int)
        sequence[:,1] = x.astype(int) + start[1] + np.random.normal(mean[1], std[1], sequence_length).astype(int)
        return sequence

    def letter_w(self, start, sequence_length, dx=9, A=28, w=np.pi/5,
                 phi=-6*np.pi/8, mean=[0,0,0], std=[2,2,3]):
        """
        Generate y and x coordinates for a w pattern. Uses a sine curve. Sine function takes radians.
        NOTE: these parameters are for seq_length of 18
        
        Inputs:
            start (array-like): the (y,x) coordinates of the starting point in pixel space
            sequence_length (int): the length of the sequence or the number of points to generate
            d_x (int): the increment to use for the distance between the y values of the
                points in the sequence. This is in pixel coordinates.
            A (int): the amplitude to use for the sine curve.
            w (float): the angular frequency of the sine curve.
            phi (float): the phase shift of the sine curve. Changing this will make it no
                longer a w.
            mean (list): (y,x, A) mean of the distribution used to add noise to pattern
            std (list): (y,x, A) std of the distribution used to add noise to pattern
        Outputs:
            sequence (np.array): (sequence_length x 2) array of the (y,x) coordinates of the pattern
        """
        A = A + np.random.normal(mean[2], std[2], 1).astype(int)
        x = np.arange(0, (sequence_length*dx), dx)
        y = utils.s_curve(x, A=A, w=w, phi=phi)
        sequence = np.zeros((sequence_length, 2))
        sequence[:,0] = y.astype(int) + start[0] + np.random.normal(mean[0], std[0], sequence_length).astype(int)
        sequence[:,1] = x.astype(int) + start[1] + np.random.normal(mean[1], std[1], sequence_length).astype(int)
        return sequence

    def u2d(self, start, sequence_length, xmean=0, xstd=2, ymean=20, ystd=3):
        """
        returns the object centers for a up to down placing pattern
        Images are flipped upside down, so it is +
        NOTE: default parameter values are for a sequence_length of around 6 to 8
        """
        sequence = np.zeros((sequence_length, 2))
        position = start
        for i in range(sequence_length):
            sequence[i, 1] = position[1] + np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] + np.random.normal(ymean, ystd, 1)
            position = sequence[i,:] 
        return sequence

    def d2u(self, start, sequence_length, xmean=0, xstd=2, ymean=20, ystd=3):
        """
        returns the object centers for a down to up placing pattern
        Images are flipped upside down, so it is -
        NOTE: default parameter values are for a sequence_length of around 6 to 8
        """
        sequence = np.zeros((sequence_length, 2))
        position = start
        for i in range(sequence_length):
            sequence[i, 1] = position[1] - np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] - np.random.normal(ymean, ystd, 1)
            position = sequence[i,:] 
        return sequence

    def check_locations(self, locations, bounds):
        """
        Check if the proposed placement pattern is within working area

        Inputs:
            locations (np.array): Nx2 array with the (y,x) values in
                each row. N is the sequence length
            bounds (array-like): a shape (2,) array-like where the 1st
                value is the height and the 2nd is the width of the image
                you are checking
        Outputs:
            output (bool): True if all objects in proposed placement
                pattern are inside of the working area, False otherwise
        """
        xmin = max(self.work_area[0], 0)
        ymin = max(self.work_area[1], 0)
        xmax = min(self.work_area[2], bounds[1])
        ymax = min(self.work_area[3], bounds[0])
        x = np.where((xmin <= locations[:,1]) & (locations[:,1] <= xmax), True, False)
        y = np.where((ymin <= locations[:,0]) & (locations[:,0] <= ymax), True, False)
        values = np.concatenate((x, y))
        assert len(values) != 0

        return np.all(values)

    def rand_class(self, remove=None):
        """
        Return a random key and list index from self.foregrounds
        Args:
            remove (int): a class index (the index of what key from self.foregrounds)
                to not include in the random choice of the key. Leave as None to include all
        """
        if remove is not None:
            classes = self.class_indexes.copy()
            classes.remove(remove)
        else:
            classes = self.class_indexes
        key_idx = np.random.choice(classes)
        rand_key = self.foreground_keys[key_idx]
        num_foregrounds = len(self.foregrounds[rand_key])
        rand_idx = np.random.choice(num_foregrounds)
       
        return rand_key, rand_idx, key_idx

    def run(self, num_samples, sequence_length, patterns=None, foreground_dim_mean=50,
            foreground_dim_std=10, label_type='mask', **kwargs):
        """
        Main run loop for generating the image sequences

        Inputs:
            num_samples (int): number of sample sequences to make.
            sequence_length (int): length of each sequence.
            patterns (array-like): list or array of ints, saying what patterns
                to use(see place_pattern)
            foreground_dim_mean (int): mean of the distribution used to sample the pixel
                size of the foreground images
            foreground_dim_std (int): standard deviation of the distribution used to 
                sample the pixel size of the foreground images
            label_type - (string): specifies what type of labels to give, default is
                binary mask of where the object center should be: 'mask'.
                If "center" gives the value of the object center as [y,x,h,w].
                If "yolo" gives the value of the object bounding box as [xmin, ymin, xmax, ymax].
                NOTE: These are values 1-4 of a yolo label so: YOLO_label[1:5]
            **kwargs: see the arguments for the different functions in place_patterns

        Outputs:
            fake_sequences - (np.array): all the generated images
            fake_labels - (np.array): labels of the training sequences
            (both are of shape -> (num_batches, sequence length, H, W, D))
            if self.label_type = center, then the label is 
                -> (num_batches, sequence_length, 4) and each label is (y,x,h,w)
        """
        fake_sequences = []
        fake_labels = []
        samples = np.zeros(((num_samples, sequence_length)+self.image_shape))
        print(f'Generating sequences...')
        status = tqdm.tqdm(initial=0, total=num_samples, file=sys.stdout, desc='Samples left to create:')
        check = 0
        while len(fake_sequences) < num_samples:
            assert check < 1000 # make sure loop doesn't hang
            # get one of the random placement patterns, change rand_pattern for what patterns to use
            if patterns is not None:
                rand_pattern = int(np.random.choice(patterns, size=1))
            else:
                # default to straight horizontal lines
                rand_pattern = int(np.random.randint(0, 2, size=1))

            # pick a random background for sequence, make sure dims are correct
            rand_background = np.random.randint(self.num_backgrounds)
            seq_image = np.squeeze(self.backgrounds[rand_background])
            temp_seq = []
            temp_labels = []

            # generate a pattern of placement locations and check validity
            initial_position = utils.rand_location(self.work_area, 1)
            place_locations = self.place_pattern(rand_pattern, initial_position, sequence_length, **kwargs)
            # check that objects don't go outside of work_area
            if not self.check_locations(place_locations, seq_image.shape):
                check += 1
                continue
            check = 0

            # randomly choose a starting object
            rand_key, rand_idx, key_idx = self.rand_class()

            # make the image sequence
            j = status.n
            for i in range(sequence_length):
                if self.masks is not None:
                    mask = self.masks[rand_key][rand_idx]
                else:
                    mask = None
                
                # place foreground image
                rand_foreground = self.foregrounds[rand_key][rand_idx].copy()
                rand_foreground = image_utils.random_resize(rand_foreground,
                                                            mean_size=foreground_dim_mean,
                                                            std_size=foreground_dim_std,
                                                            pad_value=0,
                                                            img_type=self.image_type)
                seq_image = image_utils.paste_image(seq_image,
                                                    rand_foreground,
                                                    place_locations[i],
                                                    alpha_mask=mask)
                # resize the image
                temp_img = image_utils.letterbox_image(seq_image,
                                                       self.image_shape[:2],
                                                       fill_value=0,
                                                       input_img_type='RGB')

                temp_seq.append(temp_img.copy()) #TODO might want to initialize an array to make this faster
                # samples[j,i,:,:,:] = temp_img.copy()

                # get label for the image
                if label_type == 'mask':
                    h, w = rand_foreground.shape[:2]
                    corners = utils.mins_max(place_locations[i], h, w, dtype=int)
                    temp_l = utils.make_mask(temp_img.shape, corners, 0, 1, dtype=int)
                    temp_labels.append(temp_l)
                elif label_type == 'center':
                    h, w = rand_foreground.shape[:2]
                    y = place_locations[i][0]
                    x = place_locations[i][1]
                    temp_l = np.array([y, x, h, w], dtype=int)
                    temp_labels.append(temp_l)
                elif label_type == 'yolo':
                    h, w = rand_foreground.shape[:2]
                    corners = utils.mins_max(place_locations[i], h, w, dtype=int)
                    temp_labels.append(corners)
                else:
                    raise ValueError('Unknown label type')

                # update values to get a different object type
                rand_key, rand_idx, key_idx = self.rand_class(key_idx)

            # import ipdb; ipdb.set_trace()
            # plt.imshow(temp_seq[-1])
            # plt.show()
            temp_seq = np.asarray(temp_seq)
            fake_sequences.append(temp_seq)
            temp_label = np.asarray(temp_labels)
            fake_labels.append(temp_label)

            # update the progress bar
            status.update(1)
            status.refresh()

        fake_sequences = np.asarray(fake_sequences, dtype=np.uint8)
        fake_labels = np.asarray(fake_labels) # should have been turned to uint earlier

        status.close()

        return fake_sequences, fake_labels


############### Deprecated ###############

def make_reference_images(image_path, resolution, label_file_name):
    """
    DEPRECATING: Use collect_ref_images

    Crop images to the specified resolution and returns background
    and foreground objects for making augmented images

    Inputs:
        image_path - (string): file path of images
        resolution - desired image reolution to crop to, keeping aspect ratio in tact
        label_file_name - string indicating the file strings to look for 

    Outputs:
        foreground - (list): list of images of the objects that vary in size 
        background - (np.array): array of images of shape (num images, H, W, D) 
        obj_id - (list): object ID of the foreground images
        alpha_mask - (list of arrays): list of the alpha layers for the 
                                       foreground images (use w/ PIL)
    """
    print('DEPRECATING: Use collect_ref_images')
    # start clock
    t0 = time.clock()

    # gather the images and labels from path
    test = preprocess.PrepTrainData(image_path, resolution, label_file_name)
    sequences = test.collect_images()
    labels, obj_id = test.create_labels()

    # seperate the background images of just the cutting board if they are present
    background = []
    foreground = []
    i = 0
    for sequence in sequences:
        if len(sequence) == len(labels[i]):
            foreground.append(np.asarray(sequence))
        else:
            background.append(np.asarray(sequence[:1]))
            foreground.append(np.asarray(sequence[1:]))
        i += 1

    # mask foreground images to just show object
    obj_foreground = np.zeros((1, resolution[0], resolution[1], 3))
    for i in range(len(foreground)):
        # labels contains masks
        temp_labels = np.asarray(labels[i])
        temp_foreground = np.asarray(foreground[i])
        obj_foreground = np.concatenate((obj_foreground, np.multiply(temp_labels, temp_foreground)))
    # remove inital image of zeros from array of images (this isn't efficient)
    obj_foreground = obj_foreground[1:]
    
    #crop foreground images
    foreground = []
    alpha_mask = []
    for i in range(obj_foreground.shape[0]):
        cropped_image, alpha = image_utils.crop_alpha_image(obj_foreground[i], mode='Circle')
        
        foreground.append(cropped_image)
        alpha_mask.append(alpha)

    # flatten the background to just the images (HxWxD)
    background = np.squeeze(np.asarray(background))

    # print process time
    t1 = time.clock()
    print(f'It took {t1-t0} seconds to gather reference images')

    return foreground, background, obj_id, alpha_mask

def make_train_data(img_reso, num_samples, backgrounds, foregrounds, foreground_ids,
                    seq_length=1, location=None, masks= None, label_type = None,
                    patterns=None):
    """
    #DEPRECATED
    Make a augmented data set for training
    #TODO just move all of this into the make_fake_data.py script no need for an extra function
    
    Inputs:
    #TODO need to adjust these
        img_reso - int that specifies the image resolution, H and W equal this value
        num_samples - int of the number of toy image sequences to make
        backgrounds - array of images of the possible backgrounds
                    [number of images, H, W, D] 
        foregrounds - list of images of the objects that can be placed
                    [number of images, H, W, D]
        foreground_ids - array of ints, stating the type of object the foregrounds are
        seq_length - int of how long to make sequence length (training length)
        location - array of ints of the area where the objects can be placed
                   [xmin, ymin, xmax, ymax]
        masks - (list of arrays): the corresponding alpha layers for the foregrounds
        label_type - (string): specifies what type of labels to give, default is
                     binary mask of where the object center should be
                     if "center" gives the value of the object center as (y,x,h,w)
        patterns: list or array of ints, saying what patterns to use(see place_pattern)

    Outputs:
        training_sequences - (np.array): all the generated images
        labels - (np.array): labels of the training sequences
        (both are of shape -> (num_batches, sequence length, H, W, D))
        unless label_type is center
    """
    # record start time
    t0 = time.clock()

    if type(backgrounds) == list:
        backgrounds = np.array(backgrounds)
    assert img_reso == backgrounds[0].shape[0] and img_reso == backgrounds[0].shape[1]
    
    # place object anywhere in image if location isn't given
    if location is None:
        location = np.array([0, 0, img_reso[0], img_reso[1]])

    #TODO just do the below check dynamically during image generation
    # # find the largest obj width/height 
    # sizes = np.zeros((len(foregrounds),2))
    # for i in range(len(foregrounds)):
    #     sizes[i] = foregrounds[i].shape[:2]
    # max_obj_size = int(np.max(sizes))
    
    # makes sure not to place objects outside of the specified location
    work_area = np.array([location[0]+max_obj_size//2, location[1]+max_obj_size//2, 
                        location[2]-max_obj_size//2, location[3]-max_obj_size//2])

    # generate the training data, Note: args=0 is just a place holder
    placements = PlacingPatterns(num_samples, seq_length, backgrounds, 
                                foregrounds, foreground_ids, work_area, 
                                masks, label_type=label_type)
    #for multiprocessing, doesn't work right now
    # training_sequences, labels = utils.many_processes(patterns.run, 8, args=[1])
    
    # change the placing patterns to use here
    training_sequences, labels = placements.run(patterns=patterns)

    # print process time
    t1 = time.clock()
    print(f'It took {t1-t0} seconds to generate the data')

    return training_sequences, labels #TODO might want to make this three channeled
                    # also check if the input images are float
                    # or out of 255, i assumed 255 here
