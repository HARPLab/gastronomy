import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image, ImageDraw

import rnns.utils as utils
import rnns.preprocess as preprocess

"""
TODO: choose reference_images from a smaller pool, some of the foreground
      images dont all turn out nice. also need to start training on the other patterns
"""

def make_train_data(img_reso, num_samples, backgrounds, foregrounds, foreground_ids,
                    seq_length=1, location=None, masks= None, label_type = None,
                    patterns=None):
    """
    Make a augmented data set for training
    
    Inputs:
    #TODO need to adjust these
        img_reso - tuple of ints that specifies the image resolution, HxWxD
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
    
    # place object anywhere in image if location isn't given
    if location is None:
        location = np.array([0, 0, img_reso[0], img_reso[1]])

    assert img_reso == backgrounds[0].shape[:2]

    # find the largest obj width/height
    sizes = np.zeros((len(foregrounds),2))
    for i in range(len(foregrounds)):
        sizes[i] = foregrounds[i].shape[:2]
    max_obj_size = int(np.max(sizes))
    
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

def reference_images(image_path, resolution, label_file_name):
    """
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
        cropped_image, alpha = utils.crop_image(obj_foreground[i], mode='Circle')
        
        foreground.append(cropped_image)
        alpha_mask.append(alpha)

    # flatten the background to just the images (HxWxD)
    background = np.squeeze(np.asarray(background))

    # print process time
    t1 = time.clock()
    print(f'It took {t1-t0} seconds to gather reference images')

    return foreground, background, obj_id, alpha_mask

class PlacingPatterns:
    def __init__(self, sample_size, sequence_length, backgrounds, 
                foregrounds, foreground_ids, work_area, 
                alpha_masks=None, label_type=None):
        """
        Inputs:
            sample_size - (int): number of sample sequences to make
            sequence_length - (int): length of image sequence
            backgrounds - (np.array): RGB image of the background to base image on
            foregrounds - (np.array): RGB image of the foregorund object to place
            foreground_ids - (np.array): array of labels for the foreground objects
            work_area - array of ints of the area where the objects can be placed
                    format is [xmin, ymin, xmax, ymax]
            alpha_masks (list of arrays): the alpha layers for foreground objects
            label_type - (string): specifies what type of labels to give, default is
                    binary mask of where the object center should be
                    if "center" gives the value of the object center as (y,x,h,w)
        """
        self.sample_size = sample_size
        self.sequence_length = sequence_length
        self.backgrounds = backgrounds
        self.num_backgrounds = backgrounds.shape[0] #TODO move
        self.work_area = work_area
        self.label_type = label_type

        # split the foreground objects up
        foreground_objs = []

        masks = []
        objs = np.unique(foreground_ids)
        for i in range(len(objs)):
            temp_idx = np.argwhere(foreground_ids == objs[i]).reshape(-1)
            foreground_objs.append(np.asarray(foregrounds)[temp_idx])
            if len(alpha_masks) > 0:
                masks.append(np.asarray(alpha_masks)[temp_idx])
        assert len(foreground_objs) == len(objs)
        self.foregrounds = foreground_objs
        self.masks = masks

    def l2r(self, start, xmean=8, xstd=2, ymean=0, ystd=1):
        """
        returns the object centers for a left to right placing pattern

        Inputs:

        Outputs:
            sequence - (np.array): (N x 2 -> sequence_length x (h,w)) 
        """
        sequence = np.zeros((self.sequence_length, 2))
        position = start
        for i in range(self.sequence_length):
            sequence[i, 1] = position[1] + np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] + np.random.normal(ymean, ystd, 1)
            position = sequence[i,:]
        return sequence

    def r2l(self, start, xmean=8, xstd=2, ymean=0, ystd=1):
        """
        returns the object centers for a right to left placing pattern
        """
        sequence = np.zeros((self.sequence_length, 2))
        position = start
        for i in range(self.sequence_length):
            sequence[i, 1] = position[1] - np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] - np.random.normal(ymean, ystd, 1)
            position = sequence[i,:] 
        return sequence

    def forward_slash(self, start):
        pass

    def back_slash(self, start):
        pass

    def letter_c(self, start):
        pass

    def letter_u(self, start):
        pass

    def letter_s(self, start, dy=3, A=10, w=2*np.pi*(1/24), ymean=8, ystd=2):
        #NOTE these parameters are for seq_length of 12
        sequence = np.zeros((self.sequence_length, 2))
        position = start
        x = lambda y : A*np.sin(w*y + 2*np.pi/3)

        y = np.arange(dy, self.sequence_length*dy+1, dy)
        sequence[:, 0] = y + position[0]

        for i in range(self.sequence_length):
            sequence[i, 1] = position[1] + x(y[i])
            position = sequence[i,:]
        return sequence

    def sin_curve(self, start, A, w, xmean=8, xstd=2):
        pass

    def u2d(self, start, xmean=0, xstd=1, ymean=8, ystd=2):
        """
        returns the object centers for a up to down placing pattern
        Images are flipped upside down, so it is +
        """
        sequence = np.zeros((self.sequence_length, 2))
        position = start
        for i in range(self.sequence_length):
            sequence[i, 1] = position[1] + np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] + np.random.normal(ymean, ystd, 1)
            position = sequence[i,:] 
        return sequence

    def d2u(self, start, xmean=0, xstd=1, ymean=8, ystd=2):
        """
        returns the object centers for a down to up placing pattern
        Images are flipped upside down, so it is -
        """
        sequence = np.zeros((self.sequence_length, 2))
        position = start
        for i in range(self.sequence_length):
            sequence[i, 1] = position[1] - np.random.normal(xmean, xstd, 1)
            sequence[i, 0] = position[0] - np.random.normal(ymean, ystd, 1)
            position = sequence[i,:] 
        return sequence

    def place_pattern(self, arg, start):
        switcher = {
            0: self.l2r,
            1: self.r2l,
            2: self.forward_slash,
            3: self.back_slash,
            4: self.letter_c,
            5: self.letter_u,
            6: self.letter_s,
            7: self.sin_curve,
            8: self.u2d,
            9: self.d2u
        }
        func = switcher.get(int(arg), "whoops")
        return func(start)

    def check_locations(self, locations):
        """
        Check if the proposed placement pattern is within working area

        Inputs:
            locations (np.array): Nx2 array with the (y,x) values in
                each row. N is the sequence length
        Outputs:
            output (bool): True if all objects in proposed placement
                pattern are inside of the working area, False otherwise
        """
        xmin = self.work_area[0]
        ymin = self.work_area[1]
        xmax = self.work_area[2]
        ymax = self.work_area[3]
        x = np.where((xmin <= locations[:,1]) & (locations[:,1] <= xmax), True, False)
        y = np.where((ymin <= locations[:,0]) & (locations[:,0] <= ymax), True, False)
        values = np.concatenate((x, y))
        assert len(values) != 0

        return np.all(values)

    def run(self, patterns=None, nothing=None):
        """
        Main run loop for generating the image sequences

        Inputs:
            patterns: list or array of ints, saying what patterns to use(see place_pattern)
            nothing: literally nothing, just here for multiproccesing purposes,
                not yet implemented
        Outputs:
            fake_sequences - (np.array): all the generated images
            fake_labels - (np.array): labels of the training sequences
            (both are of shape -> (num_batches, sequence length, H, W, D))
            if self.label_type = center, then the label is 
                -> (num_batches, sequence_length, 4) and each label is (y,x,h,w)
        """
        fake_sequences = []
        fake_labels = []
        status = 1
        # print(f'Currently generating sequence number: {status}') #print current sequence
        print(f'Generating sequences...')
        while len(fake_sequences) < self.sample_size:
            # print function status
            if (status - len(fake_sequences)) <= 0:
                status += 1
                # print(f'Currently generating sequence number: {status}')
            # get one of the random placement patterns, change rand for what patterns to use
            if patterns is not None:
                rand = np.random.choice(patterns, size=1)
            else:
                # default to straight horizontal lines
                rand = np.random.randint(0, 2, size=1)
            initial_position = utils.rand_location(self.work_area, 1)
            place_locations = self.place_pattern(rand, initial_position)
            # check that objects don't go outside of work_area
            if not self.check_locations(place_locations):
                continue

            # pick a random background for sequence, make sure dims are correct
            rand_background = np.random.randint(0, self.num_backgrounds, 1)
            temp_img = np.squeeze(self.backgrounds[rand_background])
            temp_seq = []
            temp_labels = []
            # pick a obj type to start placement with
            options = np.arange(0, len(self.foregrounds))
            which_obj = np.random.choice(options)
            # make the image sequence
            while len(temp_seq) < self.sequence_length:
                # set foreground image
                foreground = self.foregrounds[which_obj]
                mask = self.masks[which_obj]
                rand_idx = np.random.randint(0, len(foreground), 1)
                rand_foreground = foreground[int(rand_idx)]
                # place foreground image
                i = len(temp_seq)
                temp_img = utils.paste_image(temp_img, rand_foreground,
                           place_locations[i], alpha_mask=mask[int(rand_idx)])
                temp_seq.append(temp_img)
                # get label for the image
                if self.label_type == None:
                    h, w = rand_foreground.shape[:2]
                    area = utils.mins_max(place_locations[i], h, w, dtype=np.uint8)
                    temp_l = utils.make_mask(temp_img.shape, area, 0, 1, dtype=np.uint8)
                    temp_labels.append(temp_l)
                elif self.label_type == 'center':
                    h, w = rand_foreground.shape[:2]
                    y, x = int(place_locations[i][0]), int(place_locations[i][1])
                    temp_l = np.array([y, x, h, w])
                    temp_labels.append(temp_l)
                # switch to another object type
                temp_options = np.copy(options)
                which_obj = np.random.choice(np.setdiff1d(temp_options, which_obj))
            # import ipdb; ipdb.set_trace()
            temp_seq = np.asarray(temp_seq)
            fake_sequences.append(temp_seq)
            temp_label = np.asarray(temp_labels)
            fake_labels.append(temp_label)
        fake_sequences = np.asarray(fake_sequences, dtype=np.uint8)
        fake_labels = np.asarray(fake_labels) # should have been turned to uint earlier
        
        return fake_sequences, fake_labels
