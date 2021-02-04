import numpy as np
import random
import glob
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import rnns.utils as utils
import rnns.image_utils as image_utils

class _RepeatSampler(object):
    """
    Sampler that repeats forever
    Args:
        sampler (Sampler)
    Ref:
     - https://github.com/pytorch/pytorch/issues/15849#issuecomment-583209012
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(DataLoader):
    """
    Wrapper for DataLoader that reuses processes instead of respawning them
    Ref:
     - https://github.com/pytorch/pytorch/issues/15849#issuecomment-583209012
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class PrepTrainData:
    """
    Preprocess training data
    
    """
    def __init__(self, train_data_path, img_resolution, label_file_suffix):
       """
       Assuming the training data are images 
       and the labels are the bounding boxes locations

       train_data_path - directory where all the training images are
                        each image sequence should be split into a folder
       """
       self.dirs = utils.sub_dirs(train_data_path)
       self.reso = img_resolution
       assert type(self.reso) is tuple
       self.suffix = label_file_suffix

    def collect_images(self):
        # Collect all of the images in the directory and sort numerically into a list
        # and resize to specified image resolution without changing scale
        # each list element should contain a sequence(list) of images
        img_sequences = []
        for dirp in self.dirs:
            temp_list = []
            # For jpg images
            for filename in sorted(glob.glob('{}/*.jpg'.format(dirp)), key=utils.numerical_sort):
                img = cv2.imread(filename, 1)
                #reshape images
                img = image_utils.letterbox_image(img, self.reso)
                temp_list.append(img)
            # For png images
            for filename in sorted(glob.glob('{}/*.png'.format(dirp)), key=utils.numerical_sort):
                img = cv2.imread(filename, 1)
                #reshape images
                img = image_utils.letterbox_image(img, self.reso)
                temp_list.append(img)
            # For jpeg images
            for filename in sorted(glob.glob('{}/*.jpeg'.format(dirp)), key=utils.numerical_sort):
                img = cv2.imread(filename, 1)
                #reshape images
                img = image_utils.letterbox_image(img, self.reso)
                temp_list.append(img)
            img_sequences.append(temp_list)
        assert len(img_sequences) == len(self.dirs)
        
        return img_sequences
    
    def create_labels(self, label_type=None):
        """
        Inputs:
        label_type - (string): specifies what type of labels to give, default(None) is
                        binary mask of where the object center should be
                        if "center" gives the value of the object center as (y,x,h,w)
        Outputs:
            labels - (list of sequences): ground truth labels for img_sequences
                        - labels[n] gives the list of labels in sequence n
                        - labels[n][m] gives the labels for corresponding 
                          images from the output of self.collect_images
                          same size as images unless label_type = center

            obj_id - (list of ints): gives the identifying class number of object(see YOLO stuff)
                        - for making the fake data
                        (corresponds to foreground output of fake_data.reference_images)
        """        
        bboxes = []
        labels = []
        obj_id = []
        # iterate through the directories and get numpy file of the bounding boxes per sequence
        for dirp in self.dirs:
            label_file = utils.sub_files(dirp, self.suffix)
            for files in label_file:
                temp = np.load(files)
                bboxes.append(temp)
        assert len(bboxes) == len(self.dirs)
        # reshape the bounding box labels to the same shape as training images
        for boxes in bboxes:
            temp_labels = []
            for i in range(boxes.shape[0]):
                box = boxes[i,:]
                x1 = int(box[1])
                y1 = int(box[2])
                x2 = int(box[3])
                y2 = int(box[4])
                if label_type == None:
                    # make base RGB image of zeros
                    label = np.zeros((self.reso + (3,)), dtype=int)
                    label[y1:y2, x1:x2] = 1
                    temp_labels.append(label)
                elif label_type == 'center':
                    w = x2 - x1
                    h = y2 - y1
                    x = x1 + w//2
                    y = y1 + h//2
                    label = np.array([y, x, h, w])
                    temp_labels.append(label)
                # for object identification
                obj_id.append(int(box[-1]))
            labels.append(temp_labels)
        return labels, obj_id
#should return a list of multiple lists. each of those lists are a sequence of images
#make sure that create_labels output and the collect_images output are the same shape

def make_train_batches(sequences, labels, training_length):
    """
    Splits the given image and label sequences into batches for training

    Inputs:
        sequences - (list): list of image sequences
        labels - (list): corresponding labels for sequences
               - (same size as the images that consists of only 1s and 0s)
               - (both the images and there labels are RGB images as np.array)
        training_length - (int): size of the trainging sequences
    Outputs:
        train_data - (np.array): all the trainging sequences
        train_labels - (np.array): labels of the training sequences
        (both are of shape -> (num_batches, sequence length, H, W, D))
        except when label type is center
    """
    assert len(sequences) == len(labels)
    # sort the training sequences into the correct size
    train_data = []
    train_labels = []
    for i in range(len(sequences)):
        temp_sequence = []
        temp_labels = []
        sequence = sequences[i]
        sequence_label = labels[i]
        # remove first blank background image if it is there
        if len(sequence) > len(sequence_label):
            sequence = sequence[1:]
        assert len(sequence) == len(sequence_label)
        order = np.arange(len(sequence))
        # get all possible ordered sequences of given length
        combos = utils.ordered_combos(order, training_length)
        # put combinations into a list
        for i in range(combos.shape[0]):
            temp_sequence.append(np.array(sequence,
                        dtype=np.uint8)[combos[i]])
            temp_labels.append(np.array(sequence_label, 
                        dtype=int)[combos[i]])
        temp_sequence = np.array(temp_sequence)
        temp_labels = np.array(temp_labels)
        train_data.append(temp_sequence)
        train_labels.append(temp_labels)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    if train_data.ndim != 5:
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)

    return train_data, train_labels

def train_test_split(data, labels, percentage, mix=False):
    """
    Split training data into training and testing

    Inputs:
        data - (np.array): data to be split up, splits along axis 0
        labels - (np.array): corresponding labels for data       
        percentage - (float): percent split of data to be split for test
        mix - (bool): set flag to true to shuffle data
    Outputs:
        train_data - (np.array): training data
        train_labels - (np.array): labels for train data
        test_data - (np.array): testing data
        test_labels - (np.array): labels for test data 
    """
    assert type(data).__module__ and type(labels).__module__ == 'numpy'
    # shuffle the data if flag is set
    idx = np.arange(data.shape[0])
    if mix is True:
        np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    # split the data up
    assert percentage < 1
    train_len = int(percentage*(data.shape[0]))

    test_data = data[:-train_len]
    test_labels = labels[:-train_len]

    train_data = data[-train_len:]
    train_labels = labels[-train_len:]

    return train_data, train_labels, test_data, test_labels

class SequenceDataset(Dataset):
    """
    Inputs:
        data - (np.array): should be 5 dims -> (batch_size, sequence_length, H, W, D)
        labels - (np.array): same as data for masks labels, and 3 dims for
                             center labels -> (batch_size, sequence_length, obj_info)
                             obj_info dim is 4 -> (y, x, height, width)
    
    Notes:
        probably would've been easier to use a TensorDataset class instead of map
        https://discuss.pytorch.org/t/make-a-tensordataset-and-dataloader-with-multiple-inputs-parameters/26605
    """
    def __init__(self, data, labels, rm_amount, transform=None):
        assert data.shape[0] == labels.shape[0]
        self.length = data.shape[0]
        self.rm_amount = rm_amount
        self.transform = transform # not using right now
        if torch.is_tensor(data) and torch.is_tensor(labels):
            self.data = data
            self.labels = labels
        else:
            # swap dimensions of images to torch format if necessary (D x H x W)
            self.data = torch.from_numpy(data).permute(0,1,4,2,3)
            if labels.ndim == 3:
                self.labels = torch.from_numpy(labels)
            else:
                self.labels = torch.from_numpy(labels).permute(0,1,4,2,3)

    def __len__(self):
        # gives total number of sample sequences
        return self.length
    
    def __getitem__(self, index):
        """
        gives one sample sequence
        x - the sequence of images to pass through network
        y - the sequence of labels (y,x) for the placement predictions
        y0 - the sequence of labels (y,x) for object detection predctions
        obj_shape - the height and width of object
        """
        obj_shape = None
        x = self.data[index].type(torch.float)
        y0 = self.labels[index].type(torch.float)
        # normalize pixels of 0-255 to 0-1, ie /255
        x = torch.div(x, 255)
        # normalize labels of 0-416 to 0-1, ie /416
        y0 = torch.div(y0, x.shape[3])

        # # want labels to go with previous image, for prediction
        y, x = utils.remove_first_last(y0, x, amount=self.rm_amount)
        y0 = y0[:-1, :]
        #sequence length is now 7
        
        if y.ndim == 2:
            obj_shape = y[:, 2:]
            y = y[:, :2]
            obj_shape0 = y0[:, 2:]
            y0 = y0[:, :2]

        if self.transform is not None:
            temp = []
            for i in range(x.shape[0]):
                temp_x = self.transform(x[i])
                temp.append(temp_x)
                #NOTE need to add something for mask labels
            x = torch.stack(temp)
        
        return x, y, obj_shape, y0, obj_shape0


class PlacementShiftDataset(Dataset):
    """Dataset for post placement shift prediction"""

    def __init__(self, image_data, ee_data, labels, transform=None):
        """
        Inputs:
            image_data (np.array): should be 4 dims -> (batch_size, H, W, D) where D = 2
            ee_data (np.array): should be 2 dims -> (batch_size, 7) where each row is ee pose
            labels (np.array): should be 4 dims (batch_size, x,y,z) distance from center of object to end effector pose
            transform (callable, optional): Optional transform to be applied
                on a sample (only the input, not the label).
        """
        assert image_data.ndim == 4 and labels.ndim == 2# and ee_data.ndim == 2
        assert image_data.shape[0] == labels.shape[0]
        assert ee_data.shape[0] == labels.shape[0]
        self.length = image_data.shape[0]
        self.transform = transform

        if torch.is_tensor(image_data):
            self.image_data = image_data
        else:
            self.image_data = torch.from_numpy(image_data).permute(0,3,1,2) # swap dims (D x H x W)

        if torch.is_tensor(labels):
            self.labels = labels
        else:
            self.labels = torch.from_numpy(labels)

        if torch.is_tensor(ee_data):
            self.ee_data = ee_data
        else:
            self.ee_data = torch.from_numpy(ee_data)  

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        input1 = self.image_data[index]
        input2 = self.ee_data[index]
        label = self.labels[index]

        # might as well just normalize in the training script
        # # normalize depth pixel values to be between 0 and 1
        # input1 = torch.div(input1, 0.05)
        # normalize labels to be between -1 and 1
        # label = torch.div(label, 0.05)

        if self.transform is not None:
            input1 = self.transform(input1)

        return input1, input2, label

def train_test_split_placing(data1, data2, labels, percentage, mix=False):
    """
    Same as train_test_split but with two inputs and one label
    """
    assert type(data1).__module__ and type(data2).__module__ and type(labels).__module__ == 'numpy'
    # shuffle the data if flag is set
    idx = np.arange(data1.shape[0])
    if mix is True:
        np.random.shuffle(idx)
    data1 = data1[idx]
    data2 = data2[idx]
    labels = labels[idx]

    # split the data up
    assert percentage <= 1
    train_len = int(percentage*(data1.shape[0]))

    test_data1 = data1[train_len:]
    test_data2 = data2[train_len:]
    test_labels = labels[train_len:]

    train_data1 = data1[:train_len]
    train_data2 = data2[:train_len]
    train_labels = labels[:train_len]

    return train_data1, train_data2, train_labels, test_data1, test_data2, test_labels

def prep_placing_train_data(filepath, file_type='.npz', keys=['states','ground_truth',
        'scene_depth_images','obj_depth_images'], bad_samples_key='failure_flags',
        remove_non_num=True, clip_keys=['scene_depth_images', 'obj_depth_images'],
        clip_values=[0.0, 0.18]):
    """
    For parsing the data files collected from the placing objects in Isaac Gym to
    train a post placement shift predictor. Assuming file(s) is made of numpy arrays
    and that axis 0 of the arrays is the number of training samples

    Inputs:
        filepath(str): string to the data file or folder of data files.
            Only looks at files in immediate directory
        file_type(str): the data file type, a .npz file by default
        keys(list): list of the keys on the data files to get the data for, returns
            these arrays in the order they are given in this arg
        bad_samples_key(str): the key of the array in the data file that says which 
            training samples are bad
        remove_non_num(bool): Set flag to true to remove any samples that contain nans or infs
        clip_keys(list): list of strings. each string is a key from the data arrays that you want
            to bound the values for
        clip_values(list): The min and max values to allow for the data. Only allows these values right now
    """
    # check if the filepath is a directory or file
    if os.path.isdir(filepath):
        data_files = utils.sub_files(filepath, file_type)
    else:
        data_files = [filepath]

    # loop through all of the files
    output_data = [None]*len(keys)
    rm_idx = np.array([]) # for removing nan's and inf's
    n_new_samples = 0 # for removing nan's and inf's
    for j in range(len(data_files)):
        data_file = data_files[j]
        data = np.load(data_file)
        # get the indexes of the training samples to remove
        if bad_samples_key is not None:
            remove_indexes = np.argwhere(data[bad_samples_key])
        else:
            remove_indexes = []

        # loop through all of the given keys
        for idx, desired_key in enumerate(keys):
            key_count = 0 # to check if the desired key exists
            # not using 'in' method since we want to return keys in order
            for key, value in data.items():
                if key == desired_key:
                    # Remove the samples that are flagged as bad
                    if len(remove_indexes) == 0:
                        temp = value
                    else:
                        temp = np.delete(value, remove_indexes, axis=0) # remove bad samples
                    # Remove samples that have inf or nan
                    if remove_non_num is True:
                        # get all axis 0 indexes of the samples that have infs and nans, remove repeats
                        temp_rm_idx = np.unique(np.concatenate((np.where(np.isnan(temp))[0], 
                                                                np.where(np.isinf(temp))[0])))
                        temp_rm_idx += n_new_samples
                        rm_idx = np.concatenate((rm_idx, temp_rm_idx))
                        rm_idx = np.unique(rm_idx)
                        # Reminder: np.where(boolean_array) will return a tuple of the indexes of where there
                        # are True values in the boolean_array. len of the tuple = ndims. So the first item in
                        # tuple is an flattened array of the axis0 indexes of all the values in boolean_array
                        # that are True.
                    # remove samples that contain values that are too high or low
                    if key in clip_keys:
                        temp_rm_idx = np.unique(np.where(temp > clip_values[1])[0])
                        temp_rm_idx += n_new_samples
                        rm_idx = np.concatenate((rm_idx, temp_rm_idx))

                        temp_rm_idx = np.unique(np.where(temp < clip_values[0])[0])
                        temp_rm_idx += n_new_samples
                        rm_idx = np.concatenate((rm_idx, temp_rm_idx))
                        
                        rm_idx = np.unique(rm_idx)
                    break
                else:
                    key_count += 1
            if key_count == len(data.items()):
                raise ValueError(f'The file "{data_file}", does not have the key "{desired_key}"')
            
            # combine the data is there were multiple files
            if output_data[idx] is not None:
                output_data[idx] = np.concatenate((output_data[idx], temp), axis=0)
            else:
                output_data[idx] = temp
        # update the value so the indexes for removing nan's/inf's are correct
        n_new_samples += temp.shape[0] #TODO need to check that this works with multiple files

    for i in range(len(output_data)):
        # remove the nan's and inf's
        if remove_non_num is True:
            rm_idx = np.unique(rm_idx)
            output_data[i] = np.delete(output_data[i], rm_idx, axis=0)

        # check that all of the arrays are the same shape
        assert output_data[0].shape[0] == output_data[i].shape[0]

    return output_data

class PlacementShiftFileDataset(Dataset):
    """
    Same as PlacementShiftDataset, but the data files are read at run time instead of being loaded in offline.
    This assumes that all of the unwanted samples have already been removed (see prep_placing_train_data).
    """
    def __init__(self, image_files, ee_files, label_files, root_dir, 
                labels_mu, labels_std,  max_rot, image_mu, image_std,
                image_channel_mu=None, image_channel_std=None, transform=None,
                multi_classifier=False, binary_classifier=False, use_curriculum=False,
                class_interval=0.01, max_class_value=0.1, min_class_value=0.0,
                return_indices=False):
        """
        Inputs:
            filenames (list): list of strings of the file path to each individual sample
            image_files (list): list of filepaths to the depth image data should be
                4 dims -> (batch_size, H, W, D) where D = 2
            ee_files (list): list of filepaths to the end effector poses should be
                2 dims -> (batch_size, 7) where each row is ee pose
            label_files(list): list of the filepaths to the data containing the delta values
                from the final end effector pose to the final pose of the object's center.
                Should be 4 dims (batch_size, x,y,z) 
            root_dir (str): Directory with all of the individual data samples 
                (just using to get normalzing values)
            labels_mu (torch.tensor): the mean value of the labels for normalizing
            labels_std (torch.tensor): the standard deviation of the labels for normalizing
            image_mu (torch.tensor): the mean value of the images, pixel-wise
            image_std (torch.tensor): the standard deviation of the images, pixel-wise 
            image_channel_mu (torch.tensor): the mean value of the images, channel-wise
            image_channel_std (torch.tensor): the standard deviation of the images, channel-wise 
            max_rot (float): the maximum rotation in radians for normalizing the ee data
            transform (callable, optional): Optional transform to be applied
                on a sample (only the image input).
            multi_classifier (bool): if True then return an extra label for each sample
                that specifies what class interval it is in
            binary_classifier (bool): if True then return an extra label that says
                whether a sample is within some distance
            use_curriculum (bool): whether you plan to use curriculum learning.
                YOU WILL NEED TO UPDATE THE DATASET MANUALLY FOR THIS
                NOTE: this has only been tested with the classifier
            class_interval (float): The interval to use when splitting the labels into classes.
                0 - class_interval is class 1, class_interval - 2*class_interval is class 2, etc.
            max_class_value (float): Label all samples that have a distance value greater than
                this as the last class
            min_class_value (float): Label all samples that have a distance value less than
                this as the first class. If zero than this is ignored and stops at zero.
        """
        self.image_files = image_files
        self.ee_files = ee_files
        self.label_files = label_files
        assert len(self.image_files) == len(self.ee_files) and len(self.image_files) == len(self.label_files)

        # Values for normalizing dataset
        self.labels_mu = labels_mu
        self.labels_std = labels_std
        self.image_mu = image_mu
        self.image_std = image_std
        self.image_channel_mu = image_channel_mu
        self.image_channel_std = image_channel_std
        self.max_rot = max_rot

        self.transform = transform
        self.multi_classifier = multi_classifier
        self.binary_classifier = binary_classifier
        self.use_curriculum = use_curriculum

        self.return_indices = return_indices

        """#TODO a few ways to do the curriculum:
        - leave any samples that don't have current curriculum labels out and
            add in as curriculum changes (multiclass classification). The
            size of the dataset changes each time you update curriculum
        - similar to above, but have a set 'mini-batch'(sub set of the dataset) size
            and sample from the entire dataset. The likelihood of sampling from the difficult
            samples is small at first and becomes more likely as the curriculum changes.
            You could change the dataset size, or keep it fixed.
            This is a multi-class classification problem
        - have a binary classifier that says whether a sample is within some error
            and keep decreasing the error values (and update the labels of samples)
            Data set is always the same size, but may cause issues since the labels 
            of a sample can change
            # will only having samples in the first and last classes cause issues?
            # Maybe we should network output size as you change the curriculum?
        """
        if self.multi_classifier:
            assert self.binary_classifier is not True
            # assign all of the samples a class label
            dataset_labels = get_dataset_labels(label_files=self.label_files,
                                                class_interval=class_interval,
                                                max_class_value=max_class_value,
                                                min_class_value=min_class_value
            )
            self.class_labels, self.near_labels, self.class_intervals, self.class_weights = dataset_labels
            # for multiclass classification, withholding classes
            # add the data that is in the first and last classes
            self.indexes = self.class_labels[0]
            key = list(self.class_labels.keys())[-1]
            self.indexes += self.class_labels[key]
            self.length = len(self.indexes)
            # keep track of which classes are being used (1.0 is used, 0.0 is not used)
            self.classes_used = [0.0 for i in range(len(self.class_intervals))]
            self.classes_used[0] = 1.0
            self.classes_used[-1] = 1.0
            
            # if sample:
            #     # need to decide how to sample from the dictionary and not reuse samples
            #     items  = [["item1", 0.2], ["item2", 0.3], ["item3", 0.45], ["item4", 0.05]
            #     elems = [i[0] for i in items]
            #     probs = [i[1] for i in items]
            #     trials = 1000
            #     numpy.random.choice(items, trials, replace=False, p=probs)
            #     #for sampling based curriculum
            #     # i think this makes sense for outputting distributions, maybe not classification
        elif self.binary_classifier:
            # assign all of the samples a class label
            dataset_labels = get_dataset_labels(label_files=self.label_files,
                                                class_interval=class_interval,
                                                max_class_value=max_class_value,
                                                min_class_value=min_class_value
            )
            self.class_labels, self.near_labels, self.class_intervals, self.class_weights = dataset_labels
            # make the list of indexes and rewrite labels to be all ones (NOTE: can make this more efficient)
            self.indexes = list(np.arange(len(self.near_labels))) #TODO might be better to just keep this as an array
            self.near_labels = list(np.ones(len(self.near_labels)))
            key = list(self.class_labels.keys())[0]
            for i in self.class_labels[key]:
                self.near_labels[i] = 0.0 
            self.length = len(self.indexes)
            # keep track of which classes are being used (1.0 is used, 0.0 is not used)
            self.classes_used = [1.0 for i in range(len(self.class_intervals))]
            # use update_dataset_binary to change what labels should be within range
        else:
            # regression problem to predict distance values directly
            self.length = len(self.label_files)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.use_curriculum:
            idx = self.indexes[index]
        else:
            idx = index
        image = np.load(f'{self.image_files[idx]}', allow_pickle=True)
        ee_pose = np.load(f'{self.ee_files[idx]}', allow_pickle=True)[3:] # just the rotation
        label = np.load(f'{self.label_files[idx]}', allow_pickle=True)
        assert label.ndim == 1

        label = label[:3]
        # just use the x and z coordinates NOTE change this if you include y or rotations
        label = np.delete(label, 1)
        assert label.shape[0] == 2

        if self.multi_classifier or self.binary_classifier:
            near_label = torch.tensor([self.near_labels[idx]])

        # Only keep the pitch rotation. Roll is captured in images and there is no yaw rotation.
        ee_pose = utils.quat_to_rpy(ee_pose)[1]
        ee_pose = np.array([ee_pose]).flatten() # make sure format is correct
        
        # Normalize the data
        # ee_pose = ee_pose / self.max_rot #TODO might want to normalize this, but need to figure out normalization value
        label =  (label - self.labels_mu) / self.labels_std 
        # normalize the images channel-wise instead of pixel-wise
        if self.image_channel_mu is not None:
            # TODO 
            image = (image * self.image_std) + self.image_mu
            image = (image - self.image_channel_mu) / self.image_channel_std

        # convert to torch tensors
        image = torch.from_numpy(image).permute(2,0,1)
        ee_pose = torch.from_numpy(ee_pose)
        label = torch.from_numpy(label)
        if self.return_indices:
            index = torch.FloatTensor([index]) # NOTE: this might be unnecessary

        if self.transform is not None:
            image = self.transform(image)

        if self.multi_classifier or self.binary_classifier:
            if self.return_indices:
                return image.float(), ee_pose.float(), label.float(), near_label.float(), index
            else:
                return image.float(), ee_pose.float(), label.float(), near_label.float()
        else:
            return image.float(), ee_pose.float(), label.float()

    def update_dataset(self, class_key):
        """
        For updating the dataset when using curriculum learning and multiclass
        classification to predict what interval the distance of the sample is in.
        Adds an entire class when it has been withhelds.
        """
        if self.classes_used[class_key] == 0.0:
            self.classes_used[class_key] = 1.0
        else:
            raise ValueError('This class had already been added')
        self.indexes += self.class_labels[class_key]
        self.length = len(self.indexes)

    def update_dataset_binary(self, class_key):
        """
        For updating the dataset when using  binary classification to predict
        whether sample is within some distance. The dataset is always the same,
        only the labels are updated. Can use this once to set what the distance is
        or call it mutliple times for curriculum learning
        Inputs:
            class_key (int): the key of the label to set to 0.0. All keys above this will
            also be set to zero (i.e. set all distance values above given key are set to False)
        """
        counter = 0
        key = list(self.class_labels.keys())[counter]
        # TODO should make changing all of them an option, so computation isn't wasted if you already changed a class
        while key <= class_key:
            for i in self.class_labels[key]:
                #less than tolerance is 1.0, greater than is 0.0
                self.near_labels[i] = 0.0
            counter += 1
            key = list(self.class_labels.keys())[counter]
        self.length = len(self.indexes) #this isn't necessary

    def update_dataset_sampling(self, class_key):
        """
        For updating the dataset when using curriculum learning and multiclass
        classification to predict what interval the distance of the sample is in.
        This uses a fixed dataset size and increases the likelihood of sampling from
        the given class.
        """
        pass
        # self.indexes += self.class_labels[class_key]
        # if shuffle:
        #     random.shuffle(self.indexes)
        # self.length = len(self.indexes)

    def remove_dataset_class(self, class_key, indexes=None):
        """
        Remove a class from the dataset, or the given indexes
        
        Inputs:
            class_key (int): the key of the class to remove from the dataset.
        """
        if indexes is not None:
            if indexes.ndim == 2:
                assert indexes.shape[1] == 1
                indexes = indexes.flatten()
            else:
                assert indexes.ndim == 1
        else:
            if self.classes_used[class_key] == 1.0:
                self.classes_used[class_key] = 0.0
            else:
                raise ValueError('This class had already been removed')
            indexes = np.array(self.class_labels[class_key])
        self.indexes = list(np.delete(np.array(self.indexes), indexes))
        self.length = len(self.indexes)


def train_test_split_filenames(root_dir, file_prefixes=['image_data','ee_data', 'label'],
                               percentage=0.8, shuffle=False, file_suffix='.npy',
                               sss=True, class_interval=0.1, max_class_value=0.01,
                               min_class_value=0.0, random_state=None):
    """
    Get lists of the filepaths to the data files in root_dir
    Inputs:
        root_dir (str): Directory with all of the individual data samples
        file_prefixes (list): list of strings that are the prefixes of the data files
            to look for and add to the dataset
        percentage (float): percentage of the data set to use for the training set
        shuffle (bool): whether or not to shuffle the data before splitting
        file_suffic (str): suffix of the files to look for
        sss (bool): whether to use StratifiedShuffleSplit
        class_interval (float): The interval to use when splitting the labels into classes.
            0 - class_interval is class 1, class_interval - 2*class_interval is class 2, etc.
        max_class_value (float): Label all samples that have a distance value greater than
            this as the last class
        min_class_value (float): Label all samples that have a distance value less than
            this as the first class. If zero than this is ignored and stops at zero.
        random_state (int): value to use for Stratified Shuffle Split
    Outputs:
        output (dict): a dict the same length as file_prefixes and that uses the 
            prefixes as the keys. Each of the prefixes has a "train' and 'test' key.
            The values are lists containing the list of strings of the filepaths
    Refs:
     - https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
    """
    file_prefixes = file_prefixes
    filenames = []
    for i in range(len(file_prefixes)):
        filenames.append(utils.sub_files(root_dir,
                                         file_suffix=file_suffix,
                                         prefix=file_prefixes[i])
        )
        if i > 0:
            # make sure the number of samples is the same across data types
            assert len(filenames[0]) == len(filenames[i])
    assert len(filenames) == len(file_prefixes)
    assert percentage <= 1

    if sss:
        # assuming the labels are the third thing in list
        label_info = get_dataset_labels(filenames[2],
                                        class_interval,
                                        max_class_value,
                                        min_class_value,
                                        message='Files left to parse for train-test split'
        )
        _, near_labels, __, ___ = label_info
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-percentage), random_state=random_state)
        fake_X = np.zeros(len(filenames[0])) #placeholder for training data
        train_idx, test_idx = next(sss.split(fake_X, near_labels))
        output = {}
        for k in range(len(filenames)):
            output[file_prefixes[k]] = {}
            output[file_prefixes[k]]['train'] = [filenames[k][i] for i in train_idx]
            output[file_prefixes[k]]['test'] = [filenames[k][i] for i in test_idx]

    else:
        # shuffle the data if flag is set
        if shuffle:
            for j in range(len(filenames)):
                random.shuffle(filenames[j])

        # split the data
        output = {}
        train_len = int(percentage * len(filenames[0]))
        for k in range(len(filenames)):
            output[file_prefixes[k]] = {}
            output[file_prefixes[k]]['train'] = filenames[k][:train_len]
            output[file_prefixes[k]]['test'] = filenames[k][train_len:]

    return output


def get_dataset_labels(label_files, class_interval, max_class_value, min_class_value=0.0,
                       consider_y=False, l2_dist=True, message=None):
    """
    Reads through all the data samples and assigns class labels to the samples
    depending on the distance from the object's landing pose to the ee release pose
    NOTE: This is pretty expensive
    Inputs:
        label_files (list): list of the filepaths to the data containing the delta values
            from the final end effector pose to the final pose of the object's center.
            Should be 2 dims (batch_size, (x,y,z)), the second dimension can also have a
            near_label after the z i.e (x,y,z,l)
        class_interval (float): The interval to use when splitting the labels into classes.
            0 - class_interval is class 1, class_interval - 2*class_interval is class 2, etc.
        max_class_value (float): Label all samples that have a distance value greater than this as
            the last class
        min_class_value (float): Label all samples that have a distance value less than
            this as the first class. If zero than this is ignored and stops at zero.
        consider_y (bool): whether to consider y in the label assigment
        l2_dist (bool): whether to use the L2 norm as the distance metric for assigning
            labels. NOTE: this is the only option right now
    Outputs:
        class_labels (dict): A dictionary where the keys are the class labels and
            the values are all of the sample indexes that have that label.
            CLASS LABELS GO FROM LARGER DISTANCE TO SMALLER DISTANCE
        near_labels (list): List of length N_samples, that has the class label for the
            corresponding sample (same order as label_files)
        class_intervals (list): list containing the distance intervals that define the classes
        class_weights (list): list that contains the number of samples for each class
    """
    # get the different distance intervals to use as class labels
    class_intervals = []
    if min_class_value == 0.0:
        interval = 1
    else:
        interval = 0
    while ((interval * class_interval) + min_class_value) < max_class_value:
        class_intervals.append(((interval * class_interval) + min_class_value))
        interval += 1
    # include the label for samples larger than the max distance
    class_intervals.append(max_class_value)
    # reverse the list to go from largest distance to smallest
    class_intervals = class_intervals[::-1]

    if message is None:
        message = 'Files left to parse in dataset'
    class_labels = {i : [] for i in range(len(class_intervals) + 1)}
    assert len(class_labels) == len(class_intervals) + 1
    near_labels = [] 
    for index in tqdm(range(len(label_files)), file=sys.stdout, desc=message):
        label = np.load(f'{label_files[index]}', allow_pickle=True)
        # don't need the near_label (older format) on the label file
        if label.ndim == 1:
            if label.shape[0] == 4:
                label = label[:-1]
        elif label.ndim == 2:
            if label.shape[1] == 4:
                label = label[:,:-1]
        else:
            raise ValueError

        # just use the x and z coordinates
        if not consider_y:
            if label.ndim == 1:
                label = np.delete(label, 1)
            elif label.ndim == 2:
                label = np.delete(label, 1, axis=1)
            else:
                raise ValueError()

        if l2_dist:
            dist = np.linalg.norm(label)
            for j in range(len(class_intervals)):
                if dist > class_intervals[j]:
                    class_labels[j].append(index)
                    near_labels.append(j)
                    break
                elif j == (len(class_intervals) - 1):
                    key = list(class_labels.keys())[-1]
                    class_labels[key].append(index)
                    near_labels.append(key)
                    break
                else:
                    pass
        else:
            for center in modes:
                dist = np.linalg.norm(label)
            raise ValueError("Didn't implement non-L2 based labeling")

    # unit test
    check = 0
    class_weights = []
    for key, value in class_labels.items():
        length = len(value)
        assert length != 0 # check that a class is not empty
        class_weights.append(length)
        check += length
    length = len(label_files)
    assert check == length
    assert len(near_labels) == length

    return class_labels, near_labels, class_intervals, class_weights
