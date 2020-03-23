import numpy as np
import glob
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import rnns.utils as utils

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
                img = utils.letterbox_image(img, self.reso)
                temp_list.append(img)
            # For png images
            for filename in sorted(glob.glob('{}/*.png'.format(dirp)), key=utils.numerical_sort):
                img = cv2.imread(filename, 1)
                #reshape images
                img = utils.letterbox_image(img, self.reso)
                temp_list.append(img)
            # For jpeg images
            for filename in sorted(glob.glob('{}/*.jpeg'.format(dirp)), key=utils.numerical_sort):
                img = cv2.imread(filename, 1)
                #reshape images
                img = utils.letterbox_image(img, self.reso)
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
                        dtype=np.uint8)[combos[i]])
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
    test_len = int(percentage*(data.shape[0]))

    test_data = data[:-test_len]
    test_labels = labels[:-test_len]

    train_data = data[-test_len:]
    train_labels = labels[-test_len:]

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
