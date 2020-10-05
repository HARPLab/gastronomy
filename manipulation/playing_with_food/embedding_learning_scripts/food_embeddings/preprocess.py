import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle 

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import food_embeddings.utils as utils

class SiameseFoodDataset(Dataset):
    """
    Dataset for food embeddings, returns a pair of positive or negative samples
    TODO need to wait until we decie how to deal with the labels

    Inputs:
        shuffle_pairs(bool): Set to False to have fixed pairs during run time
        This still makes different pairs each time you run this though. Only fixed per instance
    Outputs(for __get_item):
        sample1(tuple): Contains the audio, image, and force data for the sample, as well as the label
        sample2(tuple): Contains the audio, image, and force data for the 2nd sample, as well as the label
        sample_id(int): identifier that says whether the two samples are negative or positive
            positive is 1, negative is 0
    Ref: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, audio_data, image_data, force_data, labels, audio_transform=None,
                 image_transform=None, force_transform=None, label_transform=None,
                 shuffle_pairs=True):
        self.length = labels.shape[0] 
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.force_transform = force_transform
        self.label_transform = label_transform

        # convert the data to correct data type
        if torch.is_tensor(audio_data):
            self.audio_data = audio_data.type(torch.float32)
        else:
            self.audio_data = torch.from_numpy(audio_data).type(torch.float32)
        if torch.is_tensor(image_data):
            self.image_data = image_data.type(torch.float32) / 255.0
        else:
            # swap dimensions to same as tensors (BxTxHxWxD) -> (BxTxDxHxW)
            self.image_data = torch.from_numpy(image_data).permute(0,1,4,2,3).type(torch.float32) / 255.0
        if torch.is_tensor(force_data):
            self.force_data = force_data.type(torch.float32)
        else:
            self.force_data = torch.from_numpy(force_data).type(torch.float32)
        if torch.is_tensor(labels):
            self.labels = labels.type(torch.int64)
        else:
            self.labels = torch.from_numpy(labels).type(torch.int64)

        # get all the unique labels, and the indices to each individual label
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # make the fixed pairs if flag is set
        self.shuffle_pairs = shuffle_pairs
        if not self.shuffle_pairs:
            random_state = np.random.RandomState(29)
            # Loop through even numbered samples and assign positve samples to them
            positive_pairs = [[i,
                            random_state.choice(self.label_to_indices[self.labels[i].item()]),
                            1]
                            for i in range(0, self.length, 2)]
            # Loop through odd numbered samples and assign negative samples to them
            negative_pairs = [[i,
                            random_state.choice(self.label_to_indices[
                                                    np.random.choice(
                                                        list(self.labels_set - set([self.labels[i].item()]))
                                                    )
                                                ]),
                            0]
                            for i in range(1, self.length, 2)]
            # combine the pairs, postive pairs have a 1 in the end and negative pairs have a 0
            self.pairs = positive_pairs + negative_pairs
        # self.pairs is list of indexes, each item in list has length 3
        else:
            pass

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Get random positive and negative pairs
        if self.shuffle_pairs:
            sample_id = np.random.randint(0,2) # 0 for negative sample, 1 for positive sample
            audio1 = self.audio_data[index]
            image1 = self.image_data[index]
            force1 = self.force_data[index]
            label1 = self.labels[index]
            # get a random positive pair
            if sample_id == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1.item()])
            # get a random negative pair
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1.item()])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            audio2 = self.audio_data[siamese_index]
            image2 = self.image_data[siamese_index]
            force2 = self.force_data[siamese_index]
            label2 = self.labels[siamese_index]
        # use set pairs if flag is not set
        else:
            audio1 = self.audio_data[self.pairs[index][0]]
            image1 = self.image_data[self.pairs[index][0]]
            force1 = self.force_data[self.pairs[index][0]]
            label1 = self.labels[self.pairs[index][0]]
            
            audio2 = self.audio_data[self.pairs[index][1]]
            image2 = self.image_data[self.pairs[index][1]]
            force2 = self.force_data[self.pairs[index][1]]
            label2 = self.labels[self.pairs[index][1]]

            sample_id = self.pairs[index][2]

        if self.audio_transform is not None:
            audio1 = self.audio_transform(audio1)
            audio2 = self.audio_transform(audio2)
        if self.image_transform is not None:
            image1 = self.image_transform(image1)
            image2 = self.image_transform(image2)
        if self.force_transform is not None:
            force1 = self.force_transform(force1)
            force2 = self.force_transform(force2)
        if self.label_transform is not None:
            label1 = self.label_transform(label1)
            label2 = self.label_transform(label2)

        return (audio1, image1, force1, label1), (audio2, image2, force2, label2), sample_id

class TripletFoodDataset(Dataset):
    """
    Dataset for food embeddings, returns a triplet of samples with an anchor, positive and negative sample
    TODO need to wait until we decie how to deal with the labels

    Inputs:
        shuffle_triplets(bool): Set to False to have fixed triplets during run time
        This still makes different triplets each time you run this though. Only fixed per instance
    Outputs(for __get_item):
        sample1(tuple): Contains the anchor sample of audio, image, and force data, as well as the label
        sample2(tuple): Contains the positive sample of audio, image, and force data, as well as the label
        sample3(tuple): Contains the negative sample of audio, image, and force data, as well as the label
    Ref: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """
    def __init__(self, audio_data, image_data, force_data, labels, audio_transform=None,
                 image_transform=None, force_transform=None, label_transform=None,
                 shuffle_triplets=True):
        self.length = labels.shape[0] 
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.force_transform = force_transform
        self.label_transform = label_transform

        # convert the data to correct data type
        if torch.is_tensor(audio_data):
            self.audio_data = audio_data.type(torch.float32)
        else:
            self.audio_data = torch.from_numpy(audio_data).type(torch.float32)
        if torch.is_tensor(image_data):
            self.image_data = image_data.type(torch.float32) / 255.0
        else:
            # swap dimensions to same as tensors (BxTxHxWxD) -> (BxTxDxHxW)
            self.image_data = torch.from_numpy(image_data).permute(0,1,4,2,3).type(torch.float32) / 255.0
        if torch.is_tensor(force_data):
            self.force_data = force_data.type(torch.float32)
        else:
            self.force_data = torch.from_numpy(force_data).type(torch.float32)
        if torch.is_tensor(labels):
            self.labels = labels.type(torch.int64)
        else:
            self.labels = torch.from_numpy(labels).type(torch.int64)

        # get all the unique labels, and the indices to each individual label
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # make the fixed triplets if flag is set
        self.shuffle_triplets = shuffle_triplets
        if not shuffle_triplets:
            random_state = np.random.RandomState(29)
            # loop through each sample and assign a positive and then negative sample to it as list
            triplets = [[i,
                        random_state.choice(self.label_to_indices[self.labels[i].item()]),
                        random_state.choice(self.label_to_indices[
                                                np.random.choice(
                                                    list(self.labels_set - set([self.labels[i].item()]))
                                                )
                                            ])
                        ]
                        for i in range(self.length)]
            self.triplets = triplets

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Get random positive and negative triplets
        if self.shuffle_triplets:
            audio1 = self.audio_data[index]
            image1 = self.image_data[index]
            force1 = self.force_data[index]
            label1 = self.labels[index]
            # get a random positive pair
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1.item()])
            audio2 = self.audio_data[positive_index]
            image2 = self.image_data[positive_index]
            force2 = self.force_data[positive_index]
            label2 = self.labels[positive_index]

            # get a random negative pair
            negative_label = np.random.choice(list(self.labels_set - set([label1.item()])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            audio3 = self.audio_data[negative_index]
            image3 = self.image_data[negative_index]
            force3 = self.force_data[negative_index]
            label3 = self.labels[negative_index]

        # use set triplets if flag is not set
        else:
            audio1 = self.audio_data[self.triplets[index][0]]
            image1 = self.image_data[self.triplets[index][0]]
            force1 = self.force_data[self.triplets[index][0]]
            label1 = self.labels[self.triplets[index][0]]
            
            audio2 = self.audio_data[self.triplets[index][1]]
            image2 = self.image_data[self.triplets[index][1]]
            force2 = self.force_data[self.triplets[index][1]]
            label2 = self.labels[self.triplets[index][1]]

            audio3 = self.audio_data[self.triplets[index][2]]
            image3 = self.image_data[self.triplets[index][2]]
            force3 = self.force_data[self.triplets[index][2]]
            label3 = self.labels[self.triplets[index][2]]

        if self.audio_transform is not None:
            audio1 = self.audio_transform(audio1)
            audio2 = self.audio_transform(audio2)
            audio3 = self.audio_transform(audio3)
        if self.image_transform is not None:
            image1 = self.image_transform(image1)
            image2 = self.image_transform(image2)
            image3 = self.image_transform(image3)
        if self.force_transform is not None:
            force1 = self.force_transform(force1)
            force2 = self.force_transform(force2)
            force3 = self.force_transform(force3)
        if self.label_transform is not None:
            label1 = self.label_transform(label1)
            label2 = self.label_transform(label2)
            label3 = self.label_transform(label3)

        return (audio1, image1, force1, label1), (audio2, image2, force2, label2), (audio3, image3, force3, label3)

class RelativeSamplesDataset(Dataset):
    """
    Dataset that returns triplets/pairs of data samples. For a sample to be
    positive, its label must be within +/- of the threshold value. Negative
    otherwise. 

    Inputs:
        data: training data, as a numpy array or torch tensor
        labels: labels for the training data, should be 1-D and same size as data.shape[0]
            Should be some sort of distance metric
        threshold(float): labels values that are within this threshold (including the threshold value)
            are considered positve samples to one another, negative elsewise
            Can also pass in an array/tensor the same shape as labels, with the threshold value per sample
        triplet(bool): return a triplet if True or a pair (w/ target) when False
    """
    def __init__(self, data, labels, threshold, data_transform=None,
                 label_transform=None, triplet=True, image=True):
        print('Creating RelativeSampleDataset object')
        #assert labels.ndim == 1
        self.length = labels.shape[0]
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.triplet = triplet

        if torch.is_tensor(data):
            self.data = data.type(torch.float32)
        else:
            self.data = torch.from_numpy(data).type(torch.float32)
        if torch.is_tensor(labels):
            self.labels = labels.type(torch.float32)
        else:
            self.labels = torch.from_numpy(labels).type(torch.float32)
        #import pdb; pdb.set_trace()
        if image:
            #import pdb; pdb.set_trace()
            self.data = self.data.permute(0,1,4,2,3) / 255.0 # convert to values between 1-0 if flag is set
            #import pdb; pdb.set_trace()
        if hasattr(threshold, "__len__"):
            if torch.is_tensor(threshold):
                self.threshold = threshold.type(torch.float32)
            else:
                self.threshold = torch.from_numpy(threshold).type(torch.float32)
        else:
            self.threshold = threshold

        self.triplet_options = []
        full_set = set(np.arange(self.length))

        #import pdb; pdb.set_trace()
        for i in range(self.length): #iterate through all the data samples
            if not hasattr(self.threshold, "__len__"): # NOTE: this prob isn't too efficient
                thresh = self.threshold            
            else:
                thresh = self.threshold[i]

            # different handling of data depending if label is scalar or vector
            if thresh != None:
                if self.labels[i].size() == torch.Size([]): # scalar labels
                    temp_diff = torch.abs(self.labels - self.labels[i]) 
                    #import pdb; pdb.set_trace()
                    temp_pos_set = set(torch.where(temp_diff <= thresh)[0].numpy()) - set([i]) #remove sample itself
                    assert not utils.is_empty(temp_pos_set)
                    temp_neg_set = full_set - temp_pos_set - set([i])
                    #TODO probably only need to save one of these, and make the other at run time since you know the index and pos
                    self.triplet_options.append([temp_neg_set, temp_pos_set])              
                
                else: #label is a vector                    
                    temp_diff = []
                    for j in range(self.length):
                        diff = np.linalg.norm(labels[i] - labels[j]) #compute L2 norm b/w current sample and all other samples
                        temp_diff.append(diff) #store L2 norms in temp_diff                
                    temp_diff = np.array(temp_diff)
                    temp_diff = torch.from_numpy(temp_diff).type(torch.float32)
                    #import pdb; pdb.set_trace()
                    temp_pos_set = set(torch.where(temp_diff <= thresh)[0].numpy()) - set([i]) #remove sample itself
                    assert not utils.is_empty(temp_pos_set)
                    temp_neg_set = full_set - temp_pos_set - set([i])
                    #TODO probably only need to save one of these, and make the other at run time since you know the index and pos
                    self.triplet_options.append([temp_neg_set, temp_pos_set])
                    #import pdb; pdb.set_trace()
            
            elif thresh == None: #is there is no thresh, and we're just training w/ veg type labels
                temp_pos_set = set(torch.where(self.labels == self.labels[i])[0].numpy()) - set([i])
                temp_neg_set = full_set - temp_pos_set - set([i])
                self.triplet_options.append([temp_neg_set, temp_pos_set])
                #import pdb; pdb.set_trace()                       

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        anchor = self.data[index]
        label = self.labels[index]
        
        if self.triplet:
            neg_idx = np.random.choice(list(self.triplet_options[index][0]))
            pos_idx = np.random.choice(list(self.triplet_options[index][1]))
            negative = self.data[neg_idx]
            positive = self.data[pos_idx]
            return anchor, positive, negative, label
        else:
            sample_id = np.random.randint(0,2) # 0 for negative sample, 1 for positive sample
            other_idx = np.random.choice(list(self.triplet_options[index][sample_id]))
            other_sample = self.data[other_idx]
            return anchor, other_sample, sample_id, label

class SilhouetteSamplesDataset(Dataset):
    """
    Dataset that returns triplets/pairs of data samples. Directly loads in saved pos/neg sets for each sample

    Inputs:
        data: training data, as a numpy array or torch tensor
        triplet(bool): return a triplet if True or a pair (w/ target) when False
    """
    def __init__(self, data, arr_pos_neg_dicts, train_test_flag, train_inds, test_inds, data_transform=None,
                triplet=True, image=True):
        print('Creating SilhouetteSamplesDataset object')
        #assert labels.ndim == 1
        self.length = arr_pos_neg_dicts.shape[0]
        self.data_transform = data_transform
        self.triplet = triplet

        if torch.is_tensor(data):
            self.data = data.type(torch.float32)
        else:
            self.data = torch.from_numpy(data).type(torch.float32)
        
        if image:
            self.data = self.data.permute(0,1,4,2,3) / 255.0 # convert to values between 1-0 if flag is set
        
        self.triplet_options = []
        full_set = set(np.arange(self.length))

        #import pdb; pdb.set_trace()
        for i in range(self.length): #iterate through all the data samples  
            if train_test_flag == 'train':   
                #import pdb; pdb.set_trace()
                temp_pos_set = set(arr_pos_neg_dicts[i]['positive']) - set([i]) - set(test_inds)
                assert not utils.is_empty(temp_pos_set)
                new_pos_idxs = []
                for j in list(temp_pos_set): #convert inds from whole dataset to train inds
                    new_pos_idx = np.where(train_inds == j)[0][0]
                    new_pos_idxs.append(new_pos_idx)
                temp_pos_set = set(new_pos_idxs)
                #import pdb; pdb.set_trace()

                temp_neg_set = set(arr_pos_neg_dicts[i]['negative']) - set(test_inds)
                #import pdb; pdb.set_trace()
                new_neg_idxs = []
                for j in list(temp_neg_set):
                    new_neg_idx = np.where(train_inds == j)[0][0]
                    new_neg_idxs.append(new_neg_idx)
                temp_neg_set = set(new_neg_idxs)
                #import pdb; pdb.set_trace()

            elif train_test_flag == 'test':
                #import pdb; pdb.set_trace()
                temp_pos_set = set(arr_pos_neg_dicts[i]['positive']) - set([i]) - set(train_inds)
                assert not utils.is_empty(temp_pos_set)
                new_pos_idxs = []
                for j in list(temp_pos_set): #convert inds from whole dataset to train inds
                    new_pos_idx = np.where(test_inds == j)[0][0]
                    new_pos_idxs.append(new_pos_idx)
                temp_pos_set = set(new_pos_idxs)
                
                temp_neg_set = set(arr_pos_neg_dicts[i]['negative']) - set(train_inds)
                new_neg_idxs = []
                for j in list(temp_neg_set):
                    new_neg_idx = np.where(test_inds == j)[0][0]
                    new_neg_idxs.append(new_neg_idx)
                temp_neg_set = set(new_neg_idxs)
                #import pdb; pdb.set_trace()
            
            #TODO probably only need to save one of these, and make the other at run time since you know the index and pos
            self.triplet_options.append([temp_neg_set, temp_pos_set])  
            
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        anchor = self.data[index]
        label = 0 # just a placeholder for now to avoid larger code changes
        
        if self.triplet:
            neg_idx = np.random.choice(list(self.triplet_options[index][0]))
            pos_idx = np.random.choice(list(self.triplet_options[index][1]))
            negative = self.data[neg_idx]
            positive = self.data[pos_idx]
            return anchor, positive, negative, label
        else:
            sample_id = np.random.randint(0,2) # 0 for negative sample, 1 for positive sample
            other_idx = np.random.choice(list(self.triplet_options[index][sample_id]))
            other_sample = self.data[other_idx]
            return anchor, other_sample, sample_id, label


class BalancedBatchSampler(BatchSampler):
    """
    Batch sampler that tries to to sample data in balanced way?
    Inputs:
        labels: 1D numpy array or torch tensor containing the labels of the data
        n_classes(int): the number of classes to use
        n_samples(int): the number of samples to use
    NOTE:
        - DONT USE THIS IF YOU ARE DEBUGGING, data is always shuffled
        - IDK how 'balanced' this is or how necessary?
          It is still just randomly sampling the data and reshuffles it whenever
          most of the data samples in a class have already been used
    Ref: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """
    def __init__(self, labels, n_classes, n_samples, shuffle=True):
        self.labels = labels
        # Get all the unique labels, and indices of where each label occurs
        if torch.is_tensor(self.labels):
            self.labels_set = list(set(self.labels.numpy()))
            self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                    for label in self.labels_set}
        else:
            self.labels_set = list(set(self.labels))
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}       
        # shuffle the indices
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # keep track of how many times each labels has been used
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset_size = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.dataset_size:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                # get the indices from the specific class and keep track of how many were used
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                     class_]+self.n_samples]
                )
                self.used_label_indices_count[class_] += self.n_samples
                # if the next iteration will use the remainder of this class of labels, shuffle the data
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        # return the number of batches(floor division, sop drop_last=True)
        return self.dataset_size // self.batch_size


def get_food_dataset(data_path, img_size=(None, None)):
    """
    Gets all of the training samples in a directory that is structured as follows:
        [vegtable type] > [slice_type] > [trail_number]
    Inputs:
        data_path(str): path to the directory containing all the subdirectories of data
        img_size(tuple): tuple of ints of size to resize images to. Leave as None to
            resize all the images to the minimum dimensions out of the dataset
    """
    audio_data = []
    image_data = []
    force_data = []
    labels = []
    veg_types = utils.sub_dirs(data_path)
    for veg_name in veg_types:
        cut_type = utils.sub_dirs(veg_name)
        for cut_name in cut_type:
            trials = utils.sub_dirs(cut_name)
            for trial in trials:
                temp_audio, temp_forces, temp_images = get_sample(trial, img_size=img_size)
                audio_data.append(temp_audio)
                image_data.append(temp_images)
                force_data.append(temp_forces)
                labels.append(trial) # store the file path, which contains directory structure mentioned above

    # TODO might need to adjuect force and labels here
    audio_data = utils.pad_and_stack(audio_data, pad_mode='constant')
    image_data = np.stack(image_data, axis=0)
    force_data = utils.pad_and_stack(force_data, pad_mode='constant')
    labels = np.asarray(labels, dtype=object)
    labels = path2label(labels) # TODO temporary labels converter
    assert audio_data.shape[0] == image_data.shape[0]
    assert audio_data.shape[0] == force_data.shape[0]
    assert audio_data.shape[0] == labels.shape[0]

    return audio_data, image_data, force_data, labels

def get_sample(sample_path, img_size=(None,None)):
    """
    Return a single data sample. Assuming 'sample_path' is a directory that contains
    folders with  the names 'images', 'audio', and 'forces'
    Inputs:
        sample_path(str): path to the directory containing the sample data
        img_size(tuple): tuple of ints of size to resize images to. Leave as None to
            resize all the images to the minimum dimensions out of the dataset
    Output is a list of len 3
    """
    data_types = utils.sub_dirs(sample_path)
    output_data = []
    for data_type in data_types:
        #TODO need to update these names/read method and make sure images are in a consistent order
        if 'audio' in data_type:
            temp_file = utils.sub_files(data_type, '.npy')
            output_data.append(np.load(temp_file[0]))
        elif 'images' in data_type:
            output_data.append(get_images(data_type, img_size=img_size))
        elif 'forces' in data_type:
            temp_file = utils.sub_files(data_type, '.npy')
            output_data.append(np.load(temp_file[0]))
        else:
            raise ValueError(f"'{data_type}' is not an expected directory")
    
    return output_data

def get_images(directory, image_type='.png', img_size=(None,None)):
    """
    Get all of the images (of given type) in the given directory.
    Sorts filenames numerically and alphabetically.
    Inputs:
        directory(str): firectory containing the images
        image_type(str): the suffix of the images to read
        img_size(tuple): tuple of ints of size to resize images to. Leave as None to
            resize all the images to the minimum dimensions out of the dataset
    
    NOTE: Assuming the images are named: 
        - ending_rgb_image, starting_rgb_image, push_#, release_#, grasp_#
    This won't work properly otherwise
    """
    #import pdb; pdb.set_trace()
    filenames = utils.sub_files(directory, image_type)
    if len(filenames) == 1:
        pass
    else:
        filenames = [filenames[-1]] + filenames[:-1] # put the starting rgb image at beginning
    
    output = []
    for filename in filenames:
        img = cv2.imread(filename)
        # remove depth channel if it is there
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        output.append(img[:,:,::-1]) #convert BGR to RGB

    if len(filenames) == 1:
        output = np.array(cv2.resize(src=output[0], dsize=img_size))
    else:
        #check if all of the images are the same size
        dims = [array.shape for array in output]
        dims = np.stack(dims, axis=0)
        max_dims = np.max(dims, axis=0)
        min_dims = np.min(dims, axis=0)
        # resize the images to the smallest size
        if not np.array_equal(max_dims, min_dims):
            for i, image in enumerate(output):
                if img_size[0] is not None:
                    image = cv2.resize(src=image, dsize=img_size)
                else:
                    image = cv2.resize(src=image, dsize=tuple((min_dims[1],min_dims[0])))
                output[i] = np.array(image)
        else:
            pass
        output = np.stack(output, axis=0) # number of images on axis 0 TODO might want to just stack on depth channel
    return output


def train_test_split(data, labels, percentage, shuffle=False):
    """
    Function to split the training data into train and validation sets
    Inputs:
        data(list): list of numpy array containing the training data. The length of this
            list should be the number of training inputs you will be using. Each item in
            list should have same size in dim 0
        labels(list): list of numpy array containing the labels for the training data. The
            length of the list should be the same as the number of labels/prediction you have.
            Each item in list should have same size in dim 0 as data
        percentage - (float): percentage of the data to use for training, rest is for valid
        shuffle - (bool): set flag to true to shuffle data
    Outputs (in this order):
        train_data - (list): training data, same length as input and in same order
        test_data - (list): testing data, same length as input and in same order
        train_labels - (list): labels for train data, same length as input and in same order
        test_labels - (list): labels for test data , same length as input and in same order
    """
    assert type(data) == list and type(labels) == list
    assert data[0].shape[0] == labels[0].shape[0]

    assert percentage <= 1 and percentage >= 0
    train_len = int(percentage*(labels[0].shape[0]))

    # shuffle the data if flag is set
    idx = np.arange(labels[0].shape[0])
    if shuffle is True:
        np.random.shuffle(idx)

    # these are just here for visual use
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    output = [[train_data, test_data], [train_labels, test_labels]]
    for i, thing in enumerate([data, labels]):
        
        # split data, then split labels
        for array in thing:
            array = array[idx] # shuffle if flag was set
            # split the data up
            if percentage == 1:
                output[i][0].append(array) # train
                output[i][1].append(np.array([])) # test
            elif percentage == 0:
                output[i][0].append(np.array([])) # train
                output[i][1].append(array) # test
            else:
                output[i][0].append(array[:train_len]) # train
                output[i][1].append(array[train_len:]) # test
    
    return output[0][0], output[0][1], output[1][0], output[1][1]

def train_test_split_even_by_veg_type(data, labels, shuffle=False):
    """
    Function to split the training data into train and validation sets and make sure at least 1 
    data point from each cut type is in training data
    Note: always assumes 80/20 train/test split (can't change right now)

    Inputs:
        data(list): list of numpy array containing the training data. The length of this
            list should be the number of training inputs you will be using. Each item in
            list should have same size in dim 0
        labels(list): list of numpy array containing the labels for the training data. The
            length of the list should be the same as the number of labels/prediction you have.
            Each item in list should have same size in dim 0 as data
        shuffle - (bool): set flag to true to shuffle data
    Outputs (in this order):
        train_data - (list): training data, same length as input and in same order
        test_data - (list): testing data, same length as input and in same order
        train_labels - (list): labels for train data, same length as input and in same order
        test_labels - (list): labels for test data , same length as input and in same order
    """
    assert type(data) == list and type(labels) == list
    assert data[0].shape[0] == labels[0].shape[0]

    num_samples = data[0].shape[0]
   
    # # shuffle the data if flag is set
    idx = np.arange(labels[0].shape[0])
  
    # these are just here for visual use
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    output = [[train_data, test_data], [train_labels, test_labels]]
    
    all_inds = np.arange(num_samples)
    set_all_inds = set(all_inds)    
    test_inds = np.arange(1, 449, 5) #pick the 2nd sample from each cut type for testing
    set_test_inds = set(test_inds)    
    set_train_inds = set_all_inds - set_test_inds
    train_inds = np.array(list(set_train_inds))
    
    for i, thing in enumerate([data, labels]):
        # split data, then split labels
        for array in thing:
            array = array[idx] # shuffle if flag was set
            
            output[i][0].append(array[train_inds]) # train
            output[i][1].append(array[test_inds]) # test

    #import pdb; pdb.set_trace()
    return output[0][0], output[0][1], output[1][0], output[1][1], train_inds, test_inds

def path2label(paths, labels=['carrot','cucumber','pear','tomato']):
    """
    Convert the file path of a sample to a label
    Using each veg type as a label for know
    NOTE: just using this to make temporary labels for testing
    Inputs:
        paths(array): an array of object strings that contain the
            full file path of the sample, should be shape (N,)
        labels(list): list of strings of the vegtable labels.
            Labels are assigned in order given, e.g. ['carrot','pear']->[0,1]
    """
    # instantiate empty array
    output = np.zeros((paths.shape[0]))
    for i in range(paths.shape[0]):
        dirs = paths[i].split('/')
        temp_veg_type = dirs[-3]
        temp_cut_type = dirs[-2] # NOTE: not using
        temp_trial = dirs[-1] #NOTE: not using

        # get index of label
        label = [i for i, label in enumerate(labels) if temp_veg_type in label][0]

        # output[0] = label*len(labels) + int(temp_cut_type)
        output[i] = label

    return output.astype(np.uint8)

def get_sample_file(data_path, filename, key=None, image=False, img_size=(None,None)):
    """
    Same as get_food_dataset, except just one data file. All the samples must have the same datafile name
        [vegtable type] > [slice_type] > [trail_number]
    Inputs:
        data_path(str): path to the directory containing all the subdirectories of data
        filename(str): name of the file to get in each sample directory. Assuming it is a npz/npy file
        key: if the file is a dict/npz object. This will return the value associated with this key
        image(bool): If True, then the script will assume the file is an image file. This ignores key
        img_size(tuple): tuple of ints of size to resize images to. Leave as None to
            resize all the images to the minimum dimensions out of the dataset
    """
    data = []
    veg_types = utils.sub_dirs(data_path)
    #print('veg_types', veg_types)
    for veg_name in veg_types:
        if 'similarity' in veg_name:
            raise ValueError('Please move the similarity matrix folder outside of this data directory.')
        cut_type = utils.sub_dirs(veg_name)
        #print('cut_type', cut_type)
        for cut_name in cut_type:            
            trials = utils.sub_dirs(cut_name)
            #print('trials', trials)
            for trial in trials:                
                if image:
                    temp_data = get_images(directory=trial, image_type=filename, img_size=img_size)
                    data.append(temp_data)
                else:
                    temp_data = np.load(f'{trial}/{filename}', allow_pickle=True)
                    if key is not None:
                        data.append(temp_data[key])
                    else:
                        data.append(temp_data)

    data = np.stack(data, axis=0)
    return data


def load_sim_matrix(dict_savepath, dict_key):
    with open(dict_savepath,'rb') as f:
        data_dict = pickle.load(f)
    
    sim_matrix = data_dict[dict_key]
    return sim_matrix

def get_thresholds_each_sample(sim_matrix, num_NNs):
    num_samples = sim_matrix.shape[0]
    thresholds = []
    #import pdb; pdb.set_trace()
    for i in range(num_samples):
        sample_row = sim_matrix[i,:]
        
        nearest_neighbors_inds = np.argsort(sample_row)
        n_NNs = nearest_neighbors_inds[0:num_NNs]
        nearest_neighbors = sample_row[n_NNs]
        thresh = max(nearest_neighbors) #threshold based on the value of the farthest (max) NN
        thresholds.append(thresh)
        print('thresh', thresh)
    return np.array(thresholds)

def scale_features(raw_features):
    #scaled data (subtract mean, div by std)
    mu = np.mean(raw_features, axis=0)
    sigma = np.std(raw_features, axis=0)
    feats_scaled = (raw_features - mu)/sigma
    return feats_scaled, mu, sigma

def parse_silhoutte_data_dicts(dict_savepath, fname):
    file = dict_savepath + fname
    with open(file,'rb') as f:
        data_dict = pickle.load(f)
    list_dicts = []
    for i in range(449): 
        list_dicts.append(data_dict[str(i)])
    return np.array(list_dicts) #list of dictionaries, 1 for each sample
    