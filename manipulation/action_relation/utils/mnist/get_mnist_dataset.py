import h5py
import numpy as np
import os
import datetime
import scipy
import numpy as np
import random
from random import randint
np.random.seed(np.random.randint(1<<30))
import matplotlib.pyplot as plt
import ipdb

class MnistDataset(object):
    def __init__(self,
                 hf,
                 canvas_size=64,
                 digit_size=28):
        self.hf = hf
        self.odd_len = len(self.hf['odd_label'])
        self.even_len = len(self.hf['even_label'])
        self.canvas_size = canvas_size # size_
        self.digit_size = digit_size
        self.relation_dict = {
                0: 'left_right',
                1: 'up_down'}

    def get_canvas(self, idx_odd, idx_even, relation, get_masks=False):
        # sample range so that there is no overlap (only required for one direction)
        return_dict = dict()
        canvas_odd = np.zeros((self.canvas_size, self.canvas_size),dtype=np.float32)
        canvas_even = np.zeros((self.canvas_size, self.canvas_size),dtype=np.float32)
        
        sample_range = self.canvas_size//2 - self.digit_size
        odd_number = np.reshape(
                self.hf['odd_data'][idx_odd],
                (self.digit_size, self.digit_size))
        even_number = np.reshape(
                self.hf['even_data'][idx_even],
                (self.digit_size, self.digit_size))
        # ==== sample the start and end position ====
        if relation == 0:
            x_odd = np.random.randint(sample_range)
            y_odd = np.random.randint(self.canvas_size-self.digit_size)
            x_even = np.random.randint(self.canvas_size//2, 
                                       self.canvas_size//2 + sample_range)
            y_even = np.random.randint(self.canvas_size-self.digit_size)

        else:
            x_odd = np.random.randint(self.canvas_size-self.digit_size)
            y_odd = np.random.randint(sample_range)
            x_even = np.random.randint(self.canvas_size-self.digit_size)
            y_even = np.random.randint(self.canvas_size//2, self.canvas_size//2 + sample_range)

        canvas_odd[x_odd:self.digit_size+x_odd,
                   y_odd:self.digit_size+y_odd] = odd_number
        canvas_even[x_even:self.digit_size+x_even,
                    y_even:self.digit_size+y_even] = even_number
        canvas = np.maximum(canvas_odd, canvas_even)
        return_dict['canvas'] = canvas

        if get_masks:
            mask_odd = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
            mask_even = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
            mask_odd[canvas_odd>0] = 1
            mask_even[canvas_even>0] = 1
            return_dict['mask_odd'] = mask_odd
            return_dict['mask_even'] = mask_even
        return return_dict

    def make_dataset(self, length=10000, num_objs=2, get_masks=False):
        dataset = np.zeros((length, self.canvas_size, self.canvas_size),dtype=np.float32)
        dataset_label = np.zeros((length, 3), dtype=np.int32)
        if get_masks:
            masks = np.zeros(
                    (length, num_objs, self.canvas_size, self.canvas_size),
                    dtype=np.float32)

        for i in range(length):
            odd_idx = np.random.randint(self.odd_len)
            even_idx = np.random.randint(self.even_len)
            relation = i%2
            return_dict = self.get_canvas(odd_idx, even_idx,
                                          relation, get_masks=get_masks)
            dataset[i] = return_dict['canvas']
            if get_masks:
                masks[i, 0] = return_dict['mask_odd']
                masks[i, 1] = return_dict['mask_even']
            dataset_label[i] = np.array([
                self.hf['odd_label'][odd_idx],
                relation,
                self.hf['even_label'][even_idx]])
        if get_masks:
            return dataset, dataset_label, masks
        else:
            return dataset, dataset_label, None

    def create_h5(self, name='final_dataset.h5', get_masks=False):
        dataset, dataset_label, masks = self.make_dataset(
                length=10000,
                num_objs=2,
                get_masks=get_masks)
        with h5py.File(name, 'w') as hf:
            hf.create_dataset('data', data=dataset)
            hf.create_dataset('label', data=dataset_label)
            if get_masks:
                hf.create_dataset('masks', data=masks)


if __name__=='__main__':
    hf_py_path = os.path.expanduser('~/acads/research/oliver/mnist_exps/mnist_train_odd_even_separated.h5')
    hf = h5py.File(hf_py_path, 'r')
    data_maker = MnistDataset(hf)
    data_maker.create_h5(
            name=os.path.expanduser('~/acads/research/oliver/final_dataset_with_mask.h5'),
            get_masks=True)
