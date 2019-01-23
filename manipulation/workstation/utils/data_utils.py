import numpy as np

import csv
import h5py
import pdb

from collections import OrderedDict

def recursively_get_dict_from_group(group_or_data):
    d = {}
    if type(group_or_data) == h5py.Dataset:
        return np.array(group_or_data)

    # Else it's still a group
    for k in group_or_data.keys():
        v = recursively_get_dict_from_group(group_or_data[k])
        d[k] = v
    return d

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """
    assert type(dic) is type({}), "must provide a dictionary"
    assert type(path) is type(''), "path must be a string"
    assert type(h5file) is h5py._hl.files.File, "must be an open h5py file"

    for key in dic:
        assert type(key) is type(''), 'dict keys must be strings to save to hdf5'
        did_save_key = False

        if type(dic[key]) in (np.int64, np.float64, type(''), int, float):
            h5file[path + key] = dic[key]
            did_save_key = True
            assert h5file[path + key].value == dic[key], \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type([]):
            h5file[path + key] = np.array(dic[key])
            did_save_key = True
        if type(dic[key]) is np.ndarray:
            h5file[path + key] = dic[key]
            did_save_key = True
            assert np.array_equal(h5file[path + key].value, dic[key]), \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type({}):
            recursively_save_dict_contents_to_group(h5file,
                                                    path + key + '/',
                                                    dic[key])
            did_save_key = True
        if not did_save_key:
            print("Dropping key from h5 file: {}".format(path + key))

def read_data_as_csv(csv_file, csv_keys_to_row_idx_ord_dict=None):
    '''Read data as dictionary of arrays from csv file.'''
    if csv_keys_to_row_idx_ord_dict is None:
        csv_keys_list = ['time', 'pose_desired', 'robot_state', 'tau_j',
                         'd_tau_j', 'q', 'dq']
        csv_keys_row_indexes = [(0, 1), (1, 17), (17, 33), (33, 40), (40, 47)]
        csv_keys_to_row_idx_ord_dict = OrderedDict()
        csv_keys_to_row_idx_ord_dict['time'] = (0, 1)
        csv_keys_to_row_idx_ord_dict['pose_desired'] = (1, 17)
        csv_keys_to_row_idx_ord_dict['robot_state'] = (17, 33)
        csv_keys_to_row_idx_ord_dict['tau_j'] = (33, 40)
        csv_keys_to_row_idx_ord_dict['d_tau_j'] = (40, 47)
        csv_keys_to_row_idx_ord_dict['q'] = (47, 54)
        csv_keys_to_row_idx_ord_dict['dq'] = (54, 61)
        
            
    with open(csv_file, 'r') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',')
        data = []
        data_dict = {}
        for k in csv_keys_to_row_idx_ord_dict.keys():
            data_dict[k] = []

        for row in csv_reader:
            # Remote the last extra , and convert to floats.
            trajectory = [float(d) for d in row[:-1]]
            for col in csv_keys_to_row_idx_ord_dict.keys():
                st_idx, end_idx = csv_keys_to_row_idx_ord_dict[col]
                data_dict[col].append(trajectory[st_idx:end_idx])

        # Convert dictionary items to arrays.
        for k in data_dict.keys():
            data_dict[k] = np.array(data_dict[k])

    return data_dict

def read_data_as_h5(h5_file):
    pass

def save_data_as_h5d(data_dict):
    pass

