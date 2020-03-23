import numpy as np

import argparse
import copy
import csv
import h5py
import os
import pdb
import pickle

from collections import OrderedDict


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def recursively_get_dict_from_group(group_or_data):
    d = {}
    if type(group_or_data) == h5py.Dataset:
        return np.array(group_or_data)

    # Else it's still a group
    for k in group_or_data.keys():
        v = recursively_get_dict_from_group(group_or_data[k])
        d[k] = v
    return d

def convert_list_of_array_to_dict_of_array_for_hdf5(arr_list):
    arr_dict = {}
    for i, a in enumerate(arr_list):
        arr_dict[str(i)] = a
    return arr_dict

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


def get_skill_desc_for_logging(csv_row):
    csv_row_items = csv_row.split(" ")
    desc_index = csv_row_items.index("desc:") + 1
    desc = ' '.join(csv_row_items[desc_index:])
    return desc

def get_skill_start_time_for_logging(csv_row):
    csv_row_items = csv_row.split(" ")
    if "curr_time:" not in csv_row_items:
        return -1
    time_idx = csv_row_items.index("curr_time:") + 1
    # -1 to remove the last comma from time_idx
    skill_time = int(csv_row_items[time_idx][:-1])
    return skill_time

def get_csv_keys_to_row_index_as_ordered_dict():
    csv_keys_to_row_idx_ord_dict = OrderedDict()
    csv_keys_to_row_idx_ord_dict['time_since_skill_started'] = (0, 1)
    csv_keys_to_row_idx_ord_dict['pose_desired'] = (1, 17)
    csv_keys_to_row_idx_ord_dict['pose'] = (17, 33)
    csv_keys_to_row_idx_ord_dict['joint_torques'] = (33, 40)
    csv_keys_to_row_idx_ord_dict['joint_torque_derivative'] = (40, 47)
    csv_keys_to_row_idx_ord_dict['joints'] = (47, 54)
    csv_keys_to_row_idx_ord_dict['joints_desired'] = (54, 61)
    csv_keys_to_row_idx_ord_dict['joint_velocities'] = (61, 68)
    return csv_keys_to_row_idx_ord_dict

def read_data_as_csv(csv_file, csv_keys_to_row_idx_ord_dict=None,
                     non_csv_rows_prefix=None):
    '''Read data as dictionary of arrays from csv file.'''
    if csv_keys_to_row_idx_ord_dict is None:
        csv_keys_to_row_idx_ord_dict = \
                get_csv_keys_to_row_index_as_ordered_dict()


    with open(csv_file, 'r') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',')
        data = []
        data_dict = {}
        for k in csv_keys_to_row_idx_ord_dict.keys():
            data_dict[k] = []
        data_dict['skill_info'] = []

        for row in csv_reader:
            if non_csv_rows_prefix is not None \
                    and row[0].startswith(non_csv_rows_prefix):
                # This is not a csv row but an info string
                data_dict['skill_info'].append(''.join(row))
                continue

            # Remote the last extra , and convert to floats.
            trajectory = [float(d) for d in row[:-1]]

            for col in csv_keys_to_row_idx_ord_dict.keys():
                st_idx, end_idx = csv_keys_to_row_idx_ord_dict[col]
                data_dict[col].append(trajectory[st_idx:end_idx])

        # Convert dictionary items to arrays.
        for k in data_dict.keys():
            if k != 'skill_info':
                data_dict[k] = np.array(data_dict[k])

    return data_dict

def read_data_as_skills_from_csv(csv_file, csv_keys_to_row_idx_ord_dict=None,
                                 non_csv_rows_prefix=None,
                                 multi_demo_in_csv=False):
    if csv_keys_to_row_idx_ord_dict is None:
        csv_keys_to_row_idx_ord_dict = \
                get_csv_keys_to_row_index_as_ordered_dict()

    all_demo_robot_state_dict = {}
    all_demo_skill_info_dict = {}
    all_demo_config_data_dict = {}

    with open(csv_file, 'r') as csv_f:
        # If a skill has multiple demonstrations we use demo_id to save data
        # for each demonstration separately.
        demo_id = 0
        csv_reader = csv.reader(csv_f, delimiter=',')
        robot_state_by_skill_desc_dict = {}
        skill_info_dict = {'skill_sequence': [], 'skill_time': {},
                           'skill_start_time': {}}
        config_data_to_save_as_csv = [['skill_id', 'use_to_train']]

        # Skill could be repeated for some weird reason (e.g. not getting a new
        # skill immediately etc.). Hence we keep a count to store all the time a
        # skill was executed and we can look at the rosbag to annotate each
        # skill as required (or something else).
        skill_desc_count_dict = {}

        last_skill_desc = None
        skill_start_time, skill_end_time = 0.0, 0.0
        will_start_new_skill = False

        for row in csv_reader:
            last_recorded_time = None
            if non_csv_rows_prefix is not None \
                    and row[0].startswith(non_csv_rows_prefix):
                # This is not a csv row but an info string. Control loop would
                # have started a new skill. Let's log this.
                skill_desc = get_skill_desc_for_logging(','.join(row))

                # Is this a new demonstration?
                if 'move_to_initial_position' in skill_desc:
                    all_demo_robot_state_dict[demo_id] = \
                            copy.deepcopy(robot_state_by_skill_desc_dict)
                    all_demo_skill_info_dict[demo_id] = \
                            copy.deepcopy(skill_info_dict)
                    all_demo_config_data_dict[demo_id] = \
                            copy.deepcopy(config_data_to_save_as_csv)
                    demo_id = demo_id + 1

                    # Now clear the data for previous demonstrations
                    robot_state_by_skill_desc_dict.clear()
                    for k in skill_info_dict.keys():
                        skill_info_dict[k].clear()
                    config_data_to_save_as_csv = [['skill_id', 'use_to_train']]

                # Save start and end time for this skill
                if last_skill_desc is not None:
                    skill_info_dict['skill_time'][last_skill_desc] = \
                            (skill_start_time, skill_end_time)
                    skill_start_time, skill_end_time = 0., 0.

                # There is a bug in the code where the skills are being
                # repeated hence we append a "_repeat" after each skill id to
                # differentiate.
                if robot_state_by_skill_desc_dict.get(skill_desc) is not None:
                    skill_count = skill_desc_count_dict[skill_desc]
                    last_skill_desc = skill_desc + '_repeat_{}'.format(skill_count)
                    skill_desc_count_dict[skill_desc] += 1
                else:
                    skill_desc_count_dict[skill_desc] = 1
                    last_skill_desc = skill_desc

                # last_skill_desc is the current skill description 
                # We have the start time for current skill, so update current # skill.
                skill_abs_start_time = get_skill_start_time_for_logging(
                        ','.join(row))
                if skill_abs_start_time != -1:
                    skill_info_dict['skill_start_time'][last_skill_desc] = \
                            skill_abs_start_time
                # Add last_skill id skill_info_dict
                skill_info_dict['skill_sequence'].append(last_skill_desc)
                config_data_to_save_as_csv.append([last_skill_desc, 0])

                robot_state_by_skill_desc_dict[last_skill_desc] = {}
                for k in csv_keys_to_row_idx_ord_dict.keys():
                    robot_state_by_skill_desc_dict[last_skill_desc][k] = []

                will_start_new_skill = True
            else:
                if last_skill_desc is None:
                    # These are initial robot states not associated with any
                    # skill we can skip them for now.
                    continue
                
                # Remote the last extra , and convert to floats.
                trajectory = [float(d) for d in row[:-1]]
                curr_recorded_time = trajectory[0]

                if will_start_new_skill and trajectory[0] <= 0.001:
                    # This is stuff  from previous log so just continue
                    continue
                elif will_start_new_skill:
                    skill_start_time = trajectory[0]
                elif curr_recorded_time - skill_end_time >= 0.01:
                    # This state is not part of the skill just continue.
                    continue
                else:
                    skill_end_time = trajectory[0] 
                will_start_new_skill = False
                last_recorded_time = curr_recorded_time

                for col in csv_keys_to_row_idx_ord_dict.keys():
                    st_idx, end_idx = csv_keys_to_row_idx_ord_dict[col]
                    robot_state_by_skill_desc_dict[last_skill_desc][col].append(
                            trajectory[st_idx:end_idx])

    # NOTE: Add the last skill too. Save start and end time for this skill.
    if last_skill_desc is not None:
        skill_info_dict['skill_time'][last_skill_desc] = \
                (skill_start_time, skill_end_time)
        skill_abs_start_time = get_skill_start_time_for_logging(
                ','.join(row))
        if skill_abs_start_time != -1:
            skill_info_dict['skill_start_time'][last_skill_desc] = \
                    skill_abs_start_time
        all_demo_robot_state_dict[demo_id] = \
                copy.deepcopy(robot_state_by_skill_desc_dict)
        all_demo_skill_info_dict[demo_id] = \
                copy.deepcopy(skill_info_dict)
        all_demo_config_data_dict[demo_id] = \
                copy.deepcopy(config_data_to_save_as_csv)

    for demo_key in all_demo_robot_state_dict.keys():
        for skill_desc, skill_value in all_demo_robot_state_dict[demo_key].items():
            for robot_state_item, robot_state_value in skill_value.items():
                if type(robot_state_value) is list:
                    all_demo_robot_state_dict[demo_key][skill_desc][robot_state_item] =\
                            np.array(robot_state_value)

    return all_demo_robot_state_dict, all_demo_skill_info_dict, \
            all_demo_config_data_dict

def read_demonstration_data_from_skill_segmented_hdf5(hdf5_path):
    '''Read data directly from HDF5 file. Assumes all data in the hdf5 file is
    stored in terms of arrays.'''
    h5_path = hdf5_path
    assert os.path.exists(hdf5_path), "HDF5 does not exist {}".format(h5_path)
    h5f = h5py.File(h5_path, 'r')
    data = recursively_get_dict_from_group(h5f['/'])
    h5f.close()
    return data

def read_skill_info_from_pickle(pkl_path):
    '''Read pickle file which contains skill info for demonstrations segmented 
    by skills.'''
    assert os.path.exists(pkl_path), "Pickle path does not exist {}".format(
            pkl_path)
    with open(pkl_path, 'rb') as pkl_f:
        data = pickle.load(pkl_f)
    return data

def write_skill_config_csv(csv_path, csv_data_list):
    pass

def read_data_config_csv(csv_path):
    '''Read the csv config for this demonstration. 
        
    This config contains the ground truth data for this demonstration. CSV uses
    '$' as the separator format.
    '''
    assert os.path.exists(csv_path), "Pickle path does not exist {}".format(
                csv_path)
    row_list, data_config_by_path_dict = [], {}
    with open(csv_path, 'r') as csv_f:
        reader = csv.DictReader(csv_f, delimiter=',')
        for row in reader:
            row_list.append(row)
            assert data_config_by_path_dict.get(row['path']) is None, \
                "Found duplicate rows in data config csv {}".format(csv_path)
            data_config_by_path_dict[row['path']] = {}
            for k, v in row.items():
                if k != 'path':
                    data_config_by_path_dict[row['path']][k] = v

    return row_list, data_config_by_path_dict

def read_skill_config_csv(csv_path):
    '''Read the csv config for this demonstration. 
    
    This config contains the ground truth data for this demonstration. CSV uses
    '$' as the separator format.
    '''
    assert os.path.exists(csv_path), "Pickle path does not exist {}".format(
            csv_path)
    row_list, skill_config_by_desc_dict = [], {}
    with open(csv_path, 'r') as csv_f:
        reader = csv.DictReader(csv_f, delimiter='$')
        for row in reader:
            row_list.append(row)
            assert skill_config_by_desc_dict.get(row['skill_id']) is None, \
                    "Found duplicate rows in skill config csv {}".format(
                            csv_path)
            skill_config_by_desc_dict[row['skill_id']] = {}
            for k, v in row.items():
                if k != 'skill_id':
                    skill_config_by_desc_dict[row['skill_id']][k] = v


    return row_list, skill_config_by_desc_dict

def get_rosbag_images_for_skills(skill_info_dict, 
                                 rosbag_images_and_timestamp_dict,
                                 delay_in_milliseconds=0):
    assert skill_info_dict.get('skill_start_time') is not None, \
            "No skill start time found cannot find rosbag images."
    img_time_by_idx_dict = \
            rosbag_images_and_timestamp_dict['image_time_by_idx_dict']
    img_idx_by_time_dict = \
            rosbag_images_and_timestamp_dict['image_idx_by_time_dict']

    img_times_sorted = np.sort(list(img_idx_by_time_dict.keys()))
    img_idx_sorted = np.sort(list(img_time_by_idx_dict.keys()))

    skill_info_dict['skill_images'] = {}

    for i in range(len(skill_info_dict['skill_sequence']) - 1):
        curr_skill_desc = skill_info_dict['skill_sequence'][i]
        next_skill_desc = skill_info_dict['skill_sequence'][i+1]
        # start time is recorded in number of milliseconds since epoch.
        curr_skill_start_time = \
                skill_info_dict['skill_start_time'][curr_skill_desc] \
                + delay_in_milliseconds
        # skill_time is recorded in number of seconds.
        skill_time = skill_info_dict['skill_time'][curr_skill_desc]
        curr_skill_end_time = curr_skill_start_time + (skill_time[1] * 1000.0)

        next_skill_start_time = \
                skill_info_dict['skill_start_time'][next_skill_desc] \
                + delay_in_milliseconds

        # Find images within this (current, next) skill time interval
        curr_skill_start_time_closest = max(
                [y for y in img_times_sorted if y <= curr_skill_start_time])
        curr_skill_end_time_closest = min(
                [y for y in img_times_sorted if y > curr_skill_end_time])
        next_skill_start_time_delay = 100  # delay of 1000 milliseconds
        next_skill_start_time_closest = max(
                [y for y in img_times_sorted if y <= 
                    (next_skill_start_time - next_skill_start_time_delay)])
        assert curr_skill_end_time_closest >= curr_skill_start_time_closest, \
            "Skill cannot really move back in time. Broke the time dimension!!"

        curr_skill_img_idx = img_idx_by_time_dict[curr_skill_start_time_closest]
        # next_skill_img_idx = img_idx_by_time_dict[curr_skill_end_time_closest]
        next_skill_img_idx = img_idx_by_time_dict[next_skill_start_time_closest]

        skill_info_dict['skill_images'][curr_skill_desc] = \
                list(range(curr_skill_img_idx, next_skill_img_idx))
        print("Added {} images for skill: {}".format(
            next_skill_img_idx - curr_skill_img_idx, curr_skill_desc))

    assert next_skill_img_idx <= img_idx_sorted[-1], \
            "Incorrect sequence for last skill"
    skill_info_dict['skill_images'][next_skill_desc] = \
            range(next_skill_img_idx, img_idx_sorted[-1])

    return skill_info_dict

def read_data_as_h5(h5_file):
    pass

def save_data_as_h5d(data_dict):
    pass

