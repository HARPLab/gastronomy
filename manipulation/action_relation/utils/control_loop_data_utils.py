import numpy as np

import argparse
import csv
import h5py
import os
import pdb
import pprint
import pickle
import shutil

from collections import OrderedDict

from utils import data_utils
from utils.data_utils import read_data_as_skills_from_csv
from utils.data_utils import recursively_save_dict_contents_to_group
from utils.data_utils import get_rosbag_images_for_skills

def update_skill_info_pickle_with_skill_config_csv(csv_path, rosbag_path, 
                                                   save_dir):
    pkl_path = os.path.join(save_dir, 'skill_info.pkl')
    csv_path = os.path.join(save_dir, 'skill_config.csv')
    assert os.path.exists(pkl_path) and os.path.exists(csv_path), \
            "pickle or csv file does not exist."

    skill_info_pkl_data = data_utils.read_skill_info_from_pickle(pkl_path)
    skill_config_csv_data = data_utils.read_skill_config_csv(csv_path)
    skill_config_row_list, skill_config_by_desc_dict = skill_config_csv_data

    for i in range(len(skill_info_pkl_data['skill_sequence'])):
        from ros_scripts.read_rosbag import RosbagUtils
        skill_desc_in_pkl = skill_info_pkl_data['skill_sequence'][i]
        skill_desc_in_csv = skill_config_row_list[i]['skill_id']
        assert skill_desc_in_pkl == skill_desc_in_csv, \
             "Skill description in CSV and pickle do not match. Cannot update"
        
        # Update pickle files with new data in csv

        # 1) Update image list initially
        skill_img_start = \
            skill_config_by_desc_dict[skill_desc_in_pkl]['skill_img_start']
        skill_img_end = \
            skill_config_by_desc_dict[skill_desc_in_pkl]['skill_image_end']
        skill_img_start, skill_img_end = int(skill_img_start), int(skill_img_end)
        assert skill_img_start < skill_img_end \
                or (skill_img_start == 0 and skill_img_end == 0), \
                "Invalid (start, end) ({}, {}) images for skill: {}".format(
                        skill_img_start, skill_img_end, skill_desc_in_pkl)
        skill_info_pkl_data['skill_images'][skill_desc_in_pkl] = \
                list(range(skill_img_start, skill_img_end+1))

    # Create a backup first and then save
    pkl_path = os.path.join(save_dir, 'skill_info.pkl')
    backup_pkl_path = os.path.join(save_dir, 'skill_info.pkl.bak')
    shutil.move(pkl_path, backup_pkl_path)
    with open(pkl_path, "wb") as pkl_f:
        pickle.dump((skill_info_pkl_data), pkl_f, protocol=2)
        print("Did save skill info at {}".format(pkl_path))

def convert_to_hdf5(csv_path, rosbag_path, save_dir):
    robot_state_by_desc_dict, skill_info_dict, config_data_to_save_as_csv = \
            read_data_as_skills_from_csv(
                    csv_path,
                    non_csv_rows_prefix='info')
    # TODO: Read image from rosbag and find images associated with each 
    if skill_info_dict.get('skill_start_time') is not None \
            and len(skill_info_dict.get('skill_start_time')) > 0 \
            and len(rosbag_path) > 0:
        rosbag_utils = RosbagUtils(rosbag_path)
        image_and_timestamp_dict = rosbag_utils.get_images_with_timestamp(
                '/camera1/color/image_raw',
                time_in_milliseconds=True)
        get_rosbag_images_for_skills(skill_info_dict,
                                     image_and_timestamp_dict,
                                     delay_in_milliseconds=500)

        # Update the config data csv based on skill images
        config_data_to_save_as_csv[0].append('skill_image_start')
        config_data_to_save_as_csv[0].append('skill_image_end')
        for i in range(1, len(config_data_to_save_as_csv)):
            skill_desc = config_data_to_save_as_csv[i][0]
            skill_images = skill_info_dict['skill_images'].get(skill_desc)
            assert skill_images is not None, \
                    "Cannot find skill images for given skill desc"
            # Add start and end images for this skill.
            skill_info_dict['skill_images'][skill_desc].append(skill_images[0])
            skill_info_dict['skill_images'][skill_desc].append(skill_images[-1])

    save_dir = os.path.join(save_dir, 'robot_data_skill_segment')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    h5_path = os.path.join(save_dir, 'robot_state_data_by_skill.h5')
    h5f = h5py.File(h5_path, 'w')
    recursively_save_dict_contents_to_group(h5f, '/', robot_state_by_desc_dict)
    h5f.flush()
    h5f.close()
    print("Did write robot state data to {}".format(h5_path))
    
    pkl_path = os.path.join(save_dir, 'skill_info.pkl')
    with open(pkl_path, "wb") as pkl_f:
        pickle.dump((skill_info_dict), pkl_f, protocol=2)
        print("Did save skill  info at {}".format(pkl_path))

    csv_path = os.path.join(save_dir, 'skill_config.csv')
    if os.path.exists(csv_path) and args.override_csv != 1:
        print("Error: Will not override CSV that already exists at {}".format(
            csv_path))
    else:
        with open(csv_path, 'w') as csv_f:
            csv_writer = csv.writer(csv_f, delimiter='$')
            for row in config_data_to_save_as_csv:
                csv_writer.writerow(row)
            print("Did write csv at {}".format(csv_path))


def main(args):
    if args.convert_to_h5 == 1:
        print("Will convert to hdf5")
        convert_to_hdf5(args.csv, args.bag, args.save_dir)
    elif args.update_skill_info_pickle_with_skill_config_csv == 1:
        print("Will update skill info pickle with csv file")
        update_skill_info_pickle_with_skill_config_csv(
            args.csv, args.bag, args.save_dir)
        pass
    else:
        print("No action required")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utils for control loop data")
    parser.add_argument("--convert_to_h5", type=int, default=0,
                        help="Convert control loop data csv to hdf5.")
    parser.add_argument("--update_skill_info_pickle_with_skill_config_csv",
                        type=int, default=0,
                        help='Update images for each skill.')
    parser.add_argument("--csv", type=str, required=True,
                        help='Path to control loop data csv.')
    parser.add_argument("--override_csv", type=int, default=0,
                        help="Set to override skill config csv if already exists")
    parser.add_argument("--bag", type=str, default='',
                        help='Path to rosbag file.')
    parser.add_argument("--save_dir", type=str, default='',
                        help='Directory to save hdf5 file in')
    args = parser.parse_args()

    # Pretty print arguments
    pprint.pprint(args.__dict__)

    main(args)
