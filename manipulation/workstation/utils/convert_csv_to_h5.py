import numpy as np
import argparse

import csv
import h5py
import os
import pdb

from data_utils import read_data_as_csv
from data_utils import recursively_save_dict_contents_to_group

def main(args):
    assert os.path.exists(args.csv_dir), "CSV directory does not exist"
    csv_paths = [os.path.join(args.csv_dir, f) 
            for f in sorted(os.listdir(args.csv_dir)) 
                if os.path.isfile(os.path.join(args.csv_dir, f))]
    all_csv_data = {}
    for i, csv_file in enumerate(csv_paths):
        data = read_data_as_csv(csv_file)
        all_csv_data[str(i)] = data

    h5_path = os.path.join(args.output_dir, 'expert_traj.h5')
    h5f = h5py.File(h5_path)
    recursively_save_dict_contents_to_group(h5f, '/', all_csv_data) 
    h5f.flush()
    h5f.close()
    print("Did save h5 data: {}".format(h5_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Convert data with csvs into h5 file')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Path to csvs directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save h5 file into.')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print("Output does not exist! Will create {}".format(args.output_dir))
        os.makedirs(args.output_dir)
    main(args)

