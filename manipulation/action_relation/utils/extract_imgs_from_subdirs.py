import numpy as np
import argparse
import os
import shutil


def main(src_dir, target_dir):
    assert os.path.exists(src_dir), "Source dir does not exist"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root_dir, sub_dir, fnames in os.walk(src_dir):
        for f in fnames:
            if f.endswith('jpg') or f.endswith('png'):
                pkl_fname = 'img_data_{}pkl'.format(f[:-3].split('_')[1])
                assert os.path.exists(os.path.join(root_dir, pkl_fname)), \
                    "pickle file does not exist"
                new_fname = '{}_{}'.format(os.path.basename(root_dir), f)
                new_f_loc = os.path.join(target_dir, new_fname)
                shutil.copy(os.path.join(root_dir, f), new_f_loc)

                new_pkl_fname = '{}_{}'.format(
                    os.path.basename(root_dir), pkl_fname)
                new_f_loc = os.path.join(target_dir, new_pkl_fname)
                shutil.copy(os.path.join(root_dir, pkl_fname), new_f_loc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Move files")
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.src_dir, args.dest_dir)
