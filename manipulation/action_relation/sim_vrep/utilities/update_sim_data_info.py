import numpy as np

import argparse
import csv
import json
import os
import pickle

def add_relative_or_absoluate_action(data):
    data['data']['action_relative_or_absolute'] = 'relative'
    return data

def update_data_for_demo_dir(demo_dir):
    print(f"Will process dir: {demo_dir}")
    demo_idx = 0
    for root, dirs, files in os.walk(demo_dir, followlinks=False):
        # Sort the data order so that we do not have randomness associated
        # with it.
        dirs.sort()
        files.sort()

        if 'all_img_data.json' in files:
            # Filter data based on the anchor object.
            all_img_data_json_path = os.path.join(root, 'all_img_data.json')
            all_init_action_idx = [int(f.split('_')[0]) 
                                   for f in files if 'img_data.pkl' in f]

            for i in sorted(all_init_action_idx):
                data_pkl_path = os.path.join(root, '{}_img_data.pkl'.format(i))
                if not os.path.exists(data_pkl_path):
                    raise ValueError(f"Data pickle does not exist: {data_pkl_path}")

                with open(data_pkl_path, 'rb') as pkl_f:
                    data = pickle.load(pkl_f)
                
                # TODO call some new method to add/update data
                if data.get('action_relative_or_absolute') is not None:
                    del data['action_relative_or_absolute']
                if data['data'].get('action_relative_or_absolute') is not None:
                    new_data = data
                else:
                    new_data = add_relative_or_absoluate_action(data)
                    new_data_pkl_path = os.path.join(root, f'{i}_img_data.pkl')
                    with open(new_data_pkl_path, 'wb') as new_pkl_f:
                        pickle.dump(new_data, new_pkl_f, protocol=2)

                json_path = os.path.join(root, f'{i}_img_data.json')
                with open(json_path, 'w') as json_f:
                    json_f.write(json.dumps(new_data))

                demo_idx += 1
                if demo_idx % 100 == 0:
                    print("Did process: {}".format(demo_idx))

def main(args):
    for demo_dir in args.dir:
        assert os.path.exists(demo_dir), f"Path does not exist: {demo_dir}"
        update_data_for_demo_dir(demo_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update simulation data info')
    parser.add_argument('--dir', action='append', required=True, 
                        help='Path to dir with data.')
    args = parser.parse_args()
    import pprint
    pprint.pprint(args.__dict__)
    main(args)
