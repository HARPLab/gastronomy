import numpy as np

import argparse

from data_utils import read_skill_info_from_pickle

def print_skill_info_from_pkl_file(pkl_path):
    skill_info_data = read_skill_info_from_pickle(pkl_path)
    for i, skill_desc in enumerate(skill_info_data['skill_sequence']):
        print("id: {}, skill: {}".format(i, skill_desc))
        skill_start_time = skill_info_data['skill_start_time'][skill_desc]
        skill_time = skill_info_data['skill_time'][skill_desc]
        skill_images = skill_info_data['skill_images'][skill_desc]

def main(args):
    assert os.path.exists(args.pickle), \
            "Pickle file does not exist {}".format(args.pickle)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Print skill info saved in pickle file.')
    parser.add_argument('--pickle', type=str, required=True,
                        help='Pickle file containing the skill info.')
    args = parser.parse_args()
    main(args)
