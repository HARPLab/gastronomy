import shutil
import os
import sys
import argparse
import tqdm

# Using TF-GPU V. 1.15
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

# Using this library: https://github.com/firephinx/image-segmentation-keras
sys.path.append(os.path.abspath("../image-segmentation-keras"))
from keras_segmentation.predict import model_from_checkpoint_path

"""
This file is for generating the binary masks of the processed food images after you
have already trained your image segmentation network. This script assumes you are
running this file from the "playing_with_food" directory.
Assuming directory structure for the source directory of data is:
    <path_to_dir>/<food_type>/<slice_type>/<trial_num>/images/*.png

Steps:
    - Labeled the masks using playing_with_food/scripts/label.py
    - Use the playing_with_food/scripts/mask_scripts/create_training_data.py to move all the files to one dir
    - Train a model using image-segmentation-keras/train_new_model.py
    - Use that model in this scripts to make the mask predictions

Steven (9/10/20)
"""

def ig_files(directory, files):
    """
    Ref: https://stackoverflow.com/a/15664273
    """
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--source_dir", type=str,
                        default="./data/Aug_2020_round2/processed_playing_data",
                        help="Directory containing the processed image data")
    parser.add_argument("-d", "--dest_dir", type=str,
                        default="./data/Aug_2020_round2/predicted_silhouettes",
                        help="Name of directory save images to. This directory cannot exist already")
    parser.add_argument("-c", "--checkpoint_path", type=str,
                        default="../playing_with_food/checkpoints/silhouette_checkpoints/checkpoints_9_10_20/all_food_types/pspnet_50",
                        help="Path to the folder containing the .json checkpoint file. Don't include the '_config.json' at the end.")

    args = parser.parse_args()

    # Copy the directory tree, but not any of the files
    shutil.copytree(args.source_dir, args.dest_dir, ignore=ig_files)
    # Get all of the dirs furthest down the tree for a unit test later
    dir_check = [path for path, dirs, files in os.walk(args.dest_dir) if len(dirs) == 0 and len(files) == 0 and path.split('/')[-1] == 'images']
    assert len(dir_check) > 0

    # set up TF backend
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # Load model
    model = model_from_checkpoint_path(args.checkpoint_path)

    prog = tqdm.tqdm(initial=0, total=len(dir_check), file=sys.stdout, desc='Directories left to parse:')
    for path, dirs, files in os.walk(args.source_dir):
        if len(dirs) == 0:
            assert len(files) != 0

            path_dirs = path.split('/')
            trial_num = int(path_dirs[-2])
            slice_num = int(path_dirs[-3])
            class_name = path_dirs[-4]

            out_dir = f"{args.dest_dir}/{class_name}/{slice_num}/{trial_num}/images"
            # keep track of the directories that were processed
            if out_dir in dir_check:
                dir_check.remove(out_dir) # this assumes there are not duplicates
                prog.update(1)
                prog.refresh()

            out = model.predict_multiple(
                inp_dir=f"{path}/",
                out_dir=out_dir,
                colors=[(0,0,0),(255,255,255)]
            )

        else:
            # Ignore all the directories except the ones containing images
            pass

    if len(dir_check) == 0:
        print('Finished!')
    else:
        print('\nSOMETHING WENT WRONG!!!\n')
        print('The following trials did not have any images saved to them:\n')
        for derp in dir_check:
            print(f'{derp}\n')
        import ipdb; ipdb.set_trace()
    prog.close()
