import glob
import sys
import cv2
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
import numpy as np

"""
Script for moving the hand labeled mask to one directory and rename them.
Will need to change the three paths below to use this.
NOTE:
    - Directory with hand labeled images should have the format:
      <path_to_dir>/<food_type>/<slice_type>/<trial_num>/images/*.png
(Steven, 9/10/20)
"""

class_label = 1
# The directories to resave this mask data to
training_image_path = './data/Aug_2020_round2/img_seg_train_data/9_10_20/training_images/'
annotation_image_path = './data/Aug_2020_round2/img_seg_train_data/9_10_20/training_annotations/'

# directory with all of the hand labeled mask data
file_paths = glob.glob('./data/silhouette_data/mask_playing_data_9_10_20/*/*/*/images/*.png')
assert len(file_paths) > 0

for i in tqdm(range(len(file_paths)), file=sys.stdout, desc='Files left to move:'):
    file_path = file_paths[i]
    path_dirs = file_path.split('/')
    file_name = path_dirs[-1]
    trial_num = int(path_dirs[-3])
    slice_num = int(path_dirs[-4])
    class_name = path_dirs[-5]

    file_suffix = None
    if 'grasp' in file_name:
        if '_0' in file_name:
            file_suffix = 'starting_grasp_image.png'
        else:
            file_suffix = 'ending_grasp_image.png'
    elif 'release' in file_name:
        if '_0' in file_name:
            file_suffix = 'starting_release_image.png'
        else:
            file_suffix = 'ending_release_image.png'
    elif 'push' in file_name:
        if '_0' in file_name:
            file_suffix = 'starting_push_image.png'
        else:
            file_suffix = 'ending_push_image.png'
    else:
        if 'starting' in file_name:
            file_suffix = 'starting_overhead_image.png'
        else:
            file_suffix = 'ending_overhead_image.png'
    
    if 'mask' in file_name:
        outfile_path = annotation_image_path + class_name+'_'+str(slice_num)+'_'+str(trial_num)+'_' + file_suffix
        img = cv2.imread(file_path)
        img[np.nonzero(img)] = class_label
        img = img.astype(float)
        cv2.imwrite(outfile_path,img)
    else:
        outfile_path = training_image_path + class_name+'_'+str(slice_num)+'_'+str(trial_num)+'_' + file_suffix
        command = "cp " + file_path + " " + outfile_path
        subprocess.run(command, shell=True)