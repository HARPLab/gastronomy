import glob
import cv2
import subprocess
import numpy as np

"""
This is the same thing as "create_training_data.py"
It's probably easier to just use that.
Also, it's not really necessary to have a test set for training the segmentation network.
(Steven, 9/10/20)
"""

class_label = 1
test_image_path = '/home/klz/food_test_images/'
annotation_image_path = '/home/klz/food_test_annotations/'

file_paths = glob.glob('/home/klz/Documents/playing_with_food/processed_data/*/*/*/images/*.png')
for file_path in file_paths:
    #print(file_path)
    first_images_idx = file_path.find('processed_data')
    second_images_idx = file_path.rfind('images')
    sub_path = file_path[first_images_idx+15:second_images_idx-1]
    class_name = sub_path[:sub_path.find('/')]
    slice_num = int(sub_path[sub_path.find('/')+1:sub_path.rfind('/')])
    trial_num = int(sub_path[sub_path.rfind('/')+1:])
    file_name = file_path[file_path.rfind('/')+1:]
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
        cv2.imwrite(outfile_path,img)
    else:
        outfile_path = test_image_path + class_name+'_'+str(slice_num)+'_'+str(trial_num)+'_' + file_suffix
        command = "cp " + file_path + " " + outfile_path
        subprocess.run(command, shell=True)