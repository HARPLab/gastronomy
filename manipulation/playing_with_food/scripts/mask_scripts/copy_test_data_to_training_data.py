import glob
import cv2
import subprocess
import numpy as np

class_label = 1
training_image_path = '/home/klz/food_training_images/'
training_annotation_path = '/home/klz/food_training_annotations/'
test_image_path = '/home/klz/food_test_images/'
test_annotation_path = '/home/klz/food_test_annotations/'

test_annotation_file_paths = glob.glob(test_annotation_path + '*.png')
test_image_file_paths = glob.glob('/home/klz/food_test_images/*.png')

for file_path in test_annotation_file_paths:
    file_name = file_path[file_path.rfind('/')+1:]

    annotation_outfile_path = training_annotation_path + file_name
    command = "mv " + file_path + " " + annotation_outfile_path
    subprocess.run(command, shell=True)

    test_image_file_path = test_image_path + file_name
    image_outfile_path = training_image_path + file_name
    command = "mv " + test_image_file_path + " " + image_outfile_path
    subprocess.run(command, shell=True)