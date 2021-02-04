import glob
import cv2
import subprocess
import numpy as np

class_label = 255
annotation_image_path = '/home/klz/labeled_food_annotations/'

file_paths = glob.glob('/home/klz/food_training_annotations/*.png')
for file_path in file_paths:
    print(file_path)
    file_name = file_path[file_path.rfind('/')+1:]
    
    outfile_path = annotation_image_path + file_name
    img = cv2.imread(file_path)
    img[np.nonzero(img)] = class_label
    cv2.imwrite(outfile_path,img)
