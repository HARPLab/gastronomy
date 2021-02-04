import glob
import cv2
import subprocess
import numpy as np

class_label = 255
annotation_image_path = '/home/klz/Documents/mask_images/cucumber/*/*/images/*_mask.png'

file_paths = glob.glob('/home/klz/Documents/mask_images/cucumber/*/*/images/*_mask.png')
for file_path in file_paths:
	if 'cucumber' in file_path:
	    print(file_path)
	    file_name = file_path[file_path.rfind('/')+1:]
	    
	    outfile_path = file_path
	    img = cv2.imread(file_path)
	    red_image = img[:,:,0]
	    nonzero_indices = np.nonzero(img)
	    for i in range(3):
	    	current_img_indices = np.ones(nonzero_indices[0].shape, dtype=int) * i
	    	appended_nonzero_indices = (nonzero_indices[0],nonzero_indices[1],current_img_indices)
	    	img[appended_nonzero_indices] = class_label
	    cv2.imwrite(outfile_path,img)
