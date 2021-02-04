import glob
import cv2
import numpy as np

class_name = 'cooked_steak'
mask_dict = {}

for i in range(1,11):
    for j in range(1,6):
        file_paths = glob.glob('/home/klz/Documents/playing_with_food/processed_data/'+class_name+'/'+str(i)+'/'+str(j)+'/images/*.png')
        for file_path in file_paths:
            #print(file_path)
            file_suffix = None
            if 'grasp' in file_path:
                if '0' in file_path:
                    file_suffix = 'starting_grasp_image.png'
                else:
                    file_suffix = 'ending_grasp_image.png'
            elif 'release' in file_path:
                if '0' in file_path:
                    file_suffix = 'starting_release_image.png'
                else:
                    file_suffix = 'ending_release_image.png'
            elif 'push' in file_path:
                if '0' in file_path:
                    file_suffix = 'starting_push_image.png'
                else:
                    file_suffix = 'ending_push_image.png'
            else:
                if 'starting' in file_path:
                    file_suffix = 'starting_overhead_image.png'
                else:
                    file_suffix = 'ending_overhead_image.png'
            
            if 'mask' in file_path:
                outfile_path = class_name+'_'+str(i)+'_'+str(j)+'_' + file_suffix
                img = cv2.imread(file_path)
                img[np.nonzero(img)] = 1
                for key in mask_dict.keys():
                    if img.shape == mask_dict[key].shape:
                        if(np.allclose(img, mask_dict[key])):
                            print(outfile_path)
                            print(key)
                mask_dict[outfile_path] = img