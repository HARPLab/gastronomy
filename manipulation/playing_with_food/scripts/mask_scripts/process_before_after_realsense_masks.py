import glob
import cv2
import subprocess
import numpy as np

class_label = 1
mask_path = '/home/klz/food_test_images/'

food_types = ['apple', 'carrot', 'cooked_steak', 'cucumber', 'onion', 'pear', 'potato', 'raw_steak', 'tomato']
file_suffices = ['starting_grasp_image.png', 'ending_grasp_image.png', 'starting_release_image.png', 'ending_release_image.png', 'starting_push_image.png', 'ending_push_image.png']

max_height = 0
max_width = 0

starting_object_centers = np.zeros((449,2))
ending_object_centers = np.zeros((449,2))

starting_overhead_images = np.zeros((449, 350, 440))
ending_overhead_images = np.zeros((449, 350, 440))
starting_object_image_size = np.zeros((449,2))
ending_object_image_size = np.zeros((449,2))
starting_object_images = np.zeros((449, 150, 150))
ending_object_images = np.zeros((449, 150, 150))
object_position_changes = np.zeros((449,2))
object_size_changes = np.zeros((449,2))

i = 0
missing_file_paths = []
file_paths = glob.glob('/home/klz/food_test_images/*.png')
for food_type in food_types:
    for slice_num in range(1,11):
        for trial_num in range(1,6):
            for file_suffix in file_suffices:
                file_path = mask_path + food_type + '_' + str(slice_num) + '_' + str(trial_num) + '_' + file_suffix
                if file_path not in file_paths:
                    missing_file_paths.append(file_path)

print(len(missing_file_paths))
print(missing_file_paths)
#             if food_type == 'tomato' and slice_num == 4 and trial_num == 2:
#                 pass
#             else:
#                 file_prefix = mask_path + food_type + '_' + str(slice_num) + '_' + str(trial_num) + '_'
#                 starting_overhead_image_file_name = file_prefix + 'starting_overhead_image.png'
#                 ending_overhead_image_file_name = file_prefix + 'ending_overhead_image.png'

#                 starting_overhead_image = cv2.imread(starting_overhead_image_file_name)
#                 ending_overhead_image = cv2.imread(ending_overhead_image_file_name)


#                 starting_overhead_image[np.nonzero(starting_overhead_image)] = class_label
#                 ending_overhead_image[np.nonzero(ending_overhead_image)] = class_label

#                 starting_overhead_images[i,:,:] = starting_overhead_image[:,:,0]
#                 ending_overhead_images[i,:,:] = ending_overhead_image[:,:,0]
#                 starting_overhead_image_nonzeros = np.nonzero(starting_overhead_image[:,:,0])
#                 ending_overhead_image_nonzeros = np.nonzero(ending_overhead_image[:,:,0])

#                 starting_object_centers[i,:] = [np.mean(starting_overhead_image_nonzeros[0]), np.mean(starting_overhead_image_nonzeros[1])]

#                 if(len(ending_overhead_image_nonzeros[0]) == 0):
#                     ending_object_centers[i,:] = [-1, -1]
#                 else:
#                     ending_object_centers[i,:] = [np.mean(ending_overhead_image_nonzeros[0]), np.mean(ending_overhead_image_nonzeros[1])]

#                 starting_object_image_size[i,:] = [np.max(starting_overhead_image_nonzeros[0]) - np.min(starting_overhead_image_nonzeros[0]),
#                                                    np.max(starting_overhead_image_nonzeros[1]) - np.min(starting_overhead_image_nonzeros[1])]
                
#                 if(len(ending_overhead_image_nonzeros[0]) == 0):
#                     ending_object_image_size[i,:] = [-1,-1]
#                 else:
#                     ending_object_image_size[i,:] = [np.max(ending_overhead_image_nonzeros[0]) - np.min(ending_overhead_image_nonzeros[0]),
#                                                      np.max(ending_overhead_image_nonzeros[1]) - np.min(ending_overhead_image_nonzeros[1])]

#                 padded_starting_image = np.zeros((450,540))
#                 padded_ending_image = np.zeros((450,540))

#                 padded_starting_image[50:400,50:490] = starting_overhead_images[i,:,:]
#                 padded_ending_image[50:400,50:490] = ending_overhead_images[i,:,:]

#                 starting_object_images[i,:,:] = padded_starting_image[int(starting_object_centers[i,0])-25:int(starting_object_centers[i,0])+125,
#                                                                       int(starting_object_centers[i,1])-25:int(starting_object_centers[i,1])+125]
#                 if(len(ending_overhead_image_nonzeros[0]) == 0):
#                     ending_object_images[i,:,:] = np.zeros((150,150))
#                 else:
#                     ending_object_images[i,:,:] = padded_ending_image[int(ending_object_centers[i,0])-25:int(ending_object_centers[i,0])+125,
#                                                                       int(ending_object_centers[i,1])-25:int(ending_object_centers[i,1])+125]
#                 if(len(ending_overhead_image_nonzeros[0]) == 0):
#                     object_position_changes[i,:] = [-500, -500]
#                     object_size_changes[i,:] = [-500, -500]
#                 else:
#                     object_position_changes[i,:] = ending_object_centers[i,:] - starting_object_centers[i,:]
#                     object_size_changes[i,:] = ending_object_image_size[i,:] - starting_object_image_size[i,:]

#                 i += 1

# np.savez('/home/klz/Documents/playing_with_food/processed_overhead_masks.npz', 
#          starting_object_centers=starting_object_centers,
#          ending_object_centers=ending_object_centers,
#          starting_overhead_images=starting_overhead_images,
#          ending_overhead_images=ending_overhead_images,
#          starting_object_image_size=starting_object_image_size,
#          ending_object_image_size=ending_object_image_size,
#          starting_object_images=starting_object_images,
#          ending_object_images=ending_object_images,
#          object_position_changes=object_position_changes,
#          object_size_changes=object_size_changes)
#print(i)

# sorted_starting_heights = np.sort(starting_object_image_size[:,0])
# sorted_starting_widths = np.sort(starting_object_image_size[:,1])
# sorted_ending_heights = np.sort(ending_object_image_size[:,0])
# sorted_ending_widths = np.sort(ending_object_image_size[:,1])
# print(sorted_starting_heights[-20:])
# print(sorted_starting_widths[-20:])
# print(sorted_ending_heights[-20:])
# print(sorted_ending_widths[-20:])
# print(np.sort(starting_object_centers[:,0]))
# print(np.sort(starting_object_centers[:,1]))
# print(np.sort(ending_object_centers[:,0]))
# print(np.sort(ending_object_centers[:,1]))