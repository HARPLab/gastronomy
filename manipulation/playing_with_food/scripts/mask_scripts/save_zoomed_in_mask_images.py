import cv2
import numpy as np
from scipy.spatial import distance
import pickle

processed_overhead_mask_data = np.load('/home/klz/Documents/playing_with_food/processed_overhead_masks.npz')

starting_object_centers = processed_overhead_mask_data['starting_object_centers']
ending_object_centers = processed_overhead_mask_data['ending_object_centers']
starting_overhead_images = processed_overhead_mask_data['starting_overhead_images']
ending_overhead_images = processed_overhead_mask_data['ending_overhead_images']
starting_object_image_size = processed_overhead_mask_data['starting_object_image_size']
ending_object_image_size = processed_overhead_mask_data['ending_object_image_size']
starting_object_images = processed_overhead_mask_data['starting_object_images']
ending_object_images = processed_overhead_mask_data['ending_object_images']
object_position_changes = processed_overhead_mask_data['object_position_changes']
object_size_changes = processed_overhead_mask_data['object_size_changes']

directory = '/home/klz/trial_masks/'

for trial_idx in range(449):
	cv2.imwrite(directory+str(trial_idx)+'.png', starting_object_images[trial_idx,:,:] * 255)