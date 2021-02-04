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

flattened_object_shape = starting_object_images.reshape((449,-1))

distances = distance.pdist(flattened_object_shape)
square_distances = distance.squareform(distances)
#print(square_distances.shape)

positive_negative_sample_dict = {}

for trial_idx in range(449):
	trial_distances = square_distances[trial_idx,:].flatten()
	sorted_distance = np.sort(trial_distances)
	average_distance = sorted_distance[20]
	#print(average_distance)
	positive_samples = np.nonzero(trial_distances <= average_distance)
	negative_samples = np.nonzero(trial_distances > average_distance)
	trial_dict = {}

	trial_dict['positive'] = positive_samples[0]
	trial_dict['negative'] = negative_samples[0]

	positive_negative_sample_dict[str(trial_idx)] = trial_dict

pick = open("positive_negative_labels_using_starting_object_shape.pkl",'wb')
pickle.dump(positive_negative_sample_dict,pick)
pick.close()