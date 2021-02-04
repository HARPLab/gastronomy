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

trials_with_no_object_in_ending_image = np.nonzero(object_size_changes[:,0].flatten() == -500)
print(trials_with_no_object_in_ending_image)
trials_with_object_in_ending_image = np.nonzero(object_size_changes[:,0].flatten() > -500)
#print(trials_with_object_in_ending_image)

positive_negative_sample_dict = {}

for trial_idx in trials_with_no_object_in_ending_image[0]:
	trial_dict = {}

	trial_dict['positive'] = trials_with_no_object_in_ending_image[0]
	trial_dict['negative'] = trials_with_object_in_ending_image[0]

	positive_negative_sample_dict[str(trial_idx)] = trial_dict

flattened_object_shape = ending_object_images.reshape((449,-1))

distances = distance.pdist(flattened_object_shape)
square_distances = distance.squareform(distances)
#print(square_distances.shape)

for trial_idx in trials_with_object_in_ending_image[0]:

	trial_distances = square_distances[trial_idx,:].flatten()
	correct_trial_distances = trial_distances[trials_with_object_in_ending_image[0]]
	sorted_distance = np.sort(correct_trial_distances)
	average_distance = sorted_distance[20]
	#print(average_distance)
	positive_samples = np.nonzero(trial_distances <= average_distance)
	intersecting_samples = np.intersect1d(positive_samples[0], trials_with_no_object_in_ending_image[0])
	negative_samples = np.nonzero(trial_distances > average_distance)
	trial_dict = {}

	trial_dict['positive'] = np.setdiff1d(positive_samples[0], trials_with_no_object_in_ending_image[0])
	trial_dict['negative'] = np.concatenate((negative_samples[0], intersecting_samples))

	positive_negative_sample_dict[str(trial_idx)] = trial_dict

pick = open("positive_negative_labels_using_ending_object_shape.pkl",'wb')
pickle.dump(positive_negative_sample_dict,pick)
pick.close()