import numpy as np
import pickle
import glob
import ipdb
import glob
st = ipdb.set_trace
data_mod = "bb"
take_glob = True
num_boxes = []
name_classes = []
if take_glob:
	root_location = "/projects/katefgroup/datasets/"
	folder_name  = f"{root_location}/replica_processed/npy/{data_mod}/*"	
	# txt_file_train = f"{root_location}/replica_processed/npy/{data_mod}t.txt"
	file_list = glob.glob(folder_name)
	all_classes = []
	for file in file_list:
		pickled_file = pickle.load(open(file,"rb"))
		bbox_origin = pickled_file["bbox_origin"]
		classes = pickled_file['object_category_names']
		all_classes = all_classes + classes
		num_bbox_origin = len(pickled_file["bbox_origin"])
		category_names = pickled_file["object_category_names"]
		num_boxes.append(num_bbox_origin)
		name_classes.append(category_names)
	num_boxes = np.array(num_boxes)
	name_classes = np.array(name_classes)
	box_ind = np.argmax(num_boxes)
	unique_classes = set(all_classes)
	print(unique_classes,len(unique_classes))
	print(name_classes[box_ind])
	print(np.max(num_boxes))
	# st()
	# print(max(num_boxes))
	# pickled_file[]