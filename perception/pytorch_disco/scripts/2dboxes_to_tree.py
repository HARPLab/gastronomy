import os
import glob
import ipdb
st = ipdb.set_trace
join = os.path.join
import pickle

folderMod_dict = {
"aa3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION","ab3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_1",
"ac3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_2","ad3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_3",
"ae3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_4","af3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_5",
"ag3d_l":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_6",
"ba3d_l":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_1","bb3d_l":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_2",
"bc3d_l":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_3","bd3d_l":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_4",
}
root_folder =  "/home/mprabhud/dataset/clevr_veggies"
folder_name  = "/home/mprabhud/dataset/clevr_veggies_shamit/npys"

all_folders = ["aa3d_l","ab3d_l","ac3d_l","ad3d_l","ae3d_l","af3d_l","ag3d_l"]
all_folders = ["ba3d_l","bb3d_l","bc3d_l","bd3d_l"]

for folder in all_folders:
	main_folder = folderMod_dict[folder]
	folder_name_use = join(folder_name,folder)
	all_files = glob.glob(folder_name_use+"/*")
	print(folder)
	for file in all_files:
		pickled_file = pickle.load(open(file,"rb"))
		filename = pickled_file['tree_seq_filename']
		print(filename)
		all_names = [main_folder]+filename.split("/")[-3:]
		tree_filename = "/".join(all_names)
		tree_filename_complete = join(root_folder,tree_filename)
		tree = pickle.load(open(tree_filename_complete,"rb"))
		tree.bbox2d_3d = pickled_file['bbox_3d_from_2d']
		# st()
		pickle.dump(tree,open(tree_filename_complete,"wb"))
		pickled_file['tree_seq_filename'] = tree_filename
		print(tree_filename)
		# pickle.dump(pickled_file,open(file,"wb"))
		# print(file)
		# print(pickled_file['bbox_3d_from_2d'].shape)
	print("done")
