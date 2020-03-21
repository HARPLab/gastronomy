import ipdb
import glob
st = ipdb.set_trace
data_mod = "temp"
take_glob = False
if take_glob:
	root_location = "/projects/katefgroup/datasets/"
	folder_name  = f"{root_location}/replica_processed/npy/{data_mod}/*"	
	txt_file_train = f"{root_location}/replica_processed/npy/{data_mod}t.txt"
	file_list = glob.glob(folder_name)

	with open(txt_file_train, 'w') as f:
		for item in file_list:
			item = "/".join(item.split("/")[-2:])
			if "*" not in item:
				f.write("%s\n" % item)
	# st()
	# print("hello")
else:
	
	root_location = "/home/mprabhud/dataset"
	txt_file_train = f"{root_location}/replica_processed/npy/{data_mod}t.txt"
	# train_data = open(txt_file_train,"r").readlines()
	import random
	# random.shuffle(train_data)
	new_file_list = ['bb/15837223625466163.p','bb/15837224377690456.p']
	# term_to_use = int(len(train_data)*0.95)
	# train_data = train_data
	with open(txt_file_train, 'w') as f:
		for item in new_file_list:
			if "*" not in item:
				f.write("%s\n" % item)