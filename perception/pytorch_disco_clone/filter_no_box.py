import numpy as np
import pickle
import ipdb
import glob
st = ipdb.set_trace
data_mod = "bb"
out_data_mod = "cc"
take_glob = True
if take_glob:
	import socket
	hostname = socket.gethostname()
	if 'compute' in hostname:
		root_location = "/home/mprabhud/dataset"
	else:
		root_location = "/projects/katefgroup/datasets/"
	folder_name  = f"{root_location}/replica_processed/npy/{data_mod}/*"	
	txt_file_train = f"{root_location}/replica_processed/npy/{out_data_mod}t.txt"
	file_list = glob.glob(folder_name)
	files_filtered = []
	for f in file_list:
		val = pickle.load(open(f,"rb"))
		if len(val['bbox_origin']) > 0:
			files_filtered.append(f)
	with open(txt_file_train, 'w') as f:
		for item in files_filtered:
			item = "/".join(item.split("/")[-2:])
			if "*" not in item:
				f.write("%s\n" % item)
