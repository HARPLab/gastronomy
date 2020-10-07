import ipdb
import os
join = os.path.join
import pickle
st = ipdb.set_trace
mod = "bf_l"
main_root_folder = "/projects/katefgroup/datasets/clevr_veggies"
root_folder = "/projects/katefgroup/datasets/clevr_veggies/npys"
loc = "/projects/katefgroup/datasets/clevr_veggies/npys/ag_lt.txt"
loc = '/projects/katefgroup/datasets/clevr_veggies/npys/be_lt.txt'
all_files = open(loc,"rb").readlines()
all_types = []
filenames = []
for file in all_files:
	filename_npy = join(root_folder,file.decode("utf-8")[:-1])
	npy = pickle.load(open(filename_npy,"rb"))
	tree_filename = join(main_root_folder,npy['tree_seq_filename'])
	tree = pickle.load(open(tree_filename,"rb"))
	try:
		all_types.append(type(tree.bbox2d_3d))
		filenames.append(file.decode("utf-8")[:-1])
	except Exception as e:
		st()
	print(set(all_types))

with open(root_folder + '/%st.txt' % mod, 'w') as f:
	for item in filenames:
		if "*" not in item:
			f.write("%s\n" % item)
# st()
print("check")
