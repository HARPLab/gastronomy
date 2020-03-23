import os
import glob
import random

mod = "bb_tv"

all_set = []
import ipdb
def prebasename(val):
	return "/".join(val.split("/")[-2:])
st = ipdb.set_trace
import socket
hostname = socket.gethostname()
if "Alien" in hostname:
	out_dir_base = "/media/mihir/dataset/clevr_veggies/npys"
	dataset = ["ca"]
elif "compute" in hostname:
	out_dir_base = "/projects/katefgroup/datasets/clevr_veggies/npys"
	out_dir_base = "/home/mprabhud/dataset/clevr_veggies/npys"
	out_dir_base = '/home/shamitl/datasets/carla/npy'
	out_dir_base = "/home/mprabhud/dataset/carla/npy"
	dataset = ["fc"]
else:
	out_dir_base = "/projects/katefgroup/datasets/carla/npy"
	dataset = ["bb","tv_updated"]	
	# dataset = ["ba_l","bb_l","bc_l","bd_l"]

for i in dataset:
	all_set =  all_set + glob.glob("%s/%s/*"%(out_dir_base,i))
	print(all_set)
print(len(all_set))
random.shuffle(all_set)
split = int(len(all_set)*0.95)

# st()

with open(out_dir_base + '/%st.txt' % mod, 'w') as f:
	for item in all_set[:split]:
		if "*" not in item:
			f.write("%s\n" % prebasename(item))

with open(out_dir_base + '/%sv.txt' % mod, 'w') as f:
	for item in all_set[split:]:
		if "*" not in item:
			f.write("%s\n" % prebasename(item))