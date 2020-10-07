import os
import glob
import random

# <<<<<<< HEAD
mod = "mc_small"
# =======
# mod = "single_obj_480_i"
# >>>>>>> 3405b31e4793c3f80b3a119a45d60cf8ff66b88e

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
	dataset = ["fc"]
	out_dir_base = "/home/shamitl/datasets/clevr_vqa/npy"
	dataset = ["ba","bb","bc","bd","be","bf","bg","bh","bi","bj","bk","bl","bm","bn"]	
	out_dir_base = "/home/mprabhud/dataset/carla/npy"
	dataset = ["mc"]
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
split = 10

# st()

with open(out_dir_base + '/%st.txt' % mod, 'w') as f:
	for item in all_set[:split]:
		if "*" not in item:
			f.write("%s\n" % prebasename(item))

with open(out_dir_base + '/%sv.txt' % mod, 'w') as f:
	for item in all_set[split:]:
		if "*" not in item:
			f.write("%s\n" % prebasename(item))