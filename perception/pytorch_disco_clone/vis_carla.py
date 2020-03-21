import pickle
import os
import errno
from scipy.misc import imsave
import ipdb

st = ipdb.set_trace
import glob
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

base_path = "/home/shamitl/datasets/carla/npy/bb/*"
all_paths = glob.glob(base_path)
folder_name = "dump_carla"
mkdir_p(folder_name)
import sys
num = sys.argv[1]
for i in range(int(num)):
	val = pickle.load(open(all_paths[i],"rb"))
	obj_name = val["obj_name"]
	name = f"{folder_name}/{obj_name}"
	mkdir_p(name)
	rgb_camx  = val['rgb_camXs_raw']
	for num in range(17):
		imsave(f"{name}/{i}_{num}.jpg",rgb_camx[num])