import pickle
import errno 
import os
import utils_geom
import torch
os.environ["MODE"] = "CLEVR_STA"   
import ipdb
st = ipdb.set_trace

import copy
import glob
import os
import utils_basic
def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python ≥ 2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


join = os.path.join
tree = pickle.load(open("template.tree","rb"))
mod_name = "fc"
out_mod_name = "fc"
import getpass
username = getpass.getuser()
if 'shamit' in username:
	base_dir =  "/home/shamitl/datasets/carla"	
else:
	base_dir =  "/home/mprabhud/dataset/carla"
carla_loc = f"{base_dir}/npy/{mod_name}/*"
carla_loc_out = f"{base_dir}/npy/{out_mod_name}/*"
trees_dir = f"{base_dir}/{out_mod_name}/trees_updated/train"
trees_display_dir = f"{out_mod_name}/trees_updated/train"
__p = lambda x: utils_basic.pack_seqdim(x, 1)
__u = lambda x: utils_basic.unpack_seqdim(x, 1)
mkdir_p(trees_dir)
mkdir_p(carla_loc_out)
lowerdim = False
all_files = glob.glob(carla_loc)
# st()
for file in all_files:
	if "*" not in file:
		val = pickle.load(open(file,"rb"))
		bbox_camR = val["bbox_origin"]
		obj_name = val["obj_name"]
		tree_copy = copy.deepcopy(tree)
		file_name = file.split("/")[-1]
		tree_copy.bbox_origin = bbox_camR
		if lowerdim:
			pix_T_cams = torch.from_numpy(val["pix_T_cams_raw"]).cuda(non_blocking=True).to(torch.float32)
			xyz_camXs = torch.from_numpy(val["xyz_camXs_raw"]).cuda(non_blocking=True).to(torch.float32)
			H = 256
			W=256
			st()
			depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(pix_T_cams, xyz_camXs, H, W)
			dense_xyz_camXs = utils_geom.depth2pointcloud(depth_camXs_, pix_T_cams)
			val["xyz_camXs_raw"] = dense_xyz_camXs.cpu().numpy()
		file = file.replace(f"/{mod_name}/",f"/{out_mod_name}/")
		tree_copy.word = obj_name
		tree_file_name = join(trees_dir,file_name.replace(".p",".tree"))
		display_file_name = join(trees_display_dir,file_name.replace(".p",".tree"))
		# st()
		pickle.dump(tree_copy,open(tree_file_name,"wb"))
		val["tree_seq_filename"] = display_file_name
		pickle.dump(val,open(file,"wb"))
		print("done",file)
	# print("done")