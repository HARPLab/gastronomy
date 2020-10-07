import Nel_Utils as nlu
import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import multiprocess_flag
multiprocess_flag.multi = True
import sys
# import ipdb
import pickle

# import ipdb
# st = ipdb.set_trace
import pathos.pools as pp
import torch
import cv2
import os
import sys
import random
import matplotlib as mpl
from matplotlib import pyplot
import os
import numpy as np
import tensorflow as tf
import glob
import scipy
import os.path as path
import utils_geom
import open3d as o3d
from easydict import EasyDict as edict
# import ipdb
# st = ipdb.set_trace
import shutil
import utils_basic
import utils_pyplot_vis
# import ipdb
# st = ipdb.set_trace
# mpl.use('Agg')

# both
# z ranges from 0.42 to 0.8
# y ranges from 0.2 to -0.3
# x ranges from -0.2 to 0.4


# camX
# z ranges from 0.38 to 0.8
# y ranges from 0.2 to -0.36
# x ranges from -0.2 to 0.3

# camR
# z ranges from 0.42 to 0.8
# y ranges from 0.2 to -0.36
# x ranges from -0.2 to 0.4
camx_zmin,camx_zmax = (0.4,0.8)
camx_ymin,camx_ymax = (-0.36,0.2)
camx_xmin,camx_xmax = (-0.2,0.4)

camr_zmin,camr_zmax = (0.4,0.8)
camr_ymin,camr_ymax = (-0.36,0.2)
camr_xmin,camr_xmax = (-0.2,0.4)

from itertools import permutations
import pickle
sync_dict_keys = [
	'colorIndex1', 'colorIndex2', 'colorIndex3', 'colorIndex4', 'colorIndex5', 'colorIndex6', \
	'depthIndex1', 'depthIndex2', 'depthIndex3', 'depthIndex4', 'depthIndex5', 'depthIndex6', \
	]

N = 5

SAMPLE_DEPTH_PTS = 200000
EPS = 1e-6
USE_ALL_VIEWS = True
MAX_DEPTH_PTS = 320000
MIN_DEPTH_RANGE = 0.01
MAX_DEPTH_RANGE = 5.0
VISUALIZE = False
SAMPLE_DEPTH_PTS =200000
H = int(480 / 2.0)
W = int(640 / 2.0)
NUM_CAMS = 6


DO_BOXES = True
empty_table = DO_BOXES

TFR_MOD = "ab"  

do_matplotlib = True
import socket 
hostname = socket.gethostname()
import sys


if len(sys.argv) > 1:
	folder_names = [sys.argv[1]]


if 'Alien' in hostname:
	o_base_dir = "/media/mihir/dataset/real_data_matching"
	base_dir = o_base_dir
else:
	# o_base_dir = "/projects/katefgroup/datasets/real_data_matching"
	o_base_dir = "/media/mihir/dataset/real_data_lang"
	base_dir = o_base_dir

out_dir_base = '{}/npys'.format(o_base_dir)
out_dir = '%s/%s' % (out_dir_base, TFR_MOD)
utils_basic.mkdir(out_dir)

def close():
	os._exit(1)
  
# to do:
## add check for if file exists; if exists, skip.
## >>>in raw data, write smaller npys, for faster debugging<<<

def process_rgbs(rgb):
	H_, W_, _ = rgb.shape
	assert (H_ == 480)  # otw i don't know what data this is
	assert (W_ == 640)  # otw i don't know what data this is
	# scale down
	rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
	return rgb


def clip(xyz_camXs_single):
	xyz_camXs_single =  xyz_camXs_single[xyz_camXs_single[:, 2] > MIN_DEPTH_RANGE]
	xyz_camXs_single = xyz_camXs_single[xyz_camXs_single[:, 2] < MAX_DEPTH_RANGE]
	V_current = xyz_camXs_single.shape[0]

	if V_current > MAX_DEPTH_PTS:
		xyz_camXs_single = xyz_camXs_single[torch.randperm(V_current)[:V]]
	elif V_current < MAX_DEPTH_PTS:
		zeros = torch.zeros(1,3).repeat(int(MAX_DEPTH_PTS-V_current),1)
		xyz_camXs_single = torch.cat([xyz_camXs_single,zeros],axis=0)
	return xyz_camXs_single

def process_depths(depth):
	depth = depth / 1000.0
	# making everything to mts
	depth = depth.astype(np.float32)
	return depth

def make_pcd(pts):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
	# if the dim is greater than 3 I expect the color
	if pts.shape[1] == 6:
		pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
			if pts[:, 3:].max() > 1. else pts[:, 3:])
	return pcd

def visualize(list_of_pcds):
	o3d.visualization.draw_geometries(list_of_pcds)

def merge_pcds(pcds):
	pts = [np.asarray(pcd.points) for pcd in pcds]
	colors = [np.asarray(pcd.colors) for pcd in pcds]
	# assert len(pts) == 5, "these is the number of supplied pcd, it should match"
	combined_pts = np.concatenate(pts, axis=0)
	combined_colors = np.concatenate(colors, axis=0)
	assert combined_pts.shape[1] == 3, "concatenation is wrong"
	return combined_pts, combined_colors



def depth2xyz(depth_camXs, pix_T_cams):

	"""
	  depth_camXs: B X H X W X 1

	"""
	depth_camXs = torch.tensor(depth_camXs)
	pix_T_cams = torch.tensor(pix_T_cams)
	depth_camXs = depth_camXs.permute([0,3,1,2])
	xyz_camXs = utils_geom.depth2pointcloud_cpu(depth_camXs, pix_T_cams)
	clipped_xyz = []
	for xyz_camX in xyz_camXs:	
		xyz_camX = clip(xyz_camX)
		clipped_xyz.append(xyz_camX)
	xyz_camXs = torch.stack(clipped_xyz)
	return xyz_camXs

def process_xyz(depth, pix_T_cam):	
	H, W = depth.shape
	pix_T_cam = pix_T_cam.astype(np.float32)
	depth = np.reshape(depth, [1, H, W, 1])
	pix_T_cams = np.expand_dims(pix_T_cam, 0)    
	xyz_camXs = depth2xyz(depth, pix_T_cams) 
	return xyz_camXs

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_from_pkl2to3(path):
	with open(path, 'rb') as f:
		data = pickle.load(f, fix_imports=True, encoding="latin1")
	f.close()
	return data

def split_intrinsics(K):
	# K is 3 x 4 or 4 x 4
	fx = K[0,0]
	fy = K[1,1]
	x0 = K[0,2]
	y0 = K[1,2]
	return fx, fy, x0, y0
					
def gen_cube_coordinates(coord):
	vals = np.split(coord,6,1)
	vals = [np.squeeze(i) for i in vals]
	x,y,z,x1,y1,z1 = vals 
	cube_coords = np.array([[x,y,z],[x,y1,z],[x1,y,z],[x,y,z1]])
	cube_coords = np.transpose(cube_coords,axes=[2,0,1])
	return cube_coords

def merge_intrinsics(fx, fy, x0, y0):
	# inputs are shaped []
	K = np.eye(4)
	K[0,0] = fx
	K[1,1] = fy
	K[0,2] = x0
	K[1,2] = y0
	# K is shaped 4 x 4
	return K
							
def scale_intrinsics(K, sx, sy):
	fx, fy, x0, y0 = split_intrinsics(K)
	fx *= sx
	fy *= sy
	x0 *= sx
	y0 *= sy
	return merge_intrinsics(fx, fy, x0, y0)

def calibrate(boxes_r):
	boxes_r[0,4] = -0.165
	return boxes_r

def intra_job_init(data_dir):
	global all_origin_T_camXs,all_rgb_camXs,all_depths,scaled_pix_T_cams,all_pix_T_cams, ar_T_camXs, sync_data, color_dir
	sync_file = '%s/syncedIndexData.pkl' % data_dir
	sync_data = pickle.load(open(sync_file, 'rb'), encoding="latin1")
	color_dir = '%s/colorData' % data_dir
	depth_dir = '%s/depthData' % data_dir

	numfiles = sync_data["depthIndex5"].shape[0]
	intrinsics_file = '%s/../ar_tag/intrinsics.pkl' % data_dir
	all_pix_T_cams = load_from_pkl2to3(intrinsics_file)

	all_depths = {}
	all_rgb_camXs = {}
	all_origin_T_camXs = {}
	scaled_pix_T_cams = {}

	extrinsics_file = '%s/../ar_tag/extrinsics.npy' % (data_dir)
	ar_T_camXs = np.load(extrinsics_file)
	for cam_num in range(4,NUM_CAMS,1):
		cam_name = '%d' % (cam_num+1)
		ar_T_camX = ar_T_camXs[cam_num]
		origin_real_T_origin_table = np.array([[0, 1, 0, 0],
								  [-1, 0, 0, 0],
								  [0, 0, 1, 0],
								  [0, 0, 0, 1]]).astype(np.float32) 		
		ar_T_camX = (ar_T_camX).astype(np.float32)
		ar_T_camX = np.matmul(origin_real_T_origin_table,ar_T_camX)
		all_origin_T_camXs[cam_name] = (ar_T_camX).astype(np.float32)
		scaled_pix_T_cams[cam_name] = scale_intrinsics(all_pix_T_cams[cam_name], 0.5, 0.5).astype(np.float32)
	return numfiles

def preprocess_points(combined_points,subtract=False):
	if subtract:
		xvalids = (combined_points[:,0]<0.45) *(combined_points[:,0] > -0.05)
		yvalids = (combined_points[:,1]<-0.17) *(combined_points[:,1] > -0.45)
		zvalids = (combined_points[:,2]<1.0) *(combined_points[:,2] > 0.50)		
		valids = xvalids * yvalids * zvalids
	else:
		xvalids = (combined_points[:,0]<0.4) *(combined_points[:,0] > -0.2)
		yvalids = (combined_points[:,1]<0.2) *(combined_points[:,1] > -0.3)
		zvalids = (combined_points[:,2]<0.78) *(combined_points[:,2] > 0.43)
		valids = xvalids * yvalids * zvalids    
	return valids

def preprocess_membounds(combined_points,camX=False):
	# camx_zmin,camx_zmax = (0.4,0.8)
	# camx_ymin,camx_ymax = (-0.36,0.2)
	# camx_xmin,camx_xmax = (-0.2,0.4)
	combined_points = np.reshape(combined_points,[-1,3])
	if camX:
		xvalids = (combined_points[:,0]<0.3) *(combined_points[:,0] > -0.2)
		yvalids = (combined_points[:,1]<0.3) *(combined_points[:,1] > -0.3)
		zvalids = (combined_points[:,2]<0.8) *(combined_points[:,2] > 0.4)
		valids = xvalids * yvalids * zvalids    
	else:
		xvalids = (combined_points[:,0]<0.5) *(combined_points[:,0] > -0.1)
		yvalids = (combined_points[:,1]<-0.15) *(combined_points[:,1] > -0.3)
		zvalids = (combined_points[:,2]<1.0) *(combined_points[:,2] > 0.4)
		valids = xvalids * yvalids * zvalids        
	return valids

def job(frame):
	print("frame done",frame)
	frame_ind = int(frame)
	all_cams = [5, 6]
	# we can output all permutations
	cams_to_use = all_cams
	out_fn = '%05d' % frame_ind

	out_fn += '.tfrecord'
	out_fn = '%s/%s_%s' % (out_dir,folder_name, out_fn)
	# st()
	# print(color_dir,"color di/r")
	print(folder_name)
	pix_T_cams_ = []
	rgb_camXs_ = []
	xyz_camXs_ = []
	depths_ = []
	origin_T_camXs_ = []
	camRs_T_origins_ = [] 

	camR_T_origin = np.array([[1, 0, 0, 0.2],
							  [0, 0, -1, -0.2],
							  [0, 1, 0, 0.6],
							  [0, 0, 0, 1]]).astype(np.float32)    		
	if empty_table:
		empty_rgb_camXs_ = []
		empty_xyz_camXs_ = []                
		all_subtracted_points_list = []
	for cam in cams_to_use:
		# print("doing cam",cam)
		## gen a filename
		cam_custom_dir = "Cam%d" %cam
		cam_custom_file = "cam_%d" %cam
		
		rgb_id = sync_data['colorIndex{}'.format(cam)][frame_ind]
		depth_id = sync_data['depthIndex{}'.format(cam)][frame_ind]

		rgb_path = "{}/colorData/Cam{}/cam_{}_color_{}.npy".format(data_dir,cam,cam,rgb_id)
		depth_path = "{}/depthData/Cam{}/cam_{}_depth_{}.npy".format(data_dir,cam,cam,depth_id)

		rgb_im = np.load(rgb_path)
		depth_im = np.load(depth_path)
		rgb_camX_ = process_rgbs(rgb_im)
		rgb_camXs_.append(rgb_camX_)

		scaled_pix_T_cam = scaled_pix_T_cams[str(cam)].astype(np.float32)
		pix_T_cams_.append(scaled_pix_T_cam)
		origin_T_camX = all_origin_T_camXs[str(cam)]
		origin_T_camXs_.append(origin_T_camX)
		# camRs_T_camXs_.append(tf.matmul(camR_T_origin,origin_T_camX))
		camRs_T_origins_.append(camR_T_origin)

		depth = process_depths(depth_im)
		depths_.append(depth)
		
		# st()
		xyz_camX = process_xyz(depth, all_pix_T_cams[str(cam)])
		xyz_camXs_.append(xyz_camX)

		if empty_table and DO_BOXES:
			empty_rgb_path = "{}/colorData/Cam{}/cam_{}_color_{}.npy".format(empty_data_dir,cam,cam,0)
			empty_depth_path = "{}/depthData/Cam{}/cam_{}_depth_{}.npy".format(empty_data_dir,cam,cam,0)

			empty_rgb_im = np.load(empty_rgb_path)
			empty_depth_im = np.load(empty_depth_path)
			empty_rgb_camX = process_rgbs(empty_rgb_im)
			empty_rgb_camXs_.append(empty_rgb_camX)
			empty_depth = process_depths(empty_depth_im)

			empty_xyz_camX = process_xyz(empty_depth, all_pix_T_cams[str(cam)])
			empty_xyz_camXs_.append(empty_xyz_camX)
			# pix_to_cam_current = pix_T_cams[cam_num]
			rgb_current = rgb_camX_
			xyz_camX_current = xyz_camX

			empty_xyz_camX_current = empty_xyz_camX
			origin_T_camXs_current = torch.from_numpy(origin_T_camX).unsqueeze(0)

			xyz_origin = utils_geom.apply_4x4(origin_T_camXs_current,xyz_camX_current)
			empty_xyz_origin = utils_geom.apply_4x4(origin_T_camXs_current,empty_xyz_camX_current)
			camR_T_origin_torch = torch.from_numpy(camR_T_origin).unsqueeze(0)

			xyz_r = utils_geom.apply_4x4(camR_T_origin_torch,xyz_origin)
			empty_xyz_r = utils_geom.apply_4x4(camR_T_origin_torch,empty_xyz_origin)

			xyz_r = np.squeeze(xyz_r.numpy())
			empty_xyz_r = np.squeeze(empty_xyz_r.numpy())
			subtracted_xyz = xyz_r - empty_xyz_r
			norm_new_pts = np.linalg.norm(subtracted_xyz, axis=1)
			subtracted_xyz = xyz_r[norm_new_pts>0.01]
			# st()
			sub_valids = preprocess_points(subtracted_xyz,subtract=True)
			subtracted_xyz = subtracted_xyz[sub_valids]

			# fig, ax_points = utils.pyplot_vis.plot_pointcloud(np.reshape(subtracted_points,[-1,3])[::100], fig_id=3, ax=None,coord="xright-ydown-topview")
			# pyplot.show()
			# pyplot.close()

			# st()
			all_subtracted_points_list.append(subtracted_xyz)
	if empty_table and DO_BOXES:
		all_subtracted_points = np.concatenate(all_subtracted_points_list,axis=0)
	
	if DO_BOXES:	
		print("finding box..")
		all_subtracted_points = all_subtracted_points[::8]
		print(all_subtracted_points.shape,"box shape")
		boxes_r,pcd,valid = nlu.cluster_using_dbscan(all_subtracted_points,5,eps=0.03,min_samples=200,vis=False)
		if not valid:
			return 0
		print("found")
		boxes_r = calibrate(boxes_r).astype(np.float32)
		# st()
		boxes_r_cube = gen_cube_coordinates(boxes_r)
	# # st()

	# print("one")
	camRs_T_origins = np.stack(camRs_T_origins_, axis=0)
	depths = np.stack(depths_,axis=0)
	pix_T_cams = np.stack(pix_T_cams_, axis=0)
	rgb_camXs = np.stack(rgb_camXs_, axis=0)
	xyz_camXs = np.stack(xyz_camXs_, axis=0)
	origin_T_camXs = np.stack(origin_T_camXs_, axis=0)

	if empty_table:
		empty_rgb_camXs = np.stack(empty_rgb_camXs_, axis=0)
		empty_xyz_camXs = np.stack(empty_xyz_camXs_, axis=0)	
	# boxes = np.zeros((N, 6)).tostring()
	# all_mask = np.zeros((24, N)).tostring()
	# all_views_boxes_theta = np.zeros((24, N, 9)).tostring()
	rgb_camXs = np.concatenate([rgb_camXs,np.ones([len(all_cams),H,W,1],np.uint8)*255],axis=-1)

	assert rgb_camXs.dtype == np.uint8
	assert xyz_camXs.dtype == np.float32
	assert origin_T_camXs.dtype == np.float32
	assert pix_T_cams.dtype == np.float32
	assert depths.dtype == np.float32

	if DO_BOXES:
		boxes_raw = boxes_r[0]
		temp_tree = pickle.load(open("temp.tree","rb"))
		temp_tree.word = folder_name
		temp_tree.bbox_origin = boxes_raw
		tree_file_name = "{}/{}".format(root_tree_file,str(frame)+".tree")
		save_tree_filename = "{}/{}".format(data_dir_trees_upd,str(frame)+".tree")
		pickle.dump(temp_tree,open(tree_file_name,"wb"))
	else:
		save_tree_filename = "temp"

	# print(data_dir_trees_upd)

	filename = "/".join(out_fn.split("/")[-2:])
	# print(filename)

	feature = {
		'filename': filename,
		'tree_seq_filename': save_tree_filename,
		'pix_T_cams_raw': pix_T_cams,
		'origin_T_camXs_raw': origin_T_camXs,
		'rgb_camXs_raw': rgb_camXs,
		'xyz_camXs_raw': np.squeeze(xyz_camXs),
		'camR_T_origin_raw' : camRs_T_origins
	}

	comptype = "GZIP"


	if VISUALIZE:
		utils_basic.mkdir("preprocess_vis/dump_tfrs_vis")
		ax_points = None
		origin_T_camXs_selected = []
		all_subtracted_points = []
		cams = list(range(len(all_cams)))
		for cam_num in cams:
			pix_to_cam_current = pix_T_cams[cam_num]
			rgb_current = rgb_camXs[cam_num]
			xyz_camX_current = xyz_camXs[cam_num]
			origin_T_camXs_current = origin_T_camXs[cam_num]
			origin_T_camXs_selected.append(origin_T_camXs_current)
			origin_T_camXs_current = torch.from_numpy(origin_T_camXs_current).unsqueeze(0)
			xyz_camX_current = torch.from_numpy(xyz_camX_current)
			xyz_origin = utils_geom.apply_4x4(origin_T_camXs_current,xyz_camX_current)
			camR_T_origin_torch = torch.from_numpy(camR_T_origin).unsqueeze(0)
			xyz_r = utils_geom.apply_4x4(camR_T_origin_torch,xyz_origin)

			scipy.misc.imsave('preprocess_vis/dump_tfrs_vis/rgb_cam%d.png' % (cam_num),rgb_current)            

			if do_matplotlib :
				if True:
					xyz_camX_current = xyz_camX_current.reshape([-1,3])
					xyz_camX_current_valid = preprocess_membounds(xyz_camX_current,camX = True)
					xyz_camX_current = xyz_camX_current[xyz_camX_current_valid]
					xyz_camX_current = xyz_camX_current.unsqueeze(0)
					xyz_origin_current = utils_geom.apply_4x4(origin_T_camXs_current,xyz_camX_current)
					xyz_r_current = utils_geom.apply_4x4(camR_T_origin_torch,xyz_origin_current)
					xyz_r_current = np.reshape(xyz_r_current,[-1,3])
					xyz_r_current = xyz_r_current
				fig, ax_points = utils_pyplot_vis.plot_pointcloud(np.reshape(xyz_r_current,[-1,3])[::10], fig_id=3, ax=ax_points,coord="xright-ydown-topview",zlims=[0.3,0.9])
				# fig, ax_points = utils_pyplot_vis.plot_pointcloud(np.reshape(xyz_camX_current,[-1,3])[::10], fig_id=3, ax=ax_points,coord="xright-ydown-topview",zlims=[0.3,0.9])

				# pyplot.clf()
			else:
				empty_xyz_camX_current = empty_xyz_camXs[cam_num]
				empty_xyz_origin = utils_geom.apply_4x4(np.expand_dims(origin_T_camXs_current,axis=0),np.expand_dims(empty_xyz_camX_current,axis=0))
				empty_xyz_r = utils.geom.apply_4x4(np.expand_dims(camR_T_origin_torch,axis=0),empty_xyz_origin)
				xyz_r = np.squeeze(xyz_r.numpy())
				empty_xyz_r = np.squeeze(empty_xyz_r.numpy())
				subtracted_xyz = xyz_r - empty_xyz_r
				norm_new_pts = np.linalg.norm(subtracted_xyz, axis=1)
				subtracted_xyz = xyz_r[norm_new_pts>0.01]
				norm_new_pcd = make_pcd(np.concatenate([subtracted_xyz,np.zeros_like(subtracted_xyz)],axis=1))
				empty_pcd = make_pcd(np.concatenate([empty_xyz_r,np.ones_like(empty_xyz_r)*0.5],axis=1))
				combined_points,combined_color =  merge_pcds([norm_new_pcd,empty_pcd])

				sub_valids = preprocess_points(subtracted_xyz,subtract=True)
				subtracted_points = subtracted_xyz[sub_valids]
				# subtracted_pcds = make_pcd(subtracted_points)
				all_subtracted_points.append(subtracted_points)
				# boxes,pcd = nlu.cluster_using_dbscan(subtracted_points,20,eps=0.08,min_samples=500,vis=True)
				# valids = preprocess_points(combined_points)
				# combined_points = combined_points[valids]
				# combined_color = combined_color[valids]
				# combined_pcd = make_pcd(np.concatenate([combined_points,combined_color],axis=1))
				# scipy.misc.imsave("vis/check_rgb.png",rgb_current)
				# print("camera ",cam_num)
				# visualize([combined_pcd])
		# st()
		# all_subtracted_pcds = make_pcd(all_subtracted_points)
		# visualize([all_subtracted_pcds])
		# st()
		# all_subtracted_pcds = merge_pcds(all_subtracted_pcds)
		# utils.pyplot_vis.plot_cam(tf.concat(origin_T_camXs_selected, 0), fig_id=2, xlims = [-13.0, 13.0], ylims = [-13.0, 13.0], zlims=[-13, 13.0], length=2.0)
		if do_matplotlib:
			if DO_BOXES:
				fig, ax_points = utils_pyplot_vis.plot_cube(boxes_r_cube,fig=3,ax=ax_points)
			# st()
			pyplot.show()
			pyplot.close()
			# pyplot.clf()
		else:
			all_subtracted_points = np.concatenate(all_subtracted_points,axis=0)
			boxes,pcd = nlu.cluster_using_dbscan(all_subtracted_points[::100],20,eps=0.08,min_samples=50,vis=True)
	# shape_dict = print_feature_shapes(feature)
	pickle.dump(feature,open(out_fn,"wb"))
	sys.stdout.write('.')
	sys.stdout.flush()

def print_feature_shapes(fs):
	shape_dict = {}
	for k,i in fs.items():
		if isinstance(i,type(np.array([]))):
			shape_dict[k] = i.shape
	print(shape_dict)
	return shape_dict
# {'pix_T_cams_raw': (2, 4, 4), 'origin_T_camXs_raw': (2, 4, 4), 'rgb_camXs_raw': (2, 240, 320, 3), 'xyz_camXs_raw': (2, 1, 320000, 3)}
# {'pix_T_cams_raw': (24, 4, 4), 'origin_T_camXs_raw': (24, 4, 4), 'rgb_camXs_raw': (24, 256, 256, 4), 'camR_T_origin_raw': (24, 4, 4), 'xyz_camXs_raw': (24, 65536, 3)}

def main(parallel):
	import sys
	global data_dir, nFrames, empty_data_dir,folder_name, data_dir_trees_upd, folder_name, root_tree_file
	# folder_name = sys.argv[1]\

	for folder_name in folder_names:
		data_dir = "{}/{}/rgb_depth_npy".format(base_dir, folder_name)
		data_dir_trees_upd = "{}/trees_updated/train".format(folder_name)
		root_tree_file = "{}/{}".format(base_dir,data_dir_trees_upd)
		utils_basic.mkdir(root_tree_file)

		empty_folder_name = "empty_table"    
		empty_data_dir = "{}/{}/rgb_depth_npy".format(base_dir, empty_folder_name)
		iFrame = 0
		num_jobs = intra_job_init(data_dir)
		print(num_jobs,"num jobs")

		if parallel:
			jobs_list = list(range(0,num_jobs,2))
			p = pp.ProcessPool(4)
			p.map(job, jobs_list, chunksize=1)
		else:
			for index in range(0,num_jobs,2):
				print("job index",index)
				job(index)

		tfrecord_list = glob.glob(out_dir+"/*")
		random.shuffle(tfrecord_list)
		split = int(len(tfrecord_list)*0.9)

		with open(out_dir_base + '/%st.txt' % TFR_MOD, 'w') as f:
			for item in tfrecord_list[:split]:
				f.write("%s/%s\n" % (TFR_MOD,os.path.basename(item)))

		with open(out_dir_base + '/%sv.txt' % TFR_MOD, 'w') as f:
			for item in tfrecord_list[split:]:
				f.write("%s/%s\n" % (TFR_MOD,os.path.basename(item)))
		print('done')
		print('done')



if __name__ == '__main__':
	main(True)
# don't use cam0
# 0:5,1:7,2:9,3:8,4:9,5:10
