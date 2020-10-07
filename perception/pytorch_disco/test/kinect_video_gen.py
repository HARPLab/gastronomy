import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "trainer_quantize_object_no_detach_rotate"
os.environ["run_name"] = "check"

import numpy as np 
import socket
import open3d as o3d
import pickle
import h5py 
from time import time

from PIL import Image
from lib_classes import Nel_Utils as nlu

import time
import matplotlib.pyplot as plt
import torch
import utils_geom
import utils_pointcloud
import utils_improc
from scipy.misc import imresize
import ipdb
import random
import matplotlib.pyplot as plt
import math
st = ipdb.set_trace
hostname = socket.gethostname()
import cv2

if "Shamit" in hostname:
    base_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/kinect"
    file_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/kinect_processed"
    dumpmod = "bb"
    dump_dir = os.path.join(file_dir, dumpmod)
elif "baxterstation" in hostname:
    base_dir = "/home/nel/shamit/carla96/PythonAPI/examples/_carla_multiview"
    file_dir = "/projects/katefgroup/datasets/carla_sm/npy"
    dumpmod = "aa"
    dump_dir = os.path.join(file_dir, dumpmod)
elif "domestation" in hostname:
    base_dir = "/hdd/carla97/PythonAPI/examples/_carla_multiview_two_vehicles"
    file_dir = "/hdd/carla97/PythonAPI/examples/pdisco_npys/npy"
    dumpmod = "tv"
    dump_dir = os.path.join(file_dir, dumpmod)
else:
    base_dir = "/projects/katefgroup/datasets/carla_objdet/_carla_multiview_single_vehicle_multiple_camRs"
    file_dir = "/home/shamitl/datasets/carla_objdet/npy"
    dumpmod = "bb"
    dump_dir = os.path.join(file_dir, dumpmod)
    
files = ['paper26','paper19','paper22','paper24','paper25']

def process_rgbs(rgbs):
    rgb_scaled = []
    for rgb in rgbs:
        rgb = imresize(rgb, (256,256), interp = "bilinear")
        rgb_scaled.append(rgb)
    rgb_scaled = np.stack(rgb_scaled)
    return rgb_scaled

def visualize_all_rgbs():
    f1 = pickle.load(open(os.path.join(base_dir,"paper19_rgb_data.pkl"),"rb"))
    f2 = pickle.load(open(os.path.join(base_dir,"paper22_rgb_data.pkl"),"rb"))
    f3 = pickle.load(open(os.path.join(base_dir,"paper24_rgb_data.pkl"),"rb"))
    f4 = pickle.load(open(os.path.join(base_dir,"paper25_rgb_data.pkl"),"rb"))
    img1 = f1['rgb_file'][-1]
    img2 = f2['rgb_file'][-1]
    img1_2 = np.concatenate((img1, img2), axis=1)
    img3 = f3['rgb_file'][-1]
    # st()
    img4 = f4['rgb_file'][-1]
    img3_4 = np.concatenate((img3, img4), axis=1)
    img_f = np.concatenate((img1_2, img3_4), axis=0)
    plt.imshow(img_f)
    plt.show(block=True)

    

if __name__ == "__main__":
    # visualize_all_rgbs()
    # text_dict = {'paper26': ['C7_R240','C1_R0','C9_R190','C13_R80','C5_R310'], 'paper19':['C4_R10','C11_R0','C10_R30'], 'paper22':['C11_R350','C7_R250','C10_R50'],'paper24':['C18_R90','C7_R30'],'paper25':['C18_R10','C7_R120','C2_R180']}
    text_dict = {'paper26': ['C7_R242','C1_R1','C9_R191','C13_R88','C5_R311'], 'paper19':['C4_R10','C11_R3','C10_R27'], 'paper22':['C11_R350','C7_R258','C10_R55'],'paper24':['C18_R92','C7_R31'],'paper25':['C18_R14','C7_R116','C2_R177']}
    img_array = []
    for pname in files:
        print("file name is: ", pname)
        maindata = pickle.load(open(os.path.join(base_dir,"{}_rgb_data.pkl".format(pname)),"rb"))
        bbox = pickle.load(open(os.path.join(base_dir,'bbox',"{}.p".format(pname)),"rb"))
        bbox_origin = bbox['bbox_origin_predicted']
        pix_T_camX = maindata['intrinsics']
        rgb_camX_single = maindata['rgb_file'][0]
        originalH, originalW = rgb_camX_single.shape[0], rgb_camX_single.shape[1]
        rgb_camXs = process_rgbs(maindata['rgb_file'])
        scaled_pix_T_camX = utils_geom.scale_intrinsics(torch.tensor(pix_T_camX).unsqueeze(0), 1*256/originalW, 1*256/originalH).squeeze(0).numpy()
        
        origin_T_camXs = maindata['extrinisics_final']
        origin_T_camXs.append(np.eye(4))
        origin_T_camXs = torch.tensor(np.stack(origin_T_camXs))
        camXs_T_origin = utils_geom.safe_inverse(origin_T_camXs)

        bbox_origin_theta = nlu.get_alignedboxes2thetaformat(torch.tensor(bbox_origin).reshape(1, -1, 2, 3))
        bbox_origin_corners = utils_geom.transform_boxes_to_corners(bbox_origin_theta).float()
        B,N,num_pts, xyz = bbox_origin_corners.shape
        assert B==1
        bbox_origin_corners = bbox_origin_corners.reshape(B, N*num_pts, xyz)
        bbox_camX_corners = utils_geom.apply_4x4(camXs_T_origin.float(), bbox_origin_corners).reshape(camXs_T_origin.shape[0],N,num_pts,xyz)
        # st()
        
        summaryWriter = utils_improc.Summ_writer(None, 10, "train")
        for rgb_camX, bbox_camX_corner in zip(rgb_camXs, bbox_camX_corners):

            rgb_for_bbox_vis = torch.tensor(rgb_camX).permute(2, 0, 1).unsqueeze(0)
            rgb_for_bbox_vis = utils_improc.preprocess_color(rgb_for_bbox_vis)
            bbox_camX_corner = bbox_camX_corner.unsqueeze(0)
            scores = torch.ones((bbox_camX_corner.shape[0], bbox_camX_corner.shape[1]))
            tids = torch.ones_like(scores)
            intrinsics = torch.tensor(scaled_pix_T_camX).unsqueeze(0)
            
            # st()
            # self.draw_boxes_using_ends(bbox_camX_ends, rgb_for_bbox_vis, intrinsics)
            info_text = text_dict[pname]
            rgb_with_bbox = summaryWriter.summ_box_by_corners_parses("2Dto3D", rgb_for_bbox_vis, bbox_camX_corner, scores, tids, intrinsics, info_text, only_return=True)
            # st()
            rgb_with_bbox = utils_improc.back2color(rgb_with_bbox)
            rgb_with_bbox = rgb_with_bbox.permute(0, 2, 3, 1).squeeze(0).numpy()
            img_array.append(rgb_with_bbox[:,:,::-1])

            # if True:
            #     plt.imshow(rgb_with_bbox)
            #     plt.show(block=True)

    print("Creating video")
    out = cv2.VideoWriter('~/Desktop/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (256, 256))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

        




