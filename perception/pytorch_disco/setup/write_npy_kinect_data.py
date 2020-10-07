import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "carla_trainer_builder1"
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
import math
st = ipdb.set_trace
hostname = socket.gethostname()

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

    base_dir = "/home/sirdome/shamit/datasets/kinect_data/kinect_raw"
    file_dir = "/home/sirdome/shamit/datasets/kinect_data/kinect_processed"

    base_dir = "/hdd/shamit/kinect/raw"
    file_dir = "/hdd/shamit/kinect/processed"
    dumpmod = "single_obj"
    dump_dir = os.path.join(file_dir, dumpmod)
else:
    base_dir = "/projects/katefgroup/datasets/carla_objdet/_carla_multiview_single_vehicle_multiple_camRs"
    file_dir = "/home/shamitl/datasets/carla_objdet/npy"
    dumpmod = "bb"
    base_dir = "/projects/katefgroup/datasets/kinect/big_points_multi"
    file_dir = "/projects/katefgroup/datasets/kinect/npys"
    dumpmod = "nips_big_multi"
    dump_dir = os.path.join(file_dir, dumpmod)
    

class write_kinect_to_npy():
    def __init__(self):
        self.H = 320#256
        self.W = 480#256
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # st()
        self.datapts = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.pkl')]
        self.visualize = False
        self.device = torch.device("cuda")
        self.fnames_stored = []
    
    def process_rgbs(self, rgbs):
        rgb_scaled = []
        for rgb in rgbs:
            rgb = imresize(rgb, (self.H, self.W), interp = "bilinear")
            rgb_scaled.append(rgb)
        rgb_scaled = np.stack(rgb_scaled)
        return rgb_scaled
    

    def process_pointclouds(self, xyz_camXs, pix_T_camX):
        pix_T_camX = torch.tensor(pix_T_camX).unsqueeze(0).float()
        processed_xyz_camXs = []
        processed_depths = []
        for xyz_camX in xyz_camXs:
            depth_camX,_ = utils_geom.create_depth_image(pix_T_camX, torch.tensor(xyz_camX).unsqueeze(0).float(), self.H, self.W)
            # st()
            depth_camX[torch.where(depth_camX == 100.0)] = 0.0
            processed_depths.append(depth_camX.squeeze(0).squeeze(0))

            processed_xyz_camX = utils_geom.depth2pointcloud_cpu(depth_camX, pix_T_camX)
            processed_xyz_camXs.append(processed_xyz_camX.squeeze(0))
            
        return torch.stack(processed_xyz_camXs).numpy(), torch.stack(processed_depths).numpy()
        


    def visualize_single_original_pcd(self, xyz_camX, rgb_camX):
        pcd = nlu.make_pcd(xyz_camX)
        o3d.visualization.draw_geometries([pcd, self.mesh_frame])
        num_pts = rgb_camX.shape[0]*rgb_camX.shape[1]
        zeros = np.zeros((num_pts-xyz_camX.shape[0], 3))
        xyz_camX = np.concatenate((zeros, xyz_camX), axis=0)
        colored_pcd = utils_pointcloud.draw_colored_pcd(xyz_camX, rgb_camX)
        o3d.visualization.draw_geometries([colored_pcd, self.mesh_frame])

    def process(self,):
        fnames = []
        for datapt in self.datapts:
            print("datapt is: ", datapt)
            f = pickle.load(open(datapt, "rb"))

            rgb_camX_single = f['rgb_file'][0]
            # st()
            originalH, originalW = rgb_camX_single.shape[0], rgb_camX_single.shape[1]

            rgb_camXs = self.process_rgbs(f['rgb_file'])

            if self.visualize:
                for i in range(rgb_camXs.shape[0]):
                    plt.imshow(rgb_camXs[i])
                    plt.show(block=True)
            pix_T_camX = f['intrinsics']
            scaled_pix_T_camX = utils_geom.scale_intrinsics(torch.tensor(pix_T_camX).unsqueeze(0), 1*self.W/originalW, 1*self.H/originalH).squeeze(0).numpy()
            num_views = rgb_camXs.shape[0]
            # st()

            xyz_camXs = f['point_cloud_file']
            old_xyz_camXs = xyz_camXs

            if self.visualize:
                for xyz_camX in xyz_camXs:
                    pcd = nlu.make_pcd(xyz_camX)
                    o3d.visualization.draw_geometries([pcd, self.mesh_frame])


            # self.visualize_single_original_pcd(old_xyz_camXs[0], rgb_camX_single)
            
            xyz_camXs, depth_camXs = self.process_pointclouds(xyz_camXs, scaled_pix_T_camX)
            
            # st()
            if self.visualize:
                for depth_camX, rgb_camX in zip(depth_camXs, rgb_camXs):
                    utils_pointcloud.visualize_colored_pcd(depth_camX, rgb_camX, scaled_pix_T_camX)
            
            origin_T_camXs = f['extrinisics_final']
            # origin_T_camXs.append(np.eye(4))
            origin_T_camXs = np.stack(origin_T_camXs)
            # st()

            xyz_camRs = utils_geom.apply_4x4(torch.tensor(origin_T_camXs).float(), torch.tensor(xyz_camXs).float()).numpy()
            if self.visualize:
                print("Total point clouds : ", xyz_camRs.shape[0])
                cnt=0
                pcd_list = [self.mesh_frame]
                for xyz_camX, xyz_camR, rgb_camR in zip(xyz_camXs, xyz_camRs, rgb_camXs):
                    print("size of pcd: ", xyz_camX.shape)
                    cnt+=1
                    print("Visualizing pcd number: ", cnt)
                    # st()
                    # Visualize RGB

                    # plt.imshow(rgb_camR)
                    # plt.show(block=True)

                    # Visualize xyz_camX

                    # pcd = nlu.make_pcd(xyz_camX)
                    # o3d.visualization.draw_geometries([pcd, self.mesh_frame])

                    # Visualize colored xyz_camX
                    # colored_pcd = utils_pointcloud.draw_colored_pcd(xyz_camX, rgb_camR)
                    # o3d.visualization.draw_geometries([colored_pcd, self.mesh_frame])

                    # # Visualize xyz_camR
                    pcd = nlu.make_pcd(xyz_camR)
                    # o3d.visualization.draw_geometries([pcd, self.mesh_frame])
                    
                    pcd_list.append(pcd)
                print("Visualizing aggregated point clouds")
                o3d.visualization.draw_geometries(pcd_list)
            data_to_save = {"pix_T_cams_raw": torch.tensor(scaled_pix_T_camX).unsqueeze(0).repeat(num_views, 1, 1).numpy(), "camR_T_origin_raw": torch.tensor(np.eye(4)).unsqueeze(0).repeat(num_views, 1, 1).numpy(), "xyz_camXs_raw": xyz_camXs, "origin_T_camXs_raw": origin_T_camXs, 'rgb_camXs_raw': rgb_camXs}
            # st()
            cur_epoch = str(time.time()).replace(".","")
            pickle_fname = datapt.split('/')[-1].split('.pkl')[0]+"__"+cur_epoch + ".p"
            fnames.append(pickle_fname)
            with open(os.path.join(dump_dir, pickle_fname), 'wb') as f:
                pickle.dump(data_to_save, f)
        return fnames
            





if __name__ == "__main__":
    datawriter = write_kinect_to_npy()
    fnames = datawriter.process()
    trainfile = open(os.path.join(file_dir, dumpmod+"t.txt"),"w")
    for fname in fnames:
        trainfile.write(os.path.join(dumpmod, fname))
        trainfile.write("\n")
    trainfile.close()
