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
import quaternion
from PIL import Image
from lib_classes import Nel_Utils as nlu

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
    base_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/habitat"
    file_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/carla/data/_carla_multiview_two_vehicles"
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

if os.path.exists(os.path.join(file_dir, dumpmod)):
    print("This datamod already exists. Terminating")
    # exit(1)
#    save_data = {'rgb_camX':rgb, 'depth_camX': depth, 'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}

class write_habitat_to_npy():
    def __init__(self):
        self.H = 256
        self.W = 256
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.scene_dirs = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        self.visualize = True
        self.device = torch.device("cuda")
        # https://github.com/facebookresearch/habitat-sim/issues/80
        self.fov = 90
        self.pix_T_camXs = self.get_pix_T_camX()

    def get_pix_T_camX(self):
        hfov = float(self.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        return pix_T_camX
        # focal = self.get_focal()
        # pix_T_camX = np.eye(4)
        # pix_T_camX[0,0] = focal
        # pix_T_camX[1,1] = -focal
        # return pix_T_camX
    
    def get_pdisco_pix_T_camX(self):
        pix_T_camX = self.get_pix_T_camX()
        pix_T_camX[0,2] = self.W/2.
        pix_T_camX[1,2] = self.H/2.
        return pix_T_camX


    def get_focal(self):
        focal = self.W / 2.0 * 1.0 / math.tan(self.fov * math.pi / 180 / 2)
        # focal = 1.0 / math.tan(self.fov * math.pi / 180 / 2)
        return focal

    
    def get_camX_T_camPdisco(self):
        # Rotate along x axis
        rot_pi = np.eye(4)
        rot_pi[1,1] = -1
        rot_pi[2,2] = -1
        camX_T_camPdisco = rot_pi
        return camX_T_camPdisco

    def get_origin_T_camX(self, pos, rot):
        rotation_0 = quaternion.as_rotation_matrix(rot)
        origin_T_camX = np.eye(4)
        origin_T_camX[0:3,0:3] = rotation_0
        origin_T_camX[0:3,3] = pos
        
        
        return origin_T_camX @ self.get_camX_T_camPdisco()
    
    def follow_habitat_documentation(self, depth_camXs, rgb_camXs, pix_T_camXs):
        xyz_camXs = []
        for i in range(depth_camXs.shape[0]):
            # st()
            K = pix_T_camXs[i]
            xs, ys = np.meshgrid(np.linspace(-1*self.W/2.,1*self.W/2.,self.W), np.linspace(1*self.W/2.,-1*self.W/2.,self.W))
            depth = depth_camXs[i].reshape(1,self.W,self.W)
            xs = xs.reshape(1,self.W,self.W)
            ys = ys.reshape(1,self.W,self.W)
            # st()
            xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
            xys = xys.reshape(4, -1)
            xy_c0 = np.matmul(np.linalg.inv(K), xys)
            xyz_camX = xy_c0.T[:,:3]
            xyz_camXs.append(xyz_camX)
            if self.visualize and False:
                pcd = nlu.make_pcd(xyz_camX)
                o3d.visualization.draw_geometries([pcd, self.mesh_frame])
        return np.stack(xyz_camXs)

        

    def test_bbox_projection(self, origin_T_camXs, pix_T_camXs, rgb_camX, f):
        camX_T_camPdisco = self.get_camX_T_camPdisco()
        pix_T_camXs = self.get_pdisco_pix_T_camX()
        objs_info = f['objects_info']
        for obj_info in objs_info:
            if obj_info['class_name'] == "lamp":
                bbox = obj_info['oriented_bbox']
                break
        center = np.array(bbox['abb']['center'])
        size = np.array(bbox['abb']['sizes'])
        # Ignore rotation for now
        ends = np.array([center[0]-size[0], center[1]-size[1], center[2]-size[2], center[0]+size[0], center[1]+size[1], center[2]+size[2]])
        


        



    def process(self):
        for scene_cnt, scene_dir in enumerate(self.scene_dirs):
            print("Processing scene {}. Scene number {}".format(scene_dir, scene_cnt))
            
            rgb_camXs = []
            depth_camXs = []
            pix_T_camXs = []
            origin_T_camXs = []
            xyz_camXs = []
            pickle_files = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith('.p')]
            for pickle_file in pickle_files:
                f = pickle.load(open(pickle_file, "rb"))
                rgb_camXs.append(f['rgb_camX'])
                depth_camXs.append(f['depth_camX'])
                pix_T_camXs.append(self.pix_T_camXs)
                origin_T_camXs.append(self.get_origin_T_camX(f['sensor_pos'], f['sensor_rot']))
            

            self.test_bbox_projection(origin_T_camXs[-1], pix_T_camXs[-1], rgb_camXs[-1], f)
            rgb_camXs = np.stack(rgb_camXs)[:,:,:,:3]
            origin_T_camXs = np.stack(origin_T_camXs)
            depth_camXs = np.stack(depth_camXs)
            pix_T_camXs = np.stack(pix_T_camXs)
            xyz_camXs = self.follow_habitat_documentation(depth_camXs, rgb_camXs, pix_T_camXs)
            if self.visualize and False:
                # st()
                for xyz_camX in xyz_camXs:
                    pcd = nlu.make_pcd(xyz_camX)
                    o3d.visualization.draw_geometries([pcd, self.mesh_frame])

            # Get xyz_camXs in pydisco coordinate frame.
            xyz_camXs = utils_geom.apply_4x4(torch.tensor(self.get_camX_T_camPdisco()).repeat(xyz_camXs.shape[0], 1, 1), torch.tensor(xyz_camXs)).numpy()

            if self.visualize and False:
                # st()
                for xyz_camX, rgb_camX in zip(xyz_camXs, rgb_camXs):
                    pcd = nlu.make_pcd(xyz_camX)
                    o3d.visualization.draw_geometries([pcd, self.mesh_frame])

                    pdisco_pix_T_camX = self.get_pdisco_pix_T_camX()
                    depth, _ = utils_geom.create_depth_image(torch.tensor(pdisco_pix_T_camX).unsqueeze(0).float(), torch.tensor(xyz_camX).unsqueeze(0).float(), self.H, self.W)
                    depth[torch.where(depth>10)]=0
                    # xyz_camX = utils_geom.depth2pointcloud_cpu(depth, torch.tensor(pdisco_pix_T_camX).unsqueeze(0).float()).squeeze(0).numpy()
                    # pcd = nlu.make_pcd(xyz_camX)
                    # o3d.visualization.draw_geometries([pcd, self.mesh_frame])
                    # st()
                    utils_pointcloud.visualize_colored_pcd(depth.squeeze(0).squeeze(0).numpy(), rgb_camX, pdisco_pix_T_camX)
                    
            xyz_camXs_origin = utils_geom.apply_4x4(torch.tensor(origin_T_camXs), torch.tensor(xyz_camXs))

            # Visualize aggregated pointcloud
            if self.visualize or False:
                pcd_list = [self.mesh_frame]
                for xyz_camX_origin in xyz_camXs_origin:
                    pcd_list.append(nlu.make_pcd(xyz_camX_origin))
                o3d.visualization.draw_geometries(pcd_list)
            
                

            st()
            # xyz_camXs = utils_geom.depth2pointcloud_cpu(torch.tensor(depth_camXs).unsqueeze(1).float(), torch.tensor(pix_T_camXs).float()).numpy()

            if self.visualize and False:
                # st()
                for rgb_camX, xyz_camX in zip(rgb_camXs, xyz_camXs):
                    plt.imshow(rgb_camX)
                    plt.show(block=True)
                    pcd = nlu.make_pcd(xyz_camX)
                    o3d.visualization.draw_geometries([pcd, self.mesh_frame])
            




    

if __name__ == "__main__":
    datawriter = write_habitat_to_npy()
    datawriter.process()

    