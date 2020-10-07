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
    file_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/habitat_processed"
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

'''
origin -> habitatCamX -> camX
habitatCamX has z and y opposite of pdisco camX i.e. 180 degree rotated along x axis (pitch)
'''
class write_habitat_to_npy():
    def __init__(self):
        self.H = 256
        self.W = 256
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.scene_dirs = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        self.visualize = False
        self.device = torch.device("cuda")
        # https://github.com/facebookresearch/habitat-sim/issues/80
        self.fov = 90
        self.num_camR_candidates = 3

    def get_habitat_pix_T_camX(self):
        hfov = float(self.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        return pix_T_camX
    
    def get_pix_T_camX(self):
        pix_T_camX = self.get_habitat_pix_T_camX()
        pix_T_camX[0,2] = self.W/2.
        pix_T_camX[1,2] = self.H/2.
        return pix_T_camX

    def get_habitatCamX_T_camX(self):
        # Rotate 180 degrees along x axis
        rot_pi = np.eye(4)
        rot_pi[1,1] = -1
        rot_pi[2,2] = -1
        habitatCamX_T_camX = rot_pi
        return habitatCamX_T_camX

    def get_origin_T_camX(self, pos, rot):
        rotation_0 = quaternion.as_rotation_matrix(rot)
        origin_T_habitatCamX = np.eye(4)
        origin_T_habitatCamX[0:3,0:3] = rotation_0
        origin_T_habitatCamX[0:3,3] = pos
        
        return origin_T_habitatCamX @ self.get_habitatCamX_T_camX()
    
    def generate_xyz_habitatCamXs(self, depth_camXs, rgb_camXs, pix_T_camXs):
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

        

    def test_bbox_projection(self, xyz_camXs_origin_agg, origin_T_camXs, pix_T_camXs, rgb_camX, xyz_camXs, f):
        rgb_camX = rgb_camX.astype(np.float32)
        
        objs_info = f['objects_info']
        st()
        for obj_info in objs_info:
            if obj_info['category_name'] == "table":
                bbox_center = obj_info['bbox_center']
                bbox_size = obj_info['bbox_size']
                print("bbox center and size are: ", bbox_center, bbox_size)

        xmin, xmax = bbox_center[0]-bbox_size[0]/2., bbox_center[0]+bbox_size[0]/2.
        ymin, ymax = bbox_center[1]-bbox_size[1]/2., bbox_center[1]+bbox_size[1]/2.
        zmin, zmax = bbox_center[2]-bbox_size[2]/2., bbox_center[2]+bbox_size[2]/2.

        ends = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
        nlu.only_visualize(nlu.make_pcd(xyz_camXs_origin_agg.numpy()), ends.reshape(1,-1))
        # ends = np.array([center[0]-size[0], center[1]-size[1], center[2]-size[2], center[0]+size[0], center[1]+size[1], center[2]+size[2]])
        ends = ends.reshape(1, 2, 3)

        print("Ends of bbox in origin are: ", ends)
        # st()
        camX_T_origin = utils_geom.safe_inverse(torch.tensor(origin_T_camXs).unsqueeze(0)).float()
        ends_camX = utils_geom.apply_4x4(camX_T_origin, torch.tensor(ends).float()).squeeze(0).reshape(-1).numpy()
        xmin, xmax = min(ends_camX[0], ends_camX[3]), max(ends_camX[0], ends_camX[3])
        ymin, ymax = min(ends_camX[1], ends_camX[4]), max(ends_camX[1], ends_camX[4])
        zmin, zmax = min(ends_camX[2], ends_camX[5]), max(ends_camX[2], ends_camX[5])
        ends_camX = np.array([[xmin, ymin, zmin, xmax, ymax, zmax]])
        print("ends in camX are: ", ends_camX)
        # st()
        nlu.only_visualize(nlu.make_pcd(xyz_camXs), ends_camX)
        plt.imshow(rgb_camX)
        plt.show(block=True)
        utils_pointcloud.draw_boxes_on_rgb(rgb_camX, pix_T_camXs, ends_camX, visualize=True)

    def get_object_info(self, f):
        objs_info = f['objects_info']

        object_dict = {}
        for obj_info in objs_info:
            bbox_center = obj_info['bbox_center']
            bbox_size = obj_info['bbox_size']
            print("bbox center and size and name are: ", bbox_center, bbox_size, obj_info['category_name'])

            xmin, xmax = bbox_center[0]-bbox_size[0], bbox_center[0]+bbox_size[0]
            ymin, ymax = bbox_center[1]-bbox_size[1], bbox_center[1]+bbox_size[1]
            zmin, zmax = bbox_center[2]-bbox_size[2], bbox_center[2]+bbox_size[2]

            ends = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
            # all_bbox_ends.append(ends)

            object_dict[obj_info['instance_id']] = {'category_id': obj_info['category_id'], 'category_name': obj_info['category_name'], 'bbox_origin': ends}

        return object_dict
    

    def process(self):
        fnames = []
        for scene_cnt, scene_dir in enumerate(self.scene_dirs):
            print("Processing scene {}. Scene number {}".format(scene_dir, scene_cnt))
            
            rgb_camXs = []
            depth_camXs = []
            pix_T_camXs = []
            origin_T_camXs = []
            xyz_camXs = []
            habitat_pix_T_camXs = []
            scene_bbox_ends = []
            scene_category_ids = []
            scene_category_names = []
            scene_instance_ids = []
            scene_dict = {}
            pickle_files = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith('.p')]
            for cnt, pickle_file in enumerate(pickle_files):
                
                f = pickle.load(open(pickle_file, "rb"))

                rgb_camXs.append(f['rgb_camX'])
                depth_camXs.append(f['depth_camX'])
                pix_T_camXs.append(self.get_pix_T_camX())
                habitat_pix_T_camXs.append(self.get_habitat_pix_T_camX())
                origin_T_camXs.append(self.get_origin_T_camX(f['sensor_pos'], f['sensor_rot']))
                print("count of pickle file is: ", cnt)
                object_dict = self.get_object_info(f)
                scene_dict.update(object_dict)

            
            for key in scene_dict.keys():
                scene_instance_ids.append(key)
                scene_category_names.append(scene_dict[key]['category_name'])
                scene_category_ids.append(scene_dict[key]['category_id'])
                scene_bbox_ends.append(scene_dict[key]['bbox_origin'])

            
            scene_bbox_ends = np.stack(scene_bbox_ends)

            habitat_pix_T_camXs = np.stack(habitat_pix_T_camXs)
            rgb_camXs_to_save = np.stack(rgb_camXs)[:,:,:,:3]
            rgb_camXs = np.stack(rgb_camXs)[:,:,:,:3].astype(np.float32)/255.
            origin_T_camXs = np.stack(origin_T_camXs)
            depth_camXs = np.stack(depth_camXs)
            pix_T_camXs = np.stack(pix_T_camXs)
            xyz_habitatCamXs = self.generate_xyz_habitatCamXs(depth_camXs, rgb_camXs, habitat_pix_T_camXs)

            if self.visualize:
                print("Showing pointclouds in habitat_camXs coordinate ref frame")
                for xyz_habitatCamX in xyz_habitatCamXs:
                    pcd = nlu.make_pcd(xyz_habitatCamX)
                    o3d.visualization.draw_geometries([pcd, self.mesh_frame])

            # Get xyz_camXs in pydisco coordinate frame.
            # Since its 180 deg rotation, habitatCamX_T_camX and it's inverse will be same. Therefore, not taking inv.
            xyz_camXs = utils_geom.apply_4x4(torch.tensor(self.get_habitatCamX_T_camX()).repeat(xyz_habitatCamXs.shape[0], 1, 1), torch.tensor(xyz_habitatCamXs)).numpy()

            if self.visualize:
                for xyz_camX, rgb_camX in zip(xyz_camXs, rgb_camXs):
                    pcd = nlu.make_pcd(xyz_camX)
                    o3d.visualization.draw_geometries([pcd, self.mesh_frame])

                    pix_T_camX = pix_T_camXs[0]
                    depth, _ = utils_geom.create_depth_image(torch.tensor(pix_T_camX).unsqueeze(0).float(), torch.tensor(xyz_camX).unsqueeze(0).float(), self.H, self.W)
                    depth[torch.where(depth>10)]=0
                    utils_pointcloud.visualize_colored_pcd(depth.squeeze(0).squeeze(0).numpy(), rgb_camX, pix_T_camX)
                    
            xyz_camXs_origin = utils_geom.apply_4x4(torch.tensor(origin_T_camXs), torch.tensor(xyz_camXs))
            xyz_camXs_origin_agg = xyz_camXs_origin.reshape(-1,3)

            # Visualize aggregated pointcloud
            if self.visualize or True:
                pcd_list = [self.mesh_frame]
                for xyz_camX_origin in xyz_camXs_origin:
                    pcd_list.append(nlu.make_pcd(xyz_camX_origin))
                o3d.visualization.draw_geometries(pcd_list)
            
            self.test_bbox_projection(xyz_camXs_origin_agg, origin_T_camXs[2], pix_T_camXs[2], rgb_camXs[2], xyz_camXs[2], f)
            # First num_camR_candidates views will be our camR candidates
            for num_save in range(self.num_camR_candidates):
                camX1_T_origin = utils_geom.safe_inverse(torch.tensor(origin_T_camXs[num_save]).unsqueeze(0)).float().repeat(origin_T_camXs.shape[0], 1, 1).numpy()
                data_to_save = {"camR_index": num_save, "object_category_ids": scene_category_ids, "object_category_names": scene_category_names, "object_instance_ids": scene_instance_ids, "bbox_origin": scene_bbox_ends, "pix_T_cams_raw": pix_T_camXs, "camR_T_origin_raw": camX1_T_origin, "xyz_camXs_raw": xyz_camXs, "origin_T_camXs_raw": origin_T_camXs, 'rgb_camXs_raw': rgb_camXs_to_save}
                cur_epoch = str(time()).replace(".","")
                pickle_fname = cur_epoch + ".p"
                fnames.append(pickle_fname)
                with open(os.path.join(dump_dir, pickle_fname), 'wb') as f:
                    pickle.dump(data_to_save, f)

        return fnames
            




def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)   

if __name__ == "__main__":
    mkdir(dump_dir)
    datawriter = write_habitat_to_npy()
    fnames = datawriter.process()
    random.shuffle(fnames)
    train_len = int(len(fnames)*0.75)
    test_len = len(fnames) - train_len

    
    trainfile = open(os.path.join(file_dir, dumpmod+"t.txt"),"w")
    valfile = open(os.path.join(file_dir, dumpmod+"v.txt"),"w")

    for i in range(len(fnames)):
        if i < train_len:
            trainfile.write(os.path.join(dumpmod, fnames[i]))
            trainfile.write("\n")
        else:
            valfile.write(os.path.join(dumpmod, fnames[i]))
            valfile.write("\n")
    trainfile.close()
    valfile.close()


