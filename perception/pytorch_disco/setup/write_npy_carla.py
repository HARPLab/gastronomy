import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "replica_multiview_builder"
os.environ["run_name"] = "check"
import numpy as np 
import socket
import open3d as o3d
import pickle
import h5py 
from time import time

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

#############################################
### IMPORTANT : Dont use this script now #####
### Use write_npy_carla_n_vehicles  ##########
#############################################



'''
Update the mod information here:
cc -> Data generated with 2 views and camR at 0 degree azimuth
dd -> Data generated with 2 views and camR at random degree azimuth
'''

if "Shamit" in hostname:
    base_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/carla/data/_carla_multiview_single_vehicle_multiple_camRs"
    file_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/carla/data/_carla_multiview_single_vehicle_multiple_camRs"
    dumpmod = "aa"
    dump_dir = os.path.join(file_dir, dumpmod)
elif "baxterstation" in hostname:
    base_dir = "/home/nel/shamit/carla96/PythonAPI/examples/_carla_multiview"
    file_dir = "/projects/katefgroup/datasets/carla_sm/npy"
    dumpmod = "aa"
    dump_dir = os.path.join(file_dir, dumpmod)
elif "domestation" in hostname:
    base_dir = "/hdd/carla97/PythonAPI/examples/_carla_multiview"
    base_dir = "/hdd/carla97/PythonAPI/examples/CarlaMultiview/_carla_multiview_single_vehicle_multiple_camRs_far_camera"
    file_dir = "/hdd/carla97/PythonAPI/examples/pdisco_npys/npy"
    dumpmod = "singleVehicle_farCam"
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
IMPORTANT ASSUMPTIONS FOR THIS SCRIPT
1. Intrinsics are same for all cameras
2. Corresponding RGB and Depth cameras have same extrinsics
3. Origin is at the car
4. CamR is the first camera
'''

class write_carla_to_npy():
    def __init__(self, episode):
        self.carla_T_cam = np.eye(4, dtype=np.float32)
        self.carla_T_cam[0,0] = -1.0
        self.carla_T_cam[1,1] = -1.0
        self.cam_T_carla = np.linalg.inv(self.carla_T_cam)
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.episode_path = os.path.join(base_dir, episode)
        self.finalH = 256
        self.finalW = 256
        
        self.pix_T_cam = np.load(os.path.join(self.episode_path, "intrinsics.npy"))
        self.pix_T_cam[0,0] *= -1
        self.pix_T_cam[1,1] *= -1

        self.scaled_pix_T_cam = utils_geom.scale_intrinsics(torch.tensor(self.pix_T_cam).unsqueeze(0), 1, 1).squeeze(0).numpy()

        self.scenes_to_process_per_vehicle = 100 # Only process these number of scenes for each vehicle.
        self.visualize = False

    
    def process_depths(self, depths):
        depths[np.where(depths>20)] = 0 
        depths[np.where(depths<-20)] = 0 
        # depths = depths.astype(np.float32)
        return depths

    def process_rgbs(self, rgbs):
        rgb_scaled = []
        for rgb in rgbs:
            rgb = imresize(rgb, (self.finalH, self.finalW), interp = "bilinear")
            rgb_scaled.append(rgb)
        rgb_scaled = np.stack(rgb_scaled)
        return rgb_scaled
    
    def get_alignedboxes2thetaformat(self, aligned_boxes):
        # st()
        B,N,_ = list(aligned_boxes.shape)
        xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
        xc = (xmin+xmax)/2.0
        yc = (ymin+ymax)/2.0
        zc = (zmin+zmax)/2.0
        w = xmax-xmin
        h = ymax - ymin
        d = zmax - zmin
        zeros = torch.zeros([B,N]).double()
        boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
        return boxes

    def process(self):
        vehicles = os.listdir(self.episode_path)
        print("Processing episode: ", self.episode_path)
        vehicles = [v for v in vehicles if os.path.isdir(os.path.join(self.episode_path, v))]
        fnames = []
        for vehicle in vehicles:
            print("Processing vehicle: ", vehicle)
            vehicle_path = os.path.join(self.episode_path, vehicle)
            scenes = os.listdir(vehicle_path)
            scenes = [scene for scene in scenes if scene.endswith(".p")]
            random.shuffle(scenes)
            for i in range(min(len(scenes), self.scenes_to_process_per_vehicle)):
                scene = scenes[i]
                print("processing scene: ", scene)
                scene_path = os.path.join(vehicle_path, scene)
                scene = pickle.load(open(scene_path, "rb"))
                depths = self.process_depths(scene['depth_data'])
                
                rgbs = scene['rgb_data']
                num_camRs = scene['num_camRs']
                rgbs = self.process_rgbs(rgbs)
                if self.visualize:
                    concat_rgbs = np.concatenate(rgbs, axis=0)
                    plt.imshow(concat_rgbs)
                    plt.show(block=True)
                
                B = depths.shape[0]
                xyz_camXs = utils_geom.depth2pointcloud_cpu(torch.tensor(depths).unsqueeze(1), torch.tensor(self.pix_T_cam).unsqueeze(0).repeat(B, 1, 1)).squeeze(0).numpy()
                if self.visualize:
                    pcd_camXs = []
                    for xyz_camX in xyz_camXs:
                        pcd_camX = nlu.make_pcd(xyz_camX)
                        pcd_camXs.append(pcd_camX)
                        # o3d.visualization.draw_geometries([pcd_camX, self.mesh_frame])
                #####################################################################
                #### WARNING: Assuming rgb and depth cameras have same extrinsics ###
                #####################################################################
                cam_to_car_transform_locs = scene['rgb_cam_to_car_transform_locs']
                cam_to_car_transform_rots = scene['rgb_cam_to_car_transform_rots']
                
                car_T_camXs = []
                for rotation, position in zip(cam_to_car_transform_rots, cam_to_car_transform_locs):
                    # print("Rotation and position are: ", rotation, position)
                    car_T_camX = self.get_unreal_transform(rotation, position) @ self.carla_T_cam
                    car_T_camXs.append(car_T_camX)
                
                # We will assume car to be the origin
                car_T_camXs = np.stack(car_T_camXs)
                camXs_T_car = utils_geom.safe_inverse(torch.tensor(car_T_camXs)).numpy()
                
                bbox_origin = np.array(scene['bounding_box'])
                # st()
                bbox = torch.tensor(bbox_origin.reshape(1, 2, 3))
                camX1_T_car = torch.tensor(camXs_T_car[0:1])
                bbox_camX1 = utils_geom.apply_4x4(camX1_T_car, bbox).squeeze(0).numpy()
                
                random_camR_num = np.random.randint(0, num_camRs)
                camRandomR_T_car = torch.tensor(camXs_T_car[random_camR_num:random_camR_num+1])
                bbox_theta = self.get_alignedboxes2thetaformat(bbox.reshape(1, 1, 6))
                bbox_corners = utils_geom.transform_boxes_to_corners(bbox_theta.float())
                
                bbox_camRandomR_corners = utils_geom.apply_4x4(camRandomR_T_car.float(), bbox_corners.squeeze(0).float()).float()
                st()
                bbox_camRandomR_ends = nlu.get_ends_of_corner(bbox_camRandomR_corners.permute(0, 2, 1)).permute(0, 2, 1).float()
                bbox_camRandomR_theta = self.get_alignedboxes2thetaformat(bbox_camRandomR_ends.reshape(1, 1, 6).double())
                bbox_camRandomR_corners = utils_geom.transform_boxes_to_corners(bbox_camRandomR_theta.float())[0]

                if self.visualize or True:
                    # Visualize boxes on randomly selected camR
                    summwriter = utils_improc.Summ_writer(None, 100, "train")
                    rgb_randomCamR_normalized = torch.tensor(utils_improc.preprocess_color(rgbs[random_camR_num:random_camR_num+1]))
                    rgb_randomCamR_normalized = rgb_randomCamR_normalized.permute(0, 3, 1, 2)
                    scores = torch.ones((bbox_corners.shape[0], bbox_corners.shape[1]), dtype=int)
                    tids = scores.clone()
                    rgb_with_bbox = summwriter.summ_box_by_corners("camR_bbox", rgb_randomCamR_normalized, bbox_camRandomR_corners.unsqueeze(0), scores, tids, torch.tensor(self.scaled_pix_T_cam).unsqueeze(0), only_return=True)
                    rgb_with_bbox = utils_improc.back2color(rgb_with_bbox)
                    rgb_with_bbox_np = rgb_with_bbox.permute(0, 2, 3, 1).squeeze(0).numpy()
                    print("Visualizing bbox")
                    plt.imshow(rgb_with_bbox_np)
                    plt.show(block=True)

                    pcd_car_list = [self.mesh_frame]
                    xyz_cars = utils_geom.apply_4x4(torch.tensor(car_T_camXs), torch.tensor(xyz_camXs)).squeeze(0).numpy()
                    for xyz_car in xyz_cars:
                        xyz_car = utils_pointcloud.truncate_pcd_outside_bounds([-3,-3,-3,3,3,3], xyz_car)
                        pcd_car = nlu.make_pcd(xyz_car)
                        pcd_car_list.append(pcd_car)
                        print("visualizing bbox in origin frame")
                        nlu.only_visualize(pcd_car, [bbox_origin])
                        # o3d.visualization.draw_geometries([pcd_car, self.mesh_frame])
                    print("Visualizing merged pcds")
                    o3d.visualization.draw_geometries(pcd_car_list)
                    bbox_on_rgb = self.draw_bbox_in_cam_coords(bbox_camX1, rgbs[0], pcd_camXs[0])

                camR_T_origin = camRandomR_T_car.repeat(B, 1, 1).numpy()
                # camR_T_origin = camX1_T_car.repeat(B, 1, 1).numpy()
                data = {"camR_index":random_camR_num, "obj_name": vehicle, "pix_T_cams_raw": torch.tensor(self.scaled_pix_T_cam).unsqueeze(0).repeat(B, 1, 1).numpy(), "camR_T_origin_raw": camR_T_origin, "xyz_camXs_raw": xyz_camXs, "origin_T_camXs_raw": car_T_camXs, 'rgb_camXs_raw': rgbs, "bbox_origin":bbox_origin}
                
                cur_epoch = str(time()).replace(".","")
                pickle_fname = cur_epoch + ".p"
                fnames.append(pickle_fname)
                with open(os.path.join(dump_dir, pickle_fname), 'wb') as f:
                    pickle.dump(data, f)
        return fnames

    def draw_bbox_in_cam_coords(self, bbox_camX, rgb_camX, pcd):
    
        bbox_camX[0, 0], bbox_camX[1, 0] = min(bbox_camX[0, 0], bbox_camX[1, 0]), max(bbox_camX[0, 0], bbox_camX[1, 0])
        bbox_camX[0, 1], bbox_camX[1, 1] = min(bbox_camX[0, 1], bbox_camX[1, 1]), max(bbox_camX[0, 1], bbox_camX[1, 1])
        bbox_camX[0, 2], bbox_camX[1, 2] = min(bbox_camX[0, 2], bbox_camX[1, 2]), max(bbox_camX[0, 2], bbox_camX[1, 2])
        bbox_camX = bbox_camX.reshape(1,-1)
        nlu.only_visualize(pcd, bbox_camX)
        return utils_pointcloud.draw_boxes_on_rgb(rgb_camX, self.scaled_pix_T_cam, bbox_camX, visualize=True)
        

    def get_unreal_transform(self, rotation, position):
        '''
        Returns the camera to [whatever the camera is attached to]
        transformation with the Unreal necessary corrections applied.
        '''
        rot_matrix = self.create_transformation_matrix(rotation, position, np.array([1,1,1]))
        to_unreal_transform = self.create_transformation_matrix(np.array([0, 90, -90]), np.array([0,0,0]), np.array([-1,1,1]))
        return rot_matrix @ to_unreal_transform

    # pitch, yaw, roll, x, y, z, scalex, scaley, scalez
    def create_transformation_matrix(self, rotation, position, scale):
        # Transformation matrix
        pitch, yaw, roll = rotation
        tx, ty, tz = position
        scalex, scaley, scalez = scale
        cy = math.cos(np.radians(yaw))
        sy = math.sin(np.radians(yaw))
        cr = math.cos(np.radians(roll))
        sr = math.sin(np.radians(roll))
        cp = math.cos(np.radians(pitch))
        sp = math.sin(np.radians(pitch))
        matrix = np.eye(4)
        matrix[0, 3] = tx
        matrix[1, 3] = ty
        matrix[2, 3] = tz
        matrix[0, 0] = scalex * (cp * cy)
        matrix[0, 1] = scaley * (cy * sp * sr - sy * cr)
        matrix[0, 2] = -scalez * (cy * sp * cr + sy * sr)
        matrix[1, 0] = scalex * (sy * cp)
        matrix[1, 1] = scaley * (sy * sp * sr + cy * cr)
        matrix[1, 2] = scalez * (cy * sr - sy * sp * cr)
        matrix[2, 0] = scalex * (sp)
        matrix[2, 1] = -scaley * (cp * sr)
        matrix[2, 2] = scalez * (cp * cr)
        return matrix

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
if __name__ == "__main__":

    episodes = os.listdir(base_dir)
    episodes = [f for f in episodes if os.path.isdir(os.path.join(base_dir, f))]
    mkdir(dump_dir)
    trainfile = open(os.path.join(file_dir, dumpmod+"t.txt"),"w")
    valfile = open(os.path.join(file_dir, dumpmod+"v.txt"),"w")
    cntnum = 0
    # st()
    for episode in episodes:
        print("Processin file: ", episode)
        print("File number is: ", cntnum)
        cntnum+=1
        carla = write_carla_to_npy(episode)
        # st()
        fnames = carla.process()
        traincnt = int(0.75*len(fnames))
        testcnt = len(fnames) - traincnt
        for i in range(0, traincnt):
            trainfile.write(os.path.join(dumpmod, fnames[i]))
            trainfile.write("\n")
        
        for j in range(traincnt, len(fnames)):
            valfile.write(os.path.join(dumpmod, fnames[j]))
            valfile.write("\n")
        
    trainfile.close()
    valfile.close()


