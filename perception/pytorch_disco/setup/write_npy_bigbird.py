import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "builder_rgb"
os.environ["run_name"] = "check"
import numpy as np 
import socket
import open3d as o3d
import pickle
import h5py 
from time import time

from PIL import Image
from lib_classes import Nel_Utils as nlu

import torch
import utils_geom
import utils_pointcloud
from scipy.misc import imresize
import ipdb
st = ipdb.set_trace
hostname = socket.gethostname()


'''
Update the mod information here:
cc -> Data generated with 2 views and camR at 0 degree azimuth
dd -> Data generated with 2 views and camR at random degree azimuth
ee -> Similar to dd but with correct bboxes
ff -> Fixed camR and both views from elevation 0
gg -> Similar to ff but with camR varying at elevation 0
'''

if "Shamit" in hostname:
    base_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/bigbird_dataset/"
    bbox_dir = "/Users/shamitlal/Desktop/vis/bigbird/npy"
    file_dir = "/Users/shamitlal/Desktop/vis/bigbird/processed/npy"
    dumpmod = "gg"
    dump_dir = os.path.join(file_dir, dumpmod)
else:
    base_dir = "/projects/katefgroup/datasets/bigbird/"
    bbox_dir = "/home/shamitl/projects/bigbird/bboxes/npy"
    file_dir = "/projects/katefgroup/datasets/bigbird_processed/npy"
    dumpmod = "gg"
    dump_dir = os.path.join(file_dir, dumpmod)

if os.path.exists(os.path.join(file_dir, dumpmod)):
    print("This datamod already exists. Terminating")
    exit(1)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=.3, origin=[0, 0, 0])


'''
Big bird dataset has objects moving on a table. Here, in order to generate
the dataset, we assume that object is stationary and the camera is moving in opposite
direction. So, we have 600 views of a stationary object at 5 elevations and 
120 azimuths. The camR is assumed to be NP1 camera at "randomly selected" azimuth. We can get xyz_camR 
from xyz_camX by first cancelling out the azimuth, then moving to the "randomly selected" azimuth 
and finally moving to NP1. Cancelling out  azimuth can be done like this: 
First warp by azimuth_T_camX. This will bring rotate you view to 0 degree azimuth. 
Now warp by NP1_T_0, to get to NP1 reference frame for 0 azimuth. 
'''
class write_bigbird_to_npy():
    def __init__(self, obj_name):
        self.obj_name = obj_name
        self.base_dir = os.path.join(base_dir, obj_name)
        self.bbox_npy = os.path.join(bbox_dir, obj_name + ".npy")
        self.calib = h5py.File(os.path.join(self.base_dir, 'calibration.h5'), 'r+')
        self.mask_dir = os.path.join(self.base_dir, "masks")
        self.poses_dir = os.path.join(self.base_dir, "poses")
        self.visualize = False
        self.num_views = 2
        self.finalH = 256
        self.finalW = 256
        self.do_random_camR_azimuth = True # If true, camR will be placed at randomly selected azimuth for each datapoint.
        self.debug = True # If debug is True, elevation will be set to camera 1.
    def load_image(self, infilename) :
        img = Image.open( infilename )
        img.load()
        data = np.asarray( img, dtype="int32" )
        return data

    def get_intrinsics_and_extrinsics(self):
        NPI_ir_T_NP5_list = []
        NPI_T_NP5_list = []
        pix_T_camNPI_list = []
        pix_T_camNPI_depth_list = []

        for i in range(1,6):
            NPI_ir_T_NP5 = self.calib['H_NP{}_ir_from_NP5'.format(i)].value
            NPI_T_NP5 = self.calib['H_NP{}_from_NP5'.format(i)].value
            pix_T_camNPI = self.calib['NP{}_rgb_K'.format(i)].value
            pix_T_camNPI_depth = self.calib['NP{}_depth_K'.format(i)].value

            pix_T_camNPI = utils_geom.scale_intrinsics(torch.tensor(pix_T_camNPI).unsqueeze(0), self.finalW/1280, self.finalH/1024.0).squeeze(0).numpy()

            NPI_ir_T_NP5_list.append(NPI_ir_T_NP5)
            NPI_T_NP5_list.append(NPI_T_NP5)
            pix_T_camNPI_list.append(pix_T_camNPI)
            pix_T_camNPI_depth_list.append(pix_T_camNPI_depth)

        NPI_ir_T_NP5 = np.stack(NPI_ir_T_NP5_list)
        NPI_T_NP5 = np.stack(NPI_T_NP5_list)
        pix_T_camNPI = np.stack(pix_T_camNPI_list)
        pix_T_camNPI_depth = np.stack(pix_T_camNPI_depth_list)

        return NPI_ir_T_NP5, NPI_T_NP5, pix_T_camNPI, pix_T_camNPI_depth


    def process_depths(self, depth):
        depth = depth / 10000.0
        # making everything to mts. Right now it's in 100 um scale.
        depth = depth.astype(np.float32)
        return depth
    
    def load_poses(self):
        H_table_from_reference_camera_list = []
        for azimuth in range(0, 360, 3):
            poseFile = "NP5_{}_pose.h5".format(azimuth)
            posePath = os.path.join(self.poses_dir, poseFile)
            poseNpy = h5py.File(posePath, 'r+')['H_table_from_reference_camera'].value
            H_table_from_reference_camera_list.append(poseNpy)
        
        table_T_camNP5 = np.stack(H_table_from_reference_camera_list)
        return table_T_camNP5

    '''
    Warps to camR. Our camR is placed at azimuth selected randomly
    (or 0 degrees based on flag 'do_random_camR_azimuth') 
    and the corresponding camera is always NP1.
    
    Shapes:
    xyz_camX -> (65536, 3)
    '''
    def warp_to_camR(self, xyz_camX, camnum, azimuth):
        camIdx = camnum - 1
        tableAzimuth_T_camNP5 = self.table_T_camNP5[int(azimuth//3)]
        table0_T_camNP5 = self.table_T_camNP5[0]
        camNP5_T_table0 = utils_geom.safe_inverse(torch.tensor(table0_T_camNP5).unsqueeze(0)).squeeze(0).numpy()
        tableCamRAzimuth_T_camNP5 = self.table_T_camNP5[int(self.camR_azimuth//3)]

        camNPI_T_camNP5 = self.NPI_T_NP5[camIdx]
        camNP5_T_camNPI = utils_geom.safe_inverse(torch.tensor(camNPI_T_camNP5).unsqueeze(0)).squeeze(0).numpy()
        
        camNP1_T_camNP5 = self.NPI_T_NP5[0]

        camNP1_T_camNPI = camNP1_T_camNP5 @ camNP5_T_table0 @ tableCamRAzimuth_T_camNP5 @ camNP5_T_table0 @ tableAzimuth_T_camNP5 @ camNP5_T_camNPI
        camNP1_T_camNPI = torch.tensor(camNP1_T_camNPI).unsqueeze(0)
        xyz_camX1 = utils_geom.apply_4x4(camNP1_T_camNPI.float(), torch.tensor(xyz_camX).unsqueeze(0).float())
        return xyz_camX1.squeeze(0).numpy(), camNP1_T_camNPI.squeeze(0).numpy()

    def warp_from_ir_to_rgb_frame(self, xyz_camX_ir, camnum):
        camIdx = camnum - 1
        camNPI_ir_T_camNP5 = torch.tensor(self.NPI_ir_T_NP5[camIdx]).unsqueeze(0)
        camNP5_T_camNPI_ir = utils_geom.safe_inverse(camNPI_ir_T_camNP5).squeeze(0).numpy()
        camNPI_T_camNP5 = self.NPI_T_NP5[camIdx]
        camNPI_T_camNPI_ir = camNPI_T_camNP5 @ camNP5_T_camNPI_ir
        camNPI_T_camNPI_ir = torch.tensor(camNPI_T_camNPI_ir).unsqueeze(0)
        xyz_camX = utils_geom.apply_4x4(camNPI_T_camNPI_ir.float(), xyz_camX_ir.float()).squeeze(0).numpy()
        return xyz_camX

    def get_bbox_in_camR(self, bbox_camNP1_azimuth0):
        bbox = bbox_camNP1_azimuth0.reshape(2, 3)
        bbox = torch.tensor(bbox).unsqueeze(0).unsqueeze(0) # BxNx2x3
        bbox_theta = nlu.get_alignedboxes2thetaformat(bbox.float())
        bbox_corners = utils_geom.transform_boxes_to_corners(bbox_theta)
        # st()
        bbox_camR_corners, _ = self.warp_to_camR(bbox_corners.squeeze(0).squeeze(0).numpy(), 1, 0) # Current bboxes are by default in camNP1 and azimuth 0
        bbox_camR_ends = nlu.get_ends_of_corner(torch.tensor(bbox_camR_corners).unsqueeze(0).unsqueeze(0))
        return bbox_camR_ends.squeeze(0).squeeze(0).numpy().reshape(1, -1)

    def process(self):
        rgb_list = []
        mask_list = []
        depth_list = []
        rgb_intrinsics = []
        camR_T_origin = []
        xyz_camXs_raw = []
        origin_T_camX = []

        # TODO: Shift these two lines to constructor maybe (after ECCV 2020)?
        self.NPI_ir_T_NP5, self.NPI_T_NP5, self.pix_T_camNPI, self.pix_T_camNPI_depth = self.get_intrinsics_and_extrinsics()
        self.table_T_camNP5 = self.load_poses()
        combined_pcd = [mesh_frame]
        combined_xyz = []
        
        if self.do_random_camR_azimuth:
            self.camR_azimuth = np.random.randint(0, 120)*3 # The azimuth at which camR will be placed.
        else:
            self.camR_azimuth = 0

        print("The camR will be placed at azimuth: ", self.camR_azimuth)
        
        for _ in range(self.num_views):
            if self.debug:
                camnum = 1
            else:
                camnum = np.random.randint(1, 5) # Not selecting 5th camera as it doesn't provide much information.
            azimuth = np.random.randint(0, 120)*3
            print("camnum and azimuth are: ", camnum, azimuth)
            
            # Process rgbs
            imgName = "NP{}_{}.jpg".format(camnum, azimuth)
            rgb = self.load_image(os.path.join(self.base_dir, imgName)) # (1024, 1280, 3)
            rgb = imresize(rgb, (self.finalH, self.finalW), interp = "bilinear")
            rgb_list.append(rgb)

            # Process masks
            maskName = "NP{}_{}_mask.pbm".format(camnum, azimuth)
            mask = self.load_image(os.path.join(self.mask_dir, maskName))
            mask = imresize(mask, (self.finalH, self.finalW), interp = "nearest") # 480, 640
            mask_list.append(mask)

            # Process depths
            depName = "NP{}_{}.h5".format(camnum, azimuth)
            depth = h5py.File(os.path.join(self.base_dir, depName), 'r+')['depth'].value
            depth = self.process_depths(depth)
            depth_list.append(depth)
            
            # Create pointcloud and get it in rgb's coordinate frame.
            xyz_camX_ir, pcd_ir = utils_pointcloud.create_pointcloud(depth, rgb, self.pix_T_camNPI_depth[camnum-1])
            xyz_camX = self.warp_from_ir_to_rgb_frame(xyz_camX_ir, camnum)

            # Project and unproject pointcloud using rgb's intrinsics.
            depth_rgb_frame, valid_depths = utils_geom.create_depth_image(torch.tensor(self.pix_T_camNPI[camnum-1]).unsqueeze(0), torch.tensor(xyz_camX).unsqueeze(0), self.finalH, self.finalW)
            xyz_camX = utils_geom.depth2pointcloud(depth_rgb_frame, torch.tensor(self.pix_T_camNPI[camnum-1]).unsqueeze(0)).squeeze(0).numpy()
            xyz_camX[np.where(xyz_camX[:, 2] == 100)] = 0
            
            # Store stuff
            xyz_camXs_raw.append(xyz_camX)
            rgb_intrinsics.append(self.pix_T_camNPI[camnum-1])
            camR_T_origin.append(np.eye(4)) # Taking camR (NP1 at 0 azimuth) to be the origin as well.
            xyz_camR, camR_T_camX = self.warp_to_camR(xyz_camX, camnum, azimuth)
            origin_T_camX.append(camR_T_camX)
            combined_pcd.append(nlu.make_pcd(xyz_camR))
            combined_xyz.append(xyz_camR)

            if self.visualize:
                pcd = utils_pointcloud.draw_colored_pcd(xyz_camX, rgb)
                o3d.visualization.draw_geometries([pcd, mesh_frame])
            

        if self.visualize:
            o3d.visualization.draw_geometries(combined_pcd)

        rgb_camX = np.stack(rgb_list)
        mask_camX = np.stack(mask_list)
        depth_camX = np.stack(depth_list)
        pix_T_camX = np.stack(rgb_intrinsics)
        camR_T_origin = np.stack(camR_T_origin)
        xyz_camXs = np.stack(xyz_camXs_raw)
        origin_T_camX = np.stack(origin_T_camX)
        
        bbox_camNP1_azimuth0 = np.load(self.bbox_npy) # shape : (1, 6) [xmin, ymin, zmin, xmax, ymax, zmax]
        bbox_camR = self.get_bbox_in_camR(bbox_camNP1_azimuth0)
        
        if self.visualize:
            imgName = "NP{}_{}.jpg".format(1, self.camR_azimuth)
            rgb = self.load_image(os.path.join(self.base_dir, imgName)) # (1024, 1280, 3)
            rgb = imresize(rgb, (self.finalH, self.finalW), interp = "bilinear")
            utils_pointcloud.draw_boxes_on_rgb(rgb, self.pix_T_camNPI[0], bbox_camR, visualize=True)
            
        data = {"pix_T_cams_raw": pix_T_camX, "camR_T_origin_raw": camR_T_origin, "xyz_camXs_raw": xyz_camXs, "origin_T_camXs_raw": origin_T_camX, 'rgb_camXs_raw': rgb_camX, "mask_camXs_raw": mask_camX, "depth_camXs_raw": depth_camX, "obj_name": self.obj_name, "bbox_camR":bbox_camR}
        
        cur_epoch = str(time()).replace(".","")
        pickle_fname = self.obj_name + "_" + cur_epoch + ".p"
        with open(os.path.join(dump_dir, pickle_fname), 'wb') as f:
            pickle.dump(data, f)
        return pickle_fname

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__ == "__main__":

    files = os.listdir(base_dir)
    files = [f for f in files if os.path.isdir(os.path.join(base_dir, f))]
    mkdir(dump_dir)
    trainfile = open(os.path.join(file_dir, dumpmod+"t.txt"),"w")
    valfile = open(os.path.join(file_dir, dumpmod+"v.txt"),"w")
    cntnum = 0
    for f in files:
        print("Processin file: ", f)
        print("File number is: ", cntnum)
        cntnum+=1
        bigbird = write_bigbird_to_npy(f)
        for trainiter in range(10):
            print("train iter: ", trainiter)
            pickle_fname = bigbird.process()
            trainfile.write(os.path.join(dumpmod, pickle_fname))
            trainfile.write("\n")
        for valiter in range(2):
            print("val iter: ", valiter)
            pickle_fname = bigbird.process()
            valfile.write(os.path.join(dumpmod, pickle_fname))
            valfile.write("\n")
    trainfile.close()
    valfile.close()


        
