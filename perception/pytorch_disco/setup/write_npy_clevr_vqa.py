import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "det_builder"
os.environ["run_name"] = "check"
import numpy as np 
import socket
import open3d as o3d
import pickle
import h5py 
from time import time
import quaternion
from PIL import Image
import json
import matplotlib.pyplot as plt
import torch
import imageio
from lib_classes import Nel_Utils as nlu
import utils_geom
import utils_pointcloud
import utils_improc
from scipy.misc import imresize
import ipdb
import random
import math
import sys
from time import time
st = ipdb.set_trace
hostname = socket.gethostname()

folderMod_dict = {
"ba":"CLEVR_VQA_256_256_A", "bb":"CLEVR_VQA_256_256_B", "bc":"CLEVR_VQA_256_256_C", "bd":"CLEVR_VQA_256_256_D",
"be":"CLEVR_VQA_256_256_E", "bf":"CLEVR_VQA_256_256_F", "bg":"CLEVR_VQA_256_256_G", "bh":"CLEVR_VQA_256_256_H",
"bi":"CLEVR_VQA_256_256_I", "bj":"CLEVR_VQA_256_256_J", "bk":"CLEVR_VQA_256_256_K", "bl":"CLEVR_VQA_256_256_L",
"bm":"CLEVR_VQA_256_256_M", "bn":"CLEVR_VQA_256_256_N"
}

folderMod_dict = {
"multi_obj_256_a":"VQA_256_3_10_OBJ_A", "multi_obj_256_b":"VQA_256_3_10_OBJ_B", "multi_obj_256_c":"VQA_256_3_10_OBJ_C", "multi_obj_256_d":"VQA_256_3_10_OBJ_D",
"single_obj_256_a":"VQA_256_1_OBJ_A", "single_obj_256_b":"VQA_256_1_OBJ_B", "single_obj_256_c":"VQA_256_1_OBJ_C",
"multi_obj_480_a":"support/train"}

folderMod_dict = {
"single_obj_480_a":"VQA_480_1_OBJ_A", "single_obj_480_b":"VQA_480_1_OBJ_B", "single_obj_480_c":"VQA_480_1_OBJ_C", "single_obj_480_d":"VQA_480_1_OBJ_D",
"single_obj_480_e":"VQA_480_1_OBJ_E", "single_obj_480_f":"VQA_480_1_OBJ_F", "single_obj_480_g":"VQA_480_1_OBJ_G", "single_obj_480_h":"VQA_480_1_OBJ_H"}

folderMod_dict = {
"single_obj_large_480_a":"VQA_480_1_LARGE_OBJ_A", "single_obj_large_480_b":"VQA_480_1_LARGE_OBJ_B", "single_obj_large_480_c":"VQA_480_1_LARGE_OBJ_C", "single_obj_large_480_d":"VQA_480_1_LARGE_OBJ_D",
"single_obj_large_480_e":"VQA_480_1_LARGE_OBJ_E", "single_obj_large_480_f":"VQA_480_1_LARGE_OBJ_F"}

folderMod_dict = {"empty_480_a":"VQA_480_EMPTY_A"}

mod = sys.argv[1]
# If split is -1, don't use splits
split = int(sys.argv[2])
if split>=0:
    mod = mod+"_"+str(split)
if "Shamit" in hostname:
    base_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/clevr_vqa"
    file_dir = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/clevr_vqa/npys"
    dumpmod = "aa"
    dump_dir = os.path.join(file_dir, dumpmod)
else:
    # base_dir = "/home/shamitl/datasets/clevr_vqa/{}".format(folderMod_dict[mod])
    # base_dir = "/home/shamitl/datasets/multiview_qa/" # Darshan's multi obj
    # file_dir = "/home/shamitl/datasets/clevr_vqa/npy"
    # base_dir = "/home/shamitl/datasets/vqa_2_3_obj/" # 2-3 obj rotated
    # file_dir = "/home/shamitl/datasets/vqa_2_3_obj/npy"
    base_dir = "/projects/katefgroup/datasets/clevr_vqa/raw/{}".format(folderMod_dict[mod])
    file_dir = "/projects/katefgroup/datasets/clevr_vqa/raw/npys"
    dumpmod = mod
    dump_dir = os.path.join(file_dir, dumpmod)

if os.path.exists(os.path.join(file_dir, dumpmod)):
    print("This datamod already exists. Terminating")
    # exit(1)

class write_clevr_vqa_to_npy():
    def __init__(self):
        self.camR_T_origin = self.get_camR_T_origin()
        self.H = 320
        self.W = 480
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.scene_dirs = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        self.visualize = False
        self.device = torch.device("cpu")
        self.image_dir = os.path.join(base_dir, "images")
        self.depth_dir = os.path.join(base_dir, "depth")
        self.splits_dir = os.path.join(base_dir, "splits")

        self.scene_info_dir = os.path.join(base_dir, "scenes")
        self.fname_list = []
        self.use_split = split >= 0
        if self.use_split:
            self.split_file = os.path.join(self.splits_dir, "train_{}.json".format(split))
            self.split_scenes = json.load(open(self.split_file))
            # st()


        self.origin_T_camX_dict = {}
        self.process()

    def get_camR_T_origin(self):
        camR_T_origin = np.array([[-2.22044605e-16,  1.00000000e+00,  0.00000000e+00,
                                    2.50104901e-15],
                                [ 0.00000000e+00,  2.22044605e-16, -1.00000000e+00,
                                    0.00000000e+00],
                                [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16,
                                    1.12637234e+01]])
        camR_T_origin = np.concatenate((camR_T_origin, np.array([[0,0,0,1]])), axis=0) 
        return torch.tensor(camR_T_origin).float()

    def process_depths(self, depth):
        depth = depth *100.0
        depth = depth.astype(np.float32)
        return depth
    
    def load_extrinsics(self, fname, fpath):
        identifier = fname.split("extrinsic_")[1].split('.')[0].strip()
        camX_T_origin = np.load(fpath)
        camX_T_origin = np.concatenate((camX_T_origin, np.array([[0,0,0,1]])), axis=0) 
        origin_T_camX = utils_geom.safe_inverse(torch.tensor(camX_T_origin).unsqueeze(0)).squeeze(0).numpy()
        return identifier, origin_T_camX

    def load_extrinsics_intrinsics(self):
        ext_dir = os.path.join(base_dir, "cameras")
        files = [f for f in os.listdir(ext_dir) if f.endswith('npy')]
        for f in files:
            fpath = os.path.join(ext_dir, f)
            if "ext" in f:
                name, origin_T_camX = self.load_extrinsics(f, fpath)
                self.origin_T_camX_dict[name] = origin_T_camX
            elif "int" in f:
                self.pix_T_camX = np.load(fpath)
        self.num_cams = len(self.origin_T_camX_dict.keys())
        
    def isvalid_scene(self, img_scene, dep_scene):
        if not os.path.exists(img_scene):
            return False
        if not os.path.exists(dep_scene):
            return False
        
        imgs = [f for f in os.listdir(img_scene) if f.endswith('png')]
        deps = [f for f in os.listdir(dep_scene) if f.endswith('exr')]
        if len(imgs) == self.num_cams and len(deps) == self.num_cams:
            return True
        return False
    
    def load_image(self, rgbpath) :
        img = Image.open( rgbpath )
        img.load()
        data = np.asarray( img, dtype="int32" )[:,:,:3]
        return data
    
    def process_rgbs(self, rgb):
        H_, W_, _ = rgb.shape
        assert (H_ == self.H)  # otw i don't know what data this is
        assert (W_ == self.W)  # otw i don't know what data this is
        return rgb
        
    def process(self):
        self.load_extrinsics_intrinsics()
        scenes = [scene for scene in os.listdir(self.image_dir) if scene.startswith('CLEVR')]
        for scene in scenes:
            scene_num = int(scene.split('_')[-1])
            # st()
            if self.use_split:
                if scene_num not in self.split_scenes:
                    continue
            if not os.path.exists(os.path.join(self.scene_info_dir, scene+".json")):
                continue
            scene_info = json.load(open(os.path.join(self.scene_info_dir, scene+".json")))
            img_scene = os.path.join(self.image_dir, scene)
            dep_scene = os.path.join(self.depth_dir, scene)
            if not self.isvalid_scene(img_scene, dep_scene):
                continue
            rgb_list = []
            dep_list = []
            origin_T_camX_list = []
            for key in self.origin_T_camX_dict:
                print("key is :", key)
                origin_T_camX_list.append(self.origin_T_camX_dict[key])

                depth_file = os.path.join(dep_scene, key+".exr")
                depth_img = np.array(imageio.imread(depth_file, format='EXR-FI'))[:,:,0]
                depth_img = self.process_depths(depth_img)

                rgb_file = os.path.join(img_scene, key+".png")
                rgb_img = self.process_rgbs(self.load_image(rgb_file))

                rgb_list.append(rgb_img)
                dep_list.append(depth_img)
            
            # st()
            xyz_camX_list = []
            rgb_camXs = torch.tensor(np.stack(rgb_list))
            pcd_list = [self.mesh_frame]

            depths = np.stack(dep_list)
            depths = torch.tensor(depths).unsqueeze(0).permute(1,0,2,3).float()
            origin_T_camXs = torch.tensor(np.stack(origin_T_camX_list)).float()

            B = origin_T_camXs.shape[0]
            self.pix_T_camXs = torch.tensor(self.pix_T_camX).unsqueeze(0).repeat(B,1,1).float()

            xyz_camXs = utils_geom.depth2pointcloud_cpu(depths, self.pix_T_camXs)

            xyz_camXs_origin = utils_geom.apply_4x4(origin_T_camXs, xyz_camXs)

            camR_T_camXs = self.camR_T_origin.unsqueeze(0).repeat(B,1,1) @ origin_T_camXs
            camXs_T_camR = utils_geom.safe_inverse(camR_T_camXs)
            xyz_camRs = utils_geom.apply_4x4(camR_T_camXs, xyz_camXs)

            
            if self.visualize and False:
                for xyz_camX_origin in xyz_camXs_origin:
                    pcd = nlu.make_pcd(xyz_camX_origin)
                    pcd_list.append(pcd)

                o3d.visualization.draw_geometries(pcd_list)

            objects = scene_info['objects']
            material_list = []
            shape_list = []
            color_list = []
            rotation_list = []
            bbox_origin_ends_list = []
            for sobj in objects:
                material_list.append(sobj['material'])
                shape_list.append(sobj['shape'])
                color_list.append(sobj['color'])
                rotation_list.append(sobj['rotation']) 
                coords_origin = sobj['3d_coords']
                zcoord = 1.3*coords_origin[2]
                bbox_origin_ends = [coords_origin[0]-zcoord, coords_origin[1]-zcoord, coords_origin[2]-zcoord,coords_origin[0]+zcoord, coords_origin[1]+zcoord, coords_origin[2]+zcoord]
                bbox_origin_ends_list.append(bbox_origin_ends)

            if self.visualize and False:
                nlu.only_visualize(pcd_list[-1], bbox_origin_ends_list)
            pix_T_camXs_to_store = np.zeros((self.pix_T_camXs.shape[0], 4, 4))
            pix_T_camXs_to_store[:,:3,:3] = self.pix_T_camXs.numpy()
            pix_T_camXs_to_store[:,3,3] = 1

            if len(bbox_origin_ends_list) > 0:
                bbox_origin_ends = np.stack(bbox_origin_ends_list)
                bbox_origin_ends = torch.tensor(bbox_origin_ends).unsqueeze(0).reshape(1,-1,2,3).float()
                bbox_origin_theta = nlu.get_alignedboxes2thetaformat(bbox_origin_ends)
                bbox_origin_corners = utils_geom.transform_boxes_to_corners(bbox_origin_theta)
                
                bbox_camR_corners = utils_geom.apply_4x4(self.camR_T_origin.unsqueeze(0).repeat(bbox_origin_corners.shape[1],1,1), bbox_origin_corners.squeeze(0))
                bbox_camR_ends = nlu.get_ends_of_corner(bbox_camR_corners.permute(0,2,1)).permute(0,2,1)
                bbox_camR_theta = nlu.get_alignedboxes2thetaformat(bbox_camR_ends.unsqueeze(0))
                bbox_camR_corners = utils_geom.transform_boxes_to_corners(bbox_camR_theta)
                
                data_to_save = {"rotation_list":rotation_list, "color_list": color_list, "shape_list": shape_list, "material_list": material_list, "bbox_origin": bbox_origin_corners.squeeze(0).numpy(), "bbox_camR": bbox_camR_corners.squeeze(0).numpy(), "camR_T_origin_raw": self.camR_T_origin.unsqueeze(0).repeat(B,1,1).numpy(), "xyz_camXs_raw": xyz_camXs.numpy(), "origin_T_camXs_raw": origin_T_camXs.numpy(), 'rgb_camXs_raw': rgb_camXs.numpy(), 'pix_T_cams_raw': pix_T_camXs_to_store}
            else:
                empty_bbox = np.array([])
                data_to_save = {"rotation_list":rotation_list, "color_list": color_list, "shape_list": shape_list, "material_list": material_list, "bbox_origin": empty_bbox, "bbox_camR": empty_bbox, "camR_T_origin_raw": self.camR_T_origin.unsqueeze(0).repeat(B,1,1).numpy(), "xyz_camXs_raw": xyz_camXs.numpy(), "origin_T_camXs_raw": origin_T_camXs.numpy(), 'rgb_camXs_raw': rgb_camXs.numpy(), 'pix_T_cams_raw': pix_T_camXs_to_store}
            # st()
            cur_epoch = str(time()).replace(".","")
            pickle_fname = mod + "_" + cur_epoch + ".p"
            with open(os.path.join(dump_dir, pickle_fname), 'wb') as f:
                pickle.dump(data_to_save, f)
                self.fname_list.append(pickle_fname)
            # bbox_camX_corners = utils_geom.apply_4x4(camXs_T_camR[0].repeat(bbox_camR_corners.shape[1],1,1), bbox_camR_corners.squeeze(0))
            # bbox_camX_ends = nlu.get_ends_of_corner(bbox_camX_corners.permute(0,2,1)).permute(0,2,1)
            # bbox_camX_theta = nlu.get_alignedboxes2thetaformat(bbox_camX_ends.unsqueeze(0))
            # bbox_camX_corners = utils_geom.transform_boxes_to_corners(bbox_camX_theta)

            # summaryWriter = utils_improc.Summ_writer(None, 10, "train")
            # rgb_for_bbox_vis = torch.tensor(rgb_list[-1]).permute(2, 0, 1).unsqueeze(0)
            # rgb_for_bbox_vis = utils_improc.preprocess_color(rgb_for_bbox_vis)
            # scores = torch.ones((bbox_camR_corners.shape[0], bbox_camR_corners.shape[1]))
            # tids = torch.ones_like(scores)
            # intrinsics = torch.tensor(self.pix_T_camX).unsqueeze(0)
            
            # st()
            # # self.draw_boxes_using_ends(bbox_camX_ends, rgb_for_bbox_vis, intrinsics)
            # rgb_with_bbox = summaryWriter.summ_box_by_corners("2Dto3D", rgb_for_bbox_vis, bbox_camX_corners, scores, tids, intrinsics, only_return=True)
            # # st()
            # rgb_with_bbox = utils_improc.back2color(rgb_with_bbox)
            # rgb_with_bbox = rgb_with_bbox.permute(0, 2, 3, 1).squeeze(0).numpy()

            # if self.visualize:
            #     plt.imshow(rgb_with_bbox)
            #     plt.show(block=True)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)  
        
if __name__ == '__main__':
    mkdir(dump_dir)
    vqa = write_clevr_vqa_to_npy()
    total_files = len(vqa.fname_list)
    train_len = int(0.8*total_files)

    trainfile = open(os.path.join(file_dir, dumpmod+"t.txt"),"w")
    valfile = open(os.path.join(file_dir, dumpmod+"v.txt"),"w")

    for i in range(total_files):
        if i < train_len:
            trainfile.write(os.path.join(dumpmod, vqa.fname_list[i]))
            trainfile.write("\n")
        else:
            valfile.write(os.path.join(dumpmod, vqa.fname_list[i]))
            valfile.write("\n")
    trainfile.close()
    valfile.close()
