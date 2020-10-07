import numpy as np 
import os 
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "replica_multiview_builder"
os.environ["run_name"] = "check"
import cv2
import matplotlib
import copy
import matplotlib.pyplot as plt
import torch
import utils_vox
import utils_geom
import lib_classes.Nel_Utils as nlu
import open3d as o3d
import time
import utils_improc
import scipy
from scipy.misc import imread, imshow
from scipy import misc
import socket
hostname = socket.gethostname()
import getpass
import utils_vox
username = getpass.getuser()
import utils_pointcloud
import ipdb 
st = ipdb.set_trace
import sys
import pickle

name = sys.argv[1]

class bbox_visualizer:
    def __init__(self, pfile):
        if 'Shamit' in hostname:
            basepath = '/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/kinect_nips'
        else:
            basepath = '/hdd/shamit/kinect/processed/single_obj'
        self.device = torch.device("cpu")
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        summaryWriter = utils_improc.Summ_writer(None, 10, "train")
        self.lines = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]
        pfile = os.path.join(basepath, pfile)
        data = pickle.load(open(pfile, 'rb'))
        xyz_camX = torch.tensor(data['xyz_camXs_raw']).to(self.device).float()
        origin_T_camX = torch.tensor(data['origin_T_camXs_raw']).to(self.device).float()
        camR_T_origin = torch.tensor(data['camR_T_origin_raw']).to(self.device).float()
        camR_T_camX = camR_T_origin @ origin_T_camX
        xyz_camR = utils_geom.apply_4x4(camR_T_camX, xyz_camX)
        pcd_list = [self.mesh_frame]
        for xyz in xyz_camR:
            pcd_list.append(nlu.make_pcd(xyz))

        points1 = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        ]
        # st()
        rgbtovis = data['rgb_camXs_raw'][0]
        plt.show(plt.imshow(rgbtovis))
        num_boxes = int(input("enter number of boxes\n"))
        # o3d.visualization.draw_geometries(pcd_list)
        # bboxes = torch.tensor(data['bbox_origin']).to(self.device)
        class_list = []
        color_list = []
        bbox_list = []
        for _ in range(num_boxes):
            bbox_temp = torch.tensor([0,-0.1+0.02,0,0.1,0.02,0.1]).to(self.device)
            while True:
                bbox_camR_theta = self.get_alignedboxes2thetaformat(bbox_temp.reshape(-1, 1, 6))
                bbox_camR_corners = utils_geom.transform_boxes_to_corners(bbox_camR_theta)
                bbox_camR_corners = bbox_camR_corners[0,0]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(bbox_camR_corners.cpu().numpy())
                line_set.lines = o3d.utility.Vector2iVector(self.lines)
                pcd_line = copy.deepcopy(pcd_list)
                pcd_line.append(line_set)
                o3d.visualization.draw_geometries(pcd_line)
                try:
                    vals = input("enter delta\n")
                    vals = vals.split()
                    vals = [float(val) for val in vals]
                except Exception as e:
                    print("got exception : ", e)
                    continue
                done = True 
                for val in vals:
                    if val != 0:
                        done=False
                
                if done:
                    # print(data.keys())
                    # st()
                    rgb = utils_improc.preprocess_color(torch.tensor(data['rgb_camXs_raw'][0:1]).permute(0,3,1,2))
                    intrinsics = torch.tensor(data['pix_T_cams_raw'][0:1])
                    bbox_for_vis = bbox_camR_corners.unsqueeze(0).unsqueeze(0)
                    scores = torch.ones([1,1]).int()
                    tids = scores.clone()
                    rgb_with_bbox = summaryWriter.summ_box_by_corners("2Dto3D", rgb, bbox_for_vis, scores, tids, intrinsics, only_return=True)
                    rgb_with_bbox = utils_improc.back2color(rgb_with_bbox)
                    rgb_with_bbox = rgb_with_bbox.permute(0, 2, 3, 1).squeeze(0).numpy()

                    
                    plt.imshow(rgb_with_bbox)
                    plt.show()
                    clas = input("enter class\n")
                    color = input("enter color\n")
                    class_list.append(clas)
                    color_list.append(color)
                    bbox_list.append(bbox_temp.cpu().numpy())
                    break
                else:
                    bbox_temp += torch.tensor(vals).to(self.device)

        bbox_list = np.stack(bbox_list)
        data['bbox_origin'] = bbox_list
        data['shape_list'] = class_list
        data['color_list'] = color_list
        with open(pfile, "wb") as fwrite:
            pickle.dump(data, fwrite)

    def visualize_pcd_open3d(self, xyz):
        pcd = nlu.make_pcd(xyz)
        o3d.visualization.draw_geometries([self.mesh_frame, pcd])


    def get_alignedboxes2thetaformat(self, aligned_boxes):
        # aligned_boxes = torch.reshape(aligned_boxes,[aligned_boxes.shape[0],aligned_boxes.shape[1],6])
        aligned_boxes = aligned_boxes.cpu()
        B,N,_ = list(aligned_boxes.shape)
        xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
        xc = (xmin+xmax)/2.0
        yc = (ymin+ymax)/2.0
        zc = (zmin+zmax)/2.0
        w = xmax-xmin
        h = ymax - ymin
        d = zmax - zmin
        zeros = torch.zeros([B,N])
        boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
        return boxes


    


if __name__ == '__main__':

    bv = bbox_visualizer(name)

