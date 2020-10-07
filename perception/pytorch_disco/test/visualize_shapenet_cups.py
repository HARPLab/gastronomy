import numpy as np
import os
os.environ["MODE"] = "NEL_STA"
os.environ["exp_name"] = "trainer_big_builder_hard_exp5_pret"
os.environ["run_name"] = "check"
import math 
import open3d as o3d
import torch
import ipdb 
from lib_classes import Nel_Utils as nlu
st = ipdb.set_trace
import utils_geom
import utils_pointcloud

data = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/shapenet/sc0248_allviews.npz"
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0])
mesh_frame_small = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])
vis = True
bounds = [-1, -1, 1, 1, 1, 3]

if __name__ == "__main__":
    d = np.load(data)
    [print(key) for key in d.keys()]
    xyz_camXs = torch.tensor(d['xyz_camXs'])
    origin_T_camXs = torch.tensor(d['origin_T_camXs'])
    origin_T_camRs = torch.tensor(d['origin_T_camRs'])
    pix_T_cams = torch.tensor(d['pix_T_cams'])
    st()
    camR_T_camXs = utils_geom.safe_inverse(origin_T_camRs) @ origin_T_camXs
    xyz_camRs = utils_geom.apply_4x4(camR_T_camXs, xyz_camXs)
    if vis:
        pcd_list = [mesh_frame]
        for xyz_camX in xyz_camRs:
            pcd = nlu.make_pcd(xyz_camX.numpy())
            # o3d.visualization.draw_geometries([mesh_frame, pcd])
            pcd_list.append(pcd)
        o3d.visualization.draw_geometries(pcd_list)
    xyz_agg_camR = xyz_camRs.reshape(-1, 3).numpy()
    xyz_agg_camR = utils_pointcloud.truncate_pcd_outside_bounds(bounds, xyz_agg_camR)
    
    pcd = nlu.make_pcd(xyz_agg_camR)
    o3d.visualization.draw_geometries([mesh_frame_small, pcd])
    
    # st()
    padded_bbox_array,pcd = nlu.cluster_using_dbscan(xyz_agg_camR[::10], 2)
    nlu.only_visualize(pcd, padded_bbox_array)


    st()
    a=1


