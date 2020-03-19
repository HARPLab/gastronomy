import numpy as np

import open3d as o3d
import os


def read_point_cloud(pcd_path):
    assert os.path.exists(pcd_path), \
        "point cloud path does not exist {}".format(pcd_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd
    

def make_pcd(pts, color=None):
    '''Create a open3d PointCloud datastructure.'''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    elif color is not None:
        colors_arr = np.zeros((pts.shape[0], 3))
        for i in range(3):
            colors_arr[:, i] = color[i]
        pcd.colors = o3d.utility.Vector3dVector(colors_arr)
    return pcd


def visualize(pcd_list):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(pcd_list + [mesh_frame])


def get_pcd_bounds_str(pcd):
    s = "PCD min_bound: {}, max_bound: {}".format(
        np.array2string(pcd.get_min_bound(), precision=4, suppress_small=True), 
        np.array2string(pcd.get_max_bound(), precision=4, suppress_small=True))
    return s
