import numpy as np
import os
import sys
import time
import struct
import ctypes
import open3d as o3d

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

from autolab_core import RigidTransform


def get_rgb(data):
    '''Get the RGB values from raw PCL data.'''
    s = struct.pack('>f', data)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = int((pack & 0x00FF0000)>> 16)
    g = int((pack & 0x0000FF00)>> 8)
    b = int(pack & 0x000000FF)
    return [r, g, b]


##convert a PointCloud or PointCloud2 to a 4xn scipy matrix (x y z 1)
def point_cloud_to_mat(point_cloud):
    if type(point_cloud) == type(PointCloud2()):
        points = [[p[0], p[1], p[2], 1] for p in point_cloud2.read_points(
            point_cloud, field_names = 'xyz', skip_nans=True)]
    else:
        raise ValueError("Invalid point cloud type: {}".format(type(point_cloud)))
        return None
    points = np.array(points).T
    return points


def point_cloud_to_color_arr(pcl, color=None):
    cloud, rgb = [], []
    for point in point_cloud2.read_points(pcl, skip_nans=True):
        cloud.append([point[0], point[1], point[2], 1])
        if color is not None:
            rgb_point = [c for c in color]
        else:
            rgb_point = get_rgb(point[3])
        rgb.append(rgb_point)
    return np.array(cloud), np.array(rgb)


# Open3d utils
def make_pcd(pts, color=None):
    '''Create a open3d PointCloud datastructure.'''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    elif color is not None:
        colors_arr = np.array((pts.shape[0], 3))
        for i in range(3):
            colors_arr[:, i] = color[i]
        pcd.colors = o3d.utility.Vector3dVector(colors_arr)
    return pcd


def visualize(list_of_pcds):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(list_of_pcds + [mesh_frame])


def get_pcd_bounds_str(pcd):
    s = "PCD min_bound: {}, max_bound: {}".format(
        np.array2string(pcd.get_min_bound(), precision=4, suppress_small=True), 
        np.array2string(pcd.get_max_bound(), precision=4, suppress_small=True))
    return s
