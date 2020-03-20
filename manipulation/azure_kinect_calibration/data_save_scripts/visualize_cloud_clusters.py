import argparse
import cv2
import logging
import numpy as np
import h5py
import os
import sys
import time
import struct
import ctypes

from autolab_core import RigidTransform
from perception import CameraIntrinsics

import cv2
import open3d as o3d
import copy

sys.path.append(os.path.abspath('../'))

from pcl_registration_utils import point_cloud_to_color_arr, make_pcd, visualize, get_pcd_bounds_str
import fusion
import matplotlib.pyplot as plt


def main(pcd_dir):
    cloud_cluster_path = []
    for fname in os.listdir(pcd_dir):
        if 'cloud_cluster' in fname:
            cloud_cluster_path.append(os.path.join(pcd_dir, fname))
    
    for pcd_path in cloud_cluster_path:
        o3d.io.read_point_cloud(pcd_path)
    pcd_list = [o3d.io.read_point_cloud(path) for path in cloud_cluster_path]

    cmap = plt.get_cmap("Set1", len(pcd_list) + 1)

    for i, pcd in enumerate(pcd_list):
        color = np.array(cmap(i))[:3]
        pcd.paint_uniform_color(color)

    if len(pcd_list) > 0:
        visualize(pcd_list)
    else:
        print("Did not find any object clusters.")


if __name__ == '__main__':
    np.random.seed(0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    parser = argparse.ArgumentParser(description="visualize clusters.")
    parser.add_argument('--pcd_dir', type=str, required=True, 
                        help='Path to data that stores the pcds.')
    args = parser.parse_args()

    assert os.path.exists(args.pcd_dir)

    main(args.pcd_dir)
