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

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

filename = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/carla/nips/15905431386487873.p"
p = pickle.load(open(filename,'rb'))
xyz = p['xyz_camXs_raw'][0]
pcd = nlu.make_pcd(xyz)
o3d.visualization.draw_geometries([self.mesh_frame, pcd])
st()
aa=1