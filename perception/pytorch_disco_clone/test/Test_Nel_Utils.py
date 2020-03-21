import os

os.environ["MODE"] = "CLEVR_STA"
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
# from utils_vox import Mem2Ref
import tensorflow as tf
import ipdb
import cv2
from sklearn.cluster import DBSCAN as skdbscan
import open3d as o3d
# from cc3d import connected_components
import utils_improc
import utils_geom
import utils_basic
import utils_vox
import copy
import torch
import torch.nn.functional as F
st = ipdb.set_trace
EPS = 1e-6
import lib_classes.Nel_Utils as nel 
from model_clevr_sta import CLEVR_STA
from model_carla_sta import CARLA_STA
from model_carla_flo import CARLA_FLO
from model_carla_obj import CARLA_OBJ
import hyperparams as hyp
import os
import cProfile
import logging
import ipdb 
st = ipdb.set_trace
logger = logging.Logger('catch_all')

def test_get_alignedboxes2thetaformat():
    a = np.random.randint(0, 10, (3,4,6))
    at = torch.from_numpy(a)
    ret = nel.get_alignedboxes2thetaformat(at)
    print("Test completed successfully")

def test_meshgrid3D_py():
    x,y,z = nel.meshgrid3D_py(3,4,5)
    print("x,y,z are: ", x, y, z)
    print("Test completed successfully")

def test_postproccess_for_ap():
    a = np.random.randint(0, 10, (3,4,6))
    at = torch.from_numpy(a)
    boxes = nel.postproccess_for_ap(at)
    print("boxes are: ", boxes)
    print("Test completed successfully")

def test_postproccess_for_scores():
    a = np.random.randint(0, 10, (2,3,4,6))
    at = torch.from_numpy(a)
    boxes = nel.postproccess_for_scores(at)
    print("boxes are: ", boxes)
    print("Test completed successfully")

def test_yxz2xyz_v2():
    a = np.random.randint(0, 10, (3,4,6))
    at = torch.from_numpy(a)
    boxes = nel.yxz2xyz_v2(at)
    print("boxes are: ", boxes)
    print("Test completed successfully")

def test_yxz2xyz():
    a = np.random.randint(0, 10, (1,2,2,3))
    at = torch.from_numpy(a)
    boxes = nel.yxz2xyz(at)
    print("boxes are: ", boxes)
    print("Test completed successfully")

def test_make_pcd():
    a = np.random.randint(0, 10, (10, 4))
    at = torch.from_numpy(a)
    pcd = nel.make_pcd(at)
    print("pcd is: ", pcd)
    print("Test completed successfully")

def test_cluster_using_dbscan():
    a = np.random.randint(0, 10, (10, 4))
    at = torch.from_numpy(a)
    padded_bbox, pcd = nel.cluster_using_dbscan(a, 10)
    print("pcd is: ", pcd)
    print("Test completed successfully")

def test_get_boxes_from_occ_mag_py():
    occ_mag = np.random.randint(0, 10, (10, 10, 10, 1))
    img = np.random.randint(0, 10, (10,10,3))
    coord_mem = np.random.randint(0, 10, (8))
    nel.get_boxes_from_occ_mag_py(occ_mag, img, coord_mem)
    print("Test completed successfully")

def test_get_boxes_from_occ_single():
    occ_mag = np.random.randint(0, 10, (10, 10, 10, 3))
    img = torch.from_numpy(np.random.randint(0, 10, (10,10,3)))
    coord_mem = torch.from_numpy(np.random.randint(0, 10, (8)))
    nel.get_boxes_from_occ_single((occ_mag, img, coord_mem))
    print("Test completed successfully")

def test_get_boxes_from_occ():
    occ_mag = torch.from_numpy(np.random.randint(0, 10, (3, 10, 10, 10, 3)))
    img = torch.from_numpy(np.random.randint(0, 10, (10,10,3)))
    coord_mem = torch.from_numpy(np.random.randint(0, 10, (8)))
    nel.get_boxes_from_occ(occ_mag, img, coord_mem)
    print("Test completed successfully")

if __name__ == '__main__':
    
    # test_get_alignedboxes2thetaformat()
    # test_meshgrid3D_py()
    # test_postproccess_for_ap()
    # test_postproccess_for_scores()
    # test_yxz2xyz_v2()
    # test_yxz2xyz()
    # test_make_pcd()
    # test_cluster_using_dbscan()
    # test_get_boxes_from_occ_mag_py()
    test_get_boxes_from_occ()
