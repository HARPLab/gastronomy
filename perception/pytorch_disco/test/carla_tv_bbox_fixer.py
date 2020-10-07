import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "replica_multiview_builder"
os.environ["run_name"] = "check"
import numpy as np 
import torch 
import torchvision.models as models
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys 
import ipdb 
import pickle
import utils_geom
import time
import random
import lib_classes.Nel_Utils as nlu
import cv2
import cross_corr
import getpass
username = getpass.getuser()
st = ipdb.set_trace


if __name__ == '__main__':
    basepath = "/projects/katefgroup/datasets/carla/npy/tv"
    storepath = "/projects/katefgroup/datasets/carla/npy/tv_updated"
    basepath = "/hdd/carla97/PythonAPI/examples/pdisco_npys/npy/twoVehicle_farCam"
    storepath = "/hdd/carla97/PythonAPI/examples/pdisco_npys/npy/twoVehicle_farCam"
    files = [os.path.join(basepath, f) for f in os.listdir(basepath) if f.endswith('.p')]
    # st()
    errors = 0
    errorlist = []
    for cnt, pf in enumerate(files):
        print("Processing filenum and filename: ", cnt, pf)
        try:
            f = pickle.load(open(pf, "rb"))
        except Exception as e:
            print(e)
            errorlist.append(pf)
            continue
        bbox_origin = f['bbox_origin']
        v1_T_v2 = f['vehicle1_T_vehicle2'].float()
        bbox2_v2origin_ends = torch.tensor(bbox_origin[1:2]).float()
        bbox2_v2origin_theta = nlu.get_alignedboxes2thetaformat(bbox2_v2origin_ends.unsqueeze(0).reshape(1,1,2,3))
        # st()
        bbox2_v2origin_corners = utils_geom.transform_boxes_to_corners(bbox2_v2origin_theta)
        bbox2_v1origin_corners = utils_geom.apply_4x4(v1_T_v2, bbox2_v2origin_corners.squeeze(0))
        bbox2_v1origin_ends = nlu.get_ends_of_corner(bbox2_v1origin_corners.permute(0,2,1)).permute(0,2,1)
        bbox2_v1origin_ends = bbox2_v1origin_ends.reshape(-1)
        bbox_origin[1] = bbox2_v1origin_ends.numpy()
        # st()
        f['bbox_origin'] = bbox_origin
        with open(os.path.join(storepath, pf.split('/')[-1]), 'wb') as storefile:
            pickle.dump(f, storefile)
        
    print("got errors: ", errors)
    print("error list: ", errorlist)


