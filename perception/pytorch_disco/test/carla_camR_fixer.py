# Change the camR for bb mod to front view

import numpy as np 
import pickle 
import socket
hostname = socket.gethostname()
import getpass
username = getpass.getuser()
import os

import ipdb 
st = ipdb.set_trace
import matplotlib.pyplot as plt
import torch

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def make_ith_view_camR(origin_T_camXs, idx):
    origin_T_camR = origin_T_camXs[idx]
    camR_T_origin = safe_inverse(torch.tensor(origin_T_camR).unsqueeze(0)).repeat(origin_T_camXs.shape[0], 1, 1).numpy()
    return camR_T_origin



if __name__ == "__main__":
    # basepath = '/home/mprabhud/dataset/carla/npy/bb'
    basepath = '/projects/katefgroup/datasets/shamit_carla_correct/npys/mc'
    # outpath = '/home/mprabhud/dataset/carla/npy/fc'
    outpath = '/projects/katefgroup/datasets/shamit_carla_correct/npys/mc'

    pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]

    fnum = 0
    for pfile in pfiles:
        # st()
        print("Processing file number: ", fnum)
        print("Processing file: ", pfile)
        fnum+=1
        file_ = open(os.path.join(basepath, pfile), "rb")
        f = pickle.load(file_)
        file_.close()

        
        B = f['camR_T_origin_raw'].shape[0]
        f['camR_T_origin_raw_fixed'] = make_ith_view_camR(f['origin_T_camXs_raw'], 1)
        f['camR_index_fixed'] = 1
        with open(os.path.join(outpath, pfile), "wb") as fwrite:
            pickle.dump(f, fwrite)