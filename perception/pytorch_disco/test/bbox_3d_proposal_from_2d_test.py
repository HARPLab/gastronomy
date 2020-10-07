import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "replica_multiview_builder"
os.environ["run_name"] = "check"
import sys
import numpy as np 
import pickle 
from bbox_3d_proposal_from_2d import bbox_3d_proposal_from_2d
import socket
hostname = socket.gethostname()
import getpass
username = getpass.getuser()
import ipdb 
st = ipdb.set_trace
import utils_geom
import matplotlib.pyplot as plt
import torch


# [-5, -5, 0, 5, 5, 10]
def test_kinect():
    # st()
    if 'Shamit' in hostname:
        basepath = '/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/habitat_processed/bb'
    else:
        # basepath = '/home/mprabhud/dataset/carla/npy/bb'
        basepath = '/hdd/datasets/kinect_processed'
        basepath = '/hdd/shamit/kinect/processed/single_obj'
        storepath = '/hdd/shamit/kinect/bbox_processed/single_obj'

    pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]
    # pfiles = ['15821948243662689.p']# ['1582194725716713.p', '15821947268549619.p']
    '''
    self.iou_thresh = 0.5
    self.vote_thresh = 2
    '''
    fnum = 0
    for pfile in pfiles:
        print("Processing file number: ", fnum)
        print("Processing file: ", pfile)
        # st()
        fnum+=1
        bounds =  [-1.5, -0.8, 0, 1.5, 0.8, 2.5]
        file_ = open(os.path.join(basepath, pfile), "rb")
        f = pickle.load(file_)
        file_.close()

        # camR_index = 0

        rgbs = f['rgb_camXs_raw']
        origin_T_camXs = f['origin_T_camXs_raw']
        camR_T_origin = f['camR_T_origin_raw']
        # camR_T_origin = make_ith_view_camR(origin_T_camXs, camR_index)
        xyz_camXs = f['xyz_camXs_raw']
        pix_T_camXs = f['pix_T_cams_raw']

        indices_to_take = np.arange(rgbs.shape[0])
        # indices_to_take = [2,3,4,5,indices_to_take[-1]]
        rgbs = rgbs[indices_to_take]
        if False:
            for rgb in rgbs:
                plt.imshow(rgb)
                plt.show()
        origin_T_camXs = origin_T_camXs[indices_to_take]
        camR_T_origin = camR_T_origin[indices_to_take]
        xyz_camXs = xyz_camXs[indices_to_take]
        pix_T_camXs = pix_T_camXs[indices_to_take]
        
        bbox3d = bbox_3d_proposal_from_2d(rgbs.shape[1], rgbs.shape[2], 4, 2500, 0.35, 3, 0.5, bounds)
        bbox_3D_origin_ends = bbox3d.get_3d_box_proposals(rgbs, camR_T_origin.astype(np.float32), origin_T_camXs.astype(np.float32), pix_T_camXs.astype(np.float32), xyz_camXs.astype(np.float32), 0)
        f['bbox_origin_predicted'] = bbox_3D_origin_ends
        f['bbox_origin'] = bbox_3D_origin_ends
        st()
        aa=1



# [-5, -5, 0, 5, 5, 10]
def test_habitat():
    # st()
    if 'Shamit' in hostname:
        basepath = '/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/habitat_processed/bb'
    else:
        # basepath = '/home/mprabhud/dataset/carla/npy/bb'
        basepath = '/hdd/datasets/replica_processed'
    pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]
    # pfiles = ['15821948243662689.p']# ['1582194725716713.p', '15821947268549619.p']
    '''
    self.iou_thresh = 0.5
    self.vote_thresh = 2
    '''
    fnum = 0
    for pfile in pfiles:
        print("Processing file number: ", fnum)
        print("Processing file: ", pfile)
        # st()
        fnum+=1
        bounds =  [-4, -4, 0, 4, 4, 8]
        file_ = open(os.path.join(basepath, pfile), "rb")
        f = pickle.load(file_)
        file_.close()

        camR_index = f['camR_index']
        # camR_index = 0

        rgbs = f['rgb_camXs_raw']
        origin_T_camXs = f['origin_T_camXs_raw']
        camR_T_origin = f['camR_T_origin_raw']
        # camR_T_origin = make_ith_view_camR(origin_T_camXs, camR_index)
        xyz_camXs = f['xyz_camXs_raw']
        pix_T_camXs = f['pix_T_cams_raw']

        indices_to_take = [camR_index,0,1,2,3,4,5]#[16, 5, 12, camR_index]
        # st()
        rgbs = rgbs[indices_to_take]
        origin_T_camXs = origin_T_camXs[indices_to_take]
        camR_T_origin = camR_T_origin[indices_to_take]
        xyz_camXs = xyz_camXs[indices_to_take]
        pix_T_camXs = pix_T_camXs[indices_to_take]
        
        # st()
        bbox3d = bbox_3d_proposal_from_2d(rgbs.shape[1], rgbs.shape[2], 5, 25000, 0.45, 3, 0.5, bounds)
        bbox_3D_origin_ends = bbox3d.get_3d_box_proposals(rgbs, camR_T_origin.astype(np.float32), origin_T_camXs.astype(np.float32), pix_T_camXs.astype(np.float32), xyz_camXs.astype(np.float32), 0)
        f['bbox_origin_predicted'] = bbox_3D_origin_ends
        # with open(os.path.join(basepath, pfile), "wb") as fwrite:
        #     pickle.dump(f, fwrite)

        # print("Difference is : ", np.linalg.norm(bbox_3D_origin_ends[0] - f['bbox_origin']))
        # print("Original ends: ",f['bbox_origin'])
        # print("Predicted ends: ", bbox_3D_origin_ends)

def test_carla():
    # st()
    if 'Shamit' in hostname:
        basepath = '/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/carla'
    else:
        basepath = '/home/mprabhud/dataset/carla/npy/bb'
        # basepath = '/home/sirdome/shubhankar/dataset/carla_test'
    pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]
    # pfiles = ['15821948243662689.p']# ['1582194725716713.p', '15821947268549619.p']
    '''
    self.iou_thresh = 0.5
    self.vote_thresh = 2
    '''
    fnum = 0
    for pfile in pfiles:
        print("Processing file number: ", fnum)
        print("Processing file: ", pfile)
        fnum+=1
        bounds = [-3.4, -3.4, 0, 3.4, 3.4, 6.8]
        file_ = open(os.path.join(basepath, pfile), "rb")
        f = pickle.load(file_)
        file_.close()

        camR_index = f['camR_index']
        camR_index = 0

        rgbs = f['rgb_camXs_raw']
        origin_T_camXs = f['origin_T_camXs_raw']
        camR_T_origin = make_ith_view_camR(origin_T_camXs, camR_index)
        xyz_camXs = f['xyz_camXs_raw']
        pix_T_camXs = f['pix_T_cams_raw']

        indices_to_take = [16, 5, 12, camR_index]
        # st()
        rgbs = rgbs[indices_to_take]
        origin_T_camXs = origin_T_camXs[indices_to_take]
        camR_T_origin = camR_T_origin[indices_to_take]
        xyz_camXs = xyz_camXs[indices_to_take]
        pix_T_camXs = pix_T_camXs[indices_to_take]
        
        
        bbox3d = bbox_3d_proposal_from_2d(rgbs.shape[1], rgbs.shape[2], 1, 30000, 0.5, 2, 0.5, bounds)
        bbox_3D_origin_ends = bbox3d.get_3d_box_proposals(rgbs, camR_T_origin, origin_T_camXs, pix_T_camXs, xyz_camXs, camR_index)
        f['bbox_origin_predicted'] = bbox_3D_origin_ends
        with open(os.path.join(basepath, pfile), "wb") as fwrite:
            pickle.dump(f, fwrite)
        # print("Difference is : ", np.linalg.norm(bbox_3D_origin_ends[0] - f['bbox_origin']))
        # print("Original ends: ",f['bbox_origin'])
        # print("Predicted ends: ", bbox_3D_origin_ends)

def test_clevr():
    # st()
    # if 'Shamit' in hostname:
    mod = sys.argv[1]
    basepath = '/projects/katefgroup/datasets/clevr_veggies/npys/' + mod
    print("Basepath is: ", basepath)
    # basepath = '/home/sirdome/shamit/datasets/clevr_veggies/single_large/npys'
    # else:
    # basepath = '/home/mprabhud/dataset/carla/npy/bb'
    # basepath = '/home/sirdome/shubhankar/dataset/carla_test'
    pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]
    # pfiles = ['15821948243662689.p']# ['1582194725716713.p', '15821947268549619.p']
    '''
    self.iou_thresh = 0.5
    self.vote_thresh = 2
    '''
    fnum = 0
    for pfile in pfiles:
        print("Processing file number: ", fnum)
        print("Processing file: ", pfile)
        fnum+=1
        bounds = [-7.5, -7.5, 5.5, 7.5, 7.5, 20.5]
        file_ = open(os.path.join(basepath, pfile), "rb")
        f = pickle.load(file_)
        file_.close()

        # camR_index = f['camR_index']
        # camR_index = 0
        # st()
        rgbs = f['rgb_camXs_raw']
        origin_T_camXs = f['origin_T_camXs_raw']
        camR_T_origin = f['camR_T_origin_raw']
        xyz_camXs = f['xyz_camXs_raw']
        pix_T_camXs = f['pix_T_cams_raw']

        # indices_to_take = [16, 5, 12, 0]
        indices_to_take = [0,5,10,13,15,20,23]
        # indices_to_take = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        # st()
        rgbs = rgbs[indices_to_take]
        rgbs = rgbs[:,:,:,:3]
        origin_T_camXs = origin_T_camXs[indices_to_take]
        camR_T_origin = camR_T_origin[indices_to_take]
        xyz_camXs = xyz_camXs[indices_to_take]
        pix_T_camXs = pix_T_camXs[indices_to_take]
        
        
        bbox3d = bbox_3d_proposal_from_2d(rgbs.shape[1], rgbs.shape[2], 1, 10000, 0.3, 3, 0.5, bounds)
        # st()
        bbox_3D_origin_ends = bbox3d.get_3d_box_proposals(rgbs, camR_T_origin, origin_T_camXs, pix_T_camXs, xyz_camXs, None)
        f['bbox_origin_predicted'] = bbox_3D_origin_ends
        with open(os.path.join(basepath, pfile), "wb") as fwrite:
            pickle.dump(f, fwrite)
        # print("Difference is : ", np.linalg.norm(bbox_3D_origin_ends[0] - f['bbox_origin']))
        # print("Original ends: ",f['bbox_origin'])
        # print("Predicted ends: ", bbox_3D_origin_ends)


def make_ith_view_camR(origin_T_camXs, idx):
    origin_T_camR = origin_T_camXs[idx]
    camR_T_origin = utils_geom.safe_inverse(torch.tensor(origin_T_camR).unsqueeze(0)).repeat(origin_T_camXs.shape[0], 1, 1).numpy()
    return camR_T_origin


def test_carla_two_vehicles():
    # 0, 2, 4,5 ,6 , 9, 12(g), 13(g), 17
    # 1
    if 'Shamit' in hostname:
        basepath = '/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/carla'
    else:
        # basepath = '/home/mprabhud/dataset/carla/npy/bb'
        basepath = '/home/sirdome/shubhankar/dataset/carla_test'
    pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]
    # pfiles = ['15821948243662689.p']# ['1582194725716713.p', '15821947268549619.p']
    fnum = 0
    for pfile in pfiles:
        print("Processing file number: ", fnum)
        print("Processing file: ", pfile)
        fnum+=1
        bounds = [-7.5, -7.5, 0, 7.5, 7.5, 15.0]
        
        f = pickle.load(open(os.path.join(basepath, pfile), "rb"))

        camR_index = f['camR_index']
        camR_index = 0

        rgbs = f['rgb_camXs_raw']
        origin_T_camXs = f['origin_T_camXs_raw']
        camR_T_origin = make_ith_view_camR(origin_T_camXs, camR_index)
        xyz_camXs = f['xyz_camXs_raw']
        pix_T_camXs = f['pix_T_cams_raw']

        indices_to_take = [0, 2, 9, 10, 12, 13, 17]
        # indices_to_take = np.arange(rgbs.shape[0])
        
        # st()
        rgbs = rgbs[indices_to_take]
        origin_T_camXs = origin_T_camXs[indices_to_take]
        camR_T_origin = camR_T_origin[indices_to_take]
        xyz_camXs = xyz_camXs[indices_to_take]
        pix_T_camXs = pix_T_camXs[indices_to_take]
        
        
        bbox3d = bbox_3d_proposal_from_2d(rgbs.shape[1], rgbs.shape[2], 2, 4000, 0.60, 2, 0.5, bounds)
        bbox3d.get_3d_box_proposals(rgbs, camR_T_origin, origin_T_camXs, pix_T_camXs, xyz_camXs, camR_index)

if __name__ == '__main__':
    # test_carla_two_vehicles()
    # test_habitat()
    test_kinect()
    # test_clevr()
    

