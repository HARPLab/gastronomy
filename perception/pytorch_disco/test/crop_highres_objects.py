import numpy as np 
import pickle 
import socket
hostname = socket.gethostname()
import getpass
username = getpass.getuser()
import torch.nn.functional as F
import os
import open3d as o3d
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "trainer_rgb_occ_no_bn_lr3_kinect"
os.environ["run_name"] = "check"
import hyperparams as hyp
import ipdb 
st = ipdb.set_trace
import utils_geom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

import torch
import utils_improc
import utils_basic
import utils_vox
from lib_classes import Nel_Utils as nlu
if 'Shamit' in hostname:
    basepath = '/Users/shamitlal/Desktop/shamit/cmu/katefgroup/datasets/kinect_nips'
else:
    basepath = '/projects/katefgroup/datasets/kinect/npys/nips_bbox'
pfiles = [f for f in os.listdir(basepath) if f.endswith('.p')]

B = 1
visualize = False

def g(tensor):
    return torch.tensor(tensor).unsqueeze(0).float().cuda()

def scatterplot_matplotlib(xyz_camX):
    # st()
    xyz_camX = xyz_camX[:, ::50]
    xyz_camX = xyz_camX.reshape(-1,3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = xyz_camX[:,0], xyz_camX[:,1], xyz_camX[:,2]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show(block=True)

summ_writer = utils_improc.Summ_writer(None, 10, "train")
cnt = 0
for pfile in pfiles:
    print("Processing file: {}, filenum: {}".format(cnt, pfile))
    cnt+=1
    # if not pfile.startswith('single'):
    #     print("File doesnt satisfy single condition. Ignoring")
    #     continue

    # if not pfile.startswith('multi'):
    #     print("File doesnt satisfy multi condition. Ignoring")
    #     continue

    # if not pfile.startswith('2_multi'):
    #     print("File doesnt satisfy 2_multi condition. Ignoring")
    #     continue

    feed = pickle.load(open(os.path.join(basepath, pfile), 'rb'))
    # st()
    # for rgb_vis in feed['rgb_camXs_raw']:
    #     plt.imshow(rgb_vis)
    #     plt.show(block=True)
    if 'cropped_occRs' in feed.keys():
        print("This file has been processed. Skipping")
        continue
    
    scatterplot_matplotlib(feed["xyz_camXs_raw"])
    gt_boxes_origin = g(feed['bbox_origin'])
    rgb_camXs = feed['rgb_camXs_raw']
    # hyp.dataset_name == "kinect"

    rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
    rgb_camXs = rgb_camXs[:,:3]
    rgb_camXs = g(utils_improc.preprocess_color(rgb_camXs))
    
    S, H, W = rgb_camXs.shape[1], rgb_camXs.shape[2], rgb_camXs.shape[3]
    rgb_camXs_all = rgb_camXs
    
    Z, Y, X = 320,320,320
    # Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
    # Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
    
    pix_T_cams = g(feed["pix_T_cams_raw"])
    camRs_T_origin = g(feed["camR_T_origin_raw"])
    origin_T_camXs = g(feed["origin_T_camXs_raw"])
    xyz_camXs = g(feed["xyz_camXs_raw"])
    
    pix_T_cams_all = pix_T_cams
    camRs_T_origin_all = camRs_T_origin
    origin_T_camXs_all = origin_T_camXs
    xyz_camXs_all = xyz_camXs

    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)
    cropped_occRs_all = []
    cropped_unpRs_all = []
    # st()
    for viewnum in range(rgb_camXs_all.shape[1]):
        print("Processing view: ", viewnum)
        rgb_camXs = rgb_camXs_all[:,viewnum:viewnum+1]
        pix_T_cams = pix_T_cams_all[:,viewnum:viewnum+1]
        camRs_T_origin = camRs_T_origin_all[:,viewnum:viewnum+1]
        origin_T_camXs = origin_T_camXs_all[:,viewnum:viewnum+1]
        xyz_camXs = xyz_camXs_all[:,viewnum:viewnum+1]

        origin_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_origin)))
        
        N = gt_boxes_origin.shape[1]

        __pb = lambda x: utils_basic.pack_boxdim(x, N)
        __ub = lambda x: utils_basic.unpack_boxdim(x, N)


        tids = torch.from_numpy(np.reshape(np.arange(B*N),[B,N]))

        # st()
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))

        # depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        # dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))

        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))

        # st()
        if visualize:
            pcd = nlu.make_pcd(xyz_camXs[0,0].numpy())
            o3d.visualization.draw_geometries([pcd])
        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))

        unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z, Y, X, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
    
        ## projected depth, and inbound mask
        # st()
        # dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        # inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        # inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        
        # depth_camXs = __u(depth_camXs_)
        # valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))

        if visualize:
            unpOccR_vis = utils_basic.reduce_masked_mean(unpRs, occRs.repeat(1,1,unpRs.shape[2],1,1,1), dim=-2)
            unpOccR_vis = unpOccR_vis[0,0].numpy().transpose(1,2,0) + 0.5
            plt.imshow(unpOccR_vis)
            plt.show(block=True)
    

        gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[B,N,2,3])
        gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
        gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
        gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))
        gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)

        gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z,Y,X))
        gt_boxesRMem_end = nlu.get_ends_of_corner(gt_boxesRMem_corners)
        gt_boxesRMem_theta = utils_geom.transform_corners_to_boxes(gt_boxesRMem_corners)
        gt_boxesRUnp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z,Y,X))
        gt_boxesRUnp_end = nlu.get_ends_of_corner(gt_boxesRUnp_corners)

        if visualize:
            unps_visRs = utils_improc.get_unps_vis(unpRs, occRs)
            unp_visRs = torch.mean(unps_visRs, dim=1)
            scores = torch.ones((B,N))
            tids = scores
            bbox_vis = summ_writer.summ_box_mem_on_unp('eval_boxes/gt_boxesR_mem', unp_visRs , gt_boxesRMem_end, scores ,tids, only_return=True)
            bbox_vis = bbox_vis[0].numpy().transpose(1,2,0) + 0.5
        
        occ_list = []
        unp_list = []
        for i in range(N):
            gt_boxesRMem_end_i = gt_boxesRMem_end[0,i].reshape(-1)
            xmin, ymin, zmin, xmax, ymax, zmax = gt_boxesRMem_end_i

            xmin = torch.floor(xmin).to(torch.int32)
            ymin = torch.floor(ymin).to(torch.int32)
            zmin = torch.floor(zmin).to(torch.int32)
            xmax = torch.ceil(xmax).to(torch.int32)
            ymax = torch.ceil(ymax).to(torch.int32)
            zmax = torch.ceil(zmax).to(torch.int32)

            print("Bbox dimensions: ", xmax-xmin, ymax-ymin, zmax-zmin)

            cropped_occ_i =  F.interpolate(occRs[0,:,:,zmin:zmax, ymin:ymax, xmin:xmax], size = 32, mode='nearest')
            cropped_unp_i =  F.interpolate(unpRs[0,:,:,zmin:zmax, ymin:ymax, xmin:xmax], size = 32, mode='trilinear')
            
            occ_list.append(cropped_occ_i)
            unp_list.append(cropped_unp_i)
        
        cropped_occRs = torch.stack(occ_list)
        cropped_unpRs = torch.stack(unp_list)
        # st()
        cropped_occRs_all.append(cropped_occRs)
        cropped_unpRs_all.append(cropped_unpRs)
        
        if visualize:
            unpOccR_vis = utils_basic.reduce_masked_mean(cropped_unpRs, cropped_occRs.repeat(1,1,cropped_unpRs.shape[2],1,1,1), dim=-2)
            unpOccR_vis = unpOccR_vis[0,0].numpy().transpose(1,2,0) + 0.5
            plt.imshow(unpOccR_vis)
            plt.show(block=True)
    feed['cropped_occRs'] = torch.stack(cropped_occRs_all).cpu().numpy()[:,:,0]
    feed['cropped_unpRs'] = torch.stack(cropped_unpRs_all).cpu().numpy()[:,:,0]
    with open(os.path.join(basepath, pfile), "wb") as fwrite:
        pickle.dump(feed, fwrite)

    



