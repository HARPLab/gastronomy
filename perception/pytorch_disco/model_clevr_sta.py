import torch
import torch.nn as nn
import hyperparams as hyp
import cross_corr
import numpy as np
import imageio
import os
import json
from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet

from nets.detnet import DetNet
from nets.embnet3D import EmbNet3D
from collections import defaultdict
import torch.nn.functional as F
from os.path import join
import time
import pickle
import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import socket
import cross_corr
import utils_basic
import ipdb
st = ipdb.set_trace
import scipy
import utils_vox
import utils_eval
from archs.vector_quantizer import VectorQuantizer,VectorQuantizer_vox,VectorQuantizer_Eval,VectorQuantizer_Instance_Vr,VectorQuantizer_Instance_Vr_All
from archs.vector_quantizer_ema import VectorQuantizerEMA
import sklearn
from DoublePool import SinglePool
import torchvision.models as models
from lib_classes import Nel_Utils as nlu
import copy
np.set_printoptions(precision=2)
np.random.seed(0)
from sklearn.cluster import MiniBatchKMeans

class CLEVR_STA(Model):

    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        self.model = ClevrStaNet()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)

class ClevrStaNet(nn.Module):
    def __init__(self):
        super(ClevrStaNet, self).__init__()
        self.device = "cuda"
        self.list_of_classes = []
        
        if hyp.do_det:
            self.detnet = DetNet()
            if hyp.self_improve_once or hyp.filter_boxes:
                self.detnet_target = DetNet()
                self.detnet_target.eval()
        
        if hyp.dataset_name == "clevr":
            self.minclasses = 20
        elif hyp.dataset_name == "carla":
            self.minclasses = 26
        elif hyp.dataset_name == "replica":
            self.minclasses = 26            
        else:
            self.minclasses = 41

        if hyp.quant_init != "":
             hyp.object_quantize_init = None

        if hyp.object_quantize or hyp.filter_boxes or hyp.self_improve_iterate:
            embed_size = hyp.BOX_SIZE*hyp.BOX_SIZE*hyp.BOX_SIZE*hyp.feat_dim
            
            self.quantizer = VectorQuantizer(num_embeddings=hyp.object_quantize_dictsize,
                                            embedding_dim=embed_size,
                                            init_embeddings=hyp.object_quantize_init,
                                            commitment_cost=hyp.object_quantize_comm_cost)

        if hyp.gt_rotate_combinations:
            self.mbr_unpr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE*2,hyp.BOX_SIZE*2,hyp.BOX_SIZE*2)
        
        self.info_dict = defaultdict(lambda:[])
        if hyp.create_example_dict:
            self.embed_dict = defaultdict(lambda:0)
            self.embed_list = []
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()

        self.is_empty_occ_generated = False

        self.avg_ap = []
        self.avg_precision = []                    
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def load_config(self,exp_name):
        path = os.path.join('experiments', exp_name, 'config.json')
        with open(path) as file:
            config = json.load(file)
        assert config['name']==exp_name
        return config


    def evaluate_filter_boxes(self,gt_boxesRMem_theta,scores,featR,summ_writer):
        if hyp.filter_boxes or hyp.self_improve_iterate:
            with torch.no_grad():
                if hyp.self_improve_iterate:
                    _, boxlist_memR_e_pre_filter, scorelist_e_pre_filter, _, _, _ = self.detnet(
                            self.axboxlist_memR,
                            self.scorelist_s,
                            featR,
                            summ_writer)                    
                else:
                    _, boxlist_memR_e_pre_filter, scorelist_e_pre_filter, _, _, _ = self.detnet_target(
                        self.axboxlist_memR,
                        self.scorelist_s,
                        featR,
                        summ_writer)
            summ_writer.summ_box_mem_on_mem('detnet/pre_filter_boxesR_mem', self.unp_visRs, boxlist_memR_e_pre_filter ,scorelist_e_pre_filter,torch.ones_like(scorelist_e_pre_filter,dtype=torch.int32))

            corners_memR_e_pred = utils_geom.transform_boxes_to_corners(boxlist_memR_e_pre_filter)
            end_memR_e_pred = nlu.get_ends_of_corner(corners_memR_e_pred)

            emb3D_e_R = utils_vox.apply_4x4_to_vox(self.camR_T_camX0, self.emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(self.camR_T_camX0, self.emb3D_g)

            emb3D_R = emb3D_e_R

            neg_boxesMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det,2,3]).cuda()
            neg_scoresMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det]).cuda()

            gt_boxesMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det,2,3]).cuda()
            gt_scoresMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det])

            gt_boxesMem_to_consider_after_q_distance  = torch.zeros([hyp.B,self.N_det,2,3]).cuda()
            gt_scoresMem_to_consider_after_q_distance  = torch.zeros([hyp.B,self.N_det])


            emb3D_e_R_object, emb3D_g_R_object, indices, end_memR_e_pred_filtered, neg_indices, neg_boxes = nlu.create_object_tensors_filter_cs([emb3D_e_R, emb3D_g_R],  end_memR_e_pred, scorelist_e_pre_filter,[hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE], cs_check= hyp.cs_filter)

            feat_mask = torch.zeros([hyp.B,1,hyp.Z2,hyp.Y2,hyp.X2]).cuda()
            feat_mask_vis = torch.ones([1,3,hyp.Z2,hyp.X2]).cuda()*-0.5

            validR_combo_object = None
            
            if emb3D_e_R_object is not None:
                if hyp.cs_filter:                            
                    for ind,index_val in enumerate(indices):
                        batch_index, box_index = index_val
                        box_val = end_memR_e_pred_filtered[ind]
                        assert  (end_memR_e_pred_filtered[ind] == end_memR_e_pred[batch_index,box_index]).all()
                        gt_scoresMem_to_consider_after_cs[batch_index,box_index] = 1.0
                        gt_boxesMem_to_consider_after_cs[batch_index,box_index] = box_val
                    
                    if neg_boxes is not None:                                    
                        for neg_ind,neg_index_val in enumerate(neg_indices):
                            batch_index, box_index = neg_index_val
                            neg_box_val = neg_boxes[neg_ind]
                            assert  (neg_boxes[neg_ind] == end_memR_e_pred[batch_index,box_index]).all()
                            neg_boxesMem_to_consider_after_cs[batch_index,box_index] = neg_box_val
                            neg_scoresMem_to_consider_after_cs[batch_index,box_index] = 1.0

                    gt_boxesMem_to_consider_after_cs_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_cs)
                    summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_cs', self.unp_visRs, gt_boxesMem_to_consider_after_cs_theta ,gt_scoresMem_to_consider_after_cs,torch.ones([hyp.B,6],dtype=torch.int32))
                    neg_boxesMem_to_consider_after_cs_theta = nlu.get_alignedboxes2thetaformat(neg_boxesMem_to_consider_after_cs)
                    summ_writer.summ_box_mem_on_mem('detnet/sudo_neg_mem_filtered_cs', self.unp_visRs, neg_boxesMem_to_consider_after_cs_theta ,neg_scoresMem_to_consider_after_cs,torch.ones([hyp.B,6],dtype=torch.int32))


                emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2

                emb3D_R_object.shape[0] == indices.shape[0]

                distances = self.quantizer(emb3D_R_object)
                min_distances = torch.min(distances,dim=1).values


                selections = 0
                for i in range(distances.shape[0]):
                    min_distance = min_distances[i]
                    if min_distance <hyp.dict_distance_thresh:
                        selections += 1
                        index_val = indices[i]
                        batch_index, box_index = index_val
                        box_val = end_memR_e_pred_filtered[i]
                        gt_scoresMem_to_consider_after_q_distance[batch_index,box_index] = 1.0
                        gt_boxesMem_to_consider_after_q_distance[batch_index,box_index] = box_val


                if selections > 0:
                    gt_boxesMem_to_consider_after_q_distance = torch.stack([torch.clamp(gt_boxesMem_to_consider_after_q_distance[:,:,:,0],min=0,max=hyp.X2),torch.clamp(gt_boxesMem_to_consider_after_q_distance[:,:,:,1],min=0,max=hyp.Y2),torch.clamp(gt_boxesMem_to_consider_after_q_distance[:,:,:,2],min=0,max=hyp.Z2)],dim=-1)
                    for b_index in range(hyp.B):
                        for n_index in range(self.N_det):
                            if gt_scoresMem_to_consider_after_q_distance[b_index,n_index] > 0.0:
                                box = gt_boxesMem_to_consider_after_q_distance[b_index,n_index]
                                lower,upper = torch.unbind(box)

                                xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
                                xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]
                                assert (xmax-xmin) >0 and (ymax-ymin) >0 and (zmax-zmin) >0

                                padding = 3
                                xmin_padded,ymin_padded,zmin_padded = (max(xmin-padding,0),max(ymin-padding,0),max(zmin-padding,0))
                                xmax_padded,ymax_padded,zmax_padded = (min(xmax+padding,hyp.X2),min(ymax+padding,hyp.Y2),min(zmax+padding,hyp.Z2))

                                feat_mask[b_index,:,zmin_padded:zmax_padded,ymin_padded:ymax_padded,xmin_padded:xmax_padded] = 1.0
                                if b_index == 0:
                                    feat_mask_vis[b_index,1,zmin_padded:zmax_padded,xmin_padded:xmax_padded] = 0.5
                                    feat_mask_vis[b_index,1,zmin:zmax,xmin:xmax] = 0.1



                if neg_boxes is not None:
                    neg_boxesMem_to_consider_after_cs = torch.stack([torch.clamp(neg_boxesMem_to_consider_after_cs[:,:,:,0],min=0,max=hyp.X2),torch.clamp(neg_boxesMem_to_consider_after_cs[:,:,:,1],min=0,max=hyp.Y2),torch.clamp(neg_boxesMem_to_consider_after_cs[:,:,:,2],min=0,max=hyp.Z2)],dim=-1)
                    for b_index in range(hyp.B):
                        for n_index in range(self.N_det):
                            if neg_scoresMem_to_consider_after_cs[b_index,n_index] > 0.0:
                                box = neg_boxesMem_to_consider_after_cs[b_index,n_index]
                                lower,upper = torch.unbind(box)

                                xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
                                xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]
                                assert (xmax-xmin) >0 and (ymax-ymin) >0 and (zmax-zmin) >0

                                padding = 0
                                xmin_padded,ymin_padded,zmin_padded = (max(xmin-padding,0),max(ymin-padding,0),max(zmin-padding,0))
                                xmax_padded,ymax_padded,zmax_padded = (min(xmax+padding,hyp.X2),min(ymax+padding,hyp.Y2),min(zmax+padding,hyp.Z2))

                                feat_mask[b_index,:,zmin_padded:zmax_padded,ymin_padded:ymax_padded,xmin_padded:xmax_padded] = 1.0
                                if b_index == 0:
                                    feat_mask_vis[b_index,0,zmin_padded:zmax_padded,xmin_padded:xmax_padded] = 0.1


                gt_boxesMem_to_consider_after_q_distance_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_q_distance)
                # st()
                summ_writer.summ_rgb('detnet/mask_vis', feat_mask_vis)
                summ_writer.summ_occ('detnet/mask_used', feat_mask)
                summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_quant', self.unp_visRs, gt_boxesMem_to_consider_after_q_distance_theta ,gt_scoresMem_to_consider_after_q_distance,torch.ones([hyp.B,6],dtype=torch.int32))
            else:
                gt_boxesMem_to_consider_after_cs_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_cs)
                gt_boxesMem_to_consider_after_q_distance_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_q_distance)
                summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_cs', self.unp_visRs, torch.zeros_like(gt_boxesRMem_theta) ,torch.zeros([hyp.B,6]),torch.ones([hyp.B,6],dtype=torch.int32))
                summ_writer.summ_rgb('detnet/mask_vis', feat_mask_vis)
                summ_writer.summ_occ('detnet/mask_used', feat_mask)            
                summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_quant', self.unp_visRs, torch.zeros_like(gt_boxesRMem_theta) ,torch.zeros([hyp.B,6]),torch.ones([hyp.B,6],dtype=torch.int32))
        return gt_boxesMem_to_consider_after_q_distance_theta, gt_scoresMem_to_consider_after_q_distance, feat_mask, gt_boxesMem_to_consider_after_cs_theta,gt_scoresMem_to_consider_after_cs

    def forward(self, feed):

        results = dict()
        if 'log_freq' not in feed.keys():
            feed['log_freq'] = None
        start_time = time.time()

        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               log_freq=feed['log_freq'],
                                               fps=8)
        writer = feed['writer']
        global_step = feed['global_step']

          

        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)

        __pb = lambda x: utils_basic.pack_boxdim(x, hyp.N)
        __ub = lambda x: utils_basic.unpack_boxdim(x, hyp.N)


        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        PH, PW = hyp.PH, hyp.PW
        K = hyp.K
        BOX_SIZE = hyp.BOX_SIZE
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9
        
        tids = torch.from_numpy(np.reshape(np.arange(B*N),[B,N]))

        rgb_camXs = feed["rgb_camXs_raw"]
        pix_T_cams = feed["pix_T_cams_raw"]

        camRs_T_origin = feed["camR_T_origin_raw"]
        origin_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_origin)))
        origin_T_camXs = feed["origin_T_camXs_raw"]

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))

        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
        camX0_T_camRs = camXs_T_camRs[:,0]
        camX1_T_camRs = camXs_T_camRs[:,1]

        camR_T_camX0  = utils_geom.safe_inverse(camX0_T_camRs)

        xyz_camXs = feed["xyz_camXs_raw"]

        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))

        if hyp.low_res:
            if hyp.dataset_name == "carla" or hyp.dataset_name == "carla_mix":
                xyz_camXs = __u(dense_xyz_camXs_)

        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))

        occXs_to_Rs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, occXs) # torch.Size([2, 2, 1, 144, 144, 144])
        occXs_to_Rs_45 = cross_corr.rotate_tensor_along_y_axis(occXs_to_Rs, 45)

        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))

        occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))

        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))

        unpX0s_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camX0_T_camXs)))))

        unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z, Y, X, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
        
        unpRs_half = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
            
            
        ## projected depth, and inbound mask
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1),maxdepth=21.0)
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))

        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        
        if hyp.profile_time:
            print("landmark time",time.time()-start_time)

        if summ_writer.save_this:
            summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
            summ_writer.summ_occs('3D_inputs/occXs_to_Rs', torch.unbind(occXs_to_Rs, dim=1))
            summ_writer.summ_occs('3D_inputs/occXs_to_Rs_45', torch.unbind(occXs_to_Rs_45, dim=1))
            summ_writer.summ_unps('3D_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))

        #####################
        ## run the nets
        #####################
        start_time = time.time()
        
        if hyp.do_eval_boxes:
            if hyp.dataset_name == "carla":
                tree_seq_filename = feed['tree_seq_filename']
                tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename]
                trees = [pickle.load(open(i,"rb")) for i in tree_filenames]           

                gt_boxes_origin,scores,classes = nlu.trees_rearrange_corners(trees)
                

                gt_boxes_origin = torch.from_numpy(gt_boxes_origin).cuda().to(torch.float)
                if hyp.use_2d_boxes:
                    prd_boxes = feed['predicted_box']
                    prd_scores = feed['predicted_scores'].detach().cpu().numpy()
                    gt_boxes_origin = prd_boxes
                    scores = prd_scores
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))

                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
                rgb_camtop = feed['rgb_camtop'].squeeze(1)
                origin_T_camXs_top = feed['origin_T_camXs_top']
                gt_boxescamXTop_corners = __ub(utils_geom.apply_4x4(utils_geom.safe_inverse(__p(origin_T_camXs_top)), __pb(gt_boxes_origin_corners)))

            elif hyp.dataset_name =="carla_mix":
                predicted_box_origin = feed['predicted_box']
                predicted_scores_origin = feed['predicted_scores']
                gt_boxes_origin = feed['gt_box']
                gt_scores_origin = feed['gt_scores']
                classes = feed['classes']
                tree_seq_filename = feed['tree_seq_filename']
                scores = gt_scores_origin
                if hyp.use_2d_boxes:
                    gt_boxes_origin = predicted_box_origin
                    scores = predicted_scores_origin
                scores = scores.detach().cpu().numpy()
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))
                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
            elif hyp.dataset_name =="replica":
                gt_boxes_origin = feed['gt_box']
                gt_scores_origin = feed['gt_scores']
                classes = feed['classes']
                scores = gt_scores_origin
                tree_seq_filename = feed['tree_seq_filename']
                if hyp.moc or hyp.do_emb3D:
                    gt_boxes_origin_f = gt_boxes_origin[:,:1].cpu().detach().numpy()
                    gt_scores_origin_f = gt_scores_origin[:,:1].cpu().detach().numpy()
                    classes_f = classes[:,:1]                    
                    N_new = 1
                    gt_boxes_origin = torch.from_numpy(np.pad(gt_boxes_origin_f,[[0,0],[0,hyp.N-N_new],[0,0]])).cuda()
                    gt_scores_origin = torch.from_numpy(np.pad(gt_scores_origin_f,[[0,0],[0,hyp.N-N_new]])).cuda()
                    classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])                    
                    scores = gt_scores_origin
                # st()
                # if hyp.use_2d_boxes:
                #     gt_boxes_origin = predicted_box_origin
                #     scores = predicted_scores_origin
                scores = scores.detach().cpu().numpy()
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))

                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)                                
            else:
                tree_seq_filename = feed['tree_seq_filename']
                tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename]
                trees = [pickle.load(open(i,"rb")) for i in tree_filenames]
                if hyp.use_det_boxes:
                    bbox_detsR = []
                    score_dets = []
                    for tree in trees:
                        if not hasattr(tree, 'bbox_det'):
                            bbox_detsR.append(torch.zeros([hyp.N,9]).cuda())
                            score_dets.append(torch.zeros([6]).cuda())
                        else:
                            bbox_detsR.append(tree.bbox_det)
                            score_dets.append(tree.score_det)                    
                    bbox_detsR = torch.stack(bbox_detsR)
                    bbox_dets_cornersR = utils_geom.transform_boxes_to_corners(bbox_detsR)
                    bbox_dets_cornersR = __ub(utils_vox.Mem2Ref(__pb(bbox_dets_cornersR),Z2,Y2,X2))
                    bbox_dets_endR = nlu.get_ends_of_corner(bbox_dets_cornersR).cpu().detach().numpy()
                    score_dets = torch.stack(score_dets).cpu().detach().numpy()
                    gt_boxesR = bbox_dets_endR
                    scores = score_dets
                    best_indices = np.flip(np.argsort(scores),axis=1)
                    sorted_scores = []
                    sorted_boxes = []

                    for ind,sorted_index in enumerate(best_indices):
                        sorted_scores.append(scores[ind][sorted_index])
                        sorted_boxes.append(gt_boxesR[ind][sorted_index])
                    sorted_scores = np.stack(sorted_scores)
                    sorted_boxes = np.stack(sorted_boxes)
                    classes = np.reshape(['temp']*hyp.N*hyp.B,[hyp.B,hyp.N])

                    # take top2
                    gt_boxesR_f = sorted_boxes[:,:2]
                    gt_scoresR_f = sorted_scores[:,:2]
                    classes_f = classes[:,:2]                    
                    N_new = gt_boxesR_f.shape[1]

                    gt_boxesR = np.pad(gt_boxesR_f,[[0,0],[0,hyp.N-N_new],[0,0],[0,0]])
                    scores = np.pad(gt_scoresR_f,[[0,0],[0,hyp.N-N_new]])
                    classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])

                elif hyp.use_2d_boxes:
                    gt_boxesR,scores,classes = nlu.trees_rearrange_2d(trees)
                else:
                    gt_boxesR,scores,classes = nlu.trees_rearrange(trees)
                gt_boxesR = torch.from_numpy(gt_boxesR).cuda() # torch.Size([2, 3, 6])
                gt_boxesR_end = torch.reshape(gt_boxesR,[hyp.B,hyp.N,2,3])
                gt_boxesR_theta = nlu.get_alignedboxes2thetaformat(gt_boxesR_end) #torch.Size([2, 3, 9])
                gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta)
            class_names_ex_1 = "_".join(classes[0])
            summ_writer.summ_text('eval_boxes/class_names', class_names_ex_1)
            
            gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))
            gt_boxesRMem_end = nlu.get_ends_of_corner(gt_boxesRMem_corners)
            
            if hyp.dataset_name == "carla" or hyp.dataset_name == "carla_mix":
                gt_boxesR_end = __ub(utils_vox.Mem2Ref(__pb(gt_boxesRMem_end),Z2,Y2,X2))

                gt_boxesR_theta = nlu.get_alignedboxes2thetaformat(gt_boxesR_end) #torch.Size([2, 3, 9])
                gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta)                
                gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))

            gt_boxesRMem_theta = utils_geom.transform_corners_to_boxes(gt_boxesRMem_corners)
            gt_boxesRUnp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z,Y,X))
            gt_boxesRUnp_end = nlu.get_ends_of_corner(gt_boxesRUnp_corners)
            
            if hyp.gt_rotate_combinations:
                gt_boxesX1_corners = __ub(utils_geom.apply_4x4(camX1_T_camRs, __pb(gt_boxesR_corners)))
                gt_boxesX1Unp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX1_corners),Z,Y,X))
                gt_boxesX1Unp_end = nlu.get_ends_of_corner(gt_boxesX1Unp_corners)
            
            gt_boxesX0_corners = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners)))
            gt_boxesX0Mem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX0_corners),Z2,Y2,X2))

            gt_boxesX0Mem_theta = utils_geom.transform_corners_to_boxes(gt_boxesX0Mem_corners)
            
            gt_boxesX0Mem_end = nlu.get_ends_of_corner(gt_boxesX0Mem_corners)
            gt_boxesX0_end = nlu.get_ends_of_corner(gt_boxesX0_corners)

            gt_cornersX0_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))

            rgb_camX0 = rgb_camXs[:,0]
            rgb_camX1 = rgb_camXs[:,1]

            if hyp.dataset_name == "carla":
                summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamXtop', rgb_camtop, gt_boxescamXTop_corners, torch.from_numpy(scores), tids, pix_T_cams[:, 0])
            summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamX0', rgb_camX0, gt_boxesX0_corners, torch.from_numpy(scores), tids, pix_T_cams[:, 0])
            unps_vis = utils_improc.get_unps_vis(unpX0s_half, occX0s_half)
            unp_vis = torch.mean(unps_vis, dim=1)
            unps_visRs = utils_improc.get_unps_vis(unpRs_half, occRs_half)
            unp_visRs = torch.mean(unps_visRs, dim=1)
            unps_visRs_full = utils_improc.get_unps_vis(unpRs, occRs)
            unp_visRs_full = torch.mean(unps_visRs_full, dim=1)
        
            summ_writer.summ_box_mem_on_unp('eval_boxes/gt_boxesR_mem', unp_visRs , gt_boxesRMem_end, scores ,tids)
            
            unpX0s_half = torch.mean(unpX0s_half, dim=1)
            unpX0s_half = nlu.zero_out(unpX0s_half,gt_boxesX0Mem_end,scores)

            occX0s_half = torch.mean(occX0s_half, dim=1)
            occX0s_half = nlu.zero_out(occX0s_half,gt_boxesX0Mem_end,scores)            

            summ_writer.summ_unp('3D_inputs/unpX0s', unpX0s_half, occX0s_half)

        if hyp.do_feat:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
           
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)
            # it is useful to keep track of what was visible from each viewpoint
            freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occXs_half))
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half+freeXs, 0.0, 1.0)
            mask_ = None            
            if(type(mask_)!=type(None)):
                assert(list(mask_.shape)[2:5]==list(featXs_input_.shape)[2:5])
            featXs_, feat_loss = self.featnet(featXs_input_, summ_writer, mask=__p(occXs))#mask_)
            total_loss += feat_loss

            validXs = torch.ones_like(visXs)
            _validX00 = validXs[:,0:1]
            _validX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], validXs[:,1:])
            validX0s = torch.cat([_validX00, _validX01], dim=1)
            validRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, validXs)
            visRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, visXs)

            featXs = __u(featXs_)
            _featX00 = featXs[:,0:1]
            _featX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], featXs[:,1:])
            featX0s = torch.cat([_featX00, _featX01], dim=1)

            emb3D_e = torch.mean(featX0s[:,1:], dim=1) # context
            vis3D_e_R = torch.max(visRs[:,1:], dim=1)[0]
            emb3D_g = featX0s[:,0] # obs
            vis3D_g_R = visRs[:,0] # obs #only select those indices which are visible and valid.
            validR_combo = torch.min(validRs,dim=1).values

            if hyp.do_save_vis:

                imageio.imwrite('%s_rgb_%06d.png' % (hyp.name, global_step), np.transpose(utils_improc.back2color(rgb_camRs)[0,0].detach().cpu().numpy(), axes=[1, 2, 0]))
                np.save('%s_emb3D_g_%06d.npy' % (hyp.name, global_step), emb3D_e.detach().cpu().numpy())

            summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), valids=torch.unbind(validXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), valids=torch.unbind(torch.ones_like(validRs), dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validRs', torch.unbind(validRs, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e_R', vis3D_e_R, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g_R', vis3D_g_R, pca=False)
            
            if hyp.profile_time:
                print("featnet time",time.time()-start_time)
            
        if hyp.do_det:
            featRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, featXs)
            featR = torch.mean(featRs, dim=1)
            self.unp_visRs = unp_visRs
            self.camR_T_camX0 = camR_T_camX0
            self.emb3D_e = emb3D_e
            self.emb3D_g = emb3D_g

            self.N_det = hyp.K*2
            self.axboxlist_memR = utils_geom.inflate_to_axis_aligned_boxlist(gt_boxesRMem_theta)
            self.scorelist_s = torch.from_numpy(scores).cuda().to(torch.float)
            
            if hyp.do_det and hyp.self_improve_iterate:
                if hyp.exp_do:
                    gt_boxesMem_to_consider_after_q_distance_theta, gt_scoresMem_to_consider_after_q_distance, feat_mask,gt_boxesMem_to_consider_after_cs_theta,gt_scoresMem_to_consider_after_cs = self.evaluate_filter_boxes(gt_boxesRMem_theta,scores,featR,summ_writer)
                    if hyp.replace_with_cs:
                        gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_cs_theta
                        gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_cs

                    gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_q_distance_theta.detach()
                    gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_q_distance.detach().cuda()
                    axboxlist_memR_filtered = []
                    scorelist_s_filtered = []
                    feat_mask_filtered = []
                    filenames_e = []
                    filenames_g = []

                    zeroth_example_presence = False

                    for ind,score_index in enumerate(gt_scoresMem_to_consider_after_q_distance):
                        if (score_index == 1).any():
                            if ind == 0:
                                zeroth_example_presence = True
                            axboxlist_memR_filtered.append(gt_boxesMem_to_consider_after_q_distance_theta[ind])
                            scorelist_s_filtered.append(gt_scoresMem_to_consider_after_q_distance[ind])
                            feat_mask_filtered.append(feat_mask[ind])
                            filenames_e.append(feed['filename_e'][ind])
                            filenames_g.append(feed['filename_g'][ind])

                    if len(axboxlist_memR_filtered) > 0:
                        axboxlist_memR_filtered = torch.stack(axboxlist_memR_filtered)
                        scorelist_s_filtered = torch.stack(scorelist_s_filtered)
                        feat_mask_filtered = torch.stack(feat_mask_filtered)

                        results["filtered_boxes"] = axboxlist_memR_filtered
                        results["gt_boxes"] = self.axboxlist_memR
                        results["gt_scores"] = self.scorelist_s
                        results["featR_masks"] = feat_mask_filtered
                        results["scores"] =scorelist_s_filtered
                        results["filenames_e"] = filenames_e
                        results["filenames_g"] = filenames_g
                    else:
                        results["filtered_boxes"] = None
                        results["featR_masks"] = None
                        results["filenames_e"] = None
                        results["filenames_g"] = None
                        results["gt_boxes"] = None
                        results["gt_scores"] = None

                    _, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                        gt_boxesMem_to_consider_after_q_distance_theta[:1],
                        gt_scoresMem_to_consider_after_q_distance[:1],
                        featR[:1],
                        summ_writer)
                    # st()
                    boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                    boxlist_camR_g = utils_vox.convert_boxlist_memR_to_camR(self.axboxlist_memR, hyp.Z2, hyp.Y2, hyp.X2)
                    summ_writer.summ_box_mem_on_mem('detnet/gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,hyp.N],dtype=torch.int32))
                    summ_writer.summ_box_mem_on_mem('detnet/pred_boxesR_mem', unp_visRs, boxlist_memR_e ,scorelist_e,torch.ones_like(scorelist_e,dtype=torch.int32))
                    scorelist_g = self.scorelist_s[0:1].detach().cpu().numpy()
                
                if hyp.max_do: 
                    self.axboxlist_memR = feed["sudo_gt_boxes"]
                    self.scorelist_s = feed["sudo_gt_scores"]
                    featR_mask = feed["feat_mask"]
                    summ_writer.summ_occ('detnet/mask_used', featR_mask)
                    if hyp.maskout:
                        detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                self.axboxlist_memR,
                                self.scorelist_s,
                                featR,
                                summ_writer,mask=featR_mask.squeeze(1))
                    else:
                        detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                            self.axboxlist_memR,
                            self.scorelist_s,
                            featR,
                            summ_writer)
                    total_loss += detect_loss            
                    summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,self.N_det],dtype=torch.int32))
                    self.axboxlist_memR = feed["gt_boxes"]
                    self.scorelist_s = feed["gt_scores"]
                    boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                    boxlist_camR_g = utils_vox.convert_boxlist_memR_to_camR(self.axboxlist_memR, hyp.Z2, hyp.Y2, hyp.X2)
                    summ_writer.summ_box_mem_on_mem('detnet/gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,self.N_det],dtype=torch.int32))
                    summ_writer.summ_box_mem_on_mem('detnet/pred_boxesR_mem', unp_visRs, boxlist_memR_e ,scorelist_e,torch.ones_like(scorelist_e,dtype=torch.int32))
                    scorelist_g = self.scorelist_s[0:1].detach().cpu().numpy()

            elif hyp.do_det:
                if hyp.filter_boxes:
                    gt_boxesMem_to_consider_after_q_distance_theta,gt_scoresMem_to_consider_after_q_distance, feat_mask,gt_boxesMem_to_consider_after_cs_theta,gt_scoresMem_to_consider_after_cs = self.evaluate_filter_boxes(gt_boxesRMem_theta,scores,featR,summ_writer)
                    if hyp.replace_with_cs:
                        gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_cs_theta
                        gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_cs

                if hyp.filter_boxes and hyp.self_improve_once:
                    gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_q_distance_theta.detach()
                    gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_q_distance.detach().cuda()
                    featR_filtered = []
                    axboxlist_memR_filtered = []
                    scorelist_s_filtered = []
                    feat_mask_filtered = []
                    zeroth_example_presence = False
                    for ind,score_index in enumerate(gt_scoresMem_to_consider_after_q_distance):
                        if (score_index == 1).any():
                            if ind == 0:
                                zeroth_example_presence = True
                            featR_filtered.append(featR[ind])
                            axboxlist_memR_filtered.append(gt_boxesMem_to_consider_after_q_distance_theta[ind])
                            scorelist_s_filtered.append(gt_scoresMem_to_consider_after_q_distance[ind])
                            feat_mask_filtered.append(feat_mask[ind])
                    if len(axboxlist_memR_filtered) > 0:
                        axboxlist_memR_filtered = torch.stack(axboxlist_memR_filtered)
                        scorelist_s_filtered = torch.stack(scorelist_s_filtered)
                        featR_filtered = torch.stack(featR_filtered)
                        feat_mask_filtered = torch.stack(feat_mask_filtered)
                        if hyp.maskout:
                            detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                    axboxlist_memR_filtered,
                                    scorelist_s_filtered,
                                    featR_filtered,
                                    summ_writer,mask=feat_mask_filtered.squeeze(1))
                        else:
                            detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                axboxlist_memR_filtered,
                                scorelist_s_filtered,
                                featR_filtered,
                                summ_writer)

                        total_loss += detect_loss                        
                    else:
                        hyp.sudo_backprop = False

                    if not zeroth_example_presence:
                        with torch.no_grad():
                            _, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                gt_boxesMem_to_consider_after_q_distance_theta[:1],
                                gt_scoresMem_to_consider_after_q_distance[:1],
                                featR[:1],
                                summ_writer)
                else:
                    detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                        self.axboxlist_memR,
                        self.scorelist_s,
                        featR,
                        summ_writer)
                    if hyp.add_det_boxes:
                        for index in range(hyp.B):
                            tree_filename_curr = tree_seq_filename[index]
                            tree = trees[index]
                            tree.bbox_det = boxlist_memR_e[index]
                            tree.score_det = scorelist_e[index]
                            tree_filename_curr = join(hyp.root_dataset,tree_filename_curr)
                            pickle.dump(tree,open(tree_filename_curr,"wb"))

                        print("check")
                    total_loss += detect_loss
                # st()
                boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                boxlist_camR_g = utils_vox.convert_boxlist_memR_to_camR(self.axboxlist_memR, hyp.Z2, hyp.Y2, hyp.X2)
                summ_writer.summ_box_mem_on_mem('detnet/gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,hyp.N],dtype=torch.int32))
                summ_writer.summ_box_mem_on_mem('detnet/pred_boxesR_mem', unp_visRs, boxlist_memR_e ,scorelist_e,torch.ones_like(scorelist_e,dtype=torch.int32))
                scorelist_g = self.scorelist_s[0:1].detach().cpu().numpy()
            if hyp.only_cs_vis:
                boxlist_memR_e = gt_boxesMem_to_consider_after_cs_theta
                scorelist_e = gt_scoresMem_to_consider_after_cs
                boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
            if hyp.only_q_vis:
                boxlist_memR_e = gt_boxesMem_to_consider_after_q_distance_theta
                scorelist_e = gt_scoresMem_to_consider_after_q_distance
                boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)

            lrtlist_camR_e = utils_geom.convert_boxlist_to_lrtlist(boxlist_camR_e)
            boxlist_e = boxlist_camR_e[0:1].detach().cpu().numpy()
            boxlist_g = boxlist_camR_g[0:1].detach().cpu().numpy()
            scorelist_e = scorelist_e[0:1].detach().cpu().numpy()

            boxlist_e, boxlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_boxes(
                boxlist_e, boxlist_g, scorelist_e, scorelist_g)

            ious = [0.3, 0.4, 0.5, 0.6, 0.7]
            maps,precisions_avg,scores_pred_val,ious_found = utils_eval.get_mAP(boxlist_e, scorelist_e, boxlist_g, ious)
            # st()
            for ind, overlap in enumerate(ious):
                summ_writer.summ_scalar('ap/%.2f_iou' % overlap, maps[ind])
                summ_writer.summ_scalar('precision/%.2f_iou' % overlap, precisions_avg[ind])
            if hyp.self_improve_iterate:
                if hyp.exp_do:
                    self.avg_ap.append(maps[2])
                    self.avg_precision.append(precisions_avg[2])
                    size = len(self.avg_ap)
                    if ((size+1) % 100) == 0.0:
                        summ_writer.summ_scalar('ap/AVG_0.5_iou' , np.mean(self.avg_ap))
                        summ_writer.summ_scalar('precision/AVG_0.5_iou' , np.mean(self.avg_precision))
                        self.avg_ap = []
                        self.avg_precision = []
            else:
                self.avg_ap.append(maps[2])
                self.avg_precision.append(precisions_avg[2])
                # st()
                if ((global_step+1) % 100) == 0.0:
                    summ_writer.summ_scalar('ap/AVG_0.5_iou' , np.mean(self.avg_ap))
                    summ_writer.summ_scalar('precision/AVG_0.5_iou' , np.mean(self.avg_precision))
     
        if hyp.create_example_dict:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])            
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            # print(object_classes)
            emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2
            minNum = 1
            if len(self.embed_dict.keys()) == self.minclasses:
                minNum = ((hyp.object_quantize_dictsize//self.minclasses)+1)
            if len(self.embed_list) == hyp.object_quantize_dictsize and len(self.embed_dict.keys()) == self.minclasses:
                embed_list = torch.stack(self.embed_list).cpu().numpy().reshape([hyp.object_quantize_dictsize,-1])
                np.save(f'offline_obj_cluster/{hyp.feat_init}_cluster_centers_Example_{hyp.object_quantize_dictsize}.npy',embed_list)
                st()

            for index,class_val in enumerate(object_classes):
                if self.embed_dict[class_val] < minNum and len(self.embed_list) < hyp.object_quantize_dictsize:
                    self.embed_dict[class_val] += 1
                    self.embed_list.append(emb3D_R_object[index])

            print("embed size",len(self.embed_list),"keys",self.embed_dict.keys(),"len keys",len(self.embed_dict.keys()))

        if hyp.object_quantize:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)

            emb3D_R = emb3D_e_R
            try:
                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            except Exception as e:
                st()
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2
            results['emb3D_e'] = emb3D_e_R_object
            results['emb3D_g'] = emb3D_g_R_object
            loss_quant, quantized, perplexity, encodings = self.quantizer(emb3D_R_object)

            e_indexes = torch.argmax(encodings,dim=1).cpu().numpy()
            for index in range(len(object_classes)):
                class_val = object_classes[index]
                e_i = e_indexes[index]
                self.info_dict[str(e_i)].append(class_val)

            if (global_step % 1000) == 0:
                scores_dict = {}
                most_freq_dict = {}
                scores_list = []
                total_mismatch = 0 
                total_obj = 0 
                for key,item in self.info_dict.items():
                    most_freq_word = utils_basic.most_frequent(item)
                    mismatch = 0 
                    for i in item:
                        total_obj += 1
                        if i != most_freq_word:
                            mismatch += 1
                            total_mismatch += 1
                    precision = float(len(item)- mismatch)/len(item)
                    scores_dict[key] = precision
                    most_freq_dict[key] = most_freq_word
                    scores_list.append(precision)
                final_precision = (total_obj - total_mismatch)/float(total_obj)
                # st()
                summ_writer.summ_scalar('precision/unsupervised_precision', final_precision)
                self.info_dict = defaultdict(lambda:[])                

            if hyp.gt_rotate_combinations:
                quantized,best_rotated_inputs,quantized_unrotated,best_rotations_index = quantized
                clusters = torch.argmax(encodings,dim=1)
                if hyp.use_gt_centers:
                    unique_indexes = []
                    for oc_ind,oc in enumerate(object_classes):
                        if oc not in self.list_of_classes:
                            self.list_of_classes.append(oc)
                        oc_index  = self.list_of_classes.index(oc)
                        unique_indexes.append(oc_index)
                # st()
                if hyp.use_gt_centers:
                    info_text = ['C'+str(int(unique_indexes[ind]))+'_R'+str(int(i)*10) for ind,i in enumerate(best_rotations_index)] 
                else:
                    info_text = ['C'+str(int(clusters[ind]))+'_R'+str(int(i)*10) for ind,i in enumerate(best_rotations_index)] 

                summ_writer.summ_box_by_corners_parses('scene_parse/boxescamX0', rgb_camX0, gt_boxesX0_corners,torch.from_numpy(scores), tids, pix_T_cams[:, 0],info_text)
                if hyp.dataset_name == "carla":
                    summ_writer.summ_box_by_corners_parses('scene_parse/boxescamR', rgb_camtop, gt_boxescamXTop_corners,torch.from_numpy(scores), tids, pix_T_cams[:, 0],info_text)

            # st()
            summ_writer.summ_scalar('feat/perplexity',perplexity)
            summ_writer.summ_histogram('feat/encodings',e_indexes)            
 
            emb3D_e_R_object = quantized
            camX1_T_R = camXs_T_camRs[:,1]
            camX0_T_R = camXs_T_camRs[:,0]
            emb3D_R_non_updated = emb3D_R
            # st()
            emb3D_R = nlu.update_scene_with_objects(emb3D_R, emb3D_e_R_object ,gt_boxesRMem_end, scores)
            emb3D_e_X1 = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R)
            emb3D_e_X0 = utils_vox.apply_4x4_to_vox(camX0_T_R, emb3D_R)
            if hyp.gt_rotate_combinations:
                emb3D_R_best_rotated = nlu.update_scene_with_objects(emb3D_R, best_rotated_inputs ,gt_boxesRMem_end, scores)
                emb3D_e_X1_best_rotated = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R_best_rotated)
                emb3D_R_quantized_unrotated = nlu.update_scene_with_objects(emb3D_R, quantized_unrotated ,gt_boxesRMem_end, scores)
                emb3D_e_X1_quantized_unrotated = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R_quantized_unrotated)
 

            object_rgb = nlu.create_object_rgbs(rgb_camXs[:,0],gt_cornersX0_pix,scores)
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            rgb = object_rgb
            rgb = utils_improc.back2color(rgb)            
            results['rgb'] = rgb
            results['classes'] = object_classes            
            results['valid3D'] = validR_combo_object.detach()
            # st()
            total_loss += hyp.quantize_loss_coef*loss_quant

        start_time = time.time()
        if hyp.do_occ and hyp.occ_do_cheap:
            occX0_sup, freeX0_sup,_, freeXs = utils_vox.prep_occs_supervision(
                camX0_T_camXs,
                xyz_camXs,
                Z2,Y2,X2,
                agg=True)

            summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))

            if hyp.object_quantize:
                occ_loss, occX0s_pred_ = self.occnet(emb3D_e_X0,
                                                     occX0_sup,
                                                     freeX0_sup,
                                                     torch.max(validX0s[:,1:], dim=1)[0],
                                                     summ_writer)

            else:
                occ_loss, occX0s_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                     occX0_sup,
                                                     freeX0_sup,
                                                     torch.max(validX0s[:,1:], dim=1)[0],
                                                     summ_writer)
            occX0s_pred = __u(occX0s_pred_)
            total_loss += occ_loss

            if hyp.profile_time:                
                print("occ time",time.time()-start_time)
        
        start_time = time.time()
        if hyp.do_view:
            assert(hyp.do_feat)
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            if hyp.object_quantize:
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1, # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)      
                if hyp.gt_rotate_combinations:
                    feat_projX00_best_rotated = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1_best_rotated, # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)                      
                    feat_projX00_quantized_unrotated = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1_quantized_unrotated, # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)                                          
            else:        
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], featXs[:,1], # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)
            rgb_X00 = utils_basic.downsample(rgb_camXs[:,0], 2)
            valid_X00 = utils_basic.downsample(valid_camXs[:,0], 2)
            # decode the perspective volume into an image
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                valid_X00,
                summ_writer,"rgb")

            if hyp.obj_multiview:
                projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))
                for i in range(hyp.S):
                    rgb_Xi = utils_basic.downsample(rgb_camXs[:,i], 2)
                    valid_Xi = utils_basic.downsample(valid_camXs[:,i], 2)                        
                    gt_boxesXi_corners = __ub(utils_geom.apply_4x4(camXs_T_camRs[:,i], __pb(gt_boxesR_corners)))
                    gt_cornersXi_pix = __ub(utils_geom.apply_pix_T_cam(projpix_T_cams[:,i], __pb(gt_boxesXi_corners)))
                    # st()
                    feat_projXi = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,i], utils_geom.eye_4x4(hyp.B), featXs[:,i], # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)
                    _, rgb_e_i, emb2D_e = self.viewnet(
                        feat_projXi,
                        rgb_Xi,
                        valid_Xi,
                        summ_writer,f"rgb_{i}")
                    object_rgb_Xi_pred = nlu.create_object_rgbs(rgb_e_i,gt_cornersXi_pix,scores)
                    summ_writer.summ_rgb(f"scene_parse/object_q_rpred_{i}",object_rgb_Xi_pred)
                    object_rgb_Xi_gt = nlu.create_object_rgbs(rgb_Xi,gt_cornersXi_pix,scores)
                    summ_writer.summ_rgb(f"scene_parse/object_q_rgt_{i}",object_rgb_Xi_gt)

                    utils_basic.save_rgb(object_rgb_Xi_pred,object_classes)
                    utils_basic.save_rgb(object_rgb_Xi_gt,object_classes,gt=True)
            if hyp.gt_rotate_combinations and hyp.object_quantize:
                _, rgb_best, _ = self.viewnet(
                    feat_projX00_best_rotated,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"rgb_best_rotated")                                
                _, rgb_quant, _ = self.viewnet(
                    feat_projX00_quantized_unrotated,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"rgb_quant_unrotated")                
                inp_best_quant = torch.cat([rgb_X00,rgb_best,rgb_quant],dim=2)
 
                summ_writer.summ_rgb("selected_rotations/inp_best_quant",inp_best_quant)
            total_loss += view_loss
            
            if hyp.profile_time:
                print("view time",time.time()-start_time)
        
        if hyp.do_emb3D:
            emb_loss_3D = self.embnet3D(
                emb3D_e_R_object,
                emb3D_g_R_object,
                vis3D_g_R,
                summ_writer)
            rgb = object_rgb
            total_loss += emb_loss_3D

        if hyp.break_constraint and not hyp.object_quantize:
        
            rgb_vis3D = utils_improc.back2color(utils_basic.reduce_masked_mean(unpRs,occRs.repeat(1, 1, 3, 1, 1, 1),dim=1))
            rgb = rgb_camXs[:, 0]
            rgb = torch.nn.functional.interpolate(rgb, size=[hyp.PH*2, hyp.PW*2], mode='bilinear')            
            rgb = utils_improc.back2color(rgb)
            try:
                results['emb3D_e'] = emb3D_e_R_object
                results['emb3D_g'] = emb3D_g_R_object
                results['rgb'] = rgb

                if validR_combo_object is not None:
                    validR_combo_object = validR_combo_object.detach()
                results['valid3D'] = validR_combo_object
            except Exception:
                pass
        if not hyp.moc:
            summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results

