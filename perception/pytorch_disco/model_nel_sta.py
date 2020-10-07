from model_base_nel import Model
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio

from nets.featnet import FeatNet
# from nets.featnet import FeatNet_NO_BN
from nets.occnet import OccNet
from nets.viewnet import ViewNet
from nets.embnet3D import EmbNet3D
import torch.nn.functional as F
from os.path import join
# from utils_basic import *
import pickle
import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic

import ipdb
st = ipdb.set_trace
# st()
from lib_classes import Nel_Utils as nlu
import torchvision.models as models
from pretrained_model import MyModel
np.set_printoptions(precision=2)
np.random.seed(0)

class NEL_STA(Model):
    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        self.model = NelStaNet()
        if hyp.emb_moc.do or hyp.max.hard_moc:
            self.model_key = NelStaNet()

class NelStaNet(nn.Module):
    def __init__(self):
        super(NelStaNet, self).__init__()
        
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ or hyp.do_occ_gt:
            self.occnet = OccNet()
        if hyp.do_view or hyp.do_view_gt:
            self.viewnet = ViewNet()
        if hyp.do_emb3D or hyp.do_emb3D_gt:
            self.embnet3D = EmbNet3D()
        elif hyp.emb_moc.do:
            print("moc")

    def forward(self, feed):
        results = dict()
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

        if hyp.dataset_name == "carla" and not hyp.max.do:
            rgb_camTop = feed['rgb_camtop']
        rgb_camXs = feed["rgb_camXs_raw"]
        pix_T_cams = feed["pix_T_cams_raw"]
        filename_g = feed["filename_g"]
        filename_e = feed["filename_e"]

        camRs_T_origin = feed["camR_T_origin_raw"]
        camRs_T_origin_g = camRs_T_origin[:,0]
        camRs_T_origin_e = camRs_T_origin[:,1]


        origin_T_camRs = __u(utils_geom.safe_inverse(__p(feed["camR_T_origin_raw"])))
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
            if hyp.dataset_name == "carla":
                xyz_camXs = __u(dense_xyz_camXs_)


        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))



        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))
        occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))

        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))

        unpX0s_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camX0_T_camXs)))))
        unpRs_half = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
        unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z, Y, X, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))


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
        if hyp.dataset_name == "carla" and not hyp.max.do:
            summ_writer.summ_rgb('2D_inputs/rgb_camTop', rgb_camTop[:,0])
        
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))

        summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))
        # get boxes
        if hyp.max.do  and  hyp.max.predicted_matching:
            filename_e = feed['filename_e']
            filename_g = feed['filename_g']
            tree_seq_filename =  filename_g + filename_e
            hyp.B = hyp.B * 2
        else:
            tree_seq_filename = feed['tree_seq_filename']
        
        if hyp.dataset_name != "replica":
            tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename]
            trees = [pickle.load(open(i,"rb")) for i in tree_filenames]

        if hyp.dataset_name == "carla":
            gt_boxes_origin,scores,classes = nlu.trees_rearrange_corners(trees)
            gt_boxes_origin = torch.from_numpy(gt_boxes_origin).cuda().to(torch.float)
            gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])                            
            gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
            gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)

            if hyp.max.do and  hyp.max.predicted_matching:
                gt_boxes_origin_corners_g = gt_boxes_origin_corners[:2]
                gt_boxes_origin_corners_e = gt_boxes_origin_corners[2:]
                gt_boxesR_corners_g = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners_g)))
                gt_boxesR_corners_e = __ub(utils_geom.apply_4x4(camRs_T_origin[:,1], __pb(gt_boxes_origin_corners_e)))
                gt_boxesR_corners = torch.cat([gt_boxesR_corners_g,gt_boxesR_corners_e],dim=0)
            else:
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))

            gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
            assert (scores[:,:1] ==1.).all()
        elif hyp.dataset_name == "replica":
            if hyp.max.do  and  hyp.max.predicted_matching:
                obj_cat_name_e = feed['classes_e']
                obj_cat_name_g = feed['classes_g']
                scores_e = feed['scores_e']
                scores_g = feed['scores_g']
                bbox_origin_gt_e = feed['bbox_origin_e']
                bbox_origin_gt_g = feed['bbox_origin_g']
                gt_boxes_origin = torch.cat([bbox_origin_gt_g,bbox_origin_gt_e],dim=0)
                classes = np.concatenate([obj_cat_name_g,obj_cat_name_e],axis=0)
                gt_scores_origin = torch.cat([scores_g,scores_e],dim=0)
                camRs_T_origin_temp = torch.cat([camRs_T_origin_g,camRs_T_origin_e],dim=0)

            else:
                gt_boxes_origin = feed['gt_box']
                gt_scores_origin = feed['gt_scores']
                classes = feed['classes']
                camRs_T_origin_temp = camRs_T_origin[:,0]

            gt_boxes_origin_f = gt_boxes_origin[:,:1].cpu().detach().numpy()
            scores_f = gt_scores_origin[:,:1].cpu().detach().numpy()
            classes_f = classes[:,:1]
            N_new = 1
            gt_boxes_origin = np.pad(gt_boxes_origin_f,[[0,0],[0,hyp.N-N_new],[0,0]])
            gt_boxes_origin = torch.from_numpy(gt_boxes_origin).cuda()
            scores = np.pad(scores_f,[[0,0],[0,hyp.N-N_new]])
            classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])
            gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
            gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
            gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
            gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin_temp, __pb(gt_boxes_origin_corners)))
            gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
        else:
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
                gt_boxesR_f = sorted_boxes[:,:1]
                gt_scoresR_f = sorted_scores[:,:1]
                classes_f = classes[:,:1]                    
                N_new = gt_boxesR_f.shape[1]

                gt_boxesR = np.pad(gt_boxesR_f,[[0,0],[0,hyp.N-N_new],[0,0],[0,0]]).reshape([hyp.B,hyp.N,6])
                scores = np.pad(gt_scoresR_f,[[0,0],[0,hyp.N-N_new]])
                classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])
            elif hyp.use_2d_boxes:
                gt_boxesR,scores,classes = nlu.trees_rearrange_2d(trees)
            else:
                gt_boxesR,scores,classes = nlu.trees_rearrange(trees)

            if hyp.use_first_bbox:
                gt_boxesR_f = gt_boxesR[:,:1]
                scores_f = scores[:,:1]
                classes_f = classes[:,:1]
                N_new = 1
                gt_boxesR = np.pad(gt_boxesR_f,[[0,0],[0,hyp.N-N_new],[0,0]])
                scores = np.pad(scores_f,[[0,0],[0,hyp.N-N_new]])
                classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])
            
            gt_boxesR = torch.from_numpy(gt_boxesR).cuda()

            gt_boxesR_end = torch.reshape(gt_boxesR,[hyp.B,hyp.N,2,3])


            gt_boxesR_theta = nlu.get_alignedboxes2thetaformat(gt_boxesR_end)
            gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta)


        gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))
        gt_boxesRMem_theta = utils_geom.transform_corners_to_boxes(gt_boxesRMem_corners)
        gt_boxesRMem_end = nlu.get_ends_of_corner(gt_boxesRMem_corners)
        if hyp.random_noise:
            shape_val = list(gt_boxesRMem_end.shape)
            noise = torch.from_numpy(np.random.randint(-2, high=2, size=shape_val)).cuda()
            gt_boxesRMem_end = gt_boxesRMem_end + noise
            gt_boxesRMem_end = torch.clamp(gt_boxesRMem_end,0,hyp.X2)


        gt_boxesRUnp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z,Y,X))
        gt_boxesRUnp_end = nlu.get_ends_of_corner(gt_boxesRUnp_corners)


        # st()
        unps_visRs_half = utils_improc.get_unps_vis(unpRs_half, occRs_half)
        unp_visRs_half = torch.mean(unps_visRs_half, dim=1)

        summ_writer.summ_box_mem_on_unp('eval_boxes/gt_boxesR_mem', unp_visRs_half , gt_boxesRMem_end, scores ,tids)
        
        if hyp.max.do and  hyp.max.predicted_matching:
            hyp.B = hyp.B//2

            gt_boxesR_corners_g = gt_boxesR_corners[:2]
            gt_boxesX0_corners = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners_g)))
            gt_boxesX0Mem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX0_corners),Z2,Y2,X2))
            gt_boxesX0Mem_theta = utils_geom.transform_corners_to_boxes(gt_boxesX0Mem_corners)       
            gt_boxesX0Mem_end = nlu.get_ends_of_corner(gt_boxesX0Mem_corners)
            gt_boxesX0_end = nlu.get_ends_of_corner(gt_boxesX0_corners)
            gt_cornersX0_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))

            gt_boxesR_corners_e = gt_boxesR_corners[2:]
            gt_boxesX1_corners = __ub(utils_geom.apply_4x4(camX1_T_camRs, __pb(gt_boxesR_corners_e)))
            gt_boxesX1Mem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX1_corners),Z2,Y2,X2))
            gt_boxesX1Mem_theta = utils_geom.transform_corners_to_boxes(gt_boxesX1Mem_corners)       
            gt_boxesX1Mem_end = nlu.get_ends_of_corner(gt_boxesX1Mem_corners)
            gt_boxesX1_end = nlu.get_ends_of_corner(gt_boxesX1_corners)
            gt_cornersX1_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,1], __pb(gt_boxesX1_corners)))

            rgb_camX1 = rgb_camXs[:,1]
            rgb_camX0 = rgb_camXs[:,0]

            summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamX0', rgb_camX0, gt_boxesX0_corners, torch.ones([B,N]), tids, pix_T_cams[:, 0])
            summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamX1', rgb_camX1, gt_boxesX1_corners, torch.ones([B,N]), tids, pix_T_cams[:, 1])
            
            unpX0s_half = torch.mean(unpX0s_half, dim=1)
            unpX0s_half = nlu.zero_out(unpX0s_half,gt_boxesX0Mem_end,scores)

            occX0s_half = torch.mean(occX0s_half, dim=1)
            occX0s_half = nlu.zero_out(occX0s_half,gt_boxesX0Mem_end,scores)

            summ_writer.summ_unp('3D_inputs/unpX0s', unpX0s_half, occX0s_half)
        else:
            gt_boxesX0_corners = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners)))
            gt_boxesX0Mem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX0_corners),Z2,Y2,X2))

            gt_boxesX0Mem_theta = utils_geom.transform_corners_to_boxes(gt_boxesX0Mem_corners)
            
            gt_boxesX0Mem_end = nlu.get_ends_of_corner(gt_boxesX0Mem_corners)
            gt_boxesX0_end = nlu.get_ends_of_corner(gt_boxesX0_corners)
            gt_cornersX0_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))
            rgb_camX0 = rgb_camXs[:,0]
            summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamX0', rgb_camX0, gt_boxesX0_corners, torch.ones([B,N]), tids, pix_T_cams[:, 0])
            
            unpX0s_half = torch.mean(unpX0s_half, dim=1)
            unpX0s_half = nlu.zero_out(unpX0s_half,gt_boxesX0Mem_end,scores)

            occX0s_half = torch.mean(occX0s_half, dim=1)
            occX0s_half = nlu.zero_out(occX0s_half,gt_boxesX0Mem_end,scores)
            summ_writer.summ_unp('3D_inputs/unpX0s', unpX0s_half, occX0s_half)

        if hyp.do_feat:
            
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)

            freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occXs_half))
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half+freeXs, 0.0, 1.0)

            mask_ = None            
            if (type(mask_)!=type(None)):
                assert(list(mask_.shape)[2:5]==list(featXs_input_.shape)[2:5])

            featXs_,  feat_loss = self.featnet(featXs_input_, summ_writer, mask=__p(occXs))#mask_)
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
            vis3D_e_R = torch.max(visRs[:,1:], dim=1).values
            emb3D_g = featX0s[:,0] # obs
            vis3D_g_R = visRs[:,0] # obs
            validR_combo = torch.min(validRs,dim=1).values
            summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), valids=torch.unbind(validXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), valids=torch.unbind(torch.ones_like(validRs), dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validRs', torch.unbind(validRs, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e_R', vis3D_e_R, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g_R', vis3D_g_R, pca=False)

        emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
        emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)

        if hyp.max.do:
            if hyp.max.predicted_matching:
                gt_boxesRMem_end_g,gt_boxesRMem_end_e = torch.split(gt_boxesRMem_end,2,0)
                scores_g,scores_e = np.split(scores,2,0)
                emb3D_g_R, vis3D_g_R = nlu.create_object_tensors([emb3D_g_R], [vis3D_g_R], gt_boxesRMem_end_g, scores_g,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
                emb3D_e_R, _ = nlu.create_object_tensors([emb3D_e_R], None,gt_boxesRMem_end_e, scores_e,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
                if hyp.emb_moc.do or hyp.max.hardmining:
                    results['emb3D_e_R'] = emb3D_e_R
                    results['emb3D_g_R'] = emb3D_g_R
                    tmp_batch = hyp.B
                    classes_e,[filename_e,filename_g] = nlu.create_object_classes(classes[:tmp_batch],[filename_e,filename_g],scores[:tmp_batch])
                    classes_g,[filename_e,filename_g] = nlu.create_object_classes(classes[tmp_batch:],[filename_e,filename_g],scores[tmp_batch:])
                    if hyp.gt:
                        assert np.all(classes_e == classes_g)
                    results['classes'] = classes_e
        else:

            emb3D_e_R, emb3D_g_R, vis3D_g_R, validR_combo = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [vis3D_g_R,validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            if hyp.dataset_name == "carla" and not hyp.max.do:

                rgb_consider = rgb_camXs[:,0]
            else:
                rgb_consider = rgb_camXs[:,0]
            object_rgb = nlu.create_object_rgbs(rgb_consider,gt_cornersX0_pix,scores)
            object_classes,filenames = nlu.create_object_classes(classes, [filename_g,filename_e], scores)

            unpRs_eg = torch.mean(unpRs,dim=1,keepdim=False)
            occRs_eg = torch.mean(occRs,dim=1,keepdim=False)

            unpRs_eg, _ = nlu.create_object_tensors([unpRs_eg], None, gt_boxesRUnp_end, scores,[BOX_SIZE*2,BOX_SIZE*2,BOX_SIZE*2])
            occRs_eg, _ = nlu.create_object_tensors([occRs_eg], None, gt_boxesRUnp_end, scores,[BOX_SIZE*2,BOX_SIZE*2,BOX_SIZE*2])

            if len(unpRs_eg) ==0:
                unp_visRs_eg = unpRs_eg
            else:
                unps_visRs_eg = utils_improc.get_unps_vis(unpRs_eg.unsqueeze(1), occRs_eg.unsqueeze(1))
                unp_visRs_eg = torch.mean(unps_visRs_eg, dim=1)
                
                rgb = object_rgb
                summ_writer.summ_unp('3D_inputs/unpRs_eg_o', unpRs_eg, occRs_eg)
                rgb = utils_improc.back2color(rgb)




            filename_g,filename_e = filenames
            rgb_vis3D = None
            results['classes'] = object_classes
            results['emb3D_e_R'] = emb3D_e_R
            results['emb3D_g_R'] = emb3D_g_R
            results['rgb'] = rgb
            results['unp_visRs'] = unp_visRs_eg
            results['valid3D'] = validR_combo
            results['vis3D_g_R'] = vis3D_g_R
            results['filenames_g'] = filename_g
            results['filenames_e'] = filename_e
            
        if hyp.max.do:
            if hyp.max.predicted_matching:
                if hyp.emb_moc.do or  hyp.max.hardmining:
                    emb_loss_3D = 0.0
                else:
                    emb_loss_3D = self.embnet3D(
                        emb3D_e_R,
                        emb3D_g_R,
                        vis3D_g_R,
                        summ_writer)
                total_loss += emb_loss_3D
            else:
                if (hyp.do_occ or hyp.do_occ_gt) and hyp.occ_do_cheap:
                    occX0_sup, freeX0_sup,_, freeXs = utils_vox.prep_occs_supervision(
                        camX0_T_camXs,
                        xyz_camXs,
                        Z2,Y2,X2,
                        agg=True)

                    summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
                    summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
                    summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
                    summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))

                    occ_loss, occX0s_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                         occX0_sup,
                                                         freeX0_sup,
                                                         torch.max(validX0s[:,1:], dim=1)[0],
                                                         summ_writer)
                    occX0s_pred = __u(occX0s_pred_)
                    total_loss += occ_loss
                if hyp.do_view or hyp.do_view_gt:
                    assert(hyp.do_feat)
                    # we warped the features into the canonical view
                    # now we resample to the target view and decode
                    PH, PW = hyp.PH, hyp.PW
                    sy = float(PH)/float(hyp.H)
                    sx = float(PW)/float(hyp.W)
                    assert(sx==0.5) # else we need a fancier downsampler
                    assert(sy==0.5)
                    projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))
                    assert(S==2) # else we should warp each feat in 1:                    
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
                        summ_writer)                    
                    total_loss += view_loss
                
                if hyp.do_emb3D_gt:
                    emb_loss_3D = self.embnet3D(
                        emb3D_e_R,
                        emb3D_g_R,
                        vis3D_g_R,
                        summ_writer)
                    total_loss += emb_loss_3D                    
        if not (hyp.emb_moc.do or  hyp.max.hardmining):
            summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results