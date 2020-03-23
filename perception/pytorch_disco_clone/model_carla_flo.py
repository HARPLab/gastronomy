import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_FLO(Model):
    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        self.model = CarlaFloNet().to(self.device)
        # print(self.model)
        # self.model = CarlaFloNet()
        
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)

class CarlaFloNet(nn.Module):
    def __init__(self):
        super(CarlaFloNet, self).__init__()
        self.featnet = FeatNet()
        self.occnet = OccNet()
        self.flownet = FlowNet()

        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.autograd.set_detect_anomaly(True)

    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8)
        
        writer = feed['writer']
        global_step = feed['global_step']

        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        PH, PW = hyp.PH, hyp.PW
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        D = 9

        rgb_camRs = feed["rgb_camRs"]
        rgb_camXs = feed["rgb_camXs"]
        pix_T_cams = feed["pix_T_cams"]
        cam_T_velos = feed["cam_T_velos"]
        
        if (not hyp.flow_do_synth_rt) or feed['set_name']=='val':
            boxlist_camRs = feed["boxes3D"]
            tidlist_s = feed["tids"] # coordinate-less and plural
            scorelist_s = feed["scores"] # coordinate-less and plural
            # # postproc the boxes:
            # scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(__p(boxlist_camRs), __p(tidlist_s), Z, Y, X))

            boxlist_camRs_, tidlist_s_, scorelist_s_ = __p(boxlist_camRs), __p(tidlist_s), __p(scorelist_s)
            boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
                boxlist_camRs_, tidlist_s_, scorelist_s_)
            boxlist_camRs = __u(boxlist_camRs_)
            tidlist_s = __u(tidlist_s_)
            scorelist_s = __u(scorelist_s_)

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs = feed["origin_T_camXs"]
        origin_T_camXs_ = __p(origin_T_camXs)

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camX0_T_camXs_ = __p(camX0_T_camXs)
        camRs_T_camXs_ = torch.matmul(origin_T_camRs_.inverse(), origin_T_camXs_)
        camXs_T_camRs_ = camRs_T_camXs_.inverse()
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)

        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        # occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occX0s = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z, Y, X))
        # occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unpX0s = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs)
        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))
        unpX0s_half = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs_half)

        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        # summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occX0s', torch.unbind(occX0s, dim=1))
        # summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpX0s', torch.unbind(unpX0s, dim=1), torch.unbind(occX0s, dim=1))


        if (not hyp.flow_do_synth_rt) or feed['set_name']=='val':
            lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(boxlist_camRs_)).reshape(B, S, N, 19)
            lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
            # stabilize boxes for ego/cam motion
            lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(camX0_T_camXs), __p(lrtlist_camXs)))
            # these are is B x S x N x 19

            summ_writer.summ_lrtlist('lrtlist_camR0', rgb_camRs[:,0], lrtlist_camRs[:,0],
                                     scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
            summ_writer.summ_lrtlist('lrtlist_camR1', rgb_camRs[:,1], lrtlist_camRs[:,1],
                                     scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
            summ_writer.summ_lrtlist('lrtlist_camX0', rgb_camXs[:,0], lrtlist_camXs[:,0],
                                     scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
            summ_writer.summ_lrtlist('lrtlist_camX1', rgb_camXs[:,1], lrtlist_camXs[:,1],
                                     scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
            (obj_lrtlist_camXs,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camXs,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='X',
                                               do_vis=True,
                                               summ_writer=summ_writer)
            (obj_lrtlist_camRs,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camRs,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='R',
                                               do_vis=True,
                                               summ_writer=summ_writer)
            (obj_lrtlist_camX0s,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camX0s,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='X0',
                                               do_vis=False)

            masklist_memR = utils_vox.assemble_padded_obj_masklist(
                lrtlist_camRs[:,0], scorelist_s[:,0], Z, Y, X, coeff=1.0)
            masklist_memX = utils_vox.assemble_padded_obj_masklist(
                lrtlist_camXs[:,0], scorelist_s[:,0], Z, Y, X, coeff=1.0)
            # obj_mask_memR is B x N x 1 x Z x Y x X
            summ_writer.summ_occ('obj/masklist_memR', torch.sum(masklist_memR, dim=1))
            summ_writer.summ_occ('obj/masklist_memX', torch.sum(masklist_memX, dim=1))

            # # to do tracking or whatever, i need to be able to extract a 3d object crop
            # cropX0_obj0 = utils_vox.crop_zoom_from_mem(occXs[:,0], lrtlist_camXs[:,0,0], Z2, Y2, X2)
            # cropX0_obj1 = utils_vox.crop_zoom_from_mem(occXs[:,0], lrtlist_camXs[:,0,1], Z2, Y2, X2)
            # summ_writer.summ_feat('crops/cropX0_obj0', cropX0_obj0, pca=False)
            # summ_writer.summ_feat('crops/cropX0_obj1', cropX0_obj1, pca=False)
            # summ_writer.summ_feat('crops/cropR0_obj0', cropR0_obj0, pca=False)
            # summ_writer.summ_feat('crops/cropR0_obj1', cropR0_obj1, pca=False)

        # if hyp.do_flow and ((not hyp.flow_do_synth_rt) or feed['set_name']=='val'):
        #     # ego-stabilized flow from X00 to X01
        flowX0 = utils_misc.get_gt_flow(obj_lrtlist_camX0s,
                                        obj_scorelist_s,
                                        utils_geom.eye_4x4s(B, S),
                                        Z2, Y2, X2,
                                        K=K, 
                                        mod='X0',
                                        vis=False,
                                        summ_writer=summ_writer)
        is_synth = False
        
        if hyp.do_feat:
            if hyp.flow_do_synth_rt:
                if feed['set_name']=='train':
                    is_synth = True

                    # # on train iters, replace the inputs with synth 
                    # packed_synth = utils_misc.get_synth_flow(occX0s,
                    #                                          unpX0s,
                    #                                          summ_writer=summ_writer,
                    #                                          sometimes_zero=False,
                    #                                          do_vis=True)
                    # occX0s, unpX0s, flowX0, _ = packed_synth
                    # # we want the flow at half res
                    # flowX0 = utils_basic.downsample3Dflow(flowX0, 2)

                    # on train iters, replace the inputs with synth 
                    packed_synth = utils_misc.get_synth_flow_v2(xyz_camXs[:,0],
                                                                occX0s[:,0],
                                                                unpX0s[:,0],
                                                                summ_writer=summ_writer,
                                                                sometimes_zero=False,
                                                                do_vis=True)
                    occX0s, unpX0s, flowX0, _ = packed_synth
                    # we want the flow at half res
                    flowX0 = utils_basic.downsample3Dflow(flowX0, 2)

            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            featX0s_input = torch.cat([occX0s, occX0s*unpX0s], dim=2)
            featX0s_input_ = __p(featX0s_input)
            featX0s_, validX0s_, feat_loss = self.featnet(featX0s_input_, summ_writer, mask=__p(occX0s))
            total_loss += feat_loss
            featX0s = __u(featX0s_)
            validX0s = __u(validX0s_)

            summ_writer.summ_feats('3D_feats/featX0s_input', torch.unbind(featX0s_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), pca=True)

        if hyp.do_occ and hyp.occ_do_cheap:
            occX0_sup, freeX0_sup, freeX0s = utils_vox.prep_occs_supervision(
                xyz_camXs[:,0:1],
                occX0s_half[:,0:1],
                occX0s_half[:,0:1],
                camX0_T_camXs[:,0:1],
                agg=True)
        
            summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            # summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            # summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))
                
            occ_loss, occX0s_pred_ = self.occnet(featX0s[:,0],
                                                 occX0_sup,
                                                 freeX0_sup,
                                                 validX0s[:,0],
                                                 summ_writer)
            occX0s_pred = __u(occX0s_pred_)
            total_loss += occ_loss
            
        if hyp.do_flow:
            flow_loss, flowX0_pred = self.flownet(
                featX0s[:,0],
                featX0s[:,1],
                # occX0s_half[:,0]*unpX0s_half[:,0],
                # occX0s_half[:,1]*unpX0s_half[:,1],
                flowX0,
                occX0s_half[:,0],#.repeat(1, 3, 1, 1, 1), # apply loss here
                # validX0s[:,0]*occX0s_half[:,0], # apply loss here
                is_synth,
                summ_writer)
            total_loss += flow_loss

            g = flowX0.clone().detach()
            e = flowX0_pred.clone().detach()
            g = g.reshape(-1)
            e = e.reshape(-1)
            g[0] = 0.02
            summ_writer.summ_histogram('flowX0_g_nonzero_hist', g[torch.abs(g)>0.01])
            summ_writer.summ_histogram('flowX0_e_nonzero_hist', e[torch.abs(g)>0.01])
            
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results

    
