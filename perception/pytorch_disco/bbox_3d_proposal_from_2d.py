import numpy as np 
import os 
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "replica_multiview_builder"
os.environ["run_name"] = "check"
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
from object_detector_2d import ClevrDetector
import utils_vox
import utils_geom
import lib_classes.Nel_Utils as nlu
import open3d as o3d
import time
import utils_improc
import scipy
from scipy.misc import imread, imshow
from scipy import misc
import socket
hostname = socket.gethostname()
import getpass
import utils_vox
username = getpass.getuser()
import utils_pointcloud
import ipdb 
st = ipdb.set_trace
class bbox_3d_proposal_from_2d():
    '''
    max_number_of_objects - max number of objects expected in a scene.
    max_mask_size_thresh - mask proposals having are greater than this will be considered invalid
    iou_thresh - IOU threshold of cone with aggregated scene occupancy
    vote_thresh - How many votes should each occupancy get (1 vote by 1 cone) to be considered part of 3d bbox.
    confidence_threshold - Detectron confidence threshold
    '''
    def __init__(self, H, W, max_number_of_objects, max_mask_size_thresh, iou_thresh, vote_thresh, confidence_threshold, bounds):
        self.visualize = False
        self.H = H
        self.W = W
        self.max_number_of_objects = max_number_of_objects
        self.max_mask_size_thresh = max_mask_size_thresh
        self.iou_thresh = iou_thresh
        self.vote_thresh = vote_thresh
        self.confidence_threshold = confidence_threshold
        self.bounds = bounds
        # Occupancies size params
        self.segmented_obj_X, self.segmented_obj_Y, self.segmented_obj_Z = 100,100,100#56, 56, 56

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        #Detectron stuff
        opts = ["MODEL.WEIGHTS","detectron2://COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl"]
        if "domestation" in hostname:
            config_file = "/home/sirdome/shubhankar/detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml"
        elif "compute" in hostname:
            if "mprabhud" in username:
                config_file = "/home/mprabhud/shamit/detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml"
            elif "shamitl" in username:
                config_file = "/home/shamitl/projects/detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml"
        self.cleverDetector = ClevrDetector(config_file, opts, self.confidence_threshold)
    
    def visualize_pcd_open3d(self, xyz):
        pcd = nlu.make_pcd(xyz)
        o3d.visualization.draw_geometries([self.mesh_frame, pcd])
        # o3d.visualization.draw_geometries([pcd])

    def get_occupancies_for_objects(self, binary_mask, depth, pix_T_camX, camR_T_camX):
        occupancies = []
        binary_mask = binary_mask.transpose(2, 0, 1) # Bring the number of objects dimension forward. Treat them as batch
        # depth = np.expand_dims(depth, axis=0)
        depth_masks = depth*binary_mask
        depth_masks = torch.tensor(depth_masks).unsqueeze(1) # Add the channel dimension.
        pix_T_camX = torch.tensor(pix_T_camX).unsqueeze(0).repeat(depth_masks.shape[0], 1, 1)
        maskxyz = utils_geom.depth2pointcloud_cpu(depth_masks, pix_T_camX)
        
        truncated_xyz_list = []
        # st()
        for xyz in maskxyz:
            xyz = utils_pointcloud.truncate_pcd_outside_bounds(self.bounds, xyz.numpy())
            padding_required = self.H*self.W - xyz.shape[0]
            zeros = np.zeros((padding_required, 3))
            xyz = np.concatenate((xyz, zeros))
            # st()
            truncated_xyz_list.append(xyz)
        maskxyz = torch.tensor(np.stack(truncated_xyz_list)).float()
        camR_T_camX = torch.tensor(camR_T_camX).unsqueeze(0).repeat(depth_masks.shape[0], 1, 1)
        maskxyz_camR = utils_geom.apply_4x4(camR_T_camX.float(), maskxyz.float())
        
        # Get everything in camR ref frame
        if self.visualize:
            xyzs = maskxyz_camR.numpy()
            for xyz in xyzs:
                # fig = pyplot.figure()
                # ax = fig.add_subplot(111, projection='3d')

                # # Generate the values
                # x_vals = xyz[:, 0:1]
                # y_vals = xyz[:, 1:2]
                # z_vals = xyz[:, 2:3]

                # # Plot the values
                # ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
                # ax.set_xlabel('X-axis')
                # ax.set_ylabel('Y-axis')
                # ax.set_zlabel('Z-axis')

                # pyplot.show()
                # st()
                print("Visualizing maskxyz_camRs")
                self.visualize_pcd_open3d(xyz)

        maskxyz_memR = utils_vox.Ref2Mem(maskxyz_camR, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)
        # st()
        occupancies = utils_vox.get_occupancy(maskxyz_memR, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X) # Set these bounds properly
        occupancies_sum = torch.sum(occupancies.reshape(occupancies.shape[0],-1), axis=1)
        print("Occupancies sum in memR ref frame: ", occupancies_sum)


        maskxyz_memX = utils_vox.Ref2Mem(maskxyz, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)
        occupancies_memX = utils_vox.get_occupancy(maskxyz_memX, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X) # Set these bounds properly
        occupancies_sum_memX = torch.sum(occupancies_memX.reshape(occupancies_memX.shape[0],-1), axis=1)
        print("Occupancies sum in memX ref frame", occupancies_sum_memX)
        
        # st()
        occupancies_memR_from_memX = utils_vox.apply_4x4_to_vox(camR_T_camX.float(), occupancies_memX.float(), already_mem=False, binary_feat=True)
        occupancies_sum_memR_from_memX = torch.sum(occupancies_memR_from_memX.reshape(occupancies_memR_from_memX.shape[0],-1), axis=1)
        print("Occupancies sum in memR_from_memX ref frame", occupancies_sum_memR_from_memX)

        occupancies_memR_from_memX_non_binary = utils_vox.apply_4x4_to_vox(camR_T_camX.float(), occupancies_memX.float(), already_mem=False, binary_feat=False)
        occupancies_sum_memR_from_memX_non_binary = torch.sum(occupancies_memR_from_memX_non_binary.reshape(occupancies_memR_from_memX_non_binary.shape[0],-1), axis=1)
        print("Occupancies sum in memR_from_memX_nonbinary ref frame", occupancies_sum_memR_from_memX_non_binary)
        # st()


        if self.visualize:
            occs = torch.sum(occupancies, axis=3).squeeze(1)
            occs_memX = torch.sum(occupancies_memX, axis=3).squeeze(1)
            occs_memR_from_memX = torch.sum(occupancies_memR_from_memX, axis=3).squeeze(1)
            occs_memR_from_memX_nb = torch.sum(occupancies_memR_from_memX_non_binary, axis=3).squeeze(1)
            for occ, occX, occR_T_X,occR_T_X_nb in zip(occs, occs_memX, occs_memR_from_memX,occs_memR_from_memX_nb):
                # st()
                occ_occX = torch.cat((occ, occX,occR_T_X,occR_T_X_nb))
                occ_occX = np.round(occ_occX)
                imshow(occ_occX)
                # plt.close('all')
                # plt.imshow(occ_occX)
                # plt.show(block=True)
        # st()
        return occupancies, maskxyz_camR, maskxyz_memR, maskxyz, maskxyz_memX, occupancies_memX, camR_T_camX #All occupancies in camR reference frame

    '''
    camR_T_camXs.shape: (24, 4, 4)
    xyz_camXs.shape: (24, 1, 65536, 3)
    '''
    def get_aggregated_xyz_camR(self, xyz_camXs, camR_T_camXs):
        xyz_camRs = utils_geom.apply_4x4(camR_T_camXs, xyz_camXs)
        # pcds = [mesh_frame]
        # for xyz_camR in xyz_camRs:
        #     xyz_camR_pcd = make_pcd(xyz_camR.numpy())
        #     pcds.append(xyz_camR_pcd)

        # o3d.visualization.draw_geometries(pcds)
        # st()
        return xyz_camRs, camR_T_camXs # torch.Size([24, 65536, 3])
    

    def get_alignedboxes2thetaformat(self, aligned_boxes):
        # aligned_boxes = torch.reshape(aligned_boxes,[aligned_boxes.shape[0],aligned_boxes.shape[1],6])
        aligned_boxes = aligned_boxes.cpu()
        B,N,_ = list(aligned_boxes.shape)
        xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
        xc = (xmin+xmax)/2.0
        yc = (ymin+ymax)/2.0
        zc = (zmin+zmax)/2.0
        w = xmax-xmin
        h = ymax - ymin
        d = zmax - zmin
        zeros = torch.zeros([B,N])
        boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
        return boxes

    def occupancy_iou(self, occ1, occ2):
        occ1 = occ1.reshape(-1)
        occ1 = occ1.bool()
        occ2 = occ2.reshape(-1)
        occ2 = occ2.bool()
        intersection = occ1 & occ2
        union = occ1 | occ2
        iou = (1.0*intersection.sum())/(union.sum())
        return iou

    def get_3d_object_proposals_from_cones(self, multiview_obj_occs_memX, xyz_camXs, camR_T_camXs, xyz_agg_camR, multiview_obj_xyz_camX, multiview_obj_camR_T_camX, multiview_obj_occs, multiview_obj_xyz_camR, rgb_for_bbox_vis, camX_T_camR_for_bbox_vis, intrinsics):
        
        # multiview_obj_occs are in memR
        '''
        Steps
        1. Get cones for each occupancy - Done
        2. Warp each cone to camR - Done
        3. Create occupancy for aggregated xyz - Done
        4. Take intersection with the aggregated occupancy - Done
        5. Think of what to do after this. - Done
        '''
        #  (torch float32), (np, 32), (torch, 32), (torch, 32), (torch, 32), (torch, 64), (torch, 32), (torch, 32), (np, uint8), (np, 64), (np, 64)
        

        
        xyz_camXs = np.array(xyz_camXs, dtype=float)
        camR_T_camXs = camR_T_camXs.float()
        xyz_agg_camR = xyz_agg_camR.float()
        # st()
        multiview_obj_occs_memR_from_memX = utils_vox.apply_4x4_to_vox(multiview_obj_camR_T_camX.float(), multiview_obj_occs_memX.unsqueeze(1).float(), already_mem=False, binary_feat=True)
        
        if self.visualize:
            obj_visibilities_camXs = utils_vox.convert_xyz_to_visibility(multiview_obj_xyz_camX, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)

            obj_visibilities_camRs_from_camXs = utils_vox.apply_4x4_to_vox(multiview_obj_camR_T_camX.float(), obj_visibilities_camXs.float(), already_mem=False, binary_feat=True)
            obj_visibilities_camRs = utils_vox.convert_xyz_to_visibility(multiview_obj_xyz_camR, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)
            
       
        xyz_agg_memR = utils_vox.Ref2Mem(xyz_agg_camR, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)
        xyz_agg_occ_memR = utils_vox.get_occupancy(xyz_agg_memR, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X) # Set these bounds properly
        xyz_agg_occ_memR = torch.clamp(torch.sum(xyz_agg_occ_memR, dim=0), 0, 1)
        # torch.Size([24, 1, 56, 56, 56])
        obj_cones_camX = utils_vox.convert_xyz_to_cone(multiview_obj_xyz_camX, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)
        obj_cones_camR_from_camX = utils_vox.apply_4x4_to_vox(multiview_obj_camR_T_camX.float(), obj_cones_camX.float(), binary_feat=True)
        
        obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR = xyz_agg_occ_memR.bool().unsqueeze(0) & obj_cones_camR_from_camX.bool()
        obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR = obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR.float()
        # st()

        if self.visualize:
            for obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR_single, obj_cone_camR_from_camX, obj_cone_camX, occ_memR_from_memX, vix_camX, occ_camX, vis_camR_from_camX, vis_camR, occ_camR  in zip(obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR, obj_cones_camR_from_camX, obj_cones_camX, multiview_obj_occs_memR_from_memX, obj_visibilities_camXs, multiview_obj_occs_memX, obj_visibilities_camRs_from_camXs, obj_visibilities_camRs, multiview_obj_occs):

                projected_obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR_single = torch.sum(obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR_single.squeeze(0), dim=1).numpy()
                projected_xyz_agg_occ_memR = torch.sum(xyz_agg_occ_memR.squeeze(0), dim=1).numpy()
                projected_cone_camR_from_camX = torch.sum(obj_cone_camR_from_camX.squeeze(0), dim=1).numpy()
                projected_cone_camX = torch.sum(obj_cone_camX.squeeze(0), dim=1).numpy()
                projected_vis_camX = torch.sum(vix_camX.squeeze(0), dim=1).numpy()
                projected_vis_camR_from_camX = torch.sum(vis_camR_from_camX.squeeze(0), dim=1).numpy()
                projected_vis_camR = torch.sum(vis_camR.squeeze(0), dim=1).numpy()
                projected_occ_camR = torch.sum(occ_camR, dim=1).numpy()
                projected_occ_camX = torch.sum(occ_camX, dim=1).numpy()
                projected_occ_memR_from_memX = torch.sum(occ_memR_from_memX.squeeze(0), dim=1).numpy()
                # projected_cone_camR_from_camX_intersect_agg_occ_memR = projected_xyz_agg_occ_memR.astype(bool) &  projected_cone_camR_from_camX.astype(bool)
                # print("Intersection sum is: ", projected_cone_camR_from_camX_intersect_agg_occ_memR.sum())
                # st()

                firstrow = np.concatenate((projected_occ_camR, projected_occ_camX), axis=1)
                secondrow = np.concatenate((projected_vis_camR, projected_vis_camX), axis=1)
                thirdrow = np.concatenate((projected_vis_camR_from_camX, projected_occ_memR_from_memX), axis=1)
                forthrow = np.concatenate((projected_cone_camR_from_camX, projected_cone_camX), axis=1)
                fifthrow = np.concatenate((projected_xyz_agg_occ_memR, projected_obj_cones_camR_from_camX_intersect_xyz_agg_occ_memR_single), axis=1)

                visualize = np.concatenate((firstrow, secondrow, thirdrow, forthrow, fifthrow), axis=0)
                print("Sums of viss are: ", np.sum(projected_vis_camR_from_camX), np.sum(projected_vis_camR), np.sum(projected_occ_camR))
                # visualize = np.concatenate((projected_vis_camR_from_camX, projected_vis_camR, projected_occ_camR))
                imshow(visualize)
                # plt.imshow(visualize)
                # plt.show(block=True)
        
        # Take intersection of cones with aggregated occupancy
        intersections = obj_cones_camR_from_camX.bool().squeeze(1) & xyz_agg_occ_memR.bool() # torch.Size([24, 56, 56, 56])
        # Find connected components 
        # thresh = 200
        thresh = 0.25
        connected_components = []
        used = np.zeros(intersections.shape[0])
        
        for i in range(intersections.shape[0]):

            if used[i]:
                continue
            used[i] = 1
            current_component = [i]
            for j in range(i+1, intersections.shape[0]):
                if used[j]:
                    continue
                # intr = torch.sum(intersections[i].bool() & intersections[j].bool())
                intr = self.occupancy_iou(intersections[i].bool(), intersections[j].bool())
                print("IOU is: ", intr)
                if intr > self.iou_thresh:
                    used[j] = 1
                    current_component.append(j)
            connected_components.append(current_component)
        print("Connected components are: ", connected_components)
        # st()
        bbox_memR = []

        # occupancies = multiview_obj_occs
        occupancies = intersections
        # merge the occupancies of connected components
        for components in connected_components:
            if len(components) < self.vote_thresh: #or len(components)>3:
                continue
            merged_occs = None
            for idx in components:
                # wt = 1
                # if idx == 0:
                #     wt = 2
                if merged_occs == None:
                    merged_occs = occupancies[idx]
                else:
                    merged_occs += occupancies[idx]

            vote_thresh = 1
            zvals, yvals, xvals = torch.where(merged_occs >= vote_thresh)
            print("zvals are: ", zvals)
            
            if zvals.shape[0] == 0 or yvals.shape[0] == 0 or xvals.shape[0] == 0:
                continue
            zmin, zmax, ymin, ymax, xmin, xmax = zvals.min(), zvals.max(), yvals.min(), yvals.max(), xvals.min(), xvals.max()
            bbox_memR.append(torch.tensor([xmin, ymin, zmin, xmax, ymax, zmax]).reshape(2, 3))
            # print("max is: ", merged_occs.max())

        if len(bbox_memR) == 0:
            return None
        bbox_memR = torch.stack(bbox_memR).float()
        bbox_camR = utils_vox.Mem2Ref(bbox_memR, self.segmented_obj_Z, self.segmented_obj_Y, self.segmented_obj_X)
        bbox_camR_processed = []
        for bbox in bbox_camR:
            if bbox[0,0]==bbox[1,0] or bbox[0,1]==bbox[1,1] or bbox[0,2]==bbox[1,2]:
                continue
            else:
                bbox_camR_processed.append(bbox)
        print("bbox camr is: ", bbox_camR)
        print("bbox_camR_processed: ", bbox_camR_processed)
        if len(bbox_camR_processed) == 0:
            return None
        bbox_camR = torch.stack(bbox_camR_processed)

        # st()
        bbox_camR_theta = self.get_alignedboxes2thetaformat(bbox_camR.reshape(-1, 1, 6))
        bbox_camR_corners = utils_geom.transform_boxes_to_corners(bbox_camR_theta)
        # bbox_camX_corners = bbox_camR_corners.squeeze(1)
        bbox_camX_corners = utils_geom.apply_4x4(torch.tensor(camX_T_camR_for_bbox_vis).unsqueeze(0).float(), bbox_camR_corners.squeeze(1).float())
        bbox_camX_ends = nlu.get_ends_of_corner(bbox_camX_corners.permute(0,2,1)).permute(0,2,1)
        bbox_camX_theta = self.get_alignedboxes2thetaformat(bbox_camX_ends.reshape(-1, 1, 6))
        bbox_camX_corners = utils_geom.transform_boxes_to_corners(bbox_camX_theta).permute(1, 0, 2, 3).squeeze(0)
        print("camX corners are: ", bbox_camX_corners)
        summaryWriter = utils_improc.Summ_writer(None, 10, "train")
        rgb_for_bbox_vis = torch.tensor(rgb_for_bbox_vis).permute(2, 0, 1).unsqueeze(0)
        rgb_for_bbox_vis = utils_improc.preprocess_color(rgb_for_bbox_vis)
        bbox_camX_corners = bbox_camX_corners.unsqueeze(0)
        scores = torch.ones((bbox_camX_corners.shape[0], bbox_camX_corners.shape[1]))
        tids = torch.ones_like(scores)
        intrinsics = torch.tensor(intrinsics).unsqueeze(0)
        
        # st()
        # self.draw_boxes_using_ends(bbox_camX_ends, rgb_for_bbox_vis, intrinsics)
        rgb_with_bbox = summaryWriter.summ_box_by_corners("2Dto3D", rgb_for_bbox_vis, bbox_camX_corners, scores, tids, intrinsics, only_return=True)
        # st()
        rgb_with_bbox = utils_improc.back2color(rgb_with_bbox)
        rgb_with_bbox = rgb_with_bbox.permute(0, 2, 3, 1).squeeze(0).numpy()
        print("Total number of boxes: ", bbox_camR.shape[0])
        if self.visualize or True:
            plt.imshow(rgb_with_bbox)
            plt.show(block=True)
        
        if self.visualize or True:
            for ii in range(bbox_camX_corners.shape[1]):
                rgb_with_bbox_i = summaryWriter.summ_box_by_corners("2Dto3D", rgb_for_bbox_vis, bbox_camX_corners[:, ii:ii+1], scores[:, ii:ii+1], tids[:, ii:ii+1], intrinsics, only_return=True)
                rgb_with_bbox_i = utils_improc.back2color(rgb_with_bbox_i)
                rgb_with_bbox_i = rgb_with_bbox_i.permute(0, 2, 3, 1).squeeze(0).numpy()
                plt.imshow(rgb_with_bbox_i)
                plt.show(block=True)

        img_name = str(int(time.time())) + ".png"
        # st()
        if "domestation" in hostname:
            scipy.misc.imsave(os.path.join("/home/sirdome/shubhankar/3dbboxResults", img_name), rgb_with_bbox)
        elif "compute" in hostname:
            if username == "shamitl":
                scipy.misc.imsave(os.path.join("/home/shamitl/dataset/3dbboxResults", img_name), rgb_with_bbox)
            elif username == "mprabhud":
                scipy.misc.imsave(os.path.join("/home/mprabhud/dataset/carla_3dbbox_results", img_name), rgb_with_bbox)
        # st()
        return bbox_camR_corners

    def draw_boxes_using_ends(self, bbox_camX_ends, rgb_for_bbox_vis, intrinsics):
        rgb = utils_improc.back2color(rgb_for_bbox_vis)
        rgb = rgb.float()
        utils_pointcloud.draw_boxes_on_rgb(rgb[0].permute(1,2,0).numpy(),intrinsics[0].numpy(),bbox_camX_ends[0].numpy().reshape(1,-1), visualize=True)

    def get_3d_box_proposals(self, rgbs, camR_T_origin, origin_T_camXs, pix_T_camXs, xyz_camXs, camR_index):
        
        multiview_obj_occs = []
        multiview_obj_xyz_camR = []
        multiview_obj_xyz_memR = []
        multiview_obj_xyz_camX = []
        multiview_obj_camR_T_camX = []
        multiview_obj_occs_memX = []
        multiview_obj_xyz_memX = []

        
        camR_T_camXs = camR_T_origin @ origin_T_camXs
        camR_T_camXs = camR_T_camXs
        # st()
        depths,_ = utils_geom.create_depth_image(torch.tensor(pix_T_camXs).float(), torch.tensor(xyz_camXs).float(), self.H, self.W)
        depths[torch.where(depths == 100.)] = 0
        
        depths = depths.squeeze(0).numpy()
        num_views = rgbs.shape[0]
        cnt = 0
        for rgb, pix_T_camX, camR_T_camX, depth in zip(rgbs, pix_T_camXs, camR_T_camXs, depths):
            # st()
            # if cnt == camR_index:#100*num_views-3:
            if cnt==0:
                rgb_for_bbox_vis = rgb[:,:,:3]
                camX_T_camR_for_bbox_vis = utils_geom.safe_inverse(torch.tensor(camR_T_camX).unsqueeze(0)).squeeze(0).numpy()

            cnt += 1

            binary_mask = self.cleverDetector.get_binary_masks(rgb, self.max_number_of_objects, self.visualize) # (256, 256, 4)
            
            # Kill masks greater than max_mask_size_thresh

            flat_binary_mask = binary_mask.reshape(-1, binary_mask.shape[-1])
            binary_mask_sum =  np.sum(flat_binary_mask, axis=0)
            invalid_masks = np.where(binary_mask_sum > self.max_mask_size_thresh)

            binary_mask[:,:,invalid_masks] = 0 # Kill masks that are abnormally large
            if binary_mask.sum() == 0:
                continue
            # valid_masks = np.setdiff1d(np.arange(binary_mask.shape[-1]), invalid_masks)
            # # st()
            # binary_mask = binary_mask[:,:,valid_masks]
            print("Done with iteration: ", cnt)
            occupancies_memR, obj_xyz_camR, obj_xyz_memR, obj_xyz_camX, obj_xyz_memX, occupancies_memX, obj_camR_T_camX = self.get_occupancies_for_objects(binary_mask, depth, pix_T_camX, camR_T_camX)
            multiview_obj_occs.append(occupancies_memR.squeeze(1))
            multiview_obj_occs_memX.append(occupancies_memX.squeeze(1))
            multiview_obj_xyz_camR.append(obj_xyz_camR)
            multiview_obj_xyz_memR.append(obj_xyz_memR)
            multiview_obj_xyz_camX.append(obj_xyz_camX)
            multiview_obj_xyz_memX.append(obj_xyz_memX)

            multiview_obj_camR_T_camX.append(obj_camR_T_camX)
        
        invalid_bbox = np.array([])
        if len(multiview_obj_xyz_camX) > 0:
            multiview_obj_xyz_camX = torch.cat(multiview_obj_xyz_camX)
            multiview_obj_camR_T_camX = torch.cat(multiview_obj_camR_T_camX)
            multiview_obj_occs = torch.cat(multiview_obj_occs)
            multiview_obj_occs_memX = torch.cat(multiview_obj_occs_memX)
            multiview_obj_xyz_camR = torch.cat(multiview_obj_xyz_camR)
            # st()
            xyz_agg_camR, camR_T_camXs = self.get_aggregated_xyz_camR(torch.tensor(xyz_camXs), torch.tensor(camR_T_camXs))
            # BxNx8x3
            bbox3D_from_2D = self.get_3d_object_proposals_from_cones(multiview_obj_occs_memX, xyz_camXs, camR_T_camXs, xyz_agg_camR, multiview_obj_xyz_camX, multiview_obj_camR_T_camX, multiview_obj_occs, multiview_obj_xyz_camR, rgb_for_bbox_vis, camX_T_camR_for_bbox_vis, pix_T_camXs[0])
            if bbox3D_from_2D == None:
                return invalid_bbox

            bbox3D_from_2D = bbox3D_from_2D.reshape(-1, 8, 3).float() # remove the batch dimension
            print("bbox3d_from_2d shape: ", bbox3D_from_2D.shape)
            origin_T_camR = utils_geom.safe_inverse(torch.tensor(camR_T_origin)).float()
            # Bx8x3
            bbox3D_from_2D_origin = utils_geom.apply_4x4(origin_T_camR[0:1].repeat(bbox3D_from_2D.shape[0],1,1), bbox3D_from_2D)
            
            print("camR corners: ", bbox3D_from_2D_origin)
            bbox3D_from_2D_origin_ends = nlu.get_ends_of_corner(bbox3D_from_2D_origin.permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 6).numpy()
            return bbox3D_from_2D_origin_ends
        else:
            return invalid_bbox