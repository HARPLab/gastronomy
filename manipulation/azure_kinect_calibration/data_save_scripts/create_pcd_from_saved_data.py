import argparse
import cv2
import logging
import numpy as np
import h5py
import os
import sys
import time
import struct
import ctypes

from autolab_core import RigidTransform
from perception import CameraIntrinsics

import cv2
import open3d as o3d
import copy

sys.path.append(os.path.abspath('../'))

from pcl_registration_utils import point_cloud_to_color_arr, make_pcd, visualize, get_pcd_bounds_str
import fusion
import matplotlib.pyplot as plt


def recursively_get_dict_from_group(group_or_data):
    d = {}
    if type(group_or_data) == h5py.Dataset:
        return np.array(group_or_data)

    # Else it's still a group
    for k in group_or_data.keys():
        v = recursively_get_dict_from_group(group_or_data[k])
        d[k] = v
    return d


def create_o3d_pinhole_camera(intrinsic, extrinsic):
    o3d_cam = o3d.camera.PinholeCameraParameters()
    o3d_cam.extrinsic = np.copy(extrinsic)
    o3d_cam.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    o3d_cam.intrinsic.set_intrinsics(
        1280,                   # width
        720,                    # height
        intrinsic[0, 0],        # fx
        intrinsic[1, 1],        # fy
        intrinsic[0, 2],        # cx
        intrinsic[1, 2],        # cy
    )
    return o3d_cam


class PCLFusion(object):
    def __init__(self, h5_path, save_path):
        self.h5_path = h5_path
        self.save_path = save_path

        h5f = h5py.File(h5_path, 'r')
        self.h5_data = recursively_get_dict_from_group(h5f['/'])
        h5f.close()

        self.cam_info = self.h5_data['camera_info']
        self.depth_img_by_camera_dict = self.h5_data['depth_to_rgb']
        self.color_img_by_camera_dict = self.h5_data['color_raw']

        # Define the max/min bound in reference to the main camera.
        self.pcl_volume_min_bound = np.array([-0.45, -0.6, 0.1])
        self.pcl_volume_max_bound = np.array([0.45, 0.0, 1.1])

    def integrate_images(self):
        # Initialize voxel volume
        print("Initializing voxel volume...")
        vol_bnds = np.array([
                [-0.6,  0.6],
                [-0.8,  0.1],
                [-0.2,  1.0]
        ])
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)
        o3d_volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=0.005,
                sdf_trunc=0.005 * 4,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8,
                volume_unit_resolution=2,
        )
        # o3d_unif_volume = o3d.integration.UniformTSDFVolume(
        #     1.0, 256, 0.04, o3d.integration.TSDFVolumeColorType.RGB8)

        o3d_volume_camera_list = list(self.depth_img_by_camera_dict.keys())
        for cam_key in self.depth_img_by_camera_dict.keys():

            depth_im = self.depth_img_by_camera_dict[cam_key]
            # depth_im[depth_im == 65.535] = 0 # set invalid depth to 0 (specific to 7-scenes dataset)
            depth_im[np.isnan(depth_im)] = 0  # This is when we use 32FC1 conversion format.

            # TODO: Should we smooth over the depth image?

            color_im = self.color_img_by_camera_dict[cam_key]

            print("rgb_to_depth size: {}, min: {}, max: {}".format(
                  depth_im.shape, depth_im.min(), depth_im.max()))

            cam_intr = self.cam_info[cam_key]['intrinsic']
            cam_extr = self.cam_info[cam_key]['extrinsic']
            o3d_cam = create_o3d_pinhole_camera(cam_intr, cam_extr)

            # ScalableTSDF uses a depth_scale of 1000.0 to convert depth image
            # to point cloud, since we use 32FC1 encoding to get the depth image
            # the values for the depth image is in meteres ~ [0, 3m]. Hence, using
            # a depth scale of 1000, causes the point cloud values to be too close.
            # To avoid this, do "depth_im * 1000" to convert it into larger
            # values.
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_im),
                o3d.geometry.Image(depth_im*1000.0),
                depth_trunc=1000.0,
                convert_rgb_to_intensity=False)
            # import matplotlib.pyplot as plt
            # plt.title('Redwood depth image')
            # plt.imshow(rgbd.depth)
            # plt.show()

            if cam_key in o3d_volume_camera_list:

                # pcd = o3d.geometry.PointCloud.create_from_depth_image(
                #     depth=rgbd.depth,
                #     intrinsic=o3d_cam_intr,
                #     extrinsic=np.linalg.inv(cam_extr),
                #     depth_scale=1000.0,
                #     depth_trunc=1000.0,
                #     stride=4,
                # )
                # o3d.visualization.draw_geometries([pcd])

                o3d_volume.integrate(
                    rgbd,
                    o3d_cam.intrinsic,
                    np.linalg.inv(cam_extr)
                )

                # Integrate observation into voxel volume (assume color aligned with depth)
                tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_extr, obs_weight=1.)
                print("Did ingegrate: {}".format(cam_key))

        # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
        print("Saving to mesh.ply...")
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        fusion.meshwrite(os.path.join(self.save_path, "fusion_mesh.ply"),
                            verts, faces, norms, colors)

        pcd = o3d_volume.extract_point_cloud()
        mesh = o3d_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])

        o3d.io.write_point_cloud(
            os.path.join(self.save_path, 'scalable_tsdf_pcl.pcd') , pcd)
        o3d.io.write_triangle_mesh(
            os.path.join(self.save_path, 'scalable_tsdf_mesh.ply'), mesh)
        
        # Downsample PCD to look at the right thing
        print("PCD bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.pcl_volume_min_bound,
            max_bound=self.pcl_volume_max_bound,
        )
        pcd_cropped = pcd.crop(bounds)
        print("PCD bounds AFTER crop: => {}".format(get_pcd_bounds_str(pcd_cropped)))
        o3d.io.write_point_cloud(
            os.path.join(self.save_path, 'scalable_tsdf_pcl_trimmed.pcd') , pcd)
        
        # Segment plane
        plane_segment_ok = True
        segment_cloud = pcd_cropped
        num_planes = 0 
        while plane_segment_ok and num_planes < 3:
            plane_model, inliers = segment_cloud.segment_plane(
                distance_threshold=0.005, ransac_n=5, num_iterations=250
            )
            [a, b, c, d] = plane_model
            print("Plane model: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(
                a, b, c, d
            ))
            if (abs(a) < 0.0001 or abs(c/a) > 10) and \
               (abs(b) < 0.0001 or abs(c/b) > 10) and \
               (abs(c) > 0.001 and  (d/c) < -0.75 and (d/c) > -1.0):
               plane_segment_ok = True
            else:
               plane_segment_ok = False

            if not plane_segment_ok:
                break

            inlier_cloud = segment_cloud.select_down_sample(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = segment_cloud.select_down_sample(inliers, invert=True)
            segment_cloud = outlier_cloud
            num_planes += 1

        # visualize([segment_cloud])

        # Now remove the points that are far away from the table center
        center = np.array([0, -0.2, -0.8])
        table_lb = np.array([-0.18, -0.4, 0.7])
        table_ub = np.array([0.18, 0.15, 0.95])
        table_bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=table_lb,
            max_bound=table_ub,
        )
        print("PCD bounds BEFORE crop: => {}".format(get_pcd_bounds_str(segment_cloud)))
        segment_cloud = segment_cloud.crop(table_bounds)
        print("PCD bounds BEFORE crop: => {}".format(get_pcd_bounds_str(segment_cloud)))

        o3d.io.write_point_cloud(
            os.path.join(self.save_path, 'final_segmented_pcl.pcd') , segment_cloud)
        visualize([segment_cloud])

        '''
        eps = 0.01
        labels = np.array(
            segment_cloud.cluster_dbscan(eps=eps, min_points=10, print_progress=True))
        max_label = max(labels)
        print("Got {} clusters" .format(max_label + 1))

        cmap = plt.get_cmap("Set1", max_label + 1)
        # colors = cmap(labels / (max_label if max_label > 0 else 1))
        colors = cmap(labels)
        # db_scan returns -1 for noisy points
        colors[labels < 0] = 0
        segment_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([segment_cloud])
        '''

        '''
        pcd_2 = o3d_unif_volume.extract_point_cloud()
        # mesh_2 = o3d_unif_volume.extract_triangle_mesh()
        # mesh_2.compute_vertex_normals()
        o3d.visualization.draw_geometries([pcd_2]) 
        '''


def main():
    print('Create k4a data from saved data!!')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    parser = argparse.ArgumentParser(description="Create fused pcd data from saved data.")
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to hdf5 with saved data.')
    parser.add_argument('--output_dir_prefix', type=str, required=True,
                        help='Path to extract data')

    args = parser.parse_args()

    save_dir = os.path.join(os.path.dirname(args.h5_path), args.output_dir_prefix)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pcl_fusion = PCLFusion(args.h5_path, save_dir)
    pcl_fusion.integrate_images()


if __name__ == '__main__':
    np.random.seed(0)
    main()
