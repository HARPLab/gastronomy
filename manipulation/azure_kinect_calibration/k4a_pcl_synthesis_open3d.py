import argparse
import cv2
import logging
import numpy as np
import os
import sys
import time
import traceback
import rospy
import struct
import ctypes
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from autolab_core import RigidTransform
from perception import CameraIntrinsics

try:
    import cv2
    import pylibfreenect2 as lf2
except:
    logging.warning('Unable to import pylibfreenect2. Python-only Kinect driver may not work properly.')

try:
    from cv_bridge import CvBridge, CvBridgeError
    import rospy
    import sensor_msgs.msg
except ImportError:
    logging.warning("Failed to import ROS in Kinect2_sensor.py. Kinect will not be able to be used in bridged mode")


import open3d as o3d
import copy

from pcl_registration_utils import point_cloud_to_color_arr, make_pcd, visualize, get_pcd_bounds_str

def max_test_overhead_idx():
    f_list = os.listdir('./train_overhead_data')
    overhead_list = [f[:-4] for f in f_list if 'data' in f and 'pcd' in f]
    if len(overhead_list) == 0:
        return -1
    overhead_num = [int(f.split('_')[-1]) for f in overhead_list]
    return max(overhead_num)


def get_transform_file_for_dir(d):
    assert os.path.exists(d), 'Dir does not exist: {}'.format(d)
    # if 'overhead' in d:
    #     return os.path.join(d, 'kinect2_overhead_to_world_calc_from_base.tf')
    # return os.path.join(d, 'kinect2_overhead_to_world.tf')
    return os.path.join(d, 'kinect2_overhead_to_world_calc_from_base.tf')



class PCLRegistrationUtils(object):
    def __init__(self): 
        self.calib_dirs = { 
            'front_right': './calib/robot_middle/azure_kinect_front_right',
            'front_left': './calib/robot_middle/azure_kinect_front_left',
            'overhead': './calib/robot_middle/azure_kinect_overhead',
        }
        self.transform_by_camera = {}    
        for k, v in self.calib_dirs.items():
            T = RigidTransform.load(get_transform_file_for_dir(v))
            T.from_frame = k
            T.to_frame = 'world'
            self.transform_by_camera[k] = T
        
        self.stereo_calib_data = [
            # (from, to, path)
            # ('front_right', 'overhead', './calib/stereo_calib_azure_cpp/Nov_19_try_3/overhead_to_front_right_x.tf')
            # ('front_right', 'overhead', './calib/stereo_calib_azure_cpp/Nov_20_matlab/overhead_to_front_right_3.tf')

            # do the inverse of this one?
            # ('front_right', 'overhead', './calib/stereo_calib_azure_cpp/Nov_21_combined/overhead_to_front_right_3.tf'),

            ('overhead', 'front_left', './calib/stereo_calib_azure_cpp/main_overhead_sec_front_left/front_left_to_overhead_5.tf'),

            # Transf between front_left and back_left camera 
            ('front_left', 'back_left', './calib/stereo_calib_azure_cpp/main_front_left_sec_back_left/back_left_to_front_left_7.tf'),

            # Transf between front_right and back_right camera
            ('front_right', 'back_right', './calib/stereo_calib_azure_cpp/main_front_right_sec_back_right/back_right_to_front_right_2.tf')
        ]
        self.stereo_calib_dict = {}
        for s in self.stereo_calib_data:
            T = RigidTransform.load(s[2])
            T.from_frame = s[0]
            T.to_frame = s[1]
            self.stereo_calib_dict['{}${}'.format(s[1], s[0])] = T

        self.T_overhead_front_right = (self.transform_by_camera['overhead'].inverse()).dot(self.transform_by_camera['front_right'])
        self.T_overhead_front_left = (self.transform_by_camera['overhead'].inverse()).dot(self.transform_by_camera['front_left'])

        rospy.loginfo("Original transf matrix: \n{}".format(
            np.array_str(self.T_overhead_front_right.matrix, precision=5, suppress_small=True)))
        rospy.loginfo("New transf matrix: \n{}".format(
            np.array_str(self.stereo_calib_dict['front_left$overhead'].inverse().matrix, 
                         precision=5, suppress_small=True)))

        rospy.loginfo("Front left -> overhead transf matri: \n"
                      "   Original     :               \n{}\n"
                      "        new     :               \n{}\n".format(
            np.array_str(self.T_overhead_front_right.matrix, precision=5, suppress_small=True),
            np.array_str(self.stereo_calib_dict['front_left$overhead'].inverse().matrix, 
                         precision=5, suppress_small=True)
                      ))

        # import pdb; pdb.set_trace()

        self.point_cloud_by_camera_dict = {}
        self.has_pcd_by_camera_dict = {}

        self.pcl_volume_min_bound = np.array([0.1, -0.4, -0.05])
        self.pcl_volume_max_bound = np.array([0.8, 0.4, 0.5])

        self.voxel_size = 0.01
        self.max_correspondence_distance_coarse = self.voxel_size * 15
        self.max_correspondence_distance_fine = self.voxel_size * 1.5


    def pairwise_registration(self, source, target):
        icp_coarse = o3d.registration.registration_icp(
            source, target, self.max_correspondence_distance_coarse, np.identity(4),
            o3d.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.registration.registration_icp(
            source, target, self.max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    def full_registration(self, pcds):
        pose_graph = o3d.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwise_registration(
                    pcds[source_id], pcds[target_id])
                print("Build o3d.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=True))
        return pose_graph


    def merge_multiple_pcd(self, pcd_list):
        for pcd in pcd_list:
            print("PCD min_bound: {}, max_bound: {}".format(
                pcd.get_min_bound(), pcd.get_max_bound()))
        voxel_size = self.voxel_size
        voxel_grids = []
        for p in pcd_list:
            # v = o3d.geometry.VoxelGrid.create_from_point_cloud(p, 0.01)
            voxel_grids.append(p)
        
        '''
        for voxel_grid in voxel_grids: rospy.loginfo("old voxel grid size: {}".format(len(voxel_grid.voxels)))
            voxel_list = voxel_grid.voxels
            for x in range(60, 120):
                for z in range(40, 80):
                    v = o3d.geometry.Voxel([x, 0, z], [255, 0, 0])
                    voxel_list.append(v)
            voxel_grid.voxels = voxel_list
            rospy.loginfo("new voxel grid size: {}".format(len(voxel_grid.voxels)))
            visualize([voxel_grid])
        '''
        visualize(voxel_grids)
        import pdb; pdb.set_trace()

        visualize(voxel_grids[:1])
        visualize(voxel_grids[1:2])
        visualize(voxel_grids[2:3])


        # Find which voxels are occupied
        all_voxel_map = np.zeros((200, 200, 100), dtype=np.int32)
        for voxel_grid in voxel_grids:
            for v in voxel_grid.voxels:
                vi = v.grid_index
                if (vi[0] < all_voxel_map.shape[0] and
                    vi[1] < all_voxel_map.shape[1] and
                    vi[2] < all_voxel_map.shape[2]):
                    all_voxel_map[vi[0], vi[1], vi[2]] += 1
                else:
                    import pdb; pdb.set_trace()

        # Now remove the voxels which are not found in the rest

        for voxel_grid in voxel_grids:
            filt_voxels = []
            for v in voxel_grid.voxels:
                vi = v.grid_index
                if (vi[0] < all_voxel_map.shape[0] and
                    vi[1] < all_voxel_map.shape[1] and
                    vi[2] < all_voxel_map.shape[2]):
                    # If this voxel was claimed as present by multiple cameras then
                    # fill it. Should we also 
                    if all_voxel_map[vi[0], vi[1], vi[2]] >= 2:
                        filt_voxels.append(v)
            voxel_grid.voxels = filt_voxels

        visualize(voxel_grids[:1])
        visualize(voxel_grids[1:2])
        visualize(voxel_grids[2:3])

        # import pdb; pdb.set_trace()

        pcds_down = [pcd.voxel_down_sample(voxel_size=self.voxel_size) for pcd in pcd_list]

        # Visualize
        visualize(pcds_down[:1])
        visualize(pcds_down[1:2])
        visualize(pcds_down[2:3])
        visualize(pcds_down)

        print("Full registration ...")
        pose_graph = self.full_registration(pcds_down)

        print("Optimizing PoseGraph ...")
        option = o3d.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        o3d.registration.global_optimization(
            pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

        print("Transform points and display")
        for point_id in range(len(pcds_down)):
            print(pose_graph.nodes[point_id].pose)
            pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        o3d.visualization.draw_geometries(pcds_down)

        print("Make a combined point cloud")
        # pcds = load_point_clouds(voxel_size)
        pcds = [pcd.voxel_down_sample(voxel_size=self.voxel_size) for pcd in pcd_list]
        pcd_combined = o3d.geometry.PointCloud()
        for point_id in range(len(pcds)):
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)
            pcd_combined += pcds[point_id]
        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=self.voxel_size)
        o3d.io.write_point_cloud("./multiway_registration.pcd", pcd_combined_down)
        o3d.visualization.draw_geometries([pcd_combined_down])


    def get_transform_point_cloud_from_raw_data(self, data, camera_key):
        if 'overhead' in camera_key:
            color = [255, 0, 0]
        else:
            color = [0, 0, 255]
        color = None
        # pcl_arr = point_cloud_to_mat(data)
        pcl_arr, rgb_arr = point_cloud_to_color_arr(data, color=color)
        # T = self.transform_by_camera[camera_key]
        # transf_pcl_arr = np.dot(T.matrix, pcl_arr.T)
        if camera_key == 'front_right_no':
            # transf_pcl_arr = np.dot(self.T_overhead_front_right.matrix, pcl_arr.T)
            M = self.stereo_calib_dict['overhead$front_right'].inverse().matrix
            # M[0, 3] = self.T_overhead_front_right.matrix[0, 3]
            # M[1, 3] = self.T_overhead_front_right.matrix[1, 3]
            # M[2, 3] = self.T_overhead_front_right.matrix[2, 3]
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'front_left': 
            # We don't take an inverse since we want to view everything in the overhead 
            # camera's frame.
            M = self.stereo_calib_dict['front_left$overhead'].matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'back_left':
            M = self.stereo_calib_dict['back_left$front_left'].matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'back_right':
            M = self.stereo_calib_dict['back_right$front_right'].matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        else:
            transf_pcl_arr = np.dot(np.eye(4), pcl_arr.T)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transf_pcl_arr[:3, :].T)
        pcd.colors = o3d.utility.Vector3dVector(rgb_arr[:, :3]/255.0)
        return pcd

    def pcl_callback(self, data):
        rospy.loginfo("Got data: {}".format(type(data)))
        # cloud, rgb = get_cloud(data)
        cloud_arr, rgb_arr = point_cloud_to_color_arr(data)

        point_cloud_arr = point_cloud_to_mat(data)
        # T_mat = transform.matrix
        # rospy.loginfo('point_cloud_arr: {}'.format(point_cloud_arr.shape))
        # transf_pc_arr = np.dot(T_mat, point_cloud_arr)
        print(cloud_arr.shape)
        print(point_cloud_arr.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_arr[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(rgb_arr[:, :3]/255.0)
        # o3d.io.write_point_cloud("./open3d_sync.ply", pcd)

        # Load saved point cloud and visualize it
        # pcd_load = o3d.io.read_point_cloud("./open3d_sync.ply")
        o3d.visualization.draw_geometries([pcd])

    def save_pcd_with_camera_key(self, data, camera_key):
        if self.has_pcd_by_camera_dict.get(camera_key):
            return
        print("Will save pcd for camera: {}".format(camera_key))
        pcd = self.get_transform_point_cloud_from_raw_data(data, camera_key)

        print("back_right bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
        # pcd = pcd.crop(self.pcl_volume_min_bound, self.pcl_volume_max_bound)
        print("back_right bounds AFTER crop: => {}".format(get_pcd_bounds_str(pcd)))

        self.point_cloud_by_camera_dict[camera_key] = pcd
        self.has_pcd_by_camera_dict[camera_key] = True
        rospy.loginfo("Did get camera data {}".format(camera_key))
        idx = max_test_overhead_idx() + 1
        pcd_path = os.path.join('./register_{}_data_{}.pcd'.format(camera_key, idx))
        o3d.io.write_point_cloud(pcd_path, pcd)
        rospy.loginfo("Did save pcd: {}".format(pcd_path))

    def back_right_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'back_right')

    def front_right_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'front_right')

    def front_left_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'front_left')

    def back_left_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'back_left')

    def overhead_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'overhead')


def main():
    rospy.init_node('test_k4a_pcl', anonymous=True)
    rospy.loginfo('Test k4a pcl!!')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcl_reg = PCLRegistrationUtils()
    rospy.loginfo("Did load transform for camers: {}".format(len(pcl_reg.transform_by_camera)))

    pc_topics_by_callback = [
        # ('/points2', pcl_callback),
        # ('/back_right_kinect/points2', pcl_reg.back_right_pcl_callback),
        # ('/front_right_kinect/points2', pcl_reg.front_right_pcl_callback),
        ('/front_left_kinect/points2', pcl_reg.front_left_pcl_callback),
        ('/overhead_kinect/points2', pcl_reg.overhead_pcl_callback),
        # ('/back_left_kinect/points2', pcl_reg.back_left_pcl_callback),
    ]
    # pc_topic = "/points2"
    # pc_sub = rospy.Subscriber(pc_topic, PointCloud2, pcl_callback)
    for p in pc_topics_by_callback:
        pc_sub = rospy.Subscriber(p[0], PointCloud2, p[1])

    while not rospy.is_shutdown():
        rospy.loginfo("point_cloud_dict size".format(len(pcl_reg.point_cloud_by_camera_dict)))
        if len(pcl_reg.point_cloud_by_camera_dict) == 2:
            # Do registration
            #  for main and secondary, the first argument should be secondary and then main 
            # pcd_list = [
            #     pcl_reg.point_cloud_by_camera_dict['front_left'],
            #     pcl_reg.point_cloud_by_camera_dict['overhead'],
            #     # pcl_reg.point_cloud_by_camera_dict['front_right'],
            # ]
            pcd_list = [
                pcl_reg.point_cloud_by_camera_dict['front_left'],
                pcl_reg.point_cloud_by_camera_dict['overhead'],
                # pcl_reg.point_cloud_by_camera_dict['front_right'],
            ]
            rospy.loginfo("Will try to register multiple point clouds.")
            pcl_reg.merge_multiple_pcd(pcd_list)
            break
        
        # if len(pcl_reg.point_cloud_by_camera_dict) == 1:
        #     print("Did save some data")
        #     break

        rospy.sleep(1.0)

        # rospy.spin()


if __name__ == '__main__':
    main()
