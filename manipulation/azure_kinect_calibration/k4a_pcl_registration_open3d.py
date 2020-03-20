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
            'back_left': './calib/robot_middle/azure_kinect_front_left',
            'back_right': './calib/robot_middle/azure_kinect_front_right',
        }
        self.transform_by_camera = {}    
        for k, v in self.calib_dirs.items():
            T = RigidTransform.load(get_transform_file_for_dir(v))
            # self.transform_by_camera[k] = T
            self.transform_by_camera[k] = RigidTransform()
        self.point_cloud_by_camera_dict = {}
        self.has_pcd_by_camera_dict = {}

        self.pcl_volume_min_bound = np.array([0.1, -0.4, -0.05])
        self.pcl_volume_max_bound = np.array([0.8, 0.4, 0.5])

        self.pcl_volume_min_bound_dict = {
            'overhead': np.array([-1.0, -2.0, 0.1]),
            # 'front_left': np.array([-0.8, -1.0, 0.0]),
        }
        self.pcl_volume_max_bound_dict = {
            'overhead': np.array([1.0, 0.1, 1.2]),
            # 'front_left': np.array([0.0, 0.2, 1.2]),
        }

        self.voxel_size = 0.005
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

    def register_pcd_pair(self, target_pcd, source_pcd):
        pcd_list = [source_pcd, target_pcd]
        pcd_crop_list = []
        for i, pcd in enumerate(pcd_list):
            print("PCD min_bound: {}, max_bound: {}".format(
                pcd.get_max_bound(), pcd.get_min_bound()))

            if i == 0:
                bounds = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.array([-0.4, -0.8, 0.0]),
                    max_bound=np.array([0.4, 0.5, 0.8]),
                )
            elif i == 1:
                bounds = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.array([-0.4, -0.8, 0.0]),
                    max_bound=np.array([0.4, 0.0, 1.2]),
                )
            pcd_crop = pcd.crop(bounds)
            pcd_crop_list.append(pcd_crop)

        # visualize(pcd_crop_list)

        pcd_down = pcd_crop_list
        pcd_down = [pcd.voxel_down_sample(voxel_size=self.voxel_size) for pcd in pcd_list]

        for pcd in pcd_down:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * 2, max_nn=30))

        # init_transform_path = './calib/stereo_calib_azure_cpp/Nov_21_combined/overhead_to_front_right_2.tf'
        init_transform_path = './calib/stereo_calib_azure_cpp/main_overhead_sec_front_left/front_left_to_overhead_4.tf'
        # init_transform_path = './calib/stereo_calib_azure_cpp/main_front_left_sec_back_left/back_left_to_front_left_5.tf'
        # init_transform_path = './calib/stereo_calib_azure_cpp/main_front_right_sec_back_right/back_right_to_front_right.tf'

        init_transform = RigidTransform.load(init_transform_path)

        reg_result = o3d.registration.registration_colored_icp(
            pcd_down[0],
            pcd_down[1],
            0.01,
            init_transform.matrix,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=1000)
        )
        print("Result: \n{}, \ntransf: \n{}".format(
            reg_result, reg_result.transformation))
        import pdb; pdb.set_trace()

    def register_multiple_pcd_2(self, pcd_list):
        for pcd in pcd_list:
            print("PCD min_bound: {}, max_bound: {}".format(
                pcd.get_max_bound(), pcd.get_min_bound()))
        pcds_down = [pcd.voxel_down_sample(voxel_size=self.voxel_size) for pcd in pcd_list]

        for pcd in pcds_down:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            o3d.visualization.draw_geometries([pcd])

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
            reference_node=1)
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
        # pcl_arr = point_cloud_to_mat(data)
        pcl_arr, rgb_arr = point_cloud_to_color_arr(data)
        T = self.transform_by_camera[camera_key]
        transf_pcl_arr = np.dot(T.matrix, pcl_arr.T)
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
        print("Will save point cloud for camera: {}".format(camera_key))
        pcd = self.get_transform_point_cloud_from_raw_data(data, camera_key)
        print("back_right bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
        # pcd = pcd.crop(self.pcl_volume_min_bound, self.pcl_volume_max_bound)
        # if self.pcl_volume_min_bound_dict.get(camera_key) is not None and \
        #     self.pcl_volume_max_bound_dict.get(camera_key) is not None:
        #     pcd = pcd.crop(self.pcl_volume_min_bound_dict[camera_key], 
        #                    self.pcl_volume_max_bound_dict[camera_key])
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

class PCLColorRegistration(PCLRegistrationUtils):

    def __init__(self):
        super(PCLColorRegistration, self).__init__()

    def draw_registration_result_original_color(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target])

    def pairwise_registration(self, source, target):
        # draw initial alignment
        current_transformation = np.identity(4)
        self.draw_registration_result_original_color(source, target,
                                                current_transformation)

        # point to plane ICP
        current_transformation = np.identity(4)
        print("2. Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. Distance threshold 0.02.")
        result_icp = o3d.registration.registration_icp(
            source, target, 0.02, current_transformation,
            o3d.registration.TransformationEstimationPointToPlane())
        print(result_icp)
        self.draw_registration_result_original_color(source, target,
                                                result_icp.transformation)
        
        # colored pointcloud registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]
        current_transformation = np.identity(4)
        print("3. Colored point cloud registration")
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            print("3-3. Applying colored point cloud registration")
            result_icp = o3d.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter))
            current_transformation = result_icp.transformation
            print(result_icp)
        self.draw_registration_result_original_color(
            source, target, result_icp.transformation)

        transformation_icp = result_icp.transformation
        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            transformation_icp)

        return transformation_icp, information_icp

def main():
    rospy.init_node('test_k4a_pcl', anonymous=True)
    rospy.loginfo('Test k4a pcl!!')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcl_reg = PCLColorRegistration()
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
            # pcd_list = [
            #     pcl_reg.point_cloud_by_camera_dict['front_left'],
            #     pcl_reg.point_cloud_by_camera_dict['overhead'],
            #     # pcl_reg.point_cloud_by_camera_dict['front_right'],
            # ]
            rospy.loginfo("Will try to register multiple point clouds.")
            # pcl_reg.register_multiple_pcd(pcd_list)

            target_pcd = pcl_reg.point_cloud_by_camera_dict['overhead']
            source_pcd = pcl_reg.point_cloud_by_camera_dict['front_left']
            pcl_reg.register_pcd_pair(target_pcd, source_pcd)
            break
        
        # if len(pcl_reg.point_cloud_by_camera_dict) == 1:
        #     print("Did save some data")
        #     break

        rospy.sleep(1.0)

        # rospy.spin()


if __name__ == '__main__':
    main()
