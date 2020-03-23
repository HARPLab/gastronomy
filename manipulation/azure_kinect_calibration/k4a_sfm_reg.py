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
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header
from aruco_board.msg import MarkerImageInfo

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


def invert_transform_matrix(T_ab):
    assert T_ab.shape == (4, 4)
    R_ab, t_ab = T_ab[:3, :3], T_ab[:3, 3:]
    T_ba = np.hstack([R_ab.T, -np.dot(R_ab.T, t_ab)])
    T_ba = np.vstack([T_ba, np.array([[0, 0, 0, 1]])])
    return T_ba


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

def get_intrinsics_for_dir(d):
    assert os.path.exists(d), 'Dir does not exist: {}'.format(d)
    intr_path = os.path.join(d, 'ir_intrinsics_azure_kinect.intr')
    import yaml
    data = yaml.load(open(intr_path, 'r'))
    K = np.array([
        [data['_fx'],   0,              data['_cx']],
        [0,             data['_fy'],    data['_cy']],
        [0,             0,              1],
    ])
    return K


class PCLRegistrationUtils(object):
    def __init__(self): 
        self.bridge = CvBridge()
        self.calib_dirs = { 
            'front_right': './calib/robot_middle/azure_kinect_front_right',
            'front_left': './calib/robot_middle/azure_kinect_front_left',
            'overhead': './calib/robot_middle/azure_kinect_overhead',
        }
        self.transform_by_camera = {}    
        self.intr_by_camera = {}
        for k, v in self.calib_dirs.items():
            T = RigidTransform.load(get_transform_file_for_dir(v))
            self.transform_by_camera[k] = T
            # transform_by_camera[k] = RigidTransform()
            K = get_intrinsics_for_dir(v)        
            self.intr_by_camera[k] = K
        
        self.dist_coeff = {
            'overhead': [0.5801, -2.8618, 0.000896, -0.000813, 1.703311, 0.4574550, 
                        -2.681080, 1.6250559],
            'front_left': [0.678091, -2.7722103, 0.0005995, -0.000470, 1.51647, 
                           0.558341, -2.606752, 1.451771],
        }

        self.point_cloud_by_camera_dict = {}
        self.has_pcd_by_camera_dict = {}

        self.raw_img_by_camera_dict = {}
        self.camera_info_by_camera_dict = {}
        self.keypoint_by_camera_dict = {}

        self.final_data_dict = {
            'img1': [],
            'img2': [],
            'kp1': [],
            'kp2': [],
            'matches': [],
        }

        self.pcl_volume_min_bound = np.array([0.1, -0.4, -0.05])
        self.pcl_volume_max_bound = np.array([0.8, 0.4, 0.5])

        self.voxel_size = 0.01
        self.max_correspondence_distance_coarse = self.voxel_size * 15
        self.max_correspondence_distance_fine = self.voxel_size * 1.5

        self.is_vis = False


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
                pcd.get_max_bound(), pcd.get_min_bound()))
        voxel_size = self.voxel_size
        voxel_grids = []
        for p in pcd_list:
            v = o3d.geometry.VoxelGrid.create_from_point_cloud(p, voxel_size)
            voxel_grids.append(v)
        
        '''
        for voxel_grid in voxel_grids:
            rospy.loginfo("old voxel grid size: {}".format(len(voxel_grid.voxels)))
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
        if self.is_vis:
            return

        pcd = self.get_transform_point_cloud_from_raw_data(data, camera_key)
        print("back_right bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
        pcd = pcd.crop(self.pcl_volume_min_bound, self.pcl_volume_max_bound)
        print("back_right bounds AFTER crop: => {}".format(get_pcd_bounds_str(pcd)))
        self.point_cloud_by_camera_dict[camera_key] = pcd
        self.has_pcd_by_camera_dict[camera_key] = True
        rospy.loginfo("Did get camera data {}".format(camera_key))
        idx = max_test_overhead_idx() + 1
        pcd_path = os.path.join('./register_{}_data_{}.pcd'.format(camera_key, idx))
        o3d.io.write_point_cloud(pcd_path, pcd)
        rospy.loginfo("Did save pcd: {}".format(pcd_path))
    
    def save_img_with_camera_key(self, data, camera_key):
        if self.is_vis:
            return
        if self.raw_img_by_camera_dict.get(camera_key) is not None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        print(cv_image.shape)
        self.raw_img_by_camera_dict[camera_key] = cv_image
    
    def save_keypoint_with_camera_key(self, data, camera_key):
        if self.is_vis:
            return

        # data is message of type MarkerInfo ?
        if self.keypoint_by_camera_dict.get(camera_key) is None:
            self.keypoint_by_camera_dict[camera_key] = {}
        kp_by_camera = self.keypoint_by_camera_dict[camera_key]
        if kp_by_camera.get(data.id) is None:
            kp_by_camera[data.id] = {'c0': [], 'c1': [], 'mid': [], 'center': []}
        c0 = (data.c0_x, data.c0_y)
        c1 = (data.c1_x, data.c1_y)
        center = (data.center_x, data.center_y)
        mid = ((c0[0]+c1[0])/2.0, (c0[1]+c1[1])/2.0)
        kp_by_camera[data.id]['c0'].append(c0)
        kp_by_camera[data.id]['c1'].append(c1)
        kp_by_camera[data.id]['mid'].append(mid)
        kp_by_camera[data.id]['center'].append(center)

        rospy.loginfo("Did save keypoint for camera: {}".format(camera_key))

    def back_right_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'front_right')

    def front_left_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'front_left')

    def overhead_pcl_callback(self, data):
        self.save_pcd_with_camera_key(data, 'overhead')
    
    def get_img_callback(self, camera_key):
        def _f(data):
            self.save_img_with_camera_key(data, camera_key)
        return _f

    def get_keypoint_callback(self, camera_key):
        def _f(data):
            self.save_keypoint_with_camera_key(data, camera_key)
        return _f
    
    def visualize_and_check_data_for_final_processing(self, cam1, cam2):
        self.is_vis = True

        assert len(self.raw_img_by_camera_dict) == 2
        assert len(self.keypoint_by_camera_dict) == 2

        def _get_mean_for_kp_list(l):
            l_arr = np.array(l)
            if np.all(np.std(l_arr, axis=0) < 200):
                pt = np.mean(l_arr, axis=0)
                cv_pt = cv2.KeyPoint(pt[0], pt[1], 2)
                return True, pt, cv_pt
            else:
                return False, None, None

        cam1_kp_list, cam2_kp_list = [], []
        cam1_cv_kp_list, cam2_cv_kp_list = [], []
        for kp in self.keypoint_by_camera_dict[cam1].keys():
            kp1 = self.keypoint_by_camera_dict[cam1][kp]
            kp2 = self.keypoint_by_camera_dict[cam2].get(kp)
            if kp1 is not None and kp2 is not None:
                kp1_status, kp1_val, cv_kp1 = _get_mean_for_kp_list(kp1['c0'])
                kp2_status, kp2_val, cv_kp2 = _get_mean_for_kp_list(kp2['c0'])
                if kp1_status and kp2_status:
                    cam1_kp_list.append(kp1_val)
                    cam2_kp_list.append(kp2_val)
                    # cam1_cv_kp_list.append(cv_kp1)
                    # cam2_cv_kp_list.append(cv_kp2)


                kp1_status, kp1_val, cv_kp1 = _get_mean_for_kp_list(kp1['c1'])
                kp2_status, kp2_val, cv_kp2_= _get_mean_for_kp_list(kp2['c1'])
                if kp1_status and kp2_status:
                    cam1_kp_list.append(kp1_val)
                    cam2_kp_list.append(kp2_val)
                    # cam1_cv_kp_list.append(cv_kp1)
                    # cam2_cv_kp_list.append(cv_kp2)

                kp1_status, kp1_val, cv_kp1 = _get_mean_for_kp_list(kp1['center'])
                kp2_status, kp2_val, cv_kp2 = _get_mean_for_kp_list(kp2['center'])
                if kp1_status and kp2_status:
                    cam1_kp_list.append(kp1_val)
                    cam2_kp_list.append(kp2_val)
                    cam1_cv_kp_list.append(cv_kp1)
                    cam2_cv_kp_list.append(cv_kp2)
                else:
                    rospy.loginfo("Skip kp: {}".format(kp))

        out_img1 = np.zeros((1536, 2048, 3))
        matches = [cv2.DMatch(i, i, 2.0) for i in range(len(cam1_cv_kp_list))]
        out_img = cv2.drawMatches(
            self.raw_img_by_camera_dict[cam1],
            cam1_cv_kp_list,
            self.raw_img_by_camera_dict[cam2],
            cam2_cv_kp_list,
            matches,
            None
        )
        cv2.destroyAllWindows()
        cv2.imwrite('./matches.png', out_img)
        accept = True
        finish = False
        while not finish:
            rospy.loginfo("Press y for accept and 'n' for Reject")
            cv2.imshow('matches', out_img)
            key = cv2.waitKey(0)
            if key == ord('y'):
                print("Accept image")
                accept = True
                finish = True
            elif key == ord('n'):
                print("Reject image")
                accept = False
                finish = True
            elif key == 27:
                print("Reject image")
                accept = False
                finish = True

        cv2.destroyAllWindows()
        rospy.loginfo("Keypoint status Accept: {}".format(accept))
        return accept, {
            'img1': self.raw_img_by_camera_dict[cam1],
            'img2': self.raw_img_by_camera_dict[cam2],
            'kp1': cam1_cv_kp_list,
            'kp2': cam2_cv_kp_list,
            'matches': matches,
        }


def get_sift_features(img1, img2):
    sift = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1.astype(np.float32),
                             des2.astype(np.float32),
                             k=2)
    rospy.loginfo("Found ORB matches: {}".format(len(matches)))
    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if matches:
        out_img1 = np.zeros((1536, 2048, 3))
        out_img = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good,
            out_img1
        )
        cv2.imwrite('./sift_matches.png', out_img)
        accept = True
        finish = False
        cv2.destroyAllWindows()
        while not finish:
            cv2.imshow('sift_matches', out_img)
            key = cv2.waitKey(0)
            if key == 27:
                finish = True
    
    return pts1, pts2

def triangulate_points(pt_1, pt_2, K1, K2, dist_coeff1, dist_coeff2, P, P1):
    corresp_img1_pt = []

    P1_ = np.array([
        [P1[0, 0], P1[0, 1], P1[0, 2], P1[0, 3]],
        [P1[1, 0], P1[1, 1], P1[1, 2], P1[1, 3]],
        [P1[2, 0], P1[2, 1], P1[2, 2], P1[2, 3]],
        [0,        0,        0,        1]
    ])
    P1_inv = np.linalg.inv(P1_)

    rospy.loginfo("Triangulating")

    pt_1 = np.array([kp.pt for kp in pt_1])
    pt_2 = np.array([kp.pt for kp in pt_2])
    pt_size = len(pt_1)

    pt_1_undistort = cv2.undistortPoints(pt_1, K1, np.array(dist_coeff1))
    pt_2_undistort = cv2.undistortPoints(pt_2, K2, np.array(dist_coeff2))

    pt_3d_h = cv2.triangulatePoints(P, P1, pt_1_undistort.T, pt_2_undistort.T)
    # Conert points to homogeneous
    pt_3d = cv2.convertPointsFromHomogeneous(np.copy(pt_3d_h.T))
    pt_3d = np.squeeze(pt_3d)

    # Reproject the 3d points back to 2d.
    R = P[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = [P[0, 3], P[1, 3], P[2, 3]]
    reproj_pt_set_1, _ = cv2.projectPoints(
        pt_3d.reshape(-1, 3), rvec, np.array(tvec), K1, np.array(dist_coeff1))

    point_cloud, reproj_err = [], []
    for i in range(pt_size):
        point_cloud.append(pt_3d[i])
        reproj_err.append(np.linalg.norm(pt_1[i] - reproj_pt_set_1[i]))
    
    point_cloud_arr = np.array(point_cloud)
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud_arr[:, 0], point_cloud_arr[:, 1], point_cloud_arr[:, 2])
    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_xlabel('Z')
    plt.show()

    import pdb; pdb.set_trace()
    
    rospy.loginfo("Reproj error: {:.6f}".format(np.mean(reproj_err)))
    return np.mean(reproj_err), point_cloud

def test_triangulation(p_cloud, P):
    p_cloud_3d = np.copy(p_cloud)
    P4x4 = np.eye(4)
    P4x4[:3, :4] = P
    
    p_cloud_3d_proj = cv2.perspectiveTransform(p_cloud_3d[None, :], P4x4)

    status = [0]*len(p_cloud)
    for i in range(len(p_cloud)):
        status[i] = 1 if (p_cloud_3d_proj[0, i, 2] > 0) else 0

    status_nnz_count = np.sum(status > 0)
    percent = status_nnz_count/len(p_cloud)
    if percent < 0.75:
        rospy.loginfo("Less than 75\% of points infront of camera: {:.4f}".format(
            percent))
        return False, None

    return True, status

def draw_epipolar_lines_helper(img1, img2, lines, pts1, pts2):
    """Helper method to draw epipolar lines and features """
    if img1.shape[2] == 1:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.shape[2] == 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    c = img1.shape[1]
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1), 5, color, -1)
        cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Notes: http://www.maths.lth.se/matematiklth/personal/calle/datorseende13/notes/forelas6.pdf
# Code taken from
# https://github.com/MasteringOpenCV/code/blob/master/Chapter4_StructureFromMotion/FindCameraMatrices.cpp
def get_fundamental_matrix(img_pts_1, img_pts_2, img_1, img_2):
    img_pts_1_good, img_pts_2_good = [], []
    kp1_good, kp2_good = [], []

    pts_1 = np.array([kp.pt for kp in img_pts_1])
    pts_2 = np.array([kp.pt for kp in img_pts_2])

    # Normalize between 0 and 1?
    normalize = False
    if normalize:
        img_shape = np.array(img_1.shape)
        pts_1_norm = pts_1 / img_shape[:2]
        pts_2_norm = pts_2 / img_shape[:2]
        max_val = np.max(pts_1_norm)

        F, status = cv2.findFundamentalMat(
            pts_1_norm, pts_2_norm, cv2.FM_RANSAC, 0.006 * max_val, 0.99) # threshold from [Snavely07 4.1]
    else:
        max_val = np.max(pts_1)
        F, status = cv2.findFundamentalMat(
            pts_1, pts_2, cv2.FM_RANSAC, 0.006 * max_val, 0.99) # threshold from [Snavely07 4.1]

    rospy.loginfo("Inliers: {}".format(status.tolist()))
    rospy.loginfo("Calc F:\n{}".format(np.sum(np.array(status) > 0)/len(status)))
    rospy.loginfo("Estimated F: \n{}".format(
        np.array_str(F, precision=6)))

    matches = [i for i,s in enumerate(status) if s]
    for i, s in enumerate(status):
        if s:
            img_pts_1_good.append(pts_1[i])
            img_pts_2_good.append(pts_2[i])
            kp1_good.append(img_pts_1[i])
            kp2_good.append(img_pts_2[i])
    # TODO: Should draw matches

    img_pts_1_good_arr = np.array(img_pts_1_good, dtype=np.int32)
    img_pts_2_good_arr = np.array(img_pts_2_good, dtype=np.int32)

    pts2re = np.array(img_pts_2_good).reshape(-1, 1, 2)
    lines1 = cv2.computeCorrespondEpilines(pts2re, 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_epipolar_lines_helper(
        img_1, img_2, lines1, img_pts_1_good_arr, img_pts_2_good_arr)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    pts1re = np.array(img_pts_1_good).reshape(-1, 1, 2)
    lines2 = cv2.computeCorrespondEpilines(pts1re, 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_epipolar_lines_helper(
        img_2, img_1, lines2, img_pts_2_good_arr, img_pts_1_good_arr)

    import matplotlib.pyplot as plt
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

    return F, matches, img_pts_1_good, img_pts_2_good, kp1_good, kp2_good

def decompose_E_to_R_and_T(E):
    U, W, Vh = np.linalg.svd(E)
    # SVD result
    rospy.loginfo("SVD of E (U) =>\n{}".format(np.array_str(U, precision=6)))
    rospy.loginfo("SVD of E (V) =>\n{}".format(np.array_str(Vh, precision=6)))
    rospy.loginfo("SVD of E (sing_values) =>\n{}".format(np.array_str(W, precision=6)))

    # check if first and second singular values are the same (as they should be)
    singular_val_ratio = W[0] / W[1]
    if singular_val_ratio > 1.0:
        # Flip to keep between 0, 1
        singular_val_ratio = 1.0/singular_val_ratio

    if singular_val_ratio < 0.7:
		raise ValueError("singular values are too far apart: {:.6f}".format(
            singular_val_ratio))
    
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    Wt = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])
    assert np.all(W.T == Wt)

    # Read: https://en.wikipedia.org/wiki/Essential_matrix
    # 4 solutions are possible. However, 3 solutions will have atleast all points behind
    # one of the cameras.
    R1 = np.matmul(U, np.matmul(W,  Vh))
    R2 = np.matmul(U, np.matmul(Wt, Vh))
    # Choose \hat{t} as the right singular vector of E
    t1 = U[:, 2:]
    t2 = -U[:, 2:]

    return R1, R2, t1, t2

def check_coherent_rotation_mat(R):
    if abs(np.linalg.det(R)) - 1.0 > 1e-6:
        return False
    return True

def find_camera_matrices(K1, K2, dist_coeff1, dist_coeff2, img_pts_1, img_pts_2, P, 
                         matches, img_1, img_2):
    import time
    start_time = time.time()

    F, matches, img_pts_1_good, img_pts_2_good, kp1_good, kp2_good = get_fundamental_matrix(
        img_pts_1, img_pts_2, img_1, img_2)

    if len(matches) < 10:
        raise ValueError("Not enough inliers after F matrix.")
        return False
    
    # Essential matrix
    E = np.matmul(K2.T,  np.matmul(F,  K1))
        
    if abs(np.linalg.det(E)) > 1e-6:
        raise ValueError("Det of E != 0")
        return False
    
    R1, R2, t1, t2 = decompose_E_to_R_and_T(E)
    if np.linalg.det(R1) + 1.0 < 1e-8:
        rospy.loginfo("det(R) == -1. Flip E's sign")
        E = -E
        R1, R2, t1, t2 = decompose_E_to_R_and_T(E)
    
    if not check_coherent_rotation_mat(R1):
        raise ValueError("Resulting rotation is incoherent")
    
    P1 = np.hstack([R1, t1])
    assert P.shape == P1.shape

    rospy.loginfo("Test triangulation with: \n{}".format(
        np.array_str(P1, precision=6, suppress_small=True)))

    # TODO: Test Triangulation
    reproj_error1, p_cloud = triangulate_points(
        kp1_good, kp2_good, K1, K2, dist_coeff1, dist_coeff2, P, P1)
    reproj_error2, p_cloud1 = triangulate_points(
        kp2_good, kp1_good, K2, K1, dist_coeff2, dist_coeff1, P1, P )
    
    if not test_triangulation(p_cloud, P1)[0] or not test_triangulation(p_cloud1, P) \
        or reproj_error1 > 100.0 or reproj_error2 > 100.0:
        P1 = np.hstack([R2, t2])
        rospy.loginfo("Test triangulation with: \n{}".format(
            np.array_str(P1, precision=6, suppress_small=True)))

        reproj_error1, p_cloud = triangulate_points(
            kp1_good, kp2_good, K1, K2, dist_coeff1, dist_coeff2, P, P1)
        reproj_error2, p_cloud1 = triangulate_points(
            kp2_good, kp1_good, K2, K1, dist_coeff2, dist_coeff1, P1, P)
        
        if not test_triangulation(p_cloud, P1) or not test_triangulation(p_cloud1, P) \
            or reproj_error1 > 100.0 or reproj_error2 > 100.0:
            if not check_coherent_rotation_mat(R2):
                raise ValueError("resulting rotation matrix is not coherent")

            P1 = np.hstack([R2, t1])
            rospy.loginfo("Test triangulation with: \n{}".format(
                np.array_str(P1, precision=6, suppress_small=True)))
        
            reproj_error1, p_cloud = triangulate_points(
                kp1_good, kp2_good, K1, K2, dist_coeff1, dist_coeff2, P, P1)
            reproj_error2, p_cloud1 = triangulate_points(
                kp2_good, kp1_good, K2, K1, dist_coeff2, dist_coeff1, P1, P)
            if not test_triangulation(p_cloud, P1) or not test_triangulation(p_cloud1, P) \
                or reproj_error1 > 100.0 or reproj_error2 > 100.0:
                P1 = np.hstack([R2, t2])
                rospy.loginfo("Test triangulation with: \n{}".format(
                    np.array_str(P1, precision=6, suppress_small=True)))

                reproj_error1, p_cloud = triangulate_points(
                    kp1_good, kp2_good, K1, K2, dist_coeff1, dist_coeff2, P, P1)
                reproj_error2, p_cloud1 = triangulate_points(
                    kp2_good, kp1_good, K2, K1, dist_coeff2, dist_coeff1, P1, P)
                if not test_triangulation(p_cloud, P1) or not test_triangulation(p_cloud1, P) \
                    or reproj_error1 > 100.0 or reproj_error2 > 100.0:
                    raise ValueError("Did not find camera matrices. BAD!!")

    rospy.loginfo("Found P1: \n{}".format(np.array_str(P1, precision=6)))
    return P1



def main():
    rospy.init_node('test_k4a_pcl', anonymous=True)
    rospy.loginfo('Test k4a pcl!!')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcl_reg = PCLRegistrationUtils()
    rospy.loginfo("Did load transform for camers: {}".format(len(pcl_reg.transform_by_camera)))

    pc_topics_by_callback = [
        # ('/points2', pcl_callback),
        ('/back_right_kinect/points2', pcl_reg.back_right_pcl_callback),
        ('/front_left_kinect/points2', pcl_reg.front_left_pcl_callback),
        ('/overhead_kinect/points2', pcl_reg.overhead_pcl_callback),
    ]
    # pc_topic = "/points2"
    # pc_sub = rospy.Subscriber(pc_topic, PointCloud2, pcl_callback)
    # for p in pc_topics_by_callback:
    #     pc_sub = rospy.Subscriber(p[0], PointCloud2, p[1])

    img_info_topics_by_callback = [
        # ('/points2', pcl_callback),
        ('/back_right_kinect/rgb/image_raw', pcl_reg.get_img_callback('back_right')),
        ('/front_left_kinect/rgb/image_raw', pcl_reg.get_img_callback('front_left')),
        ('/overhead_kinect/rgb/image_raw', pcl_reg.get_img_callback('overhead')),
    ]
    # pc_topic = "/points2"
    # pc_sub = rospy.Subscriber(pc_topic, PointCloud2, pcl_callback)
    for p in img_info_topics_by_callback:
        img_sub = rospy.Subscriber(p[0], Image, p[1])
    
    marker_info_topics_by_callback = [
        ('/aruco_board_overhead/marker_info', pcl_reg.get_keypoint_callback('overhead')),
        ('/aruco_board_front_left/marker_info', pcl_reg.get_keypoint_callback('front_left')),
    ]
    for p in marker_info_topics_by_callback:
        kp_sub = rospy.Subscriber(p[0], MarkerImageInfo, p[1])

    while not rospy.is_shutdown():
        rospy.loginfo("point_cloud_dict size".format(len(pcl_reg.point_cloud_by_camera_dict)))
        cam1_key = 'front_left'
        cam2_key = 'overhead'
        if len(pcl_reg.raw_img_by_camera_dict) == 2 and \
           len(pcl_reg.keypoint_by_camera_dict) == 2 and \
           len(pcl_reg.keypoint_by_camera_dict[cam1_key]) > 20 and \
           len(pcl_reg.keypoint_by_camera_dict[cam2_key]) > 20:
            rospy.sleep(1.0)
            status, result_dict = pcl_reg.visualize_and_check_data_for_final_processing(
                cam1_key, cam2_key)
            
            # sift_pts1, sift_pts2 = get_sift_features(
            #     result_dict['img1'], 
            #     result_dict['img2'])

            # Calculate approx transformation between the two cameras
            T_base_cam2 = pcl_reg.transform_by_camera[cam2_key].matrix
            T_base_cam1 = pcl_reg.transform_by_camera[cam1_key].matrix
            T_cam1_base = invert_transform_matrix(T_base_cam1)
            T_cam1_cam2 = np.dot(T_cam1_base, T_base_cam2)
            T_cam2_cam1 = invert_transform_matrix(T_cam1_cam2)

            rospy.loginfo("Approx tranf from cam2 to cam1 => \n{}".format(
                np.array_str(T_cam1_cam2, precision=6, suppress_small=True)))
            rospy.loginfo("Approx tranf from cam1 to cam2 => \n{}".format(
                np.array_str(T_cam2_cam1, precision=6, suppress_small=True)))
            pcl_reg.T_cam1_cam2 = T_cam1_cam2
            pcl_reg.T_cam2_cam1 = T_cam2_cam1
            
            if status:
                pcl_reg.final_data_dict['img1'].append(result_dict['img1'])
                pcl_reg.final_data_dict['img2'].append(result_dict['img2'])
                pcl_reg.final_data_dict['kp1'] += result_dict['kp1']
                pcl_reg.final_data_dict['kp2'] += result_dict['kp2']
                pcl_reg.final_data_dict['matches'] += result_dict['matches']

                P = np.hstack([np.eye(3), np.zeros((3, 1))])
                F = find_camera_matrices(
                    pcl_reg.intr_by_camera[cam1_key], 
                    pcl_reg.intr_by_camera[cam2_key], 
                    pcl_reg.dist_coeff[cam1_key], 
                    pcl_reg.dist_coeff[cam2_key], 
                    result_dict['kp1'], 
                    result_dict['kp2'], 
                    P,
                    result_dict['matches'], 
                    result_dict['img1'], 
                    result_dict['img2'])
            else:
                pcl_reg.raw_img_by_camera_dict = {}
                pcl_reg.keypoint_by_camera_dict = {}
                pcl_reg.is_vis = False
        
        # if len(pcl_reg.point_cloud_by_camera_dict) == 1:
        #     print("Did save some data")
        #     break

        rospy.sleep(1.0)

        # rospy.spin()


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    main()
