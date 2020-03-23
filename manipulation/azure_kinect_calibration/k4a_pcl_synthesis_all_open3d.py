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
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header

from autolab_core import RigidTransform
from perception import CameraIntrinsics

try:
    import cv2
    import pylibfreenect2 as lf2
except:
    logging.warning('Unable to import pylibfreenect2. Python-only Kinect driver may not work properly.')

try:
    import rospy
    from cv_bridge import CvBridge, CvBridgeError
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


def get_intr_file_for_dir(d):
    assert os.path.exists(d), 'Dir does not exist: {}'.format(d)
    # if 'overhead' in d:
    #     return os.path.join(d, 'depth_to_rgb_pinhole_720p.intr')

    return os.path.join(d, 'depth_to_rgb_720p.intr')

class PCLRegistrationUtils(object):
    def __init__(self): 
        # This is not correct
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
            T.from_frame = k
            T.to_frame = 'world'
            self.transform_by_camera[k] = T
        
        self.intr_calib_dirs = {
            'back_left': './calib/robot_middle/azure_kinect_back_left',
            'front_left': './calib/robot_middle/azure_kinect_front_left',
            'overhead': './calib/robot_middle/azure_kinect_overhead',
            'front_right': './calib/robot_middle/azure_kinect_front_right',
            'back_right': './calib/robot_middle/azure_kinect_back_right',
        }
        self.intr_by_camera = {}
        for k, v in self.intr_calib_dirs.items():
            ir_intrinsics = CameraIntrinsics.load(get_intr_file_for_dir(v))
            self.intr_by_camera[k] = ir_intrinsics
        
        self.stereo_calib_data = [
            # (from, to, path) <=> (main_camera, secondary_camera, 'calib_data')

            # ('front_right', 'overhead', './calib/stereo_calib_azure_cpp/Nov_19_try_3/overhead_to_front_right_x.tf')
            # ('front_right', 'overhead', './calib/stereo_calib_azure_cpp/Nov_20_matlab/overhead_to_front_right_3.tf')

            # do the inverse of this one?
            ('overhead', 'front_right', './calib/stereo_calib_azure_cpp/Nov_21_combined/overhead_to_front_right_3.tf'),

            # Transf between front_left and back_left camera 
            ('overhead', 'front_left', './calib/stereo_calib_azure_cpp/main_overhead_sec_front_left/front_left_to_overhead_5.tf'),

            # Transf between front_left and back_left camera 
            ('front_left', 'back_left', './calib/stereo_calib_azure_cpp/main_front_left_sec_back_left/back_left_to_front_left_7.tf'),

            # Transf between front_right and back_right camera 
            ('front_right', 'back_right', './calib/stereo_calib_azure_cpp/main_front_right_sec_back_right/back_right_to_front_right_2.tf')
        ]
        self.stereo_calib_dict = {}
        self.extr_by_camera = {'overhead': np.eye(4)}
        for s in self.stereo_calib_data:
            T = RigidTransform.load(s[2])
            T.to_frame = s[0]
            T.from_frame = s[1]
            self.stereo_calib_dict['{}${}'.format(s[1], s[0])] = T
            rospy.loginfo("From {} => {} transf matrix: \n"
                          "        new    i           : \n{}\n".format(
                s[0], s[1],
                np.array_str(T.matrix, precision=5, suppress_small=True)
                ))
        
        for s in self.stereo_calib_data:
            T = self.stereo_calib_dict['{}${}'.format(s[1], s[0])]
            if s[1] == 'front_right':
                M = np.copy(T.matrix)
            elif s[1] == 'back_right':
                T1 = self.stereo_calib_dict['back_right$front_right']
                T2 = self.stereo_calib_dict['front_right$overhead']
                M = np.copy((T2.dot(T1)).matrix)
            elif s[1] == 'front_left': 
                M = np.copy(self.stereo_calib_dict['front_left$overhead'].matrix)
            elif s[1] == 'back_left':
                T1 = self.stereo_calib_dict['back_left$front_left']
                T2 = self.stereo_calib_dict['front_left$overhead']
                M = np.copy((T2.dot(T1)).matrix)
            elif s[1] == 'overhead':
                M = np.eye(4)
            else:
                raise ValueError("Invalid camera key")
            self.extr_by_camera[s[1]] = M

        # import pdb; pdb.set_trace()

        self.o3d_camera_params_by_camera = {}
        assert sorted(self.extr_by_camera.keys()) == sorted(self.intr_by_camera.keys())
        for k in self.intr_by_camera.keys():
            extr = self.extr_by_camera[k]
            intr = self.intr_by_camera[k]

            o3d_cam = o3d.camera.PinholeCameraParameters()
            o3d_cam.extrinsic = extr
            o3d_cam.intrinsic = o3d.camera.PinholeCameraIntrinsic()
            o3d_cam.intrinsic.set_intrinsics(
                intr.width,
                intr.height,
                intr.fx,
                intr.fy,
                intr.cx,
                intr.cy    
            )
            self.o3d_camera_params_by_camera[k] = o3d_cam

        self.point_cloud_by_camera_dict = {}
        self.has_pcd_by_camera_dict = {}
        self.depth_img_by_camera_dict = {}
        self.raw_depth_img_by_camera_dict = {}
        self.carved_voxel_grid_by_camera_dict = {}
        self.color_img_by_camera_dict = {}

        # Define the max/min bound in reference to the main camera.
        self.pcl_volume_min_bound = np.array([-0.5, -0.8, 0.1])
        self.pcl_volume_max_bound = np.array([0.5, 0.1, 1.1])

        self.voxel_size = 0.01
        self.max_correspondence_distance_coarse = self.voxel_size * 15
        self.max_correspondence_distance_fine = self.voxel_size * 1.5

        self.bridge = CvBridge()



    def merge_saved_pcd(self):
        for pcd in self.point_cloud_by_camera_dict.values():
            print("PCD min_bound: {}, max_bound: {}".format(
                pcd.get_min_bound(), pcd.get_max_bound()))

        visualize([self.carved_voxel_grid_by_camera_dict['overhead']])

        voxel_size = self.voxel_size

        pcd_list = [
            self.point_cloud_by_camera_dict['back_left'],
            self.point_cloud_by_camera_dict['front_left'],
            self.point_cloud_by_camera_dict['back_right'],
            self.point_cloud_by_camera_dict['front_right'],
            self.point_cloud_by_camera_dict['overhead'],
        ]

        downsample_pcd_list = []
        downsample_pcd_combined = o3d.geometry.PointCloud()
        pcd_combined = o3d.geometry.PointCloud()
        for pcd_idx, pcd in enumerate(pcd_list):
            pcd_combined += pcd
            downsample_pcd = pcd.voxel_down_sample(voxel_size=0.01)
            downsample_pcd_combined += downsample_pcd
            downsample_pcd_list.append(pcd)

        o3d.io.write_point_cloud('/home/klz/datasets/ms_point_cloud_data/try_0_downsample.pcd', downsample_pcd_combined)
        o3d.io.write_point_cloud('/home/klz/datasets/ms_point_cloud_data/try_0.pcd', pcd_combined)
        # import pdb; pdb.set_trace()

        # visualize(downsample_pcd_list)
        # visualize(downsample_pcd_list[:2])
        # visualize(downsample_pcd_list[2:4])
        # visualize(downsample_pcd_list)

        voxel_grids = []
        for p in downsample_pcd_list:
            v = o3d.geometry.VoxelGrid.create_from_point_cloud(p, 0.01)
            voxel_grids.append(v)
        visualize(voxel_grids[:2])
        
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

        # visualize(voxel_grids)
        # import pdb; pdb.set_trace()

        # Find which voxels are occupied
        all_voxel_map = np.zeros((500, 500, 500), dtype=np.int32)
        for voxel_grid in voxel_grids[:2]:
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

            visualize([voxel_grid])        

        visualize(voxel_grids)


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
        if camera_key == 'front_right':
            # transf_pcl_arr = np.dot(self.T_overhead_front_right.matrix, pcl_arr.T)
            M = self.stereo_calib_dict['front_right$overhead'].matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'back_right':
            T1 = self.stereo_calib_dict['back_right$front_right']
            T2 = self.stereo_calib_dict['front_right$overhead']
            M = (T2.dot(T1)).matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'front_left': 
            M = self.stereo_calib_dict['front_left$overhead'].matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'back_left':
            T1 = self.stereo_calib_dict['back_left$front_left']
            T2 = self.stereo_calib_dict['front_left$overhead']
            M = (T2.dot(T1)).matrix
            transf_pcl_arr = np.dot(M, pcl_arr.T)
        elif camera_key == 'overhead':
            transf_pcl_arr = np.dot(np.eye(4), pcl_arr.T)
        else:
            raise ValueError("Invalid camera key")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transf_pcl_arr[:3, :].T)
        pcd.colors = o3d.utility.Vector3dVector(rgb_arr[:, :3]/255.0)
        return pcd

    def save_pcd_with_camera_key(self, data, camera_key):
        if self.has_pcd_by_camera_dict.get(camera_key):
            return
        print("Will save pcd for camera: {}".format(camera_key))
        pcd = self.get_transform_point_cloud_from_raw_data(data, camera_key)

        print("{} bounds BEFORE crop: => {}".format(camera_key, get_pcd_bounds_str(pcd)))
        pcd = pcd.crop(self.pcl_volume_min_bound, self.pcl_volume_max_bound)
        print("{} bounds AFTER crop: => {}".format(camera_key, get_pcd_bounds_str(pcd)))

        self.point_cloud_by_camera_dict[camera_key] = pcd
        self.has_pcd_by_camera_dict[camera_key] = True
        rospy.loginfo("Did get camera data {}".format(camera_key))
        idx = max_test_overhead_idx() + 1
        pcd_path = os.path.join('./pcd_data/register_{}_data_{}.pcd'.format(camera_key, idx))
        o3d.io.write_point_cloud(pcd_path, pcd)
        rospy.loginfo("Did save pcd: {}".format(pcd_path))

    def save_raw_depth_with_camera_key(self, data, camera_key):
        if self.raw_depth_img_by_camera_dict.get(camera_key) is not None:
            return
        print("Will save depth for camera: {}".format(camera_key))

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, '32FC1')
        except CvBridgeError as e:
            print("Could not convert raw depth image: {}".format(e))

        self.raw_depth_img_by_camera_dict[camera_key] = cv_image
        # "32F1" gives value in m?  and OpenCV bitmap only store integer values.
        cv_img_to_save = np.array(cv_image, dtype=np.float32)
        cv2.normalize(cv_img_to_save, cv_img_to_save, 0, 1, cv2.NORM_MINMAX)
        cv_img_to_save = cv_img_to_save * 255
        cv2.imwrite('/home/klz/datasets/ms_point_cloud_data/depth_rgb/{}_raw_depth.png'.format(
            camera_key), cv_img_to_save * 255)

    def save_depth_with_camera_key(self, data, camera_key):
        if self.depth_img_by_camera_dict.get(camera_key) is not None:
            return
        print("Will save depth for camera: {}".format(camera_key))

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, '32FC1')
        except CvBridgeError as e:
            print("Could not convert depth image: {}".format(e))

        self.depth_img_by_camera_dict[camera_key] = cv_image
        # "32F1" gives value in m?
        cv_img_to_save = np.array(cv_image, dtype=np.float32)
        cv2.normalize(cv_img_to_save, cv_img_to_save, 0, 1, cv2.NORM_MINMAX)
        cv_img_to_save = cv_img_to_save * 255
        cv2.imwrite('/home/klz/datasets/ms_point_cloud_data/depth_rgb/{}_depth.png'.format(
            camera_key), cv_img_to_save)

        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        # cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
        # cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)

        # center = [0., 0., 0.]
        # cubic_size, voxel_size = 2.0, 0.01
        # dense_grid = o3d.geometry.VoxelGrid.create_dense(
        #     center, 0.01, cubic_size, cubic_size, cubic_size)
        # origin = [0.0, 0.0, 0.0 ]
        # dense_grid = o3d.geometry.VoxelGrid.create_dense( 
        #      width=cubic_size, 
        #      height=cubic_size, 
        #      depth=cubic_size, 
        #      voxel_size=voxel_size, # cubic_size / voxel_resolution,
        #      origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0]
        # )
        # visualize([dense_grid])

        
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(
        #     depth=o3d.geometry.Image(cv_image),
        #     intrinsic=self.o3d_camera_params_by_camera[camera_key].intrinsic,
        #     extrinsic=self.o3d_camera_params_by_camera[camera_key].extrinsic,
        #     depth_scale=0.05, 
        #     depth_trunc=1.0,
        # )
        # pcd = pcd.transform(self.extr_by_camera[camera_key])
        # visualize([pcd])
        
        # import pdb; pdb.set_trace()

        # img = o3d.geometry.Image(cv_image)
        # dense_grid = dense_grid.carve_depth_map(img, self.o3d_camera_params_by_camera[camera_key])
        # dense_grid = dense_grid.carve_silhouette(img, self.o3d_camera_params_by_camera[camera_key])
        # self.carved_voxel_grid_by_camera_dict[camera_key] = dense_grid
        # visualize([dense_grid])

    def save_rgb_with_camera_key(self, data, camera_key):
        if self.color_img_by_camera_dict.get(camera_key) is not None:
            return
        print("Will save depth for camera: {}".format(camera_key))

        # Convert image to OpenCV format
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data,'bgr8')
        except CvBridgeError as e:
            print("Could not convert rgb image: {}".format(e))

        self.color_img_by_camera_dict[camera_key] = cv_img
        cv2.imwrite('/home/klz/datasets/ms_point_cloud_data/depth_rgb/{}_color.png'.format(camera_key), cv_img)

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

    def back_right_depth_callback(self, data):
        self.save_depth_with_camera_key(data, 'back_right')

    def front_right_depth_callback(self, data):
        self.save_depth_with_camera_key(data, 'front_right')

    def front_left_depth_callback(self, data):
        self.save_depth_with_camera_key(data, 'front_left')

    def back_left_depth_callback(self, data):
        self.save_depth_with_camera_key(data, 'back_left')

    def overhead_depth_callback(self, data):
        self.save_depth_with_camera_key(data, 'overhead')

    # Raw depth img callback
    def back_right_depth_raw_callback(self, data):
        self.save_raw_depth_with_camera_key(data, 'back_right')

    def front_right_depth_raw_callback(self, data):
        self.save_raw_depth_with_camera_key(data, 'front_right')

    def front_left_depth_raw_callback(self, data):
        self.save_raw_depth_with_camera_key(data, 'front_left')

    def back_left_depth_raw_callback(self, data):
        self.save_raw_depth_with_camera_key(data, 'back_left')

    def overhead_depth_raw_callback(self, data):
        self.save_raw_depth_with_camera_key(data, 'overhead')

    # Raw image callback
    def back_right_rgb_callback(self, data):
        self.save_rgb_with_camera_key(data, 'back_right')

    def front_right_rgb_callback(self, data):
        self.save_rgb_with_camera_key(data, 'front_right')

    def front_left_rgb_callback(self, data):
        self.save_rgb_with_camera_key(data, 'front_left')

    def back_left_rgb_callback(self, data):
        self.save_rgb_with_camera_key(data, 'back_left')

    def overhead_rgb_callback(self, data):
        self.save_rgb_with_camera_key(data, 'overhead')

def main():
    rospy.init_node('test_k4a_pcl', anonymous=True)
    rospy.loginfo('Test k4a pcl!!')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcl_reg = PCLRegistrationUtils()
    rospy.loginfo("Did load transform for camers: {}".format(len(pcl_reg.transform_by_camera)))

    pc_topics_by_callback = [
        # ('/back_right_kinect/points2', pcl_reg.back_right_pcl_callback),
        # ('/front_right_kinect/points2', pcl_reg.front_right_pcl_callback),
        # ('/front_left_kinect/points2', pcl_reg.front_left_pcl_callback),
        # ('/overhead_kinect/points2', pcl_reg.overhead_pcl_callback),
        # ('/back_left_kinect/points2', pcl_reg.back_left_pcl_callback),
    ]
    for p in pc_topics_by_callback:
        pc_sub = rospy.Subscriber(p[0], PointCloud2, p[1])

    depth_map_topics_by_callback = [
        ('/back_right_kinect/depth_to_rgb/image_raw', pcl_reg.back_right_depth_callback),
        ('/front_right_kinect/depth_to_rgb/image_raw', pcl_reg.front_right_depth_callback),
        ('/front_left_kinect/depth_to_rgb/image_raw', pcl_reg.front_left_depth_callback),
        ('/overhead_kinect/depth_to_rgb/image_raw', pcl_reg.overhead_depth_callback),
        ('/back_left_kinect/depth_to_rgb/image_raw', pcl_reg.back_left_depth_callback),
    ]
    for p in depth_map_topics_by_callback:
        rospy.Subscriber(p[0], Image, p[1])

    color_img_topics_by_callback = [
        ('/back_right_kinect/rgb/image_raw', pcl_reg.back_right_rgb_callback),
        ('/front_right_kinect/rgb/image_raw', pcl_reg.front_right_rgb_callback),
        ('/front_left_kinect/rgb/image_raw', pcl_reg.front_left_rgb_callback),
        ('/overhead_kinect/rgb/image_raw', pcl_reg.overhead_rgb_callback),
        ('/back_left_kinect/rgb/image_raw', pcl_reg.back_left_rgb_callback),
    ]
    for p in color_img_topics_by_callback:
        rospy.Subscriber(p[0], Image, p[1])
    
    depth_raw_topics_by_callback = [
        ('/back_right_kinect/depth/image_raw', pcl_reg.back_right_depth_raw_callback),
        ('/front_right_kinect/depth/image_raw', pcl_reg.front_right_depth_raw_callback),
        ('/front_left_kinect/depth/image_raw', pcl_reg.front_left_depth_raw_callback),
        ('/overhead_kinect/depth/image_raw', pcl_reg.overhead_depth_raw_callback),
        ('/back_left_kinect/depth/image_raw', pcl_reg.back_left_depth_raw_callback),
    ]
    for p in depth_raw_topics_by_callback:
        rospy.Subscriber(p[0], Image, p[1])

    while not rospy.is_shutdown():
        rospy.loginfo("point_cloud_dict size".format(
            len(pcl_reg.point_cloud_by_camera_dict)))
        if len(pcl_reg.point_cloud_by_camera_dict) == 5 and len(pcl_reg.carved_voxel_grid_by_camera_dict) == 5:
            rospy.loginfo("Will try to register multiple point clouds.")
            pcl_reg.merge_saved_pcd()
            break
    
        if (len(pcl_reg.depth_img_by_camera_dict) == 5
            and len(pcl_reg.color_img_by_camera_dict) == 5
            and len(pcl_reg.raw_depth_img_by_camera_dict) == 5):

            rospy.loginfo("Try to merge using tsdf")
            import fusion
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
            
            o3d_volume_camera_list = list(pcl_reg.depth_img_by_camera_dict.keys())
            for cam_key in pcl_reg.depth_img_by_camera_dict.keys():

                depth_im = pcl_reg.depth_img_by_camera_dict[cam_key]
                # depth_im[depth_im == 65.535] = 0 # set invalid depth to 0 (specific to 7-scenes dataset)                                   
                depth_im[np.isnan(depth_im)] = 0  # This is when we use 32FC1 conversion format.

                raw_depth_im = pcl_reg.raw_depth_img_by_camera_dict[cam_key]
                # depth_im[depth_im == 65.535] = 0 # set invalid depth to 0 (specific to 7-scenes dataset)                                   
                raw_depth_im[np.isnan(raw_depth_im)] = 0  # This is when we use 32FC1 conversion format.

                # TODO: Should we smooth over the depth image?

                color_im = pcl_reg.color_img_by_camera_dict[cam_key]

                rospy.loginfo("rgb_to_depth size: {}, min: {}, max: {}".format(
                    depth_im.shape, depth_im.min(), depth_im.max()))

                rospy.loginfo("raw depth size: {}, min: {}, max: {}".format(
                    raw_depth_im.shape, raw_depth_im.min(), raw_depth_im.max()))

                cam_intr = pcl_reg.intr_by_camera[cam_key].K
                cam_extr = pcl_reg.extr_by_camera[cam_key]
                o3d_cam_intr = pcl_reg.o3d_camera_params_by_camera[cam_key].intrinsic

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
                    print("HERE ONCE")

                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        depth=rgbd.depth,
                        intrinsic=o3d_cam_intr,
                        extrinsic=np.linalg.inv(cam_extr),
                        depth_scale=1000.0,
                        depth_trunc=1000.0,
                        stride=4,
                    )
                    o3d.visualization.draw_geometries([pcd])

                    rospy.loginfo("pcd min: {}, max: {}".format(
                        pcd.get_min_bound(), pcd.get_max_bound()))

                    # import pdb; pdb.set_trace()

                    o3d_volume.integrate(
                        rgbd,
                        o3d_cam_intr,
                        np.linalg.inv(cam_extr)
                    )

                # Integrate observation into voxel volume (assume color aligned with depth)                                                                                                                                
                tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_extr, obs_weight=1.)
                rospy.loginfo("Did ingegrate: {}".format(cam_key))
            
            # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)                                                                                                                                     
            save_dir = '/home/klz/datasets/ms_point_cloud_data/tsdf/tsdf_org_intr'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            print("Saving to mesh.ply...")                                                                                                                                                                                 
            verts, faces, norms, colors = tsdf_vol.get_mesh()                                                                                                                                                                
            fusion.meshwrite(os.path.join(save_dir, "fusion_mesh.ply"), verts, faces, norms, colors)

            pcd = o3d_volume.extract_point_cloud()
            mesh = o3d_volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])


            o3d.io.write_point_cloud( os.path.join(save_dir, 'scalable_tsdf_pcl.pcd') , pcd)
            o3d.io.write_triangle_mesh( os.path.join(save_dir, 'scalable_tsdf_mesh.ply'), mesh)

            '''
            pcd_2 = o3d_unif_volume.extract_point_cloud()
            # mesh_2 = o3d_unif_volume.extract_triangle_mesh()
            # mesh_2.compute_vertex_normals()
            o3d.visualization.draw_geometries([pcd_2])
            '''

            break

        if len(pcl_reg.carved_voxel_grid_by_camera_dict) == 1:
            print("Will visualize")
            visualize([pcl_reg.carved_voxel_grid_by_camera_dict['front_left']])
            break
            import pdb; pdb.set_trace()
        
        # if len(pcl_reg.point_cloud_by_camera_dict) == 1:
        #     print("Did save some data")
        #     break

        rospy.sleep(1.0)

        # rospy.spin()


if __name__ == '__main__':
    main()
