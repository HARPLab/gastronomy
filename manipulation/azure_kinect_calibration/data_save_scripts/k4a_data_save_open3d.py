import argparse
import cv2
import logging
import numpy as np
import h5py
import os
import sys
import time
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

sys.path.append(os.path.abspath('../'))

from pcl_registration_utils import point_cloud_to_color_arr, make_pcd, visualize, get_pcd_bounds_str


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """
    assert type(dic) is type({}), "must provide a dictionary"
    assert type(path) is type(''), "path must be a string"
    assert type(h5file) is h5py._hl.files.File, "must be an open h5py file"

    for key in dic:
        assert type(key) is type(''), 'dict keys must be strings to save to hdf5'
        did_save_key = False

        if type(dic[key]) in (np.int64, np.float64, type(''), int, float):
            h5file[path + key] = dic[key]
            did_save_key = True
            assert h5file[path + key].value == dic[key], \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type([]):
            h5file[path + key] = np.array(dic[key])
            did_save_key = True
        if type(dic[key]) is np.ndarray:
            h5file[path + key] = dic[key]
            did_save_key = True
            assert np.array_equal(h5file[path + key].value, dic[key]), \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type({}):
            recursively_save_dict_contents_to_group(h5file,
                                                    path + key + '/',
                                                    dic[key])
            did_save_key = True
        if not did_save_key:
            print("Dropping key from h5 file: {}".format(path + key))


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
    def __init__(self, save_path): 
        self.save_path = save_path

        # This is not correct
        main_dir_path = '../calib/robot_middle'
        osp_join = os.path.join
        self.calib_dirs = { 
            'front_right': osp_join(main_dir_path,'azure_kinect_front_right'),
            'front_left': osp_join(main_dir_path, 'azure_kinect_front_left'),
            'overhead':  osp_join(main_dir_path,'azure_kinect_overhead'),
            'back_left':  osp_join(main_dir_path, 'azure_kinect_front_left'),
            'back_right':  osp_join(main_dir_path, 'azure_kinect_front_right'),
        }
        self.transform_by_camera = {}    
        for k, v in self.calib_dirs.items():
            T = RigidTransform.load(get_transform_file_for_dir(v))
            T.from_frame = k
            T.to_frame = 'world'
            self.transform_by_camera[k] = T
        
        main_dir_path = '../calib/robot_middle'
        self.intr_calib_dirs = {
            'back_left': osp_join(main_dir_path, 'azure_kinect_back_left'),
            'front_left': osp_join(main_dir_path, 'azure_kinect_front_left'),
            'overhead': osp_join(main_dir_path, 'azure_kinect_overhead'),
            'front_right': osp_join(main_dir_path, 'azure_kinect_front_right'),
            'back_right': osp_join(main_dir_path, 'azure_kinect_back_right'),
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
            ('overhead', 'front_right', '../calib/stereo_calib_azure_cpp/Nov_21_combined/overhead_to_front_right_3.tf'),

            # Transf between front_left and back_left camera 
            ('overhead', 'front_left', '../calib/stereo_calib_azure_cpp/main_overhead_sec_front_left/front_left_to_overhead_5.tf'),

            # Transf between front_left and back_left camera 
            ('front_left', 'back_left', '../calib/stereo_calib_azure_cpp/main_front_left_sec_back_left/back_left_to_front_left_7.tf'),

            # Transf between front_right and back_right camera 
            ('front_right', 'back_right', '../calib/stereo_calib_azure_cpp/main_front_right_sec_back_right/back_right_to_front_right_2.tf')
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

        self.h5_save_dict = {
            'camera_info': {},
            'color_raw': {},
            'depth_to_rgb': {}
        }
        for cam_key, cam_params in self.o3d_camera_params_by_camera.items():
            self.h5_save_dict['camera_info'][cam_key] = {
                'intrinsic': np.copy(cam_params.intrinsic.intrinsic_matrix),
                'extrinsic': np.copy(cam_params.extrinsic)
            }

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
        cv_img_to_save = (cv_img_to_save * 255).astype(np.uint8)
        cv2.imwrite('/home/klz/datasets/ms_point_cloud_data/depth_rgb/{}_raw_depth.png'.format(
            camera_key), cv_img_to_save)
        

    def save_depth_with_camera_key(self, data, camera_key):
        if self.depth_img_by_camera_dict.get(camera_key) is not None:
            return
        print("Will save depth for camera: {}".format(camera_key))

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, '32FC1')
        except CvBridgeError as e:
            print("Could not convert depth image: {}".format(e))

        cv_image[np.isnan(cv_image)] = 0.0
        self.depth_img_by_camera_dict[camera_key] = cv_image
        img_arr = np.array(cv_image, dtype=np.float32)
        self.h5_save_dict['depth_to_rgb'][camera_key] = img_arr

        # "32F1" gives value in m?
        cv_img_to_save = np.array(cv_image, dtype=np.float32)
        cv2.normalize(cv_img_to_save, cv_img_to_save, 0, 1, cv2.NORM_MINMAX)
        cv_img_to_save = (cv_img_to_save * 255).astype(np.uint8)
        img_path = os.path.join(
            self.save_path, 
            '{}_depth_to_rgb.png'.format( camera_key))
        cv2.imwrite(img_path, cv_img_to_save)

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
        self.h5_save_dict['color_raw'][camera_key] = np.copy(cv_img)
        cv2.imwrite(os.path.join(self.save_path, '{}_color_raw.png'.format(camera_key)),
                    cv_img)
    
    # PCL callback
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

    # depth_to_rgb img callback
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
    

def get_initial_data_index_for_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return [-1]

    idx_list = []
    for d in os.listdir(dir_path):
        if ('try' in d) and (not os.path.isfile(d)):
            assert d.split('_')[0] == 'try', "Invalid dir"
            idx = int(d.split('_')[1])
            idx_list.append(idx)
    if len(idx_list) == 0:
        idx_list = [-1]
    return idx_list


def main():
    rospy.init_node('test_k4a_pcl', anonymous=True)
    rospy.loginfo('Test k4a pcl!!')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    parser = argparse.ArgumentParser(description="Save data.")
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='Directory to save data in.')

    args = parser.parse_args()

    data_idx_list = get_initial_data_index_for_dir(args.save_dir)
    new_data_dix = max(data_idx_list) + 1
    timestamp_str = time.strftime('%b_%02d_%Y_%02l_%M_%p')
    save_path = os.path.join(args.save_dir, 'try_{}_{}'.format(new_data_dix, timestamp_str))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pcl_reg = PCLRegistrationUtils(save_path)
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

            # Save data
            h5_path = os.path.join(pcl_reg.save_path, 'input_data.h5')
            h5f = h5py.File(h5_path, 'w')
            recursively_save_dict_contents_to_group(h5f, '/', pcl_reg.h5_save_dict)
            h5f.flush()
            h5f.close()
            rospy.loginfo("Did save input data: {}".format(h5_path))

            break

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

                    # pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    #     depth=rgbd.depth,
                    #     intrinsic=o3d_cam_intr,
                    #     extrinsic=np.linalg.inv(cam_extr),
                    #     depth_scale=1000.0,
                    #     depth_trunc=1000.0,
                    #     stride=4,
                    # )
                    # o3d.visualization.draw_geometries([pcd])

                    # rospy.loginfo("pcd min: {}, max: {}".format(
                    #     pcd.get_min_bound(), pcd.get_max_bound()))

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
            print("Saving to mesh.ply...")                                                                                                                                                                                 
            verts, faces, norms, colors = tsdf_vol.get_mesh()                                                                                                                                                                
            fusion.meshwrite(os.path.join(pcl_reg.save_path, "fusion_mesh.ply"), 
                             verts, faces, norms, colors)

            pcd = o3d_volume.extract_point_cloud()
            mesh = o3d_volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])

            o3d.io.write_point_cloud(
                os.path.join(pcl_reg.save_path, 'scalable_tsdf_pcl.pcd') , pcd)
            o3d.io.write_triangle_mesh(
                os.path.join(pcl_reg.save_path, 'scalable_tsdf_mesh.ply'), mesh)

            '''
            pcd_2 = o3d_unif_volume.extract_point_cloud()
            # mesh_2 = o3d_unif_volume.extract_triangle_mesh()
            # mesh_2.compute_vertex_normals()
            o3d.visualization.draw_geometries([pcd_2])
            '''


            break

        rospy.sleep(1.0)


if __name__ == '__main__':
    main()
