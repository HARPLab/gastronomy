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
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    logging.warning("Failed to import ROS in Kinect2_sensor.py. Kinect will not be able to be used in bridged mode")

from geometry_msgs.msg import Transform, Pose, PoseStamped, Point, Point32, PointStamped, Vector3, Vector3Stamped, Quaternion, QuaternionStamped
import tf.transformations
import tf
import scipy
import open3d as o3d


def get_transform_file_for_dir(d):
    assert os.path.exists(d), 'Dir does not exist: {}'.format(d)
    return os.path.join(d, 'kinect2_overhead_to_world.tf')


calib_dirs = {
    'back_right': './calib/azure_kinect_back_right',
    'front_left': './calib/azure_kinect_front_left',
    'overhead': './calib/azure_kinect_overhead',
}
transform_by_camera = {}    
for k, v in calib_dirs.items():
    T = RigidTransform.load(get_transform_file_for_dir(v))
    transform_by_camera[k] = T
    # transform_by_camera[k] = RigidTransform()
point_cloud_by_camera_dict = {}
has_pcd_by_camera_dict = {}

##convert a PointCloud or PointCloud2 to a 4xn scipy matrix (x y z 1)
def point_cloud_to_mat(point_cloud):
    if type(point_cloud) == type(PointCloud2()):
        points = [[p[0], p[1], p[2], 1] for p in point_cloud2.read_points(
            point_cloud, field_names = 'xyz', skip_nans=True)]
    else:
        raise ValueError("Invalid point cloud type: {}".format(type(point_cloud)))
        return None
    points = np.array(points).T
    return points

def point_cloud_to_color_arr(pcl):
    cloud, rgb = [], []
    for point in point_cloud2.read_points(pcl, skip_nans=True):
        cloud.append([point[0], point[1], point[2], 1])
        rgb_point = get_rgb(point[3])
        rgb.append(rgb_point)
    return np.array(cloud), np.array(rgb)

def transform_point_cloud(tf_listener, point_cloud, frame):        
    '''Transform a PointCloud or PointCloud2 to be a 4xn matrix (x y z 1) 
    in a new frame.

    Uses tf_transform to convert between frames.
    '''
    points = point_cloud_to_mat(point_cloud)
    transform = get_transform(tf_listener, point_cloud.header.frame_id, frame)
    if transform == None:
        return (None, None)
    points = transform * points
    return (points, transform)


def get_transform(tf_listener, frame1, frame2):
    '''Get the (4, 4) transformation matrix from frame1 to frame2 using TF.
    '''
    temp_header = Header()
    temp_header.frame_id = frame1
    temp_header.stamp = rospy.Time(0)

    tf_listener.waitForTransform(
        frame1, frame2, rospy.Time.now(), rospy.Duration(5.0))
        
    frame1_to_frame2 = tf_listener.asMatrix(frame2, temp_header)
    return np.array(frame1_to_frame2)


def get_rgb(data):
    '''Get the RGB values from raw PCL data.'''
    s = struct.pack('>f', data)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = int((pack & 0x00FF0000)>> 16)
    g = int((pack & 0x0000FF00)>> 8)
    b = int(pack & 0x000000FF)
    return [r, g, b]


transform = None

# Open3d utils
def make_pcd(pts):
    '''Create a open3d PointCloud datastructure.'''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd

def visualize(list_of_pcds):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(list_of_pcds + [mesh_frame])

def get_pcd_bounds_str(pcd):
    s = "PCD min_bound: {}, max_bound: {}".format(
        np.array2string(pcd.get_min_bound(), precision=4, suppress_small=True), 
        np.array2string(pcd.get_max_bound(), precision=4, suppress_small=True))
    return s

def pairwise_registration(source, target,
                          voxel_size=0.02,
                          max_correspondence_distance_coarse=0.3,
                          max_correspondence_distance_fine=0.03):
    icp_coarse = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
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


def register_multiple_pcd(pcd_list):
    voxel_size = 0.005
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    for pcd in pcd_list:
        print("PCD min_bound: {}, max_bound: {}".format(
            pcd.get_max_bound(), pcd.get_min_bound()))
    pcds_down = [pcd.voxel_down_sample(voxel_size=voxel_size) for pcd in pcd_list]

    # Visualize
    visualize(pcds_down[:1])
    visualize(pcds_down[1:2])
    visualize(pcds_down[2:3])
    visualize(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
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
    pcds = [pcd.voxel_down_sample(voxel_size=voxel_size) for pcd in pcd_list]
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])


def pcl_callback(data):
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


def _get_transform_point_cloud_from_raw_data(data, camera_key):
    # pcl_arr = point_cloud_to_mat(data)
    pcl_arr, rgb_arr = point_cloud_to_color_arr(data)
    T = transform_by_camera[camera_key]
    transf_pcl_arr = np.dot(T.matrix, pcl_arr.T)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transf_pcl_arr[:3, :].T)
    pcd.colors = o3d.utility.Vector3dVector(rgb_arr[:, :3]/255.0)
    return pcd

def back_right_pcl_callback(data):
    camera_key = 'back_right'
    if has_pcd_by_camera_dict.get(camera_key):
        return
    pcd = _get_transform_point_cloud_from_raw_data(data, camera_key)
    print("back_right bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
    pcd = pcd.crop(np.array([-0.2, -1.0, -0.1]),
                   np.array([1.0, 0.4, 0.5]))
    print("back_right bounds AFTER crop: => {}".format(get_pcd_bounds_str(pcd)))
    point_cloud_by_camera_dict[camera_key] = pcd
    has_pcd_by_camera_dict[camera_key] = True
    rospy.loginfo("Did get camera data {}".format(camera_key))

def front_left_pcl_callback(data):
    camera_key = 'front_left'
    if has_pcd_by_camera_dict.get(camera_key):
        return
    pcd = _get_transform_point_cloud_from_raw_data(data, camera_key)
    # visualize([pcd])
    print("front_left bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
    pcd = pcd.crop(np.array([-0.2, -1.0, -0.1]),
                   np.array([1.0, 0.4, 0.5]))
    print("front_left bounds AFTER crop: => {}".format(get_pcd_bounds_str(pcd)))
    # visualize([pcd])
    point_cloud_by_camera_dict[camera_key] = pcd
    has_pcd_by_camera_dict[camera_key] = True
    rospy.loginfo("Did get camera data {}".format(camera_key))

def overhead_pcl_callback(data):
    camera_key = 'overhead'
    if has_pcd_by_camera_dict.get(camera_key):
        return
    pcd = _get_transform_point_cloud_from_raw_data(data, camera_key)
    # visualize([pcd])
    print("overhead bounds BEFORE crop: => {}".format(get_pcd_bounds_str(pcd)))
    pcd = pcd.crop(np.array([-0.2, -1.0, -0.1]),
                   np.array([1.0, 0.4, 0.5]))
    print("overhead bounds AFTER crop: => {}".format(get_pcd_bounds_str(pcd)))
    # visualize([pcd])
    point_cloud_by_camera_dict[camera_key] = pcd
    has_pcd_by_camera_dict[camera_key] = True
    rospy.loginfo("Did get camera data {}".format(camera_key))


def main():
    rospy.init_node('test_k4a_pcl', anonymous=True)
    rospy.loginfo('Test k4a pcl!!')

    rospy.loginfo("Did load transform for camers: {}".format(len(transform_by_camera)))
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pc_topics_by_callback = [
        # ('/points2', pcl_callback),
        ('/back_right_kinect/points2', back_right_pcl_callback),
        ('/front_left_kinect/points2', front_left_pcl_callback),
        ('/overhead_kinect/points2', overhead_pcl_callback),
    ]
    # pc_topic = "/points2"
    # pc_sub = rospy.Subscriber(pc_topic, PointCloud2, pcl_callback)
    for p in pc_topics_by_callback:
        pc_sub = rospy.Subscriber(p[0], PointCloud2, p[1])

    while not rospy.is_shutdown():
        rospy.loginfo("point_cloud_dict size".format(len(point_cloud_by_camera_dict)))
        if len(point_cloud_by_camera_dict) == 3:
            # Do registration
            pcd_list = [
                point_cloud_by_camera_dict['overhead'],
                point_cloud_by_camera_dict['front_left'],
                point_cloud_by_camera_dict['back_right'],
            ]
            rospy.loginfo("Will try to register multiple point clouds.")
            register_multiple_pcd(pcd_list)
            break

        rospy.sleep(1.0)

        # rospy.spin()


if __name__ == '__main__':
    main()
