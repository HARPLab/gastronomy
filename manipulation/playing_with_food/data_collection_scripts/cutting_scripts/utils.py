import os
import subprocess
import numpy as np
import rospy
import glob
from sensor_msgs.msg import Image
from perception import CameraIntrinsics
import cv2
from cv_bridge import CvBridge, CvBridgeError

from frankapy import FrankaArm

from autolab_core import RigidTransform, Point

def get_azure_kinect_rgb_image(cv_bridge, topic='/rgb/image_raw'):
    """
    Grabs an RGB image for the topic as argument
    """
    rgb_image_msg = rospy.wait_for_message(topic, Image)
    try:
        rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    except CvBridgeError as e:
        print(e)
    
    return rgb_cv_image

def get_azure_kinect_depth_image(cv_bridge, topic='/depth_to_rgb/image_raw'):
    """
    Grabs an Depth image for the topic as argument
    """
    depth_image_msg = rospy.wait_for_message(topic, Image)
    try:
        depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)
    
    return depth_cv_image

def get_realsense_rgb_image(cv_bridge, topic='/camera/color/image_raw'):
    """
    Grabs an RGB image for the topic as argument
    """
    rgb_image_msg = rospy.wait_for_message(topic, Image)
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
        rgb_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    except CvBridgeError as e:
        print(e)
    
    return rgb_cv_image

def get_realsense_depth_image(cv_bridge, topic='/camera/depth/image_rect_raw'):
    """
    Grabs an Depth image for the topic as argument
    """
    depth_image_msg = rospy.wait_for_message(topic, Image)
    try:
        depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)
    
    return depth_cv_image

def createFolder(directory, delete_previous=False):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory) and delete_previous:
            for file_path in glob.glob(directory + '*'):
                os.remove(file_path)
            os.rmdir(directory)
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def get_object_center_point_in_world(object_image_center_x, object_image_center_y, depth_image, intrinsics, transform):    
    
    object_center = Point(np.array([object_image_center_x, object_image_center_y]), 'azure_kinect_overhead')
    object_depth = depth_image[object_image_center_y, object_image_center_x]
    print("x, y, z: ({:.4f}, {:.4f}, {:.4f})".format(
        object_image_center_x, object_image_center_y, object_depth))
    
    object_center_point_in_world = transform * intrinsics.deproject_pixel(object_depth, object_center)    
    print(object_center_point_in_world)

    return object_center_point_in_world 

def save_audio(dir_path, filename):
    cmd = "python scripts/sound_subscriber.py " + dir_path + filename 
    audio_p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    return audio_p

def save_finger_vision(dir_path, filename):
    cmd = "python scripts/finger_vision_subscriber.py " + dir_path + filename 
    finger_vision_p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    return finger_vision_p

def save_realsense(dir_path, filename, topic='/camera', use_depth=True):
    if use_depth:
        cmd = "python scripts/cutting_scripts/realsense_subscriber.py " + dir_path + filename + ' -d -t ' + topic
    else:
        cmd = "python scripts/cutting_scripts/realsense_subscriber.py " + dir_path + filename + ' -t ' + topic
    realsense_p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    return realsense_p

def get_robot_positions_and_forces(franka, run_time):
    robot_positions = np.zeros((0,3))
    robot_forces = np.zeros((0,6))

    start_time = rospy.get_rostime()
    current_time = rospy.get_rostime()
    duration = rospy.Duration(run_time)
    while current_time - start_time < duration:
        robot_positions = np.vstack((robot_positions, franka.get_pose().translation.reshape(1,3)))
        robot_forces = np.vstack((robot_forces, franka.get_ee_force_torque().reshape(1,6)))
        rospy.sleep(0.01)
        current_time = rospy.get_rostime()

    return (robot_positions, robot_forces)