import argparse
import cv2
import IPython
import logging
import numpy as np
import os
import sys
import time
import traceback
import rospy

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

from abc import ABCMeta, abstractmethod
from perception import ColorImage, DepthImage
from perception.camera_intrinsics import CameraIntrinsics
from perception.camera_sensor import CameraSensor


class KinectSensorBridged(CameraSensor):
    """Class for interacting with a Kinect v2 RGBD sensor through the kinect bridge
    https://github.com/code-iai/iai_kinect2. This is preferrable for visualization and debug
    because the kinect bridge will continuously publish image and point cloud info.
    """

    def __init__(self, frame='kinect2_rgb_optical_frame'):
        """Initialize a Kinect v2 sensor which connects to the iai_kinect2 bridge
        ----------
        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be set to 'kinect2_rgb_optical_frame'
       """
        # set member vars
        self._frame = frame

        # self.topic_image_color = '/kinect2/%s/image_color_rect' %(quality)
        # self.topic_image_depth = '/kinect2/%s/image_depth_rect' %(quality)
        # self.topic_info_camera = '/kinect2/%s/camera_info' %(quality)

        self.topic_image_color = '/rgb/image_raw'
        self.topic_image_depth = '/depth_to_rgb/image_raw'
        self.topic_info_camera = '/rgb/camera_info'
        
        self._initialized = False
        self._format = None
        self._camera_intr = None
        self._cur_depth_im = None
        self._running = False
        self._bridge = CvBridge()
        
    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

    def _set_camera_properties(self, msg):
        """ Set the camera intrinsics from an info msg. """
        focal_x = msg.K[0]
        focal_y = msg.K[4]
        center_x = msg.K[2]
        center_y = msg.K[5]
        im_height = msg.height
        im_width = msg.width
        self._camera_intr = CameraIntrinsics(self._frame, focal_x, focal_y,
                                             center_x, center_y,
                                             height=im_height,
                                             width=im_width)
            
    def _process_image_msg(self, msg):
        """ Process an image message and return a numpy array with the image data
        Returns
        -------
        :obj:`numpy.ndarray` containing the image in the image message

        Raises
        ------
        CvBridgeError
            If the bridge is not able to convert the image
        """
        encoding = msg.encoding
        try:
            image = self._bridge.imgmsg_to_cv2(msg, encoding)
        except CvBridgeError as e:
            rospy.logerr(e)
        return image
        
    def _color_image_callback(self, image_msg):
        """ subscribe to image topic and keep it up to date
        """
        color_arr = self._process_image_msg(image_msg)
        color_arr = cv2.cvtColor(color_arr, cv2.COLOR_RGBA2RGB)
        # import matplotlib.pyplot as plt
        # plt.imshow(color_arr[:, :, ::-1])
        # plt.show()
        self._cur_color_im = ColorImage(color_arr[:,:,::-1], self._frame)
 
    def _depth_image_callback(self, image_msg):
        """ subscribe to depth image topic and keep it up to date
        """
        encoding = image_msg.encoding
        try:
            depth_arr = self._bridge.imgmsg_to_cv2(image_msg, encoding)

        except CvBridgeError as e:
            rospy.logerr(e)
        # depth = np.array(depth_arr*MM_TO_METERS, np.float32)
        depth = np.array(depth_arr, np.float32)
        self._cur_depth_im = DepthImage(depth, self._frame)

    def _camera_info_callback(self, msg):
        """ Callback for reading camera info. """
        self._camera_info_sub.unregister()
        self._set_camera_properties(msg)
    
    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Ensenso IR camera.
        """
        return self._camera_intr

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor.
        """
        return self._frame

    @property
    def ir_frame(self):
        """:obj:`str` : The reference frame of the sensor.
        """
        return self._frame

    def start(self):
        """ Start the sensor """
        # initialize subscribers
        self._image_sub = rospy.Subscriber(self.topic_image_color, sensor_msgs.msg.Image, self._color_image_callback)
        self._depth_sub = rospy.Subscriber(self.topic_image_depth, sensor_msgs.msg.Image, self._depth_image_callback)
        self._camera_info_sub = rospy.Subscriber(self.topic_info_camera, sensor_msgs.msg.CameraInfo, self._camera_info_callback)
        
        timeout = 10
        try:
            rospy.loginfo("waiting to recieve a message from the Kinect")
            rospy.wait_for_message(self.topic_image_color, sensor_msgs.msg.Image, timeout=timeout)
            rospy.wait_for_message(self.topic_image_depth, sensor_msgs.msg.Image, timeout=timeout)
            rospy.wait_for_message(self.topic_info_camera, sensor_msgs.msg.CameraInfo, timeout=timeout)
            rospy.loginfo("Done")
        except rospy.ROSException as e:
            print("KINECT NOT FOUND")
            rospy.logerr("Kinect topic not found, Kinect not started")
            rospy.logerr(e)

        while self._camera_intr is None:
            time.sleep(0.1)
        
        self._running = True

    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self._running:
            logging.warning('Kinect not running. Aborting stop')
            return False

        # stop subs
        self._image_sub.unregister()
        self._depth_sub.unregister()
        self._camera_info_sub.unregister()

        self._running = False
        return True

    def frames(self):
        """Retrieve a new frame from the Ensenso and convert it to a ColorImage,
        a DepthImage, IrImage is always none for this type

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        # wait for a new image
        while self._cur_depth_im is None or self._cur_color_im is None:
            time.sleep(0.01)
            
        # read next image
        depth_im = self._cur_depth_im
        color_im = self._cur_color_im

        self._cur_color_im = None
        self._cur_depth_im = None

        #TODO add ir image
        return color_im, depth_im, None

    def median_depth_img(self, num_img=1, fill_depth=0.0):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The median DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        median_depth = Image.median_images(depths)
        median_depth.data[median_depth.data == 0.0] = fill_depth
        return median_depth


def main():
    logging.getLogger().setLevel(logging.INFO)

    rospy.init_node('test_kinect', anonymous=True)

    rospy.loginfo('Test saving camera image')
    
    sensor = KinectSensorBridged()
    rospy.loginfo('Starting sensor')
    sensor.start()

    output_dir = '/home/mohit/Desktop/azure_kinect_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(10):
        color_img, depth_img, _ = sensor.frames()
        img = color_img.raw_data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./image_{}.png'.format(t), img_rgb)

        depth_img.save('./image_depth_{}.png'.format(t))

        cv2.imshow('hello world', img_rgb)
        # rospy.sleep(1.0)


if __name__ == '__main__':
    main()
