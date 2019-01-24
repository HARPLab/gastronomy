#!/usr/bin/env python
"""Calibration from camera to robot"""
import rospkg
import yaml
import tf
import numpy as np
import rospy

class CameraCalibration:
  def __init__(self, world_frame_name = "base_link", target_frame_name="camera_rgb_optical_frame"):
    """
    cameraCalibration.yml is something like:
    Rotation:
    - - 0.9989181262157819
      - 0.01931543906095123
      - -0.04230237500938314
    - - 0.04514525926005835
      - -0.18453994506190463
      - 0.981787611575381
    - - 0.011157180821961618
      - -0.9826351929838623
      - -0.18521229663733005
    Translation:
    - - 0.5391208676662875
    - - -0.6604205518057711
    - - 0.4549267234161455
    """
    rospack = rospkg.RosPack()
    self.ada_tut_path = rospack.get_path("feedbot_trajectory_logic")

    self.world_frame_name = world_frame_name
    self.target_frame_name = target_frame_name

    self.br = tf.TransformBroadcaster()
    rospy.logwarn("camera calibration initialized")

    rospy.Timer(rospy.Duration(0.01), self.broadcastTransform)
    rospy.logwarn("sent first message")

  def loadTransform(self):
    if rospy.has_param("camera_calib_params"):
      calibParams = { 
        "QuaternionXYZW": rospy.get_param("camera_calib_params/QuaternionXYZW"),
        "TranslationXYZ": rospy.get_param("camera_calib_params/TranslationXYZ")}
    else:
      with open(self.ada_tut_path + '/config/calibrationParameters.yml','r') as f:
        calibParams = yaml.load(f)
        rospy.set_param("camera_calib_params", calibParams)

    #rospy.logwarn("calibParams %s"% calibParams)
    q = np.array(calibParams["QuaternionXYZW"])
    translation = np.array(calibParams["TranslationXYZ"])
    # last value should be w
    rotation = tf.transformations.quaternion_matrix(q)
    rotation[0:3,3] = translation[:]
    self.camera_to_robot = rotation 


  def broadcastTransform(self, event):
    #rospy.logwarn("broadcasting tf")
    self.loadTransform()
    self.br.sendTransform(self.camera_to_robot[0:3,3],
                          # note we pass in 4x4 to this method... https://github.com/ros/geometry/issues/64
                          tf.transformations.quaternion_from_matrix(self.camera_to_robot),
                          rospy.Time.now(),
                          self.target_frame_name,
                          self.world_frame_name)

  # point should be a length 4 np.array giving the location of the target point in the camera frame
  def convert_to_robot_frame(self, point):
    return self.camera_to_robot.dot(point.transpose())

if __name__ == "__main__":
  rospy.init_node('camera_calibration', anonymous=True)
  c = CameraCalibration(rospy.get_param("~world_frame_name"), rospy.get_param("~target_frame_name"))
  rospy.spin()
