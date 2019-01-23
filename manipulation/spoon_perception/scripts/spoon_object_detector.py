#!/usr/bin/env python
import rospy
import cv2
import numpy as np

import spoon_perception.spoon_camera_interface as sci
import spoon_perception.histogram_utilities as hu

from spoon_perception.srv import ObjectSpoon, ObjectSpoonResponse


class SpoonObjectDetector:
  """class that detects an object on the spoon
  """

  def __init__(self, spoon_camera, color_channel="H", number_frames=5):
    """

    """
    self.spoon_camera = spoon_camera

    self.spoon_histogram = None #this will be a dict of H S V histograms
    self.calibrated = False #flag to know if the detection is calibrated
    self.detect_object_srv = None

    self.number_frames = number_frames # number of frames that it takes to make the decision
    self.color_channel = color_channel

    self.spoon_histogram = self.getEmptySpoonHistogram()
    return

  def startDetectObjectSrv(self):
    self.detect_object_srv = rospy.Service('detect_object_spoon',ObjectSpoon,self.detectObject)
    return

  def detectObject(self,req):
    distances = []
    for i in range(self.number_frames):
        image = self.spoon_camera.getImage()
        histogram = hu.getHSVHistogram(image, self.spoon_camera.spoon_mask)
        d = cv2.compareHist(self.spoon_histogram[self.color_channel], histogram[self.color_channel],
            # the module cv in cv2 disappeared in later version of OpenCV
            #                cv2.cv.CV_COMP_CORREL)
            cv2.HISTCMP_CORREL)
        distances.append(d)

    image_dir = ""

    return ObjectSpoonResponse(np.mean(distances), image_dir)

  def getEmptySpoonHistogram(self):
    """this function gets the color histogram for the empty spoon
       must be called after getSpoonMask is called
    """
    # get the histogram without sheet of paper there
    image = self.spoon_camera.getSpoonImage()
    rospy.logwarn('Remove the  white sheet of paper from the background'
                            '--- Press ENTER to accept the image'
                            '--- Press ANY key to reject')
    while( not displayWait(image, 'Calibration') ):
      image = self.spoon_camera.getSpoonImage()
      rospy.logwarn('Remove the  white sheet of paper from the background'
                            '--- Press ENTER to accept the image'
                            '--- Press ANY key to reject')

    histogram = hu.getHSVHistogram(image, self.spoon_camera.spoon_mask)

    return histogram

def displayWait(image,window_name='default'):
  """Function that displays a image and wait until key pressed
      if key pressed is enter it returns true otherwise false
  """
  image_accepted = False

  cv2.imshow(window_name,image)
  k = cv2.waitKey(0)
  rospy.logwarn("you pressed key %s"% k)
  if k == 13 or k == 10: #enter key code
      image_accepted = True

  cv2.destroyAllWindows()
  return image_accepted

if __name__ == '__main__':
  rospy.init_node('spoon_object_detector',anonymous = True)

  spoon_camera = sci.SpoonCameraInterface('/camera/image_raw')
  col_channel = rospy.get_param('~col_channel')
  object_detector = SpoonObjectDetector(spoon_camera, color_channel=col_channel)

  object_detector.startDetectObjectSrv()
  rospy.spin()
