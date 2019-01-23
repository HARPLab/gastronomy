#!/usr/bin/env python
import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class SpoonCameraInterface:
  """class that wraps helper methods around a camera topic
     including accessing images of just the spoon and just the background
  """

  def __init__(self,image_topic_name):
    """

    """
    self.last_image = None
    self.image_topic = None
    self.image_topic_name = image_topic_name
    self.bridge = CvBridge()
    self.spoon_image_pub = rospy.Publisher('/spoon_perception/spoon_image',Image, queue_size=10)
    self.background_image_pub = rospy.Publisher('/spoon_perception/background_image',Image, queue_size=10)
  
    self.spoon_mask = self.createSpoonMask()
    self.background_mask = 255 - self.spoon_mask # grayscale goes from 0 to 255 
    self.startRepublishingCroppedImages()
    return
  

  def startRepublishingCroppedImages(self):
    self.image_topic = rospy.Subscriber(self.image_topic_name, Image, self.republishCroppedImages)
    return

  def getImage(self, wait_for_next=True):
    if wait_for_next or self.last_image is None:
      image_msg = rospy.wait_for_message(self.image_topic_name, Image,
                                         timeout=None)
      image = convertMsgImageCv(self.bridge, image_msg)
    else:
      image = self.last_image 
    return image

  def republishCroppedImages(self,img_msg):
    """ this method is the calback to the topic of the image, it converts
    the image to opencv image
    """
    cv_image = convertMsgImageCv(self.bridge,img_msg)
    self.last_image = cv_image

    # if we have a spoon mask, publish the masked image 
    if self.spoon_mask is not None:
        image = cv2.bitwise_and(self.last_image,self.last_image,mask=self.spoon_mask)
        self.spoon_image_pub.publish(self.bridge.cv2_to_imgmsg(image,'bgr8'))
        image = cv2.bitwise_and(self.last_image,self.last_image,mask=self.background_mask)
        self.background_image_pub.publish(self.bridge.cv2_to_imgmsg(image,'bgr8'))
    return

  def createSpoonMask(self):
    """this function gets the mask for the spoon
    """
    image = self.getImage()
    mask, th = retrieveMask(image)
    rospy.logwarn('Put a white sheet of paper in the background'
                            '--- Press ENTER to accept the image'
                            '--- Press ANY key to reject')
    # wait until accepted the image to retrieve the mask
    while( not displayWait(mask,'Calibration') ):
      image = self.getImage()
      mask, th = retrieveMask(image)
      rospy.logwarn('Put a white sheet of paper in the background'
                            '--- Press ENTER to accept the image'
                            '--- Press ANY key to reject')
    return mask

  def getSpoonImage(self, wait_for_next=True):
    image = self.getImage(wait_for_next)
    spoon_image = cv2.bitwise_and(image, image, mask=self.spoon_mask)
    return spoon_image


  def saveMaskedImage(self,image_msg):
    image = convertMsgImageCv(self.bridge, image_msg)
    image = cv2.bitwise_and(image, image, mask=self.spoon_mask)

    image_dir = self.record_dir + 'image_'  + str(self.record_frame) + '.png'
    print(image_dir)
    cv2.imwrite(image_dir, image)

    self.record_frame = self.record_frame+1

    return image_dir

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


def convertMsgImageCv(bridge, ros_image_msg):
  """ function that given a cv bridge and a ros_image_msg converts the image to a opencv bgr8 image
  """
  try:
      cv_image = bridge.imgmsg_to_cv2(ros_image_msg,'bgr8')
  except CvBridgeError as e:
      rospy.logerr(e)
      cv_image = None

  return cv_image

def retrieveMask(image, intensity_th=0):
  """
  function return the image mask based on the threshold, the image
  should be in bgr8. Based in OTSU's method, returns the
  threshold calculated and the mask with the object
  """
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  th, mask  = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  return mask,th

if __name__ == '__main__':
  rospy.init_node('spoon_camera',anonymous = True)

  spoon_camera = SpoonCameraInterface('/camera/image_raw')
  rospy.spin()
