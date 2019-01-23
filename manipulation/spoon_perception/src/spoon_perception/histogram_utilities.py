#!/usr/bin/env python
import rospy
import cv2

from matplotlib import pyplot as plt

def getHSVHistogram(image, mask=None):
  """
  function that gets the HSV histogram of a bgr8 image, optinaly can include
  a mask for the image
  returns in the form of a dictionary with H: S: V:
  """

  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  hist_H = cv2.calcHist( [hsv_image], [0], mask, [180], [0, 179])
  hist_S = cv2.calcHist( [hsv_image], [1], mask, [256], [0, 255])
  hist_V = cv2.calcHist( [hsv_image], [2], mask, [256], [0, 255])

  histogram = {'H':hist_H, 'S':hist_S, 'V':hist_V}

  return histogram

def histogramHSVPlot(histogram):
  """
  plot the HSV histogram, histogram must be a dictionary with H: S: V:
  """

  hist_H = histogram['H']
  hist_S = histogram['S']
  hist_V = histogram['V']

  plt.subplot(3, 1, 1)
  plt.plot(hist_H)
  plt.xlim([0, 179])
  plt.title('Histograma H')

  plt.subplot(3, 1, 2)
  plt.plot(hist_S)
  plt.xlim([0, 255])
  plt.title('Histograma S')

  plt.subplot(3, 1, 3)
  plt.plot(hist_V)
  plt.xlim([0, 255])
  plt.title('Histograma V')

  plt.show()
  return

