#!/usr/bin/env python

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K

import tensorflow as tf
import cv2
import glob

from networks.dextr import DEXTR
from mypath import Path
from helpers import helpers as helpers
import time

modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0

file_paths = glob.glob('/home/klz/Documents/playing_with_food/processed_data/carrot/3/*/images/*.png')

# Handle input and output args
sess = tf.Session()
K.set_session(sess)

with sess.as_default():
    net = DEXTR(nb_classes=1, resnet_layers=101, input_shape=(512, 512), weights=modelName,
                num_input_channels=4, classifier='psp', sigmoid=True)

    
    for file_path in file_paths:
        mask_file_path = file_path[:-4] + '_mask.png'
        if 'mask' not in file_path and mask_file_path not in file_paths:
            print(file_path)
            #  Read image and click the points
            image = cv2.imread(file_path)
            print(image.shape)
            if(image.shape[2] == 3):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if(image.shape[2] == 4):
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            results = []

            while len(results) == 0:
                plt.ion()
                plt.axis('off')
                plt.imshow(image)
                plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')

                extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)
                if extreme_points_ori.shape[0] == 0:
                    mask = np.zeros((image.shape[0], image.shape[1]))

                    results.append(mask)
                else:

                    #  Crop image to the bounding box from the extreme points and resize
                    bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
                    crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
                    resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

                    #  Generate extreme point heat map normalized to image values
                    extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                                  pad]
                    extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
                    extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
                    extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

                    #  Concatenate inputs and convert to tensor
                    input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

                    # Run a forward pass
                    pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
                    result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

                    results.append(result)

                    # Plot the results
                    plt.imshow(helpers.overlay_masks(image / 255, results))
                    plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')

                    extreme_points_ori = np.array(plt.ginput(4, timeout=3)).astype(np.int)
                    if(len(extreme_points_ori) >= 2):
                        results = []
                    plt.close()

            result_copy = results[0].copy()
            im = result_copy.astype(np.float32)
            mask = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(mask_file_path, mask * 255)
