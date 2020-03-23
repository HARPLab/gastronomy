import numpy as np

import argparse
import glob
import os
import sys

import cv2

from utils.image_utils import is_valid_image_path

def get_all_images_for_dir(data_dir):
    image_path_list = []
    for fname in glob.iglob(data_dir+'/**/*', recursive=True):
        if 'camera2' not in fname:
            continue
        if is_valid_image_path(fname):
            image_path_list.append(fname)
    return image_path_list

def main(args):
    image_path_list = get_all_images_for_dir(args.data_dir)

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    for i, image_path in enumerate(image_path_list):
        img = cv2.imread(image_path)
        fg_mask = fgbg.apply(img)

        if fg_mask is not None:
            cv2.imshow('frame', fg_mask)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            print('fg_mask is  none')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare data for training.')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Dir to get image data.')

    args = parser.parse_args()

    main(args)