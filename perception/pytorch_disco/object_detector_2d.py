# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron_predictor_2d import VisualizationDemo
import ipdb
st = ipdb.set_trace
# constants
WINDOW_NAME = "COCO detections"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def create_binary_mask(img_og, proposals):
    bmask = np.zeros((img_og.shape),np.uint8)
    binary_mask_shape = (img_og.shape[0],img_og.shape[1],len(proposals))
    binary_mask = np.zeros((binary_mask_shape), np.uint8)

    for i in range(len(proposals)):
        bmask[int(proposals[i][1]):int(proposals[i][3]),int(proposals[i][0]):int(proposals[i][2])] = (255,255,255)
        binary_mask[int(proposals[i][1]):int(proposals[i][3]), int(proposals[i][0]):int(proposals[i][2]), i] = 1
    
    return bmask, binary_mask

class ClevrDetector:

    def __init__(self, config_file, opts, confidence_threshold):

        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.merge_from_list(opts)
        # Set score_threshold for builtin models
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.cfg.freeze()

        self.num_top_proposals = 4


    def get_binary_masks(self, rgb, max_num_of_obs, display_flag = True):

        demo = VisualizationDemo(self.cfg)
        
        # for rgb in rgbs:
        # st()
        # use PIL, to be consistent with evaluation
        # print("Name of image is: ", path.split('/')[-1])
        # img = read_image(path, format="BGR")
        # img_og = cv2.imread(path, cv2.IMREAD_COLOR)

        img = rgb[:,:,::-1] #Convert to BGR
        img_og = img
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        proposals = predictions["proposals"].get_fields()['proposal_boxes'][0:max_num_of_obs].tensor.cpu().numpy()
        
        
        bmask, binary_mask = create_binary_mask(img_og, proposals)
        flat_binary_mask = binary_mask.reshape(-1, binary_mask.shape[-1])
        binary_mask_sum =  np.sum(flat_binary_mask, axis=0)
        print("Points covered by binary mask: ", np.sum(flat_binary_mask, axis=0))

        if(display_flag):

            for i in range(len(proposals)):
                st_pt = (int(proposals[i][0]),int(proposals[i][1]))
                en_pt = (int(proposals[i][2]),int(proposals[i][3]))
                img_og = cv2.UMat(img_og).get()
                cv2.rectangle(img_og, st_pt, en_pt,(0,0,255),5)

            # cv2.imwrite("~/")
            cv2.imshow('RGB image', img_og)
            if cv2.waitKey(0) == 27:
                pass  # esc to quit
            cv2.imshow('Binary Image',bmask)
            if cv2.waitKey(0) == 27:
                pass  # esc to quit
        
        return binary_mask
    
if __name__ == "__main__":

    config_file = "/home/sirdome/shubhankar/detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml"
    display_flag = True
    image_paths = ["/home/sirdome/shubhankar/detectron2/input_1.png","/home/sirdome/shubhankar/detectron2/input_2.png"]
    opts = ["MODEL.WEIGHTS","detectron2://COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl"]
    confidence_threshold = 0.5

    cd = ClevrDetector(config_file, opts, confidence_threshold)
    cd.get_binary_masks(image_paths,True)