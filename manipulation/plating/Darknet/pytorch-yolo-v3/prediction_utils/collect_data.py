import numpy as np
import cv2
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import matplotlib.patches as patches
from matplotlib.ticker import NullFormatter
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import sys
import torch

# Custom libraries
from prediction_utils import preprocess
from prediction_utils import heatmap
from prediction_utils import kernel_density
from prediction_utils import predict
from prediction_utils import gaussian
from prediction_utils import util
from rnns import fake_data
from rnns import utils

class Preperation:

    def __init__(self, directory, resolution):
        """
        Class for preprocessing the images and gathering the bounding box information

        Inputs:
            directory (string): the file path of where the images are located
            resolution (tuple): the image resolution to transform the images to
                                (width, height)
        Notes:
            - Make sure only the original image and the predictions.npy file
                are in directory
        """        
        self.directory = directory
        self.resolution = resolution

    def numerical_sort(self, value):
        """
        key function for numerically sorting filenames
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def images(self):
        """
        Collect all of the images in the directory, crop, and sort numerically
        Outputs:
            image_list (list): list of the images in the numpy file of bounding boxes
                each element in list is a HXWXD image as np.array
        Note:
            - YOLO format is 416x416x3 and the images should be cropped
               w/o changing the aspect ratio
            - Only reads .jpg and .png images right now, add another loop to
                except more formats
        """
        image_list = []
        for filename in sorted(glob.glob('{}/*.jpg'.format(self.directory)), key=self.numerical_sort):
            img = cv2.imread(filename, 1)
            #reshape images into YOLO format
            img = preprocess.letterbox_image(img, self.resolution)
            image_list.append(img)
        for filename in sorted(glob.glob('{}/*.png'.format(self.directory)), key=self.numerical_sort):
            img = cv2.imread(filename, 1)
            #reshape images into YOLO format
            img = preprocess.letterbox_image(img, self.resolution)
            image_list.append(img)
        return image_list

    def boxes(self, file_name='predictions.npy'):
        """
        Load the numpy array created from running detect.py
        It should be a Nx8 np.array, with N bounding boxes
        - [0] the index of the image in the batch where the bounding box is located
        - [1],[2] x_min and y_min coordinates
        - [3],[4] x_max and y_max coordinates
        - [5] the objectness score
        - [6] the score of the class with maximum confidence
        - [7] the prediction of which class, check the .names file

        Inputs:
            file_name (str): the filename of the numpy file containing the
                bounding box information
        Outputs:
            self.box (np.array): the bounding box information, same format as input
        """
        num_files = 0
        for filename in glob.glob('{}/{}'.format(self.directory, file_name)):
            self.box = np.load(filename)
            num_files += 1
        assert num_files == 1
        
        return self.box

    def board_index(self, board_label=3):
        """
        Returns the indexes of any cutting boards detected in predictions.npy

        Inputs:
            board_label (int): the number label of what class the cutting board is
        Outputs:
            board_idx (list): list of ints, returns index in self.box
        Note:
            - ImageData class only works with one cutting board right now
        """
        food_boxes = np.copy(self.box)
        board_idx = []
        for i in range(self.box.shape[0]):
            b = food_boxes[i,:]
            #the cutting bboard index is 3
            if b[7] == board_label:
                board_idx.append(i)
       
        return board_idx
        
class ImageData:

    def __init__(self, images, boxes, board_index, cutting_board, n=1,
                 remove_first=True):
        """
        Class for gathering the distance values between object centers

        Inputs:
            images (list): list of np.array images, use Preperation.images
            boxes (np.array): Nx8 array containing the bounding boxes, use
                            Preperation.boxes as input
            board_index (list): list of indexes of where the cutting boards
                            are in boxes input, use Preperation.board_index
            cutting_board - either the cutting board dimension in meters
                            [width,height]. Or an integer, 0 or 1, 
                            identifying the premeasured cutting board. 0 is 
                            the smaller one and 1 is the larger one
            n (int) - object to measure distance from. ie: n=1 is previous
                            object, n=2 is two object before, etc.
                            n=3 includes 3, 2, and 1
            
        Notes:
            - Assuming only one cutting board is in the image
            - might want to put something in here for the case when a
                cutting board is identified wrong
            - Not doing a proper transform for the image location in image
                coordinates to real world coordinates. Just doing an estimation
        """
        self.n = n
        self.images = images
        self.boxes = boxes
        #make sure the number of cutting boards is the same as the number of images
        assert len(board_index) == len(self.images)
        self.board_index = board_index
        #remove the cutting board from the list of boxes
        self.food_boxes = np.copy(self.boxes)
        self.food_boxes = np.delete(self.food_boxes, board_index, axis=0)
        assert type(cutting_board) is list or type(cutting_board) is int
        #assign the cutting obard dimensions
        if type(cutting_board) == int:
            assert 0 <= cutting_board < 2
            if cutting_board == 0:
                self.cutting_board_width = 0.358
                self.cutting_board_height = 0.280 
            elif cutting_board == 1:
                self.cutting_board_width = 0.455
                self.cutting_board_height = 0.304
        else:
            assert len(cutting_board) == 2
            self.cutting_board_width = cutting_board[0]
            self.cutting_board_height = cutting_board[1]
        # Remove the first image that only shows the cutting board if
        if self.boxes.shape[0] % self.food_boxes.shape[0] != 0:
            if remove_first == True:
                self.boxes = self.boxes[1:,:]
                self.images = self.images[1:]
                self.board_index = [x-1 for x in self.board_index][1:]
    
    def subtract_images(self, threshold=10):
        """
        Take sequential images and return the most recently placed object
            by subtracting image(i) with image(i-1)
        
        Inputs:
            threshold (int): threshold value to expect as a large enough
                difference between the images, value between 0-255
        Outputs:
            newest_boxes (np.array): array of the most recently placed
                objects, same format as self.food_boxes
        """
        newest_boxes =[]
        #iterate through the images, starts with comparing the 2nd image to the 1st
        for i in range(1, len(self.images)):
            It1_boxes = np.zeros((self.food_boxes.shape))
            It1 = self.images[i] #current image
            It = self.images[i-1] #previous image AKA template
            #make binary mask of the difference between images
            diff = It1 - It
            diff = np.where(diff >= threshold, False, True)
            # diff = np.where(diff >= threshold, 0, 255) #useful for visualization
            # return the detected ojects in the current image, turn everything else to zero
            temp_It1_boxes = np.where(self.food_boxes[:,0] == i, 1, 0)
            temp_It1_boxes = np.expand_dims(temp_It1_boxes, axis = 1)
            temp_It1_boxes = np.hstack([temp_It1_boxes]*self.food_boxes.shape[1])
            It1_boxes = np.multiply(self.food_boxes, temp_It1_boxes)
            # get cutting board coordinates
            board = self.boxes[self.board_index[i]]
            #return only the most recently placed item            
            patches = np.zeros((It1_boxes.shape[0]))
            for j in range(It1_boxes.shape[0]):
                #get object location and append the most recent object
                x1 = int(It1_boxes[j,1])
                x2 = int(It1_boxes[j,3])
                y1 = int(It1_boxes[j,2])
                y2 = int(It1_boxes[j,4])
                #make sure the object is on the cutting board
                if x1>board[1] and y1>board[2] and x2<board[3] and y2<board[4]:
                    patch = diff[y1:y2, x1:x2]
                    patches[j] = np.sum(patch)
                else:
                    patches[j] = 0
            newest_boxes.append(It1_boxes[np.argmax(patches),:])
        #for a vectorized version, make two arrays, one is list[0->len(list)-1]
        #the other one is list[1 -> len(list)] then list two minus list one
        newest_boxes =  np.vstack(newest_boxes)

        #make sure the array is of correct size, add the obj in first image
        assert newest_boxes.shape[0] == len(self.images)-1
        newest_boxes = np.insert(newest_boxes, 0, self.food_boxes[0,:], axis = 0)

        assert newest_boxes.shape[1] == 8

        return newest_boxes

    def delta_center(self, newest_boxes, return_abs=False):
        """
        Calculate the distance between the centers of sequential objects
        
        Inputs:
            newest_boxes (np.array): Nx8 array, where N is the number of
                images in the sequence and each row is the label for the
                object. Usually the output of subtract images
            return_abs (bool): flag of whether to return absolute distance
                instead of distance relative to last object
        Outputs:
            dcx (list): List of N-1 values. The x distance in meters
            dcy (list): List of N-1 values. The y distance in meters
        """
        dcx = []
        dcy = []
        for i in range(len(self.images)-self.n):
            #get conversion ratio to convert pixels to meters
            board1 = self.boxes[self.board_index[i+self.n]] #idk if this is right
            board = self.boxes[self.board_index[i]]
            #assuming the long side of the cutting board is along side the width of image
            #height and width of the cutting board in pixels
            board1_width = board1[3] - board1[1]
            board_width = board[3] - board[1]
            #conversion ratio for pixels to meters
            # ratio1 is transformation for the other image. Only using the one ratio right now
            ratio1 = self.cutting_board_width/board1_width
            ratio = self.cutting_board_width/board_width
            # print('The meters/pixels conversion ratio is {:0.6f}'.format(ratio))

            #get centers of current image and previous image
            box1 = newest_boxes[i+self.n,:]
            box1_width = box1[3] - box1[1]
            box1_height = box1[4] - box1[2]
            box1_centerx = box1[1] + box1_width/2
            box1_centery = box1[2] + box1_height/2

            box = newest_boxes[i,:]
            box_width = box[3] - box[1]
            box_height = box[4] - box[2]
            box_centerx = box[1] + box_width/2
            box_centery = box[2] + box_height/2
            #find distance from center of object to center of cutting board in meters
            delta_cx = (box1_centerx - box_centerx)*ratio
            delta_cy = (box1_centery - box_centery)*ratio
            if return_abs:
                dcx.append(abs(delta_cx))
                dcy.append(abs(delta_cy))
            else:
                dcx.append((delta_cx))
                dcy.append((delta_cy))
        return dcx, dcy
    
    def delta_plate(self, newest_boxes):
        """
        Calculate the distance between the center of the plate/cutting
            board and the objects
        
        Inputs:
            newest_boxes (np.array): Nx8 array, where N is the number of
                images in the sequence and each row is the label for the
                object. Usually the output of subtract images
        Outputs:
            dpx (list): List of N-1 values. The x distance in meters
            dpy (list): List of N-1 values. The y distance in meters
        """
        dpx = []
        dpy = []

        for i in range(len(self.images)):
            #get bounding box info of the cutting board, assuming there is only/at least one
            board_box = self.boxes[self.board_index[i]]
            #assuming the long side of the cutting board is along side the width of image
            boardx1 = board_box[1]
            boardx2 = board_box[3]
            boardy1 = board_box[2]
            boardy2 = board_box[4]
            #height and width of the cutting board in pixels
            board_width = boardx2 - boardx1
            board_height = boardy2 - boardy1
            #x and y coordinate of cutting board center in pixels
            board_centerx = boardx1 + board_width/2
            board_centery = boardy1 + board_height/2
            #conversion ratio for pixels to meters
            ratio = self.cutting_board_width/board_width
            #calculate the dpx and dpy values
            box = newest_boxes[i,:]
            x1 = box[1]
            x2 = box[3]
            y1 = box[2]
            y2 = box[4]
            box_width = x2 - x1
            box_height = y2 - y1
            box_centerx = x1 + box_width/2
            box_centery = y1 + box_height/2
            #find distance from center of object to center of cutting board in meters
            delta_px = (board_centerx - box_centerx)*ratio
            delta_py = (board_centery - box_centery)*ratio
            dpx.append(delta_px)
            dpy.append(delta_py)
        return dpx, dpy

    def delta_edge(self, newest_boxes, return_abs=False):
        """
        Calculate the distance between the bottom right (xmax,ymax)
            corners of sequential objects
        
        Inputs:
            newest_boxes (np.array): Nx8 array, where N is the number of
                images in the sequence and each row is the label for the
                object. Usually the output of subtract images
            return_abs (bool): flag of whether to return absolute distance
                instead of distance relative to last object
        Outputs:
            dex (list): List of N-1 values. The x distance in meters
            dey (list): List of N-1 values. The y distance in meters
        """
        dex = []
        dey = []

        for i in range(len(self.images)-self.n):
            #some redundancy here, not using board1, assuming the cutting boards are the same iin each image
            #get conversion ratio to convert pixels to meters
            board1 = self.boxes[self.board_index[i+self.n]]
            board = self.boxes[self.board_index[i]]
            #assuming the long side of the cutting board is along side the width of image
            #height and width of the cutting board in pixels
            board1_width = board1[3] - board1[1]
            board_width = board[3] - board[1]
            #conversion ratio for pixels to meters
            ratio1 = self.cutting_board_width/board1_width
            ratio = self.cutting_board_width/board_width

            #get centers of current image and previous image
            box1 = newest_boxes[i+self.n,:]
            box = newest_boxes[i,:]

            delta_ex = (box1[3] - box[3])*ratio
            delta_ey = (box1[4] - box[4])*ratio

            if return_abs:
                dex.append(abs(delta_ex))
                dey.append(abs(delta_ey))
            else:
                dex.append((delta_ex))
                dey.append((delta_ey))
        return dex, dey

def gather_data(working_dir, n=1, suffix_list=None, save_dir=None,
                remove_first=True, return_n=None):
    """
    Sort through the given working directory and subdirectories to 
    gather all of the "predictions.npy" files (can change file name
    in collect_data.Preperation.boxes())

    Inputs:
        working_dir (str): the directory where the images for training are
        n (int): The number of time steps to include (e.g. for n=3, the data
                 measurements for 3, 2, and 1 images prior to the current
                 time step in the sequence will be included)
        suffix_list (list): a list of strings that represent the suffixes
                 of the sub-directories to gather the data from. If none, then
                 working directory should contain the images
        save_dir (str): the directory to save the n npy files to, leave
                 as None to not save anything
        remove_first (bool): Flag remove first cutting board image
        return_n (int): return the n-th object in each sequence instead of the
                 entire sequence. Starts at 1 not zero since we
                 assume first image is a cutting board. Set to None to
                 return entire sequence

    Outputs:
        data (list): list of length 6*n, where each group of six contains the
            data for n=1, n=2, ... Each element in the list is a 1-D array
            containing distance measurements. First measurement in group of 6 
            is dcx, then dcy, dpx, dpy, dex, and dey. (arrays may vary in size)
    """
    data = [] # list of len 6n, each element is a 1-D array of the delta values
    if suffix_list is None:
        suffix_list = list([1])
    for m in range(1,n+1):
        dcx_total = []
        dcy_total = []
        dpx_total = []
        dpy_total = []
        dex_total = []
        dey_total = []
        for i in range(len(suffix_list)):
            if len(suffix_list) == 1:
                prep = Preperation('{}'.format(working_dir), (416, 416))
            else:
                prep = Preperation('{}/set{}'.format(working_dir,
                    suffix_list[i]), (416, 416))
            img_list = prep.images()
            bounding_boxes = prep.boxes()
            cutting_board_index = prep.board_index()
            data_temp = ImageData(img_list, bounding_boxes,
                cutting_board_index, cutting_board=0, n=m,
                remove_first=remove_first)
            last_food_item = data_temp.subtract_images()

            if save_dir is not None:
                np.save('{}/set{}/food_boxes_n{}.npy'.format(
                    save_dir, suffix_list[i], m), data_temp.food_boxes) 

            dcx, dcy = data_temp.delta_center(last_food_item)
            dpx, dpy = data_temp.delta_plate(last_food_item)
            dex, dey = data_temp.delta_edge(last_food_item)
            dcx_total.append(dcx)
            dcy_total.append(dcy)
            dpx_total.append(dpx)
            dpy_total.append(dpy)
            dex_total.append(dex)
            dey_total.append(dey)
        dcx_total = np.concatenate(dcx_total)
        dcy_total = np.concatenate(dcy_total)
        dpx_total = np.concatenate(dpx_total)
        dpy_total = np.concatenate(dpy_total)
        dex_total = np.concatenate(dex_total)
        dey_total = np.concatenate(dey_total)
        data.append(dcx_total)
        data.append(dcy_total)
        data.append(dpx_total)
        data.append(dpy_total)
        data.append(dex_total)
        data.append(dey_total)

    # return only one element in the sequence if return_n is given
    if return_n is not None:
        assert type(return_n) is int
        idx = return_n
        # for the dp values only trained on n=10, should remove later
        new_data = []
        #loop through different dps
        for i in range(len(data)):
            temp = []
            #loop through 
            for j in range(len(suffix_list)): #suffix_list gives number of datasets
                #NOTE this is assuming the first image is a cutting board
                # and the sequences are of the same length
                length = len(data[i])//len(suffix_list) #number of data points in one sequence
                temp.append(data[i][length*j + idx])
            new_data.append(np.array(temp))
        data = new_data

    # for i in range(n):
    #     print('Number of samples for n={}:\ndcx: {}\ndcy: {}\ndpx: {}' \
    #           '\ndpy: {}\ndex: {}\ndey: {}'.format(
    #           i+1, data[0+i*len(data)//n].shape[0], data[1+i*len(data)//n].shape[0],
    #           data[2+i*len(data)//n].shape[0], data[3+i*len(data)//n].shape[0],
    #           data[4+i*len(data)//n].shape[0], data[5+i*len(data)//n].shape[0]))

    return data

def split_data(data, n, num_features=6):
    """
    Split a list of data into a list of length num_features, with 
    each item in the the list being a list of length n.
        eg. list of [1,2,3,1,2,3] with n=2 and 6 features would be
        [[1,1],[2,2],[3,3]]
    
    Inputs:
        data (list): list of data to be sorted
        n (int): number of sub features
        num_features (int): number of features
    Outputs:
        features (list): list of sorted values
    """
    features = []

    for i in range(num_features):
        temp = []
        for j in range(n):
            temp.append(data[j*num_features + i])
        features.append(temp)
    
    return features
