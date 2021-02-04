import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import sys
import time
from multiprocessing import Pool

import cProfile
import pstats

from rnns import utils, image_utils
from gmm_placing import gaussian, collect_data, heatmap, placing_utils

class GMMSequencePredictions:
    def __init__(self, init_obj, plate, ref_plate_dims, gaussians, image, 
                 obj_image, obj_mask=None, init_loc=None, transform=None):
        """
        Uses cross entropy optimization to get a single placement prediction

        Inputs:
            init_obj (np.array): shape (4,) array describing the bounding
                box of the object that was used to make the initial placement.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
            plate (np.array): shape (4,) array describing the bounding box of the
                plate/cutting board that was used to make the initial placement.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
            ref_plate_dims (list or int): info that describes plates real world 
                dimensions in meters. either an int (0 or 1) specifying which cutting
                board was used (0 is smaller and 1 in larger) or a list of 
                len 2 containing the width and height of the cutting board
                in meters
            gaussians (dict): object to use for scoring data.
                Use the gaussians.SequenceGMMs.gaussians.
            image (np.array): the background image of the scene to make placements on
            obj_image (np.array): the image of the initial object being placed
            obj_mask (np.array): the alpha layer mask of the initial object being
                placed. #TODO should remove this at some point since new image format has alpha layer as 4th channel
            init_loc (np.array): shape (2,) giving the (y,x) coordinate of where to 
                place the initial object if init_obj doesn't represent the placement
                location and is only for the dimensions
            transform (TODO)

            # (list): the list of the previous n objects that were placed
            #     in this sequence. in the format nx2, where each row is (y,x) coordinate
            #     see collect_data.ImageData for more info on n
            next_obj (np.array): same as init_obj, but is the object to be placed next,
                only being used for its height and width measurements
        """
        self.seq_idx = 1
        # get dimensions of inital object
        if init_loc is not None:
            init_height, init_width = placing_utils.get_bbox_dims(init_obj)
            init_centerx = init_loc[1]
            init_centery = init_loc[0]
        else:
            init_centerx, init_centery, init_height, init_width = placing_utils.get_bbox_center(init_obj, return_dims=True)
        # array of the positions of the previous objs in the sequence
        self.prev_obj_centers = np.array([init_centery, init_centerx]).reshape(1, 2)
        self.prev_obj_dims = np.array([init_height, init_width]).reshape(1, 2)
        self.prev_objs = np.array([init_obj]).reshape(1, 4)

        # dimensions of the plate or cutting board descriptor that object is on
        self.plate = plate
        self.plate_width, self.plate_height, self.plate_centerx, self.plate_centery = placing_utils.get_bbox_center(plate, return_dims=True)
        # Make the image of the initial placement
        self.img = image_utils.paste_image(image, obj_image, [init_centerx, init_centery], obj_mask)
        # assign the cutting obard dimensions
        assert type(ref_plate_dims) is list or type(ref_plate_dims) is int
        if type(ref_plate_dims) == int:
            assert 0 <= ref_plate_dims < 2
            if ref_plate_dims == 0:
                plate_ref_width = 0.358
                plate_ref_heigsht = 0.280 
            elif ref_plate_dims == 1:
                plate_ref_width = 0.455
                plate_ref_height = 0.304
        else:
            assert len(ref_plate_dims) == 2
            plate_ref_width = ref_plate_dims[0]
            plate_ref_height = ref_plate_dims[1]
        # ratio to convert pixels to meters (meters/pixels)
        if transform is None:
            self.ratio = plate_ref_width/self.plate_width
        else:
            #TODO get the camera intrinsics/extrinsics to get more accruate conversion
            raise ValueError('Not implemented yet')
        # initialize the variables for next object to be placed 
        self.next_obj = None
        self.next_obj_width = None
        self.next_obj_height = None
        self.gaussians = gaussian

    def update_next_obj(self, next_obj):
        """
        Set dimensions of next object to be placed
        Inputs:
            next_obj (np.array): shape (4,) array describing the bounding
                box of the next object to be placed.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
        """
        self.next_obj = next_obj
        self.next_obj_height, self.next_obj_width = placing_utils.get_bbox_dims(next_obj)

    def update_seq_idx(self):
        """
        Increment the sequence index
        """
        self.seq_idx += 1

    def update_prev_objs(self, placement_loc, obj):
        """
        Update the arrays containing placements that have already been made
        Inputs
            placement_loc (np.array): shape (2,) or 1x2 of the (y,x) coordinate
                of where the placement that was just made was
            obj (np.array): shape (4,) array describing the bounding
                box of the object that was just placed.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
        """
        #TODO change the prev_objs so the bounding box is at actual placement loc
        self.prev_objs = np.vstack((self.prev_objs, obj.reshape(1,4)))
        self.prev_obj_centers = np.vstack((self.prev_obj_centers, placement_loc.reshape(1,2)))
        obj_height, obj_width = placing_utils.get_bbox_dims(obj)
        temp_dims = np.array([obj_height, obj_width]).reshape(1,2)
        self.prev_obj_dims = np.vstack((self.prev_obj_dims, temp_dims))

    def update_image(self, next_obj_img, loc, alpha_mask=None, background_image=None, viz=False):
        """
        Paste the object onto self.img at given location
        Inputs:
            next_obj_img (np.array): the image of the next object to be placed.
                it should already be sized to fit on self.img and should have a
                4th channel alpha layer specifiying the transparency. Can use
                alpha_mask instead of the 4th channel
            loc (np.array): the (y,x) coordinates of the placement
            alpha_mask (np.array): the alpha layer specifiying the transparency
                of each pixel. should be the same (h,w) size as self.img
            background_img (np.array): can replace self.img with this argument
                and next_obj_img will be pasted onto that instead
            viz (bool): if True, the updated image will be shown
        """
        # update image and make placement 
        if background_image is not None:
            assert self.img.shape == background_image.shape
            self.img = background_image
        else:
            self.img = image_utils.paste_image(self.img, next_obj_img, loc, alpha_mask)
        #TODO put a check here to see if the image is getting pasted outside of image range
        if viz:
            plt.imshow(self.img)
            plt.show()

    def rand_samples(self, num_samples):
        """
        Randomly generates array of pixel coordinates to be sampled from
        
        Outputs: 
            samples (np.array): is a Nx2 array, where each row gives
                the Y, X coordinates (height/width)
        """
        x1 = int(self.plate[1] + self.next_obj_width/2)
        y1 = int(self.plate[2] + self.next_obj_height/2)
        x2 = int(self.plate[3] - self.next_obj_width/2)
        y2 = int(self.plate[4] - self.next_obj_height/2)
        #get a coordinate map of the image pixels
        imgX = np.arange(self.img.shape[1])
        imgY = np.arange(self.img.shape[0])
        meshX, meshY = np.meshgrid(imgX, imgY)
        #get coordinate map of the plate
        sample_areaX = meshX[y1:y2,x1:x2]
        sample_areaY = meshY[y1:y2,x1:x2]
        #create the random sample points
        pattern = np.random.randint(0, sample_areaX.shape[0]*sample_areaX.shape[1], num_samples)
        patternX = pattern % sample_areaX.shape[1]
        patternY = pattern // sample_areaX.shape[1]
        #instantiate array of random sample coordinates
        samples = np.zeros((num_samples,2))
        samples[:,0] = sample_areaY[patternY, patternX]
        samples[:,1] = sample_areaX[patternY, patternX]
        return samples

    def get_sample_values(self, samples, key, ref_idx):
        """
        Helper function to call functions that will get the delta values
        """
        assert type(key) == str
        if 'dc' in key:
            return self.d_centers(samples, ref_idx)
        elif 'dp' in key:
            return self.d_plate(samples)
        elif 'de' in key:
            return self.d_edges(samples, ref_idx)
        else:
            raise ValueError('Invalid key')

    def d_centers(self, samples, ref_idx):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between the center of an object in the sequence and the samples
        
        Outputs:
            dcx (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dcy (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
            ref_idx (int): the sequence index of the object in the sequence
                to take the distance to. Zero indexed, so 0 is initial object.
        """ 
        obj_center = self.prev_obj_centers[ref_idx, :]
        dcx = (samples[:,1] - obj_center[1])*self.ratio
        dcy = (samples[:,0] - obj_center[0])*self.ratio
        return np.hstack((dcx.reshape(-1,1), dcy.reshape(-1,1)))

    def d_plate(self, samples):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between the plate/cutting board center and the samples' centers
        
        Outputs:
            dpx (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dpy (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
        """ 
        dpx = (self.plate_centerx - samples[:,1])*self.ratio
        dpy = (self.plate_centery - samples[:,0])*self.ratio
        return np.hstack((dpx.reshape(-1,1), dpy.reshape(-1,1)))
    
    def d_edges(self, samples, ref_idx):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between self.obj's and the samples' bottom right edges (xmax,ymax)
        
        Outputs:
            dex (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dey (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
            ref_idx (int): the sequence index of the object in the sequence
                to take the distance to. Zero indexed, so 0 is initial object.
        """ 
        dex = (self.prev_objs[ref_idx, 2] - (samples[:,1] + self.next_obj_width/2))*self.ratio
        dey = (self.prev_objs[ref_idx, 3] - (samples[:,0] + self.next_obj_height/2))*self.ratio
        return np.hstack((dex.reshape(-1,1), dey.reshape(-1,1)))

    def make_2D_predictions(self, next_obj, seq_idx=None, num_neighbors=1, mode=['dc', 'de', 'dp'],
                            num_samples=None, seq_weights='relative', feat_weights=None, future=False,
                            viz_figs=False, save_fig_name=None, norm_feats=False, fig_title=None):
        """
        Get the position of where to place the next object in the sequence.
        Return object placement with highest score, using 2-D gaussians
        
        Inputs:
            next_obj (np.array): shape (4,) array describing the bounding
                box of the next object to place. This is just for its dimensions.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
            num_neighbors (int): the number of neighbors in the sequence to 
                take into account when making the prediction. e.g. for a 
                seq_idx=3 and num_neighbors=2, the prediction will be based
                on the guassians created from the data of the distance between
                the seq_idx=3 to seq_idx=2 and seq_idx=3 and seq_idx=1.
            mode (string): string specifying how to score. Can be
                "dc, "dp", "de", or a combination of the 3.
                if passing more than one, put inside a list.
                Can also pass in "all" to score based on all 3.
                See gaussian.SequenceGMMs.fit_gaussians for more info.
            num_samples (int): number of pixels to sample from when calculating
                score for cross entropy optimization. Uses all pixels in image if None
            seq_weights (array_like): the weights to use for adding the scores from
                different sequence indexes. Should be of the shape (num_neighbors,).
                Can give the string 'relative' to way indexes that are closer more.
            feat_weights (array-like): weights to use when summing the different features.
                Should of the shape (num_features,)
            future (bool): whether to take into account future placements.
                By default, the predictions are made from the gaussians with
                a sequence index < given seq_idx. If true then the neighbors
                can include indexes > given seq_idx.
            viz_figs (bool): whether to visualize the 2d Gaussian
            save_fig_name (str): name to save figure of gaussian as, leave as None to not save
            norm_feats (bool): whether to average the weights features if they
                have more than one gaussian to score.
            
            The maximum number of previous objects you want
                to take into account, ie if you provide a n > num_neighbors, then
                n = num_neighbors.
            n (int): number of previously placed objects to look back
                at. (see collect_data.gather_data)
        Outputs:
            output is a 1-D array of 2 elements, the (x,y) coordinates
                of the sample with the highest score, in image coordinates
        """
        # Check format of arguments
        if type(mode) is not list:
            mode = list([mode])
        if 'all' in mode:
            mode = ['dc', 'dp', 'de']

        if viz_figs or save_fig_name is not None:
            assert num_samples is None

        if feat_weights is not None:
            assert len(feat_weights) == len(mode)
        else:
            feat_weights = np.ones(len(mode))

        if seq_idx is None:
            seq_idx = self.seq_idx
        else:
            assert seq_idx > 0
            if seq_idx > self.prev_objs.shape[0]+1:
                print(f'WARNING: Making placement for sequence index {seq_idx}, but only {self.prev_objs.shape[0]} placements have been made.')

        if num_samples is None:
            w, h = self.img.shape[1], self.img.shape[0]
            y, x = np.mgrid[0:h, 0:w]
            samples = np.stack((y.ravel(), x.ravel())).T
            num_samples = samples.shape[0]
        else:
            samples = self.rand_samples(num_samples)

        print(f'Making prediction for sequence index {self.seq_idx}...')
        self.update_next_obj(next_obj)

        feature_scores = []
        total_score = np.zeros(num_samples)
        for i, feature in enumerate(mode):
            sample_values = self.get_sample_values(samples, feature, seq_idx-1)
            if 'dp' in feature:
                #TODO this weights the 'dp' feature a lot less than the others since the
                #'dc' and 'de' features have a set of gaussians for each seq_idx and 'dp' only has one
                # this is only if the seq_weights are not normalized though I think
                # NOTE the second seq_idx doesn't matter in the below, they are all the same value
                feat_scores = np.exp(self.gaussians[feature][seq_idx][0].score_samples(sample_values))
                # TODO you aren't using the score samples function here, you're just calling the gaussain directly
                assert np.sum(feat_scores) != 0
            else:
                max_seq_len = len(list(self.gaussians[feature].keys()))
                # make predictions on sequence indexes not contained in training data
                if seq_idx >= max_seq_len:
                    print(f'WARNING: Making placement for sequence index value that does not exist in training data. Clipping value, you can make prediction with other sequence indexes.')
                    #TODO might want to change to mode to ignore dp for predictions > max_seq_length
                    seq_idx = max_seq_len - 1

                if (num_neighbors is None) or (future and num_neighbors > (max_seq_len-1)):
                    print('WARNING: Training data does not contain sequence lengths large enough for given number of neighbors, clipping value.')
                    n_neighbors = max_seq_len - 1
                elif num_neighbors > (seq_idx):
                    print('WARNING: Not enough predecessors for given number of neighbors, clipping value.')
                    n_neighbors = seq_idx
                
                if 'relative' in seq_weights:
                    s_weights = placing_utils.get_relative_weights(n_neighbors, exponent=2, normalize=True)
                elif seq_weights is not None:
                    assert len(seq_weights) >= n_neighbors
                    s_weights = seq_weights
                else:
                    s_weights = np.ones(n_neighbors)

                if future:
                    neighbors = list(self.gaussians[feature][seq_idx].keys())
                    neighbors = placing_utils.get_n_nearest(seq_idx, neighbors, n_neighbors, remove_value=False) # value has already been removed
                else:
                    neighbors = np.arange(1, seq_idx+1)[-n_neighbors:]
                
                feat_scores = np.zeros(num_samples)
                for j, ref_idx in enumerate(neighbors):
                    # Get the samples to score
                    #TODO you are using the score samples functions, you're just calling the gaussian directly
                    temp_score = s_weights[j] * np.exp(self.gaussians[feature][seq_idx][ref_idx].score_samples(sample_values))
                    assert np.sum(temp_score) != 0
                    feat_scores += temp_score
                if norm_feats:
                    feat_scores /= n_neighbors #TODO

            total_score += feat_weights[i]*feat_scores

        winner = np.argmax(total_score)
        placement_loc = samples[winner, :] # TODO double check that this is right format, (y,x)

        # update arrays
        self.update_prev_objs(placement_loc, self.next_obj)

        self.update_seq_idx()

        if viz_figs or save_fig_name is not None:
            Z = (-total_score).reshape(self.img.shape[:2])
            _ = self.plot_2D_gaussian(Z, mode=mode, viz=viz_figs, save_path=save_fig_name, title=fig_title, convert=False)

        return placement_loc

    def plot_2D_gaussian(self, scores, mode, viz=True, save_path=None, title=None, convert=True):
        """
        Plots the mulivariate, multimodal gaussian

        Inputs:
            scores (np.array): Array of scores for each pixel in self.img.
                It should be the same shape as the image
            mode (string): string specifying how to score. Can be
                "dc, "dp", "de", or a combination of the 3.
                if passing more than one, put inside a list.
                Can also pass in "all" to score based on all 3
            viz (bool): whether to show the figure
            save_path (string): Path to save the plot to, set to None
                to just display the figure
        """
        #Use base cmap to create transparent
        mycmap = heatmap.transparent_cmap(plt.cm.inferno)

        img = self.img.copy()
        if convert:
            img = img[:,:,::-1] #convert BGR to RGB

        w, h = img.shape[1], img.shape[0]
        y, x = np.mgrid[0:h, 0:w]

        #Plot image and overlay colormap
        plt.close()
        fig, ax = plt.subplots(1, 1)
        plt.imshow(img)
        # CB = ax.contour(x, y, Z, norm=LogNorm(vmin=0.001, vmax=1000.0),
            # levels=np.logspace(0, 3, 10), cmap=mycmap, extend='min')

        #TODO fix this log scale for the new predictions (9/29/20)
        CB = ax.contour(x, y, scores, norm=Normalize(),#LogNorm(),#vmin=np.min(Z), vmax=np.max(Z)),
            levels=100, cmap=mycmap)#, extend='min')
        # import ipdb; ipdb.set_trace()
        # CB = ax.contour(x, y, Z, norm=LogNorm(vmin=1, vmax=10000.0),
        #     levels=np.logspace(1, 4, 10), cmap=mycmap, extend='min')
        plt.colorbar(CB)
        
        plt.title(title)
        if save_path is not None:
            plt.savefig(f'{save_path}')
        if viz:
            plt.show()

        return fig

class LocalGMMPredictions(GMMSequencePredictions):
    def __init__(self, init_obj, plate, ref_plate_dims, image, 
                 obj_image, obj_mask=None, init_loc=None, transform=None):
        """
        Uses cross entropy optimization to get a single placement prediction

        Inputs:
            init_obj (np.array): shape (4,) array describing the bounding
                box of the object that was used to make the initial placement.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
            plate (np.array): shape (4,) array describing the bounding box of the
                plate/cutting board that was used to make the initial placement.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
            ref_plate_dims (list or int): info that describes plates real world 
                dimensions in meters. either an int (0 or 1) specifying which cutting
                board was used (0 is smaller and 1 in larger) or a list of 
                len 2 containing the width and height of the cutting board
                in meters
            image (np.array): the background image of the scene to make placements on
            obj_image (np.array): the image of the initial object being placed
            obj_mask (np.array): the alpha layer mask of the initial object being
                placed. #TODO should remove this at some point since new image format has alpha layer as 4th channel
            init_loc (np.array): shape (2,) giving the (y,x) coordinate of where to 
                place the initial object if init_obj doesn't represent the placement
                location and is only for the dimensions
            transform (TODO)

            # (list): the list of the previous n objects that were placed
            #     in this sequence. in the format nx2, where each row is (y,x) coordinate
            #     see collect_data.ImageData for more info on n
            next_obj (np.array): same as init_obj, but is the object to be placed next,
                only being used for its height and width measurements
        """
        self.seq_idx = 1
        # get dimensions of inital object
        if init_loc is not None:
            init_height, init_width = placing_utils.get_bbox_dims(init_obj)
            init_centerx = init_loc[1]
            init_centery = init_loc[0]
        else:
            init_centerx, init_centery, init_height, init_width = placing_utils.get_bbox_center(init_obj, return_dims=True)
        # array of the positions of the previous objs in the sequence
        self.prev_obj_centers = np.array([init_centery, init_centerx]).reshape(1, 2)
        self.prev_obj_dims = np.array([init_height, init_width]).reshape(1, 2)
        self.prev_objs = np.array([init_obj]).reshape(1, 4)

        # dimensions of the plate or cutting board descriptor that object is on
        self.plate = plate
        self.plate_width, self.plate_height, self.plate_centerx, self.plate_centery = placing_utils.get_bbox_center(plate, return_dims=True)
        # Make the image of the initial placement
        self.img = image_utils.paste_image(image, obj_image, np.array([init_centery, init_centerx]), obj_mask)
        # assign the cutting obard dimensions
        assert type(ref_plate_dims) is list or type(ref_plate_dims) is int
        if type(ref_plate_dims) == int:
            assert 0 <= ref_plate_dims < 2
            if ref_plate_dims == 0:
                plate_ref_width = 0.358
                plate_ref_heigsht = 0.280 
            elif ref_plate_dims == 1:
                plate_ref_width = 0.455
                plate_ref_height = 0.304
        else:
            assert len(ref_plate_dims) == 2
            plate_ref_width = ref_plate_dims[0]
            plate_ref_height = ref_plate_dims[1]
        # ratio to convert pixels to meters (meters/pixels)
        if transform is None:
            self.ratio = plate_ref_width/self.plate_width
        else:
            #TODO get the camera intrinsics/extrinsics to get more accruate conversion
            raise ValueError('Not implemented yet')
        # initialize the variables for next object to be placed 
        self.next_obj = None
        self.next_obj_width = None
        self.next_obj_height = None

 #TODO DOES MODE NEED TO HAVE dp in the MIDDLE

    def make_2D_predictions(self, next_obj, gaussians, seq_idx=None, n_time_neighbors=1,
                            n_pos_neighbors=10, mode=['dc', 'de', 'dp'], num_samples=None,
                            time_neighbor_weights='relative', pos_neighbor_weights='relative',
                            feat_weights=None, future=False, viz_figs=False, save_fig_name=None,
                            norm_feats=False, fig_title=None, num_processes=4, bandwidth_samples=50):
        """
        Get the position of where to place the next object in the sequence.
        Return object placement with highest score, using 2-D gaussians
        
        Inputs:
            next_obj (np.array): shape (4,) array describing the bounding
                box of the next object to place. This is just for its dimensions.
                Format (in pixel space): 
                - [0],[1] x_min and y_min coordinates
                - [2],[3] x_max and y_max coordinates
                NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
            gaussians: gaussian.LocallyWeightedGMMs object, containing the training data
            seq_idx (int): the sequence index to make the prediction for. self.seq_idx is
                used by default, so leave this as None unless you wish to write that value over.
            n_time_neighbors (int): the number of neighbors in the sequence to 
                take into account when making the prediction. e.g. for a 
                seq_idx=3 and num_neighbors=2, the prediction will be based
                on the guassians created from the data of the distance between
                the seq_idx=3 to seq_idx=2 and seq_idx=3 and seq_idx=1.
            n_pos_neighbors (int): number of neighbors to include w.r.t. their
                spatial distance (i.e. the dp values)
            mode (string): string specifying how to score. Can be
                "dc, "dp", "de", or a combination of the 3.
                if passing more than one, put inside a list.
                Can also pass in "all" to score based on all 3.
                See gaussian.SequenceGMMs.fit_gaussians for more info.
            num_samples (int): number of pixels to sample from when calculating
                score for cross entropy optimization. Uses all pixels in image if None
            time_neighbor_weights (np.array): a (n_time_neighbors) size array
                of the weights for each sample in dataset. The weights will be applied in
                the order that this array is given in. They are set to one if None.
                Can give the string 'relative' to weight indexes that are closer spatially (dp values)
                more instead of giving explicit weights.
            pos_neighbor_weights (np.array): a (n_pos_neighbors) size array
                of the weights for each sample in dataset. The weights will be applied in
                the order that this array is given in. They are set to one if None.
                Can give the string 'relative' to weight indexes that are closer spatially (dp values)
                more instead of giving explicit weights.
            feat_weights (array-like): weights to use when summing the different features.
                Should of the shape (num_features,)
            future (bool): whether to take into account future placements.
                By default, the predictions are made from the gaussians with
                a sequence index < given seq_idx. If true then the neighbors
                can include indexes > given seq_idx.
            viz_figs (bool): whether to visualize the 2d Gaussian
            save_fig_name (str): name to save figure of gaussian as, leave as None to not save
            norm_feats (bool): whether to average the weights features if they
                have more than one gaussian to score.
            fig_title (str): string to use for the figure title if viz_figs or save_fig_name is used.
            num_processes (int): number of CPU processes to use for sample scoring.
            bandwidth_samples (int): the number of bandwidth values to use for cross validation
                if an optimal bandwidth value needs to be calculated.

        Outputs:
            output is a 1-D array of 2 elements, the (x,y) coordinates
                of the sample with the highest score, in image coordinates
        """
        # Check format of arguments
        if type(mode) is not list:
            mode = list([mode])
        if 'all' in mode:
            mode = ['dc', 'dp', 'de']

        if viz_figs or save_fig_name is not None:
            assert num_samples is None

        if feat_weights is not None:
            assert len(feat_weights) == len(mode)
        else:
            feat_weights = np.ones(len(mode))

        if seq_idx is None:
            seq_idx = self.seq_idx
            #TODO need to put checks here to change cap the seq_idx
            # # below is +1 because it is one indexed
            # if seq_idx > self.prev_objs.shape[0]+1:
            #     seq_idx = self.max_sequence_length
            #     print(f'WARNING: Making placement for sequence index {seq_idx}, but only {self.prev_objs.shape[0]} placements have been made.')
        else:
            assert seq_idx > 0
            if seq_idx > self.prev_objs.shape[0]+1:
                # seq_idx = self.max_sequence_length
                print(f'WARNING: Making placement for sequence index {seq_idx}, but only {self.prev_objs.shape[0]} placements have been made.')

        if num_samples is None:
            # sample across entire image if a range isn't given
            w, h = self.img.shape[1], self.img.shape[0]
            y, x = np.mgrid[0:h, 0:w]
            samples = np.stack((y.ravel(), x.ravel())).T
            num_samples = samples.shape[0]
        else:
            samples = self.rand_samples(num_samples)

        print(f'Making prediction for sequence index {self.seq_idx}...')
        self.update_next_obj(next_obj)

        feature_scores = []
        total_score = np.zeros(num_samples)
        # TODO probably a better way to do the below line
        ref_dp = self.get_sample_values(self.prev_obj_centers[seq_idx-1,:].reshape(1,2), 'dp', seq_idx-1)

        # Get the indices of the relavent data samples
        t_neighbor_idxs = gaussians.get_time_neighbor_data(seq_idx, n_time_neighbors, future=future)
        p_neighbor_idxs, p_neighbor_dist = gaussians.get_pos_neighbor_data(ref_dp, seq_idx, n_pos_neighbors)

        # Get the sample weights for temporal neighbor data points
        if time_neighbor_weights is None:
            time_neighbor_weights = np.ones(t_neighbor_idxs.shape[0])
        elif time_neighbor_weights == 'relative':
            #TODO change this weighting to gaussian
            time_neighbor_weights = placing_utils.get_relative_weights(
                n_time_neighbors,
                exponent=2,
                normalize=True
            )
            time_neighbor_weights = np.repeat(time_neighbor_weights, (t_neighbor_idxs.shape[0] / n_time_neighbors))
            time_neighbor_weights /= time_neighbor_weights.shape[0]
        assert time_neighbor_weights.shape[0] == t_neighbor_idxs.shape[0]

        # Get the sample weights for spatial neighbor data points
        if pos_neighbor_weights is None:
            pos_neighbor_weights = np.ones(p_neighbor_idxs.shape[0])
        elif pos_neighbor_weights == 'relative':
            pos_neighbor_weights = placing_utils.get_relative_weights(
                n_pos_neighbors,
                delta_values=p_neighbor_dist,
                exponent=2,
                normalize=True
            )
            pos_neighbor_weights /= pos_neighbor_weights.shape[0]
        
        assert pos_neighbor_weights.shape[0] == p_neighbor_idxs.shape[0]

        for i, feature in enumerate(mode):
            sample_values = self.get_sample_values(samples, feature, seq_idx-1)
                
            max_seq_len = list(gaussians.data[feature].keys())[-1]
            #TODO double check these if statements
            if seq_idx >= max_seq_len:
                print(f'WARNING: Making placement for sequence index value that does not exist in training data. Clipping value, you can make prediction with other sequence indexes.')
                #TODO might want to change to mode to ignore dp for predictions > max_seq_length
                seq_idx = max_seq_len - 1

            if (n_time_neighbors is None) or (future and n_time_neighbors > (max_seq_len-1)):
                print('WARNING: Training data does not contain sequence lengths large enough for given number of neighbors, clipping value.')
                n_time_neighbors = max_seq_len - 1
            elif n_time_neighbors > (seq_idx):
                print('WARNING: Not enough predecessors for given number of neighbors, clipping value.')
                n_time_neighbors = seq_idx
            else:
                pass
            
            # Gaussian regression
            gmm = gaussians.fit_gaussian(feature=feature,
                                        seq_idx=seq_idx,
                                        time_neighbor_idxs=t_neighbor_idxs,
                                        pos_neighbor_idxs=p_neighbor_idxs,
                                        mode='kde',
                                        future=future,
                                        bandwidths=None,
                                        covariance_type="full",
                                        time_neighbor_weights=time_neighbor_weights,
                                        pos_neighbor_weights=pos_neighbor_weights,
                                        kernel_type='gaussian',
                                        num_samples=50,
                                        n_jobs=num_processes
            )
            
            # Score all of the sampled placement locations, split code for multiprocessing
            sample_values = np.array_split(sample_values, num_processes, axis=0)
            with Pool(processes=num_processes) as p:
                feat_scores = p.map(gmm.score_samples, sample_values)
            feat_scores = [np.exp(scores) for scores in feat_scores]
            feat_scores = np.concatenate(feat_scores, axis=0)

            assert np.sum(feat_scores) != 0
            total_score += feat_weights[i]*feat_scores

        winner = np.argmax(total_score)
        placement_loc = samples[winner, :]

        # update arrays
        self.update_prev_objs(placement_loc, self.next_obj)

        self.update_seq_idx()

        if viz_figs or save_fig_name is not None:
            Z = (-total_score).reshape(self.img.shape[:2])
            _ = self.plot_2D_gaussian(Z, mode=mode, viz=viz_figs, save_path=save_fig_name, title=fig_title, convert=False)

        return placement_loc

class Prediction:
    def __init__(self, num_samples, last_obj, plate, plate_dims, image, 
                 new_obj, scoring, n_objs):
        """

        Uses cross entropy optimization to get a single placement prediction

        Inputs:
            num_samples (int): number of pixels to sample from when calculating
                score for cross entropy optimization
            last_obj (np.array): 1-D array with 8 values describing the object
                was just placed on the board/plate (ie. obj label)
            plate (np.array): 1-D array with 8 values describing the cutting
                board or plate detected in image (ie. plate/board label)
            plate_dims (list or int): info that describes plates real world 
                dimensions. either an int (0 or 1) specifying which cutting
                board was used (0 is smaller and 1 in larger) or a list of 
                len 2 containing the width and height of the cutting board
                jin meters
            image (np.array): the image of the scene
            new_obj (np.array): same as last_obj, but is the object to be placed,
                only being used for its height and width measurements
            scoring: class object to use for scoring. class should have 
                a 'forward' method that outputs score values
            n_obj (list): the list of the previous n objects that were placed
                in this sequence. in the format nx2, where each row is (y,x) coordinate
                see collect_data.ImageData for more info on n
        """
        self.num_samples = int(num_samples) #number of iterations/samples to do
        #get dimension of the most recently identified object
        self.last_obj = last_obj
        self.last_obj_width = last_obj[3] - last_obj[1]
        self.last_obj_height = last_obj[4] - last_obj[2]
        self.last_obj_centerx = last_obj[1] + self.last_obj_width/2
        self.last_obj_centery = last_obj[2] + self.last_obj_height/2
        #dimensions of the plate or cutting board descriptor that object is on
        self.plate = plate
        self.plate_width = plate[3] - plate[1]
        self.plate_height = plate[4] - plate[2]
        self.plate_centerx = plate[1] + self.plate_width/2
        self.plate_centery = plate[2] + self.plate_height/2
        #the image of the object that was just placed
        self.img = image 
        #assign the cutting obard dimensions
        assert type(plate_dims) is list or type(plate_dims) is int
        if type(plate_dims) == int:
            assert 0 <= plate_dims < 2
            if plate_dims == 0:
                self.plate_dims_width = 0.358
                self.plate_dims_height = 0.280 
            elif plate_dims == 1:
                self.plate_dims_width = 0.455
                self.plate_dims_height = 0.304
        else:
            assert len(plate_dims) == 2
            self.plate_dims_width = plate_dims[0]
            self.plate_dims_height = plate_dims[1]
        #ratio to convert pixels to meters (meters/pixels)
        self.ratio = self.plate_dims_width/self.plate_width
        #get dimensions of object to be placed
        self.new_obj = new_obj
        self.new_obj_width = new_obj[3] - new_obj[1]
        self.new_obj_height = new_obj[4] - new_obj[2]
        self.score = scoring
        #make all of the random samples
        self.samples = self.rand_sample()

        self.n_objs = n_objs

    def rand_sample(self):
        """
        Randomly generates array of pixel coordinates to be sampled from
        
        Outputs: 
            self.samples (np.array): is a Nx2 array, where each row gives
                the Y, X coordinates (height/width)
        """
        x1 = int(self.plate[1] + self.new_obj_width/2)
        y1 = int(self.plate[2] + self.new_obj_height/2)
        x2 = int(self.plate[3] - self.new_obj_width/2)
        y2 = int(self.plate[4] - self.new_obj_height/2)
        #get a coordinate map of the image pixels
        imgX = np.arange(self.img.shape[1])
        imgY = np.arange(self.img.shape[0])
        meshX, meshY = np.meshgrid(imgX, imgY)
        #get coordinate map of the plate
        sample_areaX = meshX[y1:y2,x1:x2]
        sample_areaY = meshY[y1:y2,x1:x2]
        #create the random sample points
        pattern = np.random.randint(0, sample_areaX.shape[0]*sample_areaX.shape[1], self.num_samples)
        patternX = pattern % sample_areaX.shape[1]
        patternY = pattern // sample_areaX.shape[1]
        #instantiate array of random sample coordinates
        samples = np.zeros((self.num_samples,2))
        samples[:,0] = sample_areaY[patternY, patternX]
        samples[:,1] = sample_areaX[patternY, patternX]
        return samples
    
    def delta_centers(self, n=1):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between self.obj's center and the samples
        
        Outputs:
            dcx (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dcy (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
            dcn (list): list of length n, where each item in list is a Nx2
                array of the dcy and dcx values (y,x pairs)
        """
        dcx, dcy = self.d_centers(self.samples[:,1], self.samples[:,0])
        if n == 1:
            return list([np.hstack((dcy.reshape(-1,1), dcx.reshape(-1,1)))])
        
        if n > 1:
            # add all of the previous n's
            dcn = []
            # assuming n starts at 1
            for i in range(n-1):
                temp_dcx = (self.samples[:,1] - self.n_objs[i,1])*self.ratio
                temp_dcy = (self.samples[:,0] - self.n_objs[i,0])*self.ratio
                temp = np.hstack((temp_dcy.reshape(-1,1), temp_dcx.reshape(-1,1)))
                dcn.append(temp)

            # append the current n
            temp = np.hstack((dcy.reshape(-1,1), dcx.reshape(-1,1)))
            dcn.append(temp)
            
            return dcn

    def delta_plate(self, n=1):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between the plate/cutting board center and the samples
        
        Outputs:
            dpx (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dpy (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
            dpn (list): list of length n, where each item in list is a Nx2
                array of the dpy and dpx values (y,x pairs)
        """ 
        dpx, dpy = self.d_plate(self.samples[:,1], self.samples[:,0])
        if n == 1:
            return list([np.hstack((dpy.reshape(-1,1), dpx.reshape(-1,1)))])
        
        if n > 1:
            dpn = []
            # assuming n starts at 1
            for i in range(n-1):
                temp_dpx = (self.plate_centerx - self.samples[:,1])*self.ratio
                temp_dpy = (self.plate_centery - self.samples[:,0])*self.ratio
                temp = np.hstack((temp_dpy.reshape(-1,1), temp_dpx.reshape(-1,1)))
                dpn.append(temp)
           
            temp = np.hstack((dpy.reshape(-1,1), dpx.reshape(-1,1)))
            dpn.append(temp)
            
            return dpn

    def delta_edge(self, n=1):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between self.obj's and the samples' bottom right edges (xmax,ymax)
        
        Outputs:
            dex (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dey (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
            den (list): list of length n, where each item in list is a Nx2
                array of the dey and dex values (y,x pairs)
        """ 
        dex, dey = self.d_edge(self.samples[:,1], self.samples[:,0])
        if n == 1:
            return list([np.hstack((dey.reshape(-1,1), dex.reshape(-1,1)))])
        
        if n > 1:
            den = []
            # assuming n starts at 1
            for i in range(n-1):
                temp_dex = ((self.n_objs[i,1] + self.new_obj_width/2) - 
                    (self.samples[:,1] + self.new_obj_width/2))*self.ratio
                temp_dey = ((self.n_objs[i,0] + self.new_obj_height/2) -
                    (self.samples[:,0] + self.new_obj_height/2))*self.ratio
                temp = np.hstack((temp_dey.reshape(-1,1), temp_dex.reshape(-1,1)))
                den.append(temp)
           
            temp = np.hstack((dey.reshape(-1,1), dex.reshape(-1,1)))
            den.append(temp)
            
            return den

    def winner(self, mode, n=1, logprob=False, epsilon=1e-8):
        """
        **DEPRECATING**
        Return object placement with highest score
        
        Inputs:
            mode (string): string specifying how to score. Can be "dcx",
            "dcy, "dpx", "dpx", "dex", "dex", or a combination of the 6.
            if passing more than one, put inside a list.
            Can also pass in "all" to score based on all 6
            n (int): number of previously placed objects to look back
                at. (see collect_data.gather_data)
            logprob (bool): if true use sum of logs for scoring
            epsilon (float): to prevent zero division
        Outputs:
            output is a 1-D array of 2 elements, the (x,y) coordinates
                of the sample with the highest score, in image coordinates
        NOTE if using KDE, should probably set logprob to False, it already returns log
        """
        if type(mode) is not list:
            mode = list([mode])

        dc = self.delta_centers()
        dp = self.delta_plate()
        de = self.delta_edge()

        total_score = np.zeros(dc[0][:,1].shape)

        for i in range(n):
           
            dcx_score = 0
            dcy_score = 0
            dpx_score = 0
            dpy_score = 0
            dex_score = 0
            dey_score = 0
            
            if 'all' in mode:
                mode = list(['dcx', 'dcy', 'dpx', 'dpy', 'dex', 'dey'])
           
            if 'dcx' in mode:
                dcx_score = self.score.forward(dc[i][:,1], 6*i)
                assert np.sum(dcx_score) != 0

            if 'dcy' in mode:
                dcy_score = self.score.forward(dc[i][:,0], 6*i + 1)
                assert np.sum(dcy_score) != 0

            if 'dpx' in mode:
                dpx_score = self.score.forward(dp[i][:,1], 6*i + 2)
                assert np.sum(dpx_score) != 0

            if 'dpy' in mode:
                dpy_score = self.score.forward(dp[i][:,0], 6*i + 3)
                assert np.sum(dpy_score) != 0

            if 'dex' in mode:
                dex_score = self.score.forward(de[i][:,1], 6*i + 4)
                assert np.sum(dex_score) != 0

            if 'dey' in mode:
                dey_score = self.score.forward(de[i][:,0], 6*i + 5)
                assert np.sum(dey_score) != 0

            if logprob == True:
                dcx_score = np.log(dcx_score + epsilon)
                dcy_score = np.log(dcy_score + epsilon)
                dpx_score = np.log(dpx_score + epsilon)
                dpy_score = np.log(dpy_score + epsilon)
                dex_score = np.log(dex_score + epsilon)
                dey_score = np.log(dey_score + epsilon)

            total_score = total_score + dcx_score + dcy_score + \
                    dpx_score + dpy_score + dex_score + dey_score

        # NOTE: might want to normalize the values

        total_winner = np.argmax(total_score)
        
        return self.samples[total_winner, :]

    def winner2D(self, mode, max_n, n=1, epsilon=1e-8):
        """
        **DEPRECATING**
        Return object placement with highest score, using 2-D gaussians
        NOTE: recommend using plot_2D_gaussian to get these outputs, This
              was not implemented ideally.
        
        Inputs:
            mode (string): string specifying how to score. Can be
                "dc, "dp", "de", or a combination of the 3.
                if passing more than one, put inside a list.
                Can also pass in "all" to score based on all 3
            max_n (int): The maximum number of previous objects you want
                to take into account, ie if you provide a n > max_n, then
                n = max_n.
            n (int): number of previously placed objects to look back
                at. (see collect_data.gather_data)
            epsilon (float): to prevent zero division
        Outputs:
            output is a 1-D array of 2 elements, the (x,y) coordinates
                of the sample with the highest score, in image coordinates
        """
        if type(mode) is not list:
            mode = list([mode])
        assert n >= 0

        if n > max_n:
            n = max_n

        dc = self.delta_centers(n=n)
        dp = self.delta_plate(n=n)
        de = self.delta_edge(n=n)
        total_score = np.zeros(self.num_samples)

        for i in range(n):
            dc_score = 0
            dp_score = 0
            de_score = 0
            
            if 'all' in mode:
                mode = list(['dc', 'dp', 'de'])
            
            if 'dc' in mode:
                dc_score = self.score.forward(dc[-i], 1, i+1)
                assert np.sum(dc_score) != 0

            if 'dp' in mode:
                dp_score = self.score.forward(dp[-i], 2, i+1)
                assert np.sum(dp_score) != 0

            if 'de' in mode:
                de_score = self.score.forward(de[-i], 3, i+1)
                assert np.sum(de_score) != 0

            total_score = total_score + dc_score + dp_score + de_score

        total_winner = np.argmax(total_score)

        last_n_objs = np.array([self.last_obj_centery,
                                self.last_obj_centerx]).reshape(-1,2)

        if n == 1:
            pass

        else:
            last_n_objs = np.vstack((self.n_objs, last_n_objs))
        
        return self.samples[total_winner, :], last_n_objs

    def plot_prediction(self, prediction, width, height):
        """
        Plots the location of the prediction

        Inputs:
            prediction (np.array): 1-D array with 2 elements, (x,y),
                which is the center coordinates of the prediction
            width (int): width of the object to be placed, in pixels
        """
        corner = (prediction[1]-height/2, prediction[0]-width/2)
        box = plt.Rectangle(corner, width, height, linewidth=1,
            edgecolor='r', fill=False)
        plt.close()
        plt.figure()
        img = self.img.copy()
        img_rgb = img[:,:,::-1] #convert BGR to RGB
        plt.imshow(img_rgb)
        plt.gca().add_patch(box)

        plt.show()

        return

    def plot_2D_gaussian(self, mode, n, i=None, save_path=None):
        """
        Plots the mulivariate, multimodal gaussian

        Inputs:
            mode (string): string specifying how to score. Can be
                "dc, "dp", "de", or a combination of the 3.
                if passing more than one, put inside a list.
                Can also pass in "all" to score based on all 3
            n (int): number of previously placed objects to look back
                at. (see collect_data.gather_data)
            i (int): for figure annotation if providing only one
                item in a sequence. i is the index of that item in the
                sequence
            save_path (string): Path to save the plot to, set to None
                to just display the figure
        """
        #Use base cmap to create transparent
        mycmap = heatmap.transparent_cmap(plt.cm.inferno)

        img = self.img.copy() # ground truth image
        img = img[:,:,::-1] #convert BGR to RGB

        w, h = img.shape[1], img.shape[0]
        y, x = np.mgrid[0:h, 0:w]
        # y, x = np.mgrid[125:225, 175:275]

        dc_score = 0
        dp_score = 0
        de_score = 0
        measure = ''

        if mode == 'all':
            mode = list(['dc', 'dp', 'de'])
        
        if 'dc' in mode:
            dcx, dcy = self.d_centers(x, y)
            inputs = np.array([dcx.ravel(), dcy.ravel()]).T
            # dc_score = self.score.forward(inputs, 1, n)
            dc_score = self.score.score_samples(inputs, 'dc', n)
            assert np.sum(dc_score) != 0
            measure = measure + '$\Delta$c '

        if 'dp' in mode:
            dpx, dpy = self.d_plate(x, y) 
            inputs = np.array([dpx.ravel(), dpy.ravel()]).T
            # dp_score = self.score.forward(inputs, 2, n)
            dp_score = self.score.score_samples(inputs, 'dp', n)
            assert np.sum(dp_score) != 0
            measure = measure + '$\Delta$p '

        if 'de' in mode:
            dex, dey = self.d_edge(x, y)
            inputs = np.array([dex.ravel(), dey.ravel()]).T
            # de_score = self.score.forward(inputs, 3, n)
            de_score = self.score.score_samples(inputs, 'de', n)
            assert np.sum(de_score) != 0
            measure = measure + '$\Delta$e '

        Z = dc_score + dp_score + de_score
        assert np.sum(Z) != 0
        Z = -Z
        Z = Z.reshape(y.shape)

        ######stuff for predictions###############
        winner = np.argmin(Z)
        winner = utils.num2yx(winner, 416,416)
        last_n_objs = np.array([self.last_obj_centery,
                        self.last_obj_centerx]).reshape(-1,2)
        if n == 1:
            pass
        else:
            last_n_objs = np.vstack((self.n_objs, last_n_objs))
        ############################################

        #Plot image and overlay colormap
        plt.close()
        fig, ax = plt.subplots(1, 1)
        plt.imshow(img)
        # CB = ax.contour(x, y, Z, norm=LogNorm(vmin=0.001, vmax=1000.0),
            # levels=np.logspace(0, 3, 10), cmap=mycmap, extend='min')

        #for sony demo
        #TODO fix this log scale for the new predictions (9/29/20)
        CB = ax.contour(x, y, Z, norm=Normalize(),#LogNorm(),#vmin=np.min(Z), vmax=np.max(Z)),
            levels=50, cmap=mycmap)#, extend='min')
        # import ipdb; ipdb.set_trace()
        # CB = ax.contour(x, y, Z, norm=LogNorm(vmin=1, vmax=10000.0),
        #     levels=np.logspace(1, 4, 10), cmap=mycmap, extend='min')
        plt.colorbar(CB)
        
        plt.title(f'Normalized negative log-likelihood predicted by GMM \n Based on {measure} and n = {n}')
        if save_path is not None:
            if i is None:
                i = ''
            plt.savefig(f'{save_path}/figure{i}_gaussian.png')
        else:
            plt.show()

        return winner, last_n_objs

    def d_centers(self, samplesx, samplesy):
        """
        Takes the randomly sampled pixles and returns the distance in meterse
        between self.obj's center and the samples
        
        Outputs:
            dcx (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dcy (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
        """ 
        dcx = (samplesx - self.last_obj_centerx)*self.ratio
        dcy = (samplesy - self.last_obj_centery)*self.ratio
        return dcx, dcy

    def d_plate(self, samplesx, samplesy):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between the plate/cutting board center and the samples' centers
        
        Outputs:
            dpx (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dpy (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
        """ 
        dpx = (self.plate_centerx - samplesx)*self.ratio
        dpy = (self.plate_centery - samplesy)*self.ratio
        return dpx, dpy
    
    def d_edge(self, samplesx, samplesy):
        """
        Takes the randomly sampled pixles and returns the distance in meters
        between self.obj's and the samples' bottom right edges (xmax,ymax)
        
        Outputs:
            dex (np.array): size N array, where N is the number of samples,
                gives distance between centers in horizontal direction
            dey (np.array): size N array, where N is the number of samples,
                gives distance between centers in vertical direction
        """ 
        dex = (self.last_obj[3] - (samplesx + self.new_obj_width/2))*self.ratio
        dey = (self.last_obj[4] - (samplesy + self.new_obj_height/2))*self.ratio
        return dex, dey
