import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys

from rnns import utils
from prediction_utils import gaussian
from prediction_utils import collect_data
from prediction_utils import heatmap

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

        dc_score = 0
        dp_score = 0
        de_score = 0
        measure = ''

        if mode == 'all':
            mode = list(['dc', 'dp', 'de'])
        
        if 'dc' in mode:
            dcx, dcy = self.d_centers(x, y)
            inputs = np.array([dcx.ravel(), dcy.ravel()]).T
            dc_score = self.score.forward(inputs, 1, n)
            assert np.sum(dc_score) != 0
            measure = measure + '$\Delta$c '

        if 'dp' in mode:
            dpx, dpy = self.d_plate(x, y) 
            inputs = np.array([dpx.ravel(), dpy.ravel()]).T
            dp_score = self.score.forward(inputs, 2, n)
            assert np.sum(dp_score) != 0
            measure = measure + '$\Delta$p '

        if 'de' in mode:
            dex, dey = self.d_edge(x, y)
            inputs = np.array([dex.ravel(), dey.ravel()]).T
            de_score = self.score.forward(inputs, 3, n)
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
        CB = ax.contour(x, y, Z, norm=LogNorm(vmin=1, vmax=1000.0),
            levels=np.logspace(1, 4, 10), cmap=mycmap, extend='min')
        plt.colorbar(CB)
        
        plt.title(f'Negative log-likelihood predicted by GMM \n Based on {measure} and n = {n}')
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
