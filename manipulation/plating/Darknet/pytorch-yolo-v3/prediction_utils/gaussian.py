import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

from prediction_utils import collect_data

"""
Some Referneces:
# https://stackoverflow.com/questions/33158726/fitting-data-to-multimodal-distributions-with-scipy-matplotlib
# https://www.youtube.com/watch?v=P-iAd8b7zl4
# https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
"""

class MultiGaussian:
    def __init__(self, train_data, centers, n, num_features=6,
                 covariance_type="full"):
        """
        *Multimodal and Multivariate*

        Inputs:
            train_data (list): list of len 2m, where m is the number of features 
                in the training set. Assuming there is an x and y for each feature
                and there are next to each other in list (ie dcx,dcy,dpx,dpy,dex,dey)
                Each of the 2m elements in list will have length of n (see below)
                Each of the elements these sublists contain a 1-D array of training data
            centers (list): same format as train_data, except the arrays are Kx1
                where K is the number of cluster centers for each seperate distribution
            n (int): see collect_data.ImageData
            num_features (int): the number of seperate features in the dataset
                (ie, dcx, dpy and dex etc. are three features. Also don't include n in the count)
            covariance_type (string): see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        """
        self.features = {}

        assert type(centers) is list
        assert len(centers) == len(train_data)
        assert len(train_data)//(n) == num_features # make sure x and y for each feature

        train_data = collect_data.split_data(train_data, n)
        centers = collect_data.split_data(centers, n)

        for j in range(0, num_features, 2):
            for i in range(n):
                xcenters = centers[j][i]
                ycenters = centers[j+1][i]
                num_components = max((xcenters.shape[0], ycenters.shape[0]))
                self.features[f'feature{j//2+1}_n={i+1}'] = GaussianMixture(
                    n_components=num_components,
                    covariance_type=covariance_type)
                xdata = train_data[j][i].reshape(-1,1)
                ydata = train_data[j+1][i].reshape(-1,1)
                self.features[f'feature{j//2+1}_n={i+1}'].fit(np.hstack((xdata, ydata)))
        assert len(self.features) == ((num_features//2)*n)

    def forward(self, x, idx, n):
        """
        Inputs:
            x (np.array): The input data to get scores for, should be (x,y) points
                ie. a Nx2 array, where N is the number of samples
            idx (int): the number of the feature to get score from
            n (int): the value of n to score from 
        Outputs:
            output (np.array): the weighted log probabilities of each sample
        """
        output = self.features[f'feature{idx}_n={n}'].score_samples(x)
        
        return output

class GaussianDistribution:
    def __init__(self, scipy=False, data=None, mu=None, sigma=None,
                 log=False, info=False, epsilon=1e-8):
        """
        Class for passing values through a Gaussian distribution, uses 
        gaussian function. made this so I could use gaussian as a method
        For this class, training occurs inside forward
        *Unimodal and univariate*

        Inputs:
            scipy (bool): if True use scipy's pdf functions, use custom is False
                see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
            data (None/np.array): if None use given mu and sigma, if data is given
                then will use data to find mu and sigma
            mu (None/float): mean of the gaussian distribution,
                don't need if data is not None
            sigma (None/float): standard deviation of the gaussian distribution,
                don't need if data is not None
            log (bool): set flag to true to return the log of the outputs
            info (bool): set flag to True to return standard deviation and mean
            epsilon (float): makes sure zero division doesn't happen
        """
        self.scipy = scipy
        self.data = data
        self.mu = mu
        self.sigma = sigma
        self.log = log
        self.info = info
        self.epsilon = epsilon

    def forward(self, x, idx):
        """
        Inputs:
            x (np.array): 1-D array of input values to calculate probabilty density for
        Outputs:
            output (np.array): 1-D array of the output values
            mu (float): mean of the distribution, only return if data arg is given
            sigma (float): standard deviation of the distribution, only return
                if data arg is given
        """
        if self.data is None:
            output = gaussian(x, scipy=self.scipy, data=self.data, mu=self.mu[idx],
                sigma=self.sigma[idx], log=self.log, info=self.info, epsilon=self.epsilon)
        else:
            output = gaussian(x, scipy=self.scipy, data=self.data[idx], mu=self.mu,
                sigma=self.sigma, log=self.log, info=self.info, epsilon=self.epsilon)
        return output

#NOTE might want to make the scoring functions classes so you 
# can use them as inputs to other functions
def gaussian(x, scipy=False, data=None, mu=None, sigma=None,
             log=False, info=False, epsilon=1e-8):
    """
    Returns output values for the elements in data when passed through
    a gaussian distribution
    *Unimodal and univariate*

    Inputs:
        x (np.array): 1-D array of input values to calculate probabilty density for
        scipy (bool): if True use scipy's pdf functions, use custom is False
            see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        data (None/np.array): if None use given mu and sigma, if data is given
            then will use data to find mu and sigma
        mu (None/float): mean of the gaussian distribution,
            don't need if data is not None
        sigma (None/float): standard deviation of the gaussian distribution,
            don't need if data is not None
        log (bool): set flag to true to return the log of the outputs
        info (bool): set flag to True to return standard deviation and mean
        epsilon (float): makes sure zero division doesn't happen
    Outputs:
        outputs (np.array): 1-D array of the output values
        mu (float): mean of the distribution, only return if data arg is given
        sigma (float): standard deviation of the distribution, only return
            if data arg is given
    """

    if scipy == False:
        f = lambda x : 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
        outputs = f(x)

    else:
        if data is not None:
            mu, sigma = norm.fit(data)
        else:
            assert mu is not None and sigma is not None
        outputs = norm.pdf(x, mu, sigma)

    if log == True:
        outputs = np.log(outputs + epsilon)
    
    if info == True:
        return outputs, mu, sigma
    
    return outputs

def plot_histogram(data, titles, xlabels, ylabels=None, bins=15, 
                   density=False, cols=None):
    """
    Plot a histogram of given data
    *Unimodal and univariate*

    Inputs:
        data_list (list): list of length N, where N is number of datasets.
            each element in list is a 1-D array of data that can vary in size
        titles (list): list of strings for the titles of subplots. Length N
        xlabels (list): list, length N, of strings of the xlabels on subplots
        ylabels (list): list, length N, of strings of the ylabels on subplots
        bins (int): number of bins to use for the histogram 
        density (bool): If True, plot the probability density, else just histogram
        cols (int): number of columns to use for the subplot
    """
    if cols is None and len(data) == 6:
        cols = 3
    elif cols is None:
        cols = 6
    else:
        pass
    h = len(data)//cols
    plt.figure()
    for i in range(len(data)):
        plt.subplot(h, cols, i+1) #specify subplot number

        #plot histogram and get x and y values
        if density == False:
            y_hist, x_hist, _ = plt.hist(data[i], bins=15, color='g') #give y axis in terms of frequency
            plt.ylabel('Frequency')

        else:
            y_hist, x_hist, _ = plt.hist(data[i], bins=15, density =True, color='g')
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            y, mu, std = gaussian(x, scipy=True, data=data[i], info=True)
            plt.plot(x, y, 'k', linewidth=2)
            plt.ylabel('Density')
            #make text box showing std and mean
            textstr = '\n'.join((
                r'$\mu=%.4f$' % (mu, ),
                r'$\sigma=%.4f$' % (std, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(x_hist.min(), y_hist.max(), textstr, fontsize=12, 
                    verticalalignment='top', bbox=props)

        plt.xlabel('{} (m)'.format(xlabels[i]))
        plt.title('{}'.format(titles[i]))
 
    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    plt.show()

class GaussianMix:
    def __init__(self, cluster_centers, sigmas, labels, weighting=False):
        """
        Class for gaussian mixture distributions, uses gauss_mixture funciton
        Made this so I could use gaussian_mix as a method
        For this class, you need to train a model first to get centers and sigmas
        *Multimodal and univariate*

        Inputs:
            cluster_centers (np.array): the mean of the distribution to
                sample from. Is a Nx1 array, where N is the number of
                cluster centers (mean of each distribution)
            sigmas (float): the standard deviation of the distributions
                NOTE only takes one right now
            labels (np.array): 1-D array of K inputs, where k is the size of the
                dataset used for clustering. Contains the labels (ie. cluster it
                belongs to) of each element in the dataset
            weighting (Bool): set flag to True to weigh the gaussians by the number
                of occurences they have in the data set (dataset used to make clusters)
        """        
        self.cluster_centers = cluster_centers
        self.sigmas = sigmas
        self.labels = labels
        self.weighting = weighting

    def forward(self, x, idx):
        """
        Inputs:
            x (np.array): 1-D array of M inputs to return outputs for
        Outputs:
            output (np.array): output of x(same shape) when passed through distribution
        """
        output = gauss_mixture(x, cluster_centers=self.cluster_centers[idx], 
            sigmas=self.sigmas[idx], labels=self.labels[idx], weighting=self.weighting)
        
        return output

def mean_shift(data, bandwidth=None):
    """
    Clusters a dataset and returns the cluster centers using meanshift
    *Multimodal and univariate*
    
    Inputs:
        data (np.array/list): 1-D array of M values, where  M is the number
            of samples. If data is a list, then each item in list is a seperate
            dataset of 1-D arrays. This is the data to be clustered.
        bandwidth (float): standard deviation or bandwidth to use for clustering
            If bandwidth is a list, then each item is the bandwidth for
            corresponding dataset in data
    Outputs:
        labels (np.array/list):1-D array of M values, where M is the number
            of samples. Each element in the array is the label
            of the corresponding data point's cluster center it is assigned
            If inputs were list, this is list of same form
        cluster_centers (np.array/list): Mx1 array, where M is the number
            of centers after clustering (ie the mean of each gaussian)
            If inputs were list, this is list of same form
    Notes:
        might need to add a threshold
            ex) thersh = 0.2 #percentage threshold, 
            remove if number of labels in cluster is below 
    """
    # format inputs so for loop works if input was a single array
    if type(data) is not list:
        data = list(data)
    if type(bandwidth) is None:
        bandwidth = list([bandwidth]*len(data))
    assert len(data) == len(bandwidth)

    labels = []
    cluster_centers = []

    for i in range(len(data)):
        temp_data = data[i]
        ms = MeanShift(bandwidth[i])
        
        if temp_data.ndim == 1:
            temp_data = temp_data.reshape(temp_data.shape[0], 1)
        ms.fit(temp_data)
        label = ms.labels_
        cluster_center = ms.cluster_centers_
        labels.append(label)
        cluster_centers.append(cluster_center)

    if len(data) == 1:
        labels = np.array(labels)
        cluster_centers = np.array(cluster_centers)

    return labels, cluster_centers
    
# gives the outputs given some data and the mean and std of each distribution 
def gauss_mixture(x, cluster_centers, sigmas, labels, weighting=False):
    """
    Passes each x value through a gaussian mixture distribution and returns output
    *Multimodal and univariate* - sum of gaussians

    Inputs:
        x (np.array): 1-D array of M inputs to return outputs for
        cluster_centers (np.array): the mean of the distribution to
            sample from. Is a Nx1 array, where N is the number of
            cluster centers (mean of each distribution)
        sigmas (float): the standard deviation of the distributions
            NOTE only takes one right now
        labels (np.array): 1-D array of K inputs, where k is the size of the
            dataset used for clustering. Contains the labels (ie. cluster it
            belongs to) of each element in the dataset
        weighting (Bool): set flag to True to weigh the gaussians by the number
            of occurences they have in the data set (dataset used to make clusters)
    Outputs:
        output (np.array): output of x(same shape) when passed through distribution

    Note: sklearn has a Gaussian mixture function you could use if you want
    """
    # get the number of unique labels (ie, number of clusters)
    uni = np.unique(labels)
    # weight the gaussian in terms of the number of samples per label
    if weighting == True:
        weight = []
        for i in range(uni.shape[0]):
            weight.append(labels[labels==uni[i]].shape[0]/labels.shape[0])
    else:
        #set weights to one so they have no affect if weights is false
        weight = np.ones(cluster_centers.shape[0]) 
    # gaussian distribution function
    f = lambda x, mu, std : 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/(2*std**2))
    # iterate through the number of cluster centers and sum of mixture of gaussians
    func = []
    for j in range(cluster_centers.shape[0]):
        func.append(f(x, cluster_centers[j,0], sigmas)*weight[j])
    
    return np.sum(func, axis=0)

def plotGaussMix(data, bandwidth, centers, labels, bins=15, cols=None, titles=None,
                 xlabels=None, ylabels=None, weighting=True, reso=200):
    """
    Plot data using meanshift cluster and mixture of gaussians
    NOTE: need to make function that returns the equation of the
        gaussians and use that for this funciton (and the labels/centers)
    *Multimodal and univariate*

    Inputs:
        data (list): list of length 6*n, where each group of six contains the
            data for n=1, n=2, ... Each element in the list is a 1-D array
            containing distance measurements. First in group of 6 is dcx, 
            then dcy, dpx, dpy, dex, and dey. (arrays may vary in size)
            (This is for histogram)
        bandwidth (list): list of floats, same length as data, where each
            element is the standard deviation of the gaussian for the 
            corresponding data element.
        centers (list): list of len(data), each item in list contains a
            1-D array of cluster centers for corresponding dataset in data
        labels (list): list of len(data), each item in list contains a
            1-D array of labels for the values in data[n]
        bins (int): number of bins for histogram
        cols (int): number of columns for subplot, set to 1 for single plot
        titles (list): list of strings of the titles for the subplots
        xlabels (list): list of strings of the x axis title for the subplots
        ylabels (list): list of strings of the y axis title for the subplots
        weighting (Bool): set flag to True to weigh the gaussians by the number
            of occurences they have in the data set (dataset used to make clusters)
        reso (int): number of x samples to use for plotting the gaussian
    """
    # format inputs so for loop works if input was a single array
    if type(data) is not list and type(centers) is not list:
        data = list(data)
        bandwidth = list(bandwidth)
        centers = list(centers)
        labels = list(labels)
    assert len(data) == len(bandwidth)

    if cols is None and len(data) == 6:
        cols = 3
    elif cols is None:
        cols = 6
    else:
        pass
    fig_h = len(data)//cols

    plt.figure()
    for i in range(len(data)):
        # plot histogram
        plt.subplot(fig_h, cols, i+1)
        plt.hist(data[i], bins=bins, density=True, color='b')
        # get the limits on the horizontal axis
        xmin, xmax = plt.xlim()
        # plot the gaussian mixture using the mean shift cluster results
        x = np.linspace(xmin,xmax,reso)
        y = gauss_mixture(x, centers[i], bandwidth[i], labels[i], weighting=weighting)
        plt.plot (x, y, 'r', linewidth=2)
        # set axis labels 
        plt.xlabel('{} (m)'.format(xlabels[i]))
        if ylabels is not None:
            plt.ylabel('{}'.format(ylabels[i]))
        else:
            # plt.ylabel('Frequency') #for histogram
            plt.ylabel('Density') #for probabilty density funciton
        plt.title('{}'.format(titles[i]))
    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    plt.show()
