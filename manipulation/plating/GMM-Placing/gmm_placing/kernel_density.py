import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from matplotlib.ticker import NullFormatter

# Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html

class KDE:
    def __init__(self, bandwidth=1.0, kernel_type='gaussian'):
        """
        Custom version of scipy's KernelDensity.

        Inputs:
            bandwidth (float): bandwidth of kernel (std for gaussian).
            kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        """        
        self.train_data = train_data
        self.bandwidth = bandwidth
        if kernel_type != 'gaussian':
            raise NotImplementedError
        self.kernel_type = kernel_type

    # def fit(self, train_data, sample_weights=None):
    #     """
    #     Inputs:
    #         train_data (np.array): A (num_samples x num_features) array containing
    #             the training data to fit a kernel density estimate distribution to.
    #             Each row represents one data sample.
    #         sample_weights (np.array): A (num_samples) array of weight for
    #             each sample in train_data array.
    #     """
    #     if kernel_type == 'gaussian':
    #         score = np.exp((-x*x)/(2*h*h)) # mulitiplication is faster than exponentiation
    #     else:
    #         raise NotImplementedError
    #     d

    # def score_samples(self, x):
    #     """
    #     Inputs:

    #     """


    def forward(self, x, idx):
        """
        Inputs:
            x (np.array): The input data to get the log probability of
        Outputs:
            output (np.array): 1-D array of the log probability values of x
        """
        # instantiate and fit the KDE model
        if self.bandwidth is None:
            output = kde_func(x, self.train_data[idx], bandwidth=self.bandwidth,
                              kernel_type=self.kernel_type)
        else:
            output = kde_func(x, self.train_data[idx], bandwidth=self.bandwidth[idx],
                              kernel_type=self.kernel_type)
        
        return output

def kde_func(train_data, x=None, bandwidth=None, kernel_type='gaussian',
             num_samples=200, return_dist=False, sample_weights=None,
             scipy=True, n_jobs=4):
    """
    Fits a kernel density estimation to data set and returns the distribution
    or it returns the log probability given input data. Wrapper function for
    scipy's KernelDensity() and KDE()

    Inputs:
        train_data (np.array): (num_samples x num_features) array of training
            samples. Each row corresponds to a single data point. The distribution
            is fit to this data.
        x (np.array): The input data to get the log probability of. Leave as None
            to return the distribution (you also need to set the return_dist flag).
        bandwidth (float): bandwidth of kernel (std for gaussian), if None
            will use opt_bandwidth to determine the bandwidth.
        kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        num_samples (int): number of bandwidth values to test for KDE distribution
            (when setting bandwidth to None).
        return_dist (bool): Set to true to return the distribution instead of density estimates.
        sample_weights (np.array): a (num_samples) size array of the weights for each sample in dataset
        scipy (bool): whether to use scipy KernelDensity() or custom KDE()
        n_jobs (int): The number of CPU processes to use if using opt_bandwidth
            to get the bandwidth values.
    Outputs:
        logprob (np.array): 1-D array of the log probability values of x
    """
    if not return_dist:
        assert x is not None

    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)

    if bandwidth is None:
        xmin = np.min(train_data) - 0.1 * np.abs(np.min(train_data))
        xmax = np.max(train_data) + 0.1 * np.abs(np.max(train_data))
        bandwidth = opt_bandwidth(data=train_data,
                                  low=xmin,
                                  high=xmax,
                                  num_samples=num_samples,
                                  kernel_type=kernel_type,
                                  sample_weights=sample_weights,
                                  n_jobs=n_jobs)

    # instantiate and fit the KDE model
    if scipy:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel_type)
    else:
        kde = KDE(bandwidth, kernel_type)
    kde.fit(train_data, sample_weight=sample_weights)
    
    if return_dist:
        return kde
    else:
        # return the log of the prob density
        if x.ndim == 1:
            x = x.reshape(-1, 1)    
        logprob = kde.score_samples(x)
        
        return logprob

def plot_KDE(data, sample_size, kernel_bandwidth=None, kernel_type='gaussian', alpha=1):
    """
    Plots the kernel density estimation for the given data
    NOTE: should probably just combine this with plot_KDEs

    Inputs:
        data (np.array): 1-D array of data to fit a kernel density estimation to
        sample_size (int): number of times to sample from KDE distribution (resolution)
        kernel_bandwidth (float) bandwidth of kernel (std for gaussian), if None
            will use opt_bandwidth to determine the bandwidth
        kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        alpha (float): transparency of the graph
    """ 
    # make sure the data is 2-D
    assert data.ndim <= 2
    if data.ndim == 1:
        data = data.reshape(-1,1) # input to kde.fit is 2-D

    # define where to sample from the model, added some space at ends
    xmin = min(data) - 0.2 * abs(min(data))
    xmax = max(data) + 0.2 * abs(max(data))
    x = np.linspace(xmin, xmax, sample_size)

    # get best bandwidth if not provided
    if kernel_bandwidth is None:
        kernel_bandwidth = np.abs(opt_bandwidth(data, xmin, xmax, num_samples=100))
        print('Calculated Bandwidth is {:0.4f}'.format(kernel_bandwidth))

    # get logprob of x values using a KDE trained on data 
    logprob = kde_func(x, data, kernel_bandwidth, kernel_type=kernel_type)

    # fill in plot, fill_between takes 1-D input
    x = x.reshape(-1) 
    plt.fill_between(x, np.exp(logprob), alpha=alpha)

    # plot the data point locations
    y = np.zeros(data.shape)
    plt.plot(data, y, '|k', markeredgewidth=1)
    plt.show()

def opt_bandwidth(data, low, high, num_samples=100, kernel_type='gaussian',
                  sample_weights=None, n_jobs=4):
    """
    Calculate the bandwidth that maximizes log-likelihood score
    The GridSearchCV doesn't have a score_samples attribute so you might need to do this
    Inputs:
        data (np.array): 1-D array of data to find bandwidth for
        low (np.array): minimum value to start sample bandwidths from
        high (np.array): maximum value to start sample bandwidths from
        num_samples (int): number of bandwidth values to test for KDE distribution
        kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        sample_weights (np.array): a (number of samples) size array of the weights for each sample in data
    Outputs:
        outputs the optimal bandwidth for the given data
    """
    # make sure the data is 2-D
    assert data.ndim <= 2
    if data.ndim == 1:
        data = data.reshape(-1,1) # input to kde.fit is 2-D

    bandwidths = np.linspace(low, high, num_samples)
    bandwidths = bandwidths.reshape(-1)
    grid = GridSearchCV(estimator=KernelDensity(kernel=kernel_type),
                        param_grid={'bandwidth': bandwidths},
                        n_jobs=n_jobs)
    grid.fit(data, sample_weight=sample_weights)

    return grid.best_params_['bandwidth']

def plot_KDEs(data_list, sample_size, xlabels, titles,
        bandwidths=None, kernel_type='gaussian', cols=None, alpha=1):
    """
    Plots the kernel density estimation for multiple sets of data

    Inputs:
        data_list (list): list of length N, where N is number of datasets.
            each element in list is a 1-D array of data to fit a kernel
            density estimation to
        sample_size (int): number of times to sample from KDE distribution (resolution)
        xlabels (list): list, length N, of strings of the xlabels on subplots
        titles (list): list of strings for the titles of subplots. Length N
        bandwidths: list of bandwidths to use. If None will use opt_bandwidth
        kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        cols (int): number of columns to use for the subplot
        alpha (float): sets the transparency of the plot
    """ 
    assert type(data_list) == list
    
    if cols is None and len(data_list) == 6:
        cols = 3
    elif cols is None:
        cols = 6
    else:
        pass
    h = len(data_list)//cols # number of rows of subplots

    for i in range(len(data_list)):
        plt.subplot(h, cols, i+1)

        # make sure the data is 2-D
        data = data_list[i]
        assert data.ndim <= 2
        if data.ndim == 1:
            data = data.reshape(-1,1) # input to kde.fit is 2-D
        
        # define where to sample from the model, added some space at ends
        xmin = min(data) - 0.2 * abs(min(data))
        xmax = max(data) + 0.2 * abs(max(data))
        x = np.linspace(xmin, xmax, sample_size)

        # calculate optimal bandwidth if not provided
        if bandwidths is None:
            bandwidth = np.abs(opt_bandwidth(data, xmin, xmax, num_samples=100))
            # print('Calculated Bandwidth for "{}" is {:0.4f}'.format(titles[i],bandwidth))
        else:
            bandwidth = bandwidths[i]

        # get logprob of x values using a KDE trained on data 
        logprob = kde_func(x, data, bandwidth, kernel_type=kernel_type)

        # plot the KDE, fill_between takes 1-D input
        x = x.reshape(-1) 
        plt.fill_between(x, np.exp(logprob), alpha=alpha)
        
        # plot the data point locations
        y = np.zeros(data.shape)
        plt.plot(data, y, '|k', markeredgewidth=1)

        plt.xlabel('{} (m)'.format(xlabels[i]))
        plt.ylabel('Density') 
        plt.title('{}'.format(titles[i]))
    # format subplots
    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    plt.show()
