import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from matplotlib.ticker import NullFormatter

# Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html

class KDE:
    def __init__(self, train_data, bandwidth=None, kernel_type='gaussian'):
        """
        Class for kernel density estimation, based on kde_func function
        Made this so I could use kde_func as a method
        For this class, training occurs inside forward

        Inputs:
            train_data (list of np.array): list of len N, with each item
                being a Mx1 array of M training samples. N is number of
                datasets to train
            bandwidth (float): bandwidth of kernel (std for gaussian), if None
                will use opt_bandwidth to determine the bandwidth
            kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        """        
        self.train_data = train_data
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type

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

def kde_func(x, train_data, bandwidth=None, kernel_type='gaussian'):
    """
    Fits a kernel density estimation to data set and returns the log
    probability given input data

    Inputs:
        x (np.array): The input data to get the log probability of
        train_data (np.array): Nx1 array of N training samples
        bandwidth (float): bandwidth of kernel (std for gaussian), if None
            will use opt_bandwidth to determine the bandwidth
        kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
    Outputs:
        logprob (np.array): 1-D array of the log probability values of x
    """
    if bandwidth is None:
        xmin = min(train_data) - 0.1 * abs(min(train_data))
        xmax = max(train_data) + 0.1 * abs(max(train_data))
        bandwidth = np.abs(opt_bandwidth(train_data, xmin, xmax, num_samples=100))

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel_type)
    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)
    kde.fit(train_data)
    
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

def opt_bandwidth(data, low, high, num_samples=100, kernel_type='gaussian'):
    """
    Calculate the bandwidth that maximizes log-likelihood score
    
    Inputs:
        data (np.array): 1-D array of data to find bandwidth for
        low (np.array): minimum value to start sample bandwidths from
        high (np.array): maximum value to start sample bandwidths from
        num_samples (int): number of times to sample from KDE distribution (resolution) 
        kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
    Outputs:
        outputs the optimal bandwidth for the given data
    """
    # make sure the data is 2-D
    assert data.ndim <= 2
    if data.ndim == 1:
        data = data.reshape(-1,1) # input to kde.fit is 2-D

    bandwidths = np.linspace(low, high, num_samples)
    bandwidths = bandwidths.reshape(-1)
    grid = GridSearchCV(KernelDensity(kernel=kernel_type),
                        {'bandwidth': bandwidths}, cv = 6,
                        iid = True) # 6-fold cross validation
    grid.fit(data)

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
