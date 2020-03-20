import numpy as np
import matplotlib.pyplot as plt
import argparse

from prediction_utils import collect_data
from prediction_utils import gaussian
from prediction_utils import kernel_density

if __name__ == "__main__":

    workin_dir = '../../images/test04/images'

    box_list = ['05', '06']
    n = 1

    iterations = 8000
    bandwidth = [0.01, 0.01, 0.1, 0.03, 0.01, 0.008]
    bandwidth2 = [0.01, 0.001, 0.02, 0.04, 0.01, 0.005]
    bandwidth = bandwidth * n
    bandwidth2 = bandwidth2 * n
    
    titles = ['Horizontal Distance\nBetween Object Centers', 
    'Veritcal Distance\nBetween Object Centers',
    'Horizontal Distance Between\nObject and Plate Centers',
    'Vertical Distance Between\nObject and Plate Centers',
    'Horizontal Distance\nBetween Object Edges',
    'Veritcal Distance\nBetween Object Edges']
    titles = titles * n

    xlabels = ['dcx', 'dcy', 'dpx', 'dpy', 'dex', 'dey']
    xlabels = xlabels * n

########################### Load Data #################################
    data = collect_data.gather_data(workin_dir, n=n, suffix_list=box_list)
    # save the training data
    # info = 'In ~/darknet/YOLO_test03/labels/data this includes \n \
    #     set01 to set03 which are the sequential images that \n \
    #     go left to right'
    # np.savez('{}/L2RData.npz'.format(workin_dir), data = data,
            #  titles = xlabels, box_list = box_list, info = info)

####################### Plot Multimodal Data ###########################
    labels, centers = gaussian.mean_shift(data, bandwidth2)
    gaussian.plotGaussMix(data, bandwidth, centers, labels, bins=12,
                           titles=titles, xlabels=xlabels)

##################### Kernel Density Estimation ########################
    kernel_density.plot_KDEs(data, 200, xlabels=xlabels, titles=titles)
    # bandwidth = None

######################### Plot Unimodal Data ###########################
    workin_dir = '../../images/test03/labels/data'   
    stuff = np.load('../../images/test03/labels/data/L2RData_caution.npz', allow_pickle=True)
    data = stuff['data']
    xlabels =stuff['titles']
    box_list = stuff['box_list']
    info = stuff['info']

    gaussian.plot_histogram(data, titles, xlabels, density=True)