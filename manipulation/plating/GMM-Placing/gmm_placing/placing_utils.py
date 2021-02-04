import numpy as np
import matplotlib.pyplot as plt
import copy

from gmm_placing import gaussian

def get_n_nearest(value, neighbors, n=None, remove_value=True):
    """
    Return the n closest neighbors from value measured by distance between
    the neighbors and value.

    Inputs:
        value (float or int): value to compare the values in neighbors to
        neighbors (array-like): 1-D array or a list of the values to compare to
        n (int): the number of neighbors to return
        remove_value (bool): if True, remove all occurences of value from neighbors
    """
    if type(neighbors) is list:
        neighbors = np.array(neighbors)
    assert neighbors.ndim == 1
    if n is not None:
        assert n <= neighbors.shape[0]
    if remove_value:
        neighbors = neighbors[neighbors != value]
    dist = np.abs(value - neighbors) #TODO needs to decide if you want to weight previous or future neighbor more heavily
    indexes = np.argsort(dist)
    if n is not None:
        indexes = indexes[:n]
    
    return neighbors[indexes]

def get_relative_weights(num_neighbors, delta_values=None, exponent=2, normalize=True):
    """
    Get scalar weight values for a set of numbers. Weights are gaussian or exponential.

    Inputs:
        num_neighbors (int): The number of neighbors to return weights for
        delta_values (np.array): The array of distance values to pick 
    """
    # TODO should change the gaussian parameters a bit to make weighting better.
    if delta_values is not None:
        delta_values /= np.max(delta_values)
        weights = gaussian.gaussian(delta_values, scipy=True, data=None, mu=0, sigma=1)
    else:
        weights = 1 / np.arange(1,(num_neighbors+1))**exponent
    if normalize:
        weights = weights / weights.sum()
    return weights

def dict_list_to_numpy(D):
    """
    Convert a dictionary whose values are dictionaries that contain
    lists to numpy arrays. This makes a copy of the dictionary.
    e.g. {key1:{key1.1:[list], key1.2:[list]}, key2:{...}...} to
    => {key1:{key1.1:[array], key1.2:[array]}, key2:{...}...}
    """
    d = copy.deepcopy(D)
    for firstkey in d.keys():
        for secondkey in d[firstkey].keys():
            d[firstkey][secondkey] = np.array((d[firstkey][secondkey]))

    return d

def combine_dict_keys(D, keys, new_keys):
    """
    Combine the array values in a dict of the format:
    e.g. {key1:{key1.1:{key1.1.1:array1, ...}, ...}, key2:{key1.1:{key2.1.1:array2, ...}, ...}, ...} to
    => {key1&2:{key1&2.1:{key1&2.1.1:array1&2, ...}, ...}, key3:{...}, ...}
    Inputs:
        D (dict): dictionary with 2 sub-dictionaries e.g. dict{dict{dict{...}}}
            assuming each sub-dictionary is the same format as the others
        keys (list): each item in this list is another list containing the 
            keys to combine together. These are the keys for the outter most dictionary
        new_keys (list): each item in this list is the new key to use. 
            should be the same length as keys
    """
    assert len(keys) == len(new_keys)
    d = {}
    #TODO this probably isn't the cleanest way to do this
    for i, combinekeys in enumerate(keys):
        d[new_keys[i]] = {}
        for key in combinekeys:
            for keykey in D[key].keys():
                if keykey not in d[new_keys[i]].keys():
                    d[new_keys[i]][keykey] = {}
                for keykeykey in D[key][keykey].keys():
                    if keykeykey in d[new_keys[i]][keykey]:
                        temp1 = d[new_keys[i]][keykey][keykeykey]
                        temp2 = D[key][keykey][keykeykey].copy().reshape(-1, 1)
                        d[new_keys[i]][keykey][keykeykey] = np.hstack((temp1, temp2))
                    else:
                        d[new_keys[i]][keykey][keykeykey] = D[key][keykey][keykeykey].copy().reshape(-1, 1)
    return d

def count_keys(dict_test):
    """
    https://stackoverflow.com/a/35428077
    """
    return sum(1+count_keys(v) if isinstance(v,dict) else 1 for _,v in dict_test.items())

def get_bbox_dims(label):
    """
    Get the height and width of a bounding box.
    Args:
        label (np.array):  shape (4,) array describing the bounding box.
            Format: 
            - [0],[1] x_min and y_min coordinates
            - [2],[3] x_max and y_max coordinates
            NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
    """
    assert label.shape == (4,)
    width = label[2] - label[0]
    height = label[3] - label[1]
    return height, width

def get_bbox_center(label, return_dims=False):
    """
    Get the center of a bounding box.
    Args:
        label (np.array):  shape (4,) array describing the bounding box.
            Format: 
            - [0],[1] x_min and y_min coordinates
            - [2],[3] x_max and y_max coordinates
            NOTE: This is values 1-4 of a yolo label so: YOLO_label[1:5]
    """
    assert label.shape == (4,)
    height, width = get_bbox_dims(label)
    centerx = label[0] + width/2
    centery = label[1] + height/2
    if return_dims:
        return centerx, centery, height, width
    else:
        return centerx, centery

def plot_deltas(X, Y, U, V, C=None, image=None, title=None, cbar_label=None,
                xlabel=None, ylabel=None, q_key_label=None, q_key_ref=None):
    """
    TODO need to double check that this is working properly and finish documentation
    """
    #plot all of the data and color the sequence indexes different colors
    fig, ax = plt.subplots()
    if image is not None:
        plt.imshow(image)
    ax.set_title(title)
    # plot the figure where arrows length is same units as axis
    Q = ax.quiver(X, Y, U, V, C, angles='xy', scale_units='xy', scale=1)
    if q_key_label is not None:
        qk = ax.quiverkey(Q, 0.2, 0.9, q_key_ref, q_key_label, labelpos='E', coordinates='figure')
    cbar = plt.colorbar(Q)
    cbar.set_label(cbar_label, rotation=270)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return fig, ax