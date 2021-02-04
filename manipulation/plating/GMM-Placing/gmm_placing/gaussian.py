import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm
from matplotlib.colors import LogNorm
from scipy.stats import norm
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture

from gmm_placing import collect_data, placing_utils, kernel_density

"""
Some Referneces:
# https://stackoverflow.com/questions/33158726/fitting-data-to-multimodal-distributions-with-scipy-matplotlib
# https://www.youtube.com/watch?v=P-iAd8b7zl4
# https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
"""

class SequenceGMMs:
    def __init__(self, train_data):
        """
        Train gaussains offline with respect to their temporal position in a sequence.
        This includes training the guassians w.r.t. all of the other temporal positions
        in a sequence. Use gaussians to score given data. Weighting occurs during scoring
        and not while training the gaussians.
        Inputs:
            train_data (dict): Dictionary of data. The keys are the features and the 
                values are dicts that contain the sequence index as keys and 
                delta values as the values. These keys represent the delta values from the 
                key value to the previous. e.g. data['dcx'][1] gives the delta center
                value on the x axis from the second to the first placement
        "Outputs:"
            self.data (dict): A dictionary of distance values between items in a sequence. The 
                format is the following:
                {feature_1:
                    {seq_idx_1 :
                        {seq_idx_2 : array(diff values from seq_idx_2 to seq_idx_1), 
                         seq_idx_3 : array(diff values from seq_idx_3 to seq_idx_1), 
                         ... 
                         seq_idx_n : array(diff values from seq_idx_n to seq_idx_1)},
                     seq_idx_2 :
                        {seq_idx_1 : array(diff values from seq_idx_1 to seq_idx_2), 
                         seq_idx_3 : array(diff values from seq_idx_3 to seq_idx_2), 
                         ... 
                         seq_idx_n : array(diff values from seq_idx_n to seq_idx_2)},
                     ...
                     seq_idx_n :
                        {seq_idx_1 : array(diff values from seq_idx_1 to seq_idx_n), 
                         seq_idx_2 : array(diff values from seq_idx_2 to seq_idx_n), 
                         ... 
                         seq_idx_n-1 : array(diff values from seq_idx_n-1 to seq_idx_n)},
                    },
                 ...
                 feature_2:
                    {
                        ... same format as feature_1 ...
                    }
                 ...
                 feature_n:
                    {
                        ... same format as feature_1 ...
                    }
                }.
                EXCEPTION! For the 'dpx' and 'dpy' features, the array values are all the same for each 
                seq_idx. e.g. self.data['dpx'][seq_idx_1][seq_idx_2] = self.data['dpx'][seq_idx_1][seq_idx_3]
                = self.data['dpx'][seq_idx_1][seq_idx_4] ... etc.
                This different because it does not make sense to compare them with respect to other
                values in the sequence, since they are being compared to the same constant position. 
        """
        self.data =  {}

        # get the delta values w.r.t. all of the other values in a sequence, not just the previous one.
        for feature in train_data.keys():
            self.data[feature] = {}
            for index in train_data[feature].keys():
                self.data[feature][index] = {}
                temp_indexes = list(train_data[feature].keys()).copy()
                temp_indexes.remove(index)
                for idx in temp_indexes:
                    if 'dpx' in feature or 'dpy' in feature:
                        #TODO i think you need to remove the multiple dp values, it is messing up prediction
                        #TODO can improve this also not sure if its right, might need to remove the dp being compared to other indexes
                        temp_data = np.array(copy.deepcopy(train_data[feature][index]))
                    else:
                        temp_data = np.zeros((len(train_data[feature][index])))
                        if idx < index:
                            idxs = np.arange(idx, index)
                            for i in idxs:
                                #TODO check that the + and - are correct for this
                                # probably dont need the copy here
                                # temp_data += np.array(copy.deepcopy(train_data[feature][i]))
                                temp_data -= np.array(copy.deepcopy(train_data[feature][i]))
                        elif idx > index:
                            idxs = np.arange(idx, index, -1)
                            for i in idxs:
                                # temp_data -= np.array(copy.deepcopy(train_data[feature][i]))
                                temp_data += np.array(copy.deepcopy(train_data[feature][i]))
                        else:
                            raise ValueError('Multiple indexes that are the same')
                        
                    self.data[feature][index][idx] = temp_data

    def fit_gaussians(self, mode='meanshift', dependence='independent', features=None,
                      new_features=None, bandwidths=None, covariance_type="full", centers=None):
        """
        Inputs:
            mode (str): what method to use to generate the gaussians.
                Can be 'meanshift' or 'kde'.
            dependence (str): how to treat dependence between features when
                generating the guassian models. Can use:
                    'independent' - seperate gaussian for each feature
                    'spatial_xy' - combine the x and y features
                    'all_features' - combine all features
                This still keeps the different indexes in a sequence indepdent
                from one another.
            features (list): mutually exclusive with dependence. It won't
                throw an error but this overides dependence argument.
                This should be a list of length < k-1, where k is the total number
                of features. Each element in the list is either a string (this feature
                won't be combined) or another list containing the strings of the features
                to combine with on another.
                e.g. for spatial features = [['dcx', 'dcy'], ['dpx', 'dpy'], ['dex', 'dey']]
            new_features (list): This is required if features is used. These are the new
                keys to use as the feature names

            covariance_type (string): see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
            centers (list): same format as train_data, except the arrays are Kx1
                where K is the number of cluster centers for each seperate distribution

        Outputs:
            self.gaussians (dict): Same format as self.data, except each data array is replaced
                with a sklearn.mixture.GaussianMixture object that is trained on the corresponding
                data in self.data
        """
        if features is not None:
            #TODO hasn't been tested yet
            assert new_features is not None and len(new_features) == len(features)
            for i in range(len(features)):
                assert features[i] in self.data.keys()
                if type(features[i]) == str:
                    features[i] = [features[i]]
                else:
                    pass
        elif dependence == 'independent':
            #TODO hasn't been tested yet
            # split features individually
            features = list(self.data.keys())
        elif dependence == 'spatial_xy':
            # split features into pairs
            features = [list(self.data.keys())[i:i + 2] for i in range(0, len(list(self.data.keys())), 2)] # no point in doing this if you hardcode the line below
            new_features = ['dc', 'dp', 'de']
        elif dependence == 'all_features':
            #TODO hasn't been tested yet'
            # group all features together
            features = [list(self.data.keys())]
            new_features = ['all']
        else:
            raise ValueError('Invalid dependence argument')

        if features is None and dependence == 'independent':
            temp_data = copy.deepcopy(self.data)
        else:
            # combine the data from the different features
            # same format as self.data except the size of final arrays increased and there are less features
            temp_data = placing_utils.combine_dict_keys(self.data, features, new_features)

        self.gaussians = {}
        #TODO fix the progress bar
        total_keys = placing_utils.count_keys(temp_data)-37
        prog = tqdm.tqdm(initial=0, total=total_keys, file=sys.stdout, desc='Gaussians left to train')
        for feature in temp_data.keys():
            self.gaussians[feature] = {}
            for index in temp_data[feature].keys():
                self.gaussians[feature][index] = {}
                for idx in temp_data[feature][index].keys():
                    if mode == 'meanshift':
                        if centers is None:
                            _, temp_centers = mean_shift_centers(temp_data[feature][index][idx], bandwidths) #TODO need to decide on bandwidth format, its constant across features right now
                            num_components = temp_centers.shape[0]
                        else:
                            raise ValueError('not implemented yet')
                            num_components = centers #TODO need to decide how to format this
                        self.gaussians[feature][index][idx] = GaussianMixture(
                            n_components=num_components,
                            covariance_type=covariance_type                                                               
                        )
                        self.gaussians[feature][index][idx].fit(temp_data[feature][index][idx])
                    elif mode == 'kde':
                        raise ValueError('not implemented yet')
                        #TODO see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
                    else:
                        raise ValueError('Invalid method type')
                    prog.update(1)
                    prog.refresh()

    def score_samples(self, x, feature, seq_idx, ref_seq_idx): #, weights=None):#'relative'):
        """
        Inputs:
            x (np.array): The input data to get scores for, should be of the shape
                (num_samples, num_features) and should be the distance values from
                seq_idx to ref_seq_idx
            feature: the key of the feature to make prediction with
            seq_idx (int): the index in the sequence to make the prediction for
            ref_seq_idx (int): the index in the sequence to make the prediction
                with respect to.
            # weights (np.array): array of weights to use for making prediction.
            #     Should be of shape (seq_length,), but the sequence length depends on the
            #     specific training data used. Leave as None to not weight or use the 
            #     string 'relative' to weight depending on how close the other values
            #     are in the sequence #TODO reword this
        Outputs:
            output (np.array): the weighted log probabilities of shape (n_samples,)
        """
        return self.gaussians[feature][seq_idx][ref_seq_idx].score_samples(x) #TODO add options for score_samples


class LocallyWeightedGMMs:
    def __init__(self, train_data, dependence='independent', features=None, new_features=None):
        """
        Perform "locally weighted" regression on data at runtime. Gather data w.r.t. their
        physical position and temporal position, weight them, and train gaussian distributions
        using this data. Can use the distributions to score samples.
        Inputs:
            train_data (dict): Dictionary of data. The keys are the features and the 
                values are dicts that contain the sequence index as keys and 
                delta values as the values. These keys represent the delta values from the 
                key value to the previous. e.g. data['dcx'][1] gives the delta center
                value on the x axis from the second to the first placement (NOTE: arrays are zero indexed)
            dependence (str): how to treat dependence between features when
                generating the guassian models. Can use:
                    'independent' - seperate gaussian for each feature
                    'spatial_xy' - combine the x and y features
                    'all_features' - combine all features
                This still keeps the different indexes in a sequence indepdent
                from one another.
            features (list): mutually exclusive with dependence. It won't
                throw an error but this overides dependence argument.
                This should be a list of length < k-1, where k is the total number
                of features. Each element in the list is either a string (this feature
                won't be combined) or another list containing the strings of the features
                to combine with on another.
                e.g. for spatial features = [['dcx', 'dcy'], ['dpx', 'dpy'], ['dex', 'dey']]
                assuming the order of the keys in the data dictionary is dcx,dcy,dpx,dpy,dex,dey
            new_features (list): This is required if features is used. These are the new
                keys to use as the feature names
        #TODO should restructure this data. It is overly complicated.
        #TODO all the copies in here are probably unnecessary
        #TODO get all of the data w.r.t each individual sample, instead of w.r.t. the feature/sequence index
        # remove the first dp value
        # might not have to combine the x and y?
        """
        self.data =  {}

        # the features are the delta values        
        for feature in train_data.keys():
            self.data[feature] = {}
            # The index values give the delta values for ith object that the "ROBOT" placed
            # (e.g. index=1 is the first object placed by robot, but 2nd object in sequence w.r.t. the initial object)
            for index in train_data[feature].keys():
                self.data[feature][index] = {}
                # the delta values don't have a "0" key since all delta values are current - previous in sequence
                # indices in temp_indexes specify the delta values from "temp_indexes[i]"" to "index"
                temp_indexes = [0]+list(train_data[feature].keys()).copy()
                temp_indexes.remove(index)
                for idx in temp_indexes:
                    if 'dpx' in feature or 'dpy' in feature:
                        # TODO This is copying the same values for the dp values to have consistent formatting but the dp values aren't different w.r.t. other objects.
                        temp_data = np.array(copy.deepcopy(train_data[feature][index]))
                    else:
                        temp_data = np.zeros((len(train_data[feature][index])))
                        # getting delta values for previously placed objects in the sequence
                        if idx < index:
                            idxs = np.arange(idx, index) + 1 # +1 is here because between (n+1) and n, is indexed by (n+1)
                            for i in idxs:
                                # 
                                temp_data += np.array(copy.deepcopy(train_data[feature][i])) 
                        # get delta values for the objects that were placed later in the sequence
                        elif idx > index:
                            # delta values were calculated by the delta values if the sequence indices (n+1) - n, so +1 indices isn't needed and the delta values are being subtracted
                            idxs = np.arange(idx, index, -1) 
                            for i in idxs:
                                temp_data -= np.array(copy.deepcopy(train_data[feature][i]))
                        else:
                            raise ValueError('Multiple indexes that are the same')
                        
                    self.data[feature][index][idx] = temp_data
        
        # combine features if needed
        if features is not None:
            #TODO hasn't been tested yet
            assert new_features is not None and len(new_features) == len(features)
            for i in range(len(features)):
                assert features[i] in self.data.keys()
                if type(features[i]) == str:
                    features[i] = [features[i]]
                else:
                    pass
        elif dependence == 'independent':
            #TODO hasn't been tested yet
            # split features individually
            features = list(self.data.keys())
        elif dependence == 'spatial_xy':
            # split features into pairs
            features = [list(self.data.keys())[i:i + 2] for i in range(0, len(list(self.data.keys())), 2)] # no point in doing this if you hardcode the line below
            new_features = ['dc', 'dp', 'de']
        elif dependence == 'all_features':
            #TODO hasn't been tested yet'
            # group all features together
            features = [list(self.data.keys())]
            new_features = ['all']
        else:
            raise ValueError('Invalid dependence argument')

        if features is None and dependence == 'independent':
            pass
        else:
            # combine the data from the different features
            # same format as self.data except the size of final arrays increased and there are less features
            self.data = placing_utils.combine_dict_keys(self.data, features, new_features)
      
    def get_pos_neighbor_data(self, position, seq_idx, num_neighbors, dist_feat_key='dp'):
        """
        Return indexes/keys for the samples in the data set that are the nearest neighbors in
        terms of their spatial position. The position is given by dp values relative to the
        center of the plate/cutting board.
        Inputs:
            position (np.array): the x,y position of the sample w.r.t. the center of the
                plate/cutting board, i.e. the dp values.
            seq_idx (int): the sequence index to get the neighbors for. NOTE: Just needed for getting indices
            num_neighbors (int): the number of closest neighbors in dataset to return
            dist_feat_key: key of the feature to use to measure the distance. Default is
                the dp values.
        Outputs:
            output_idx (np.array): A (num_neighbors x 2) array, where each row is the
                sequence index and the sample index. So the correspoinding
                delta values can be obtained by:
                self.data[whatever_feature][sequence_index][reference_sequence_index][sample_index],
                where reference_sequence_idx is the the sequence index you want to measure the delta
                values from. This value doesn't matter for dp values, so you'll need to decide which
                index to use as reference when you want to use these for indices for dc or de calculations.
                (e.g. you can use seq_idx input value or sequence_index-1).
            dist (np.array): A (num_neighbors x 1) array of the distance (dp) values for each sample.
        """
        indexes = self.data[dist_feat_key].keys()
        data = [] # For the dp values
        output_idx = []
        for idx in indexes:
            # assuming all of values are the same (this is how init sets up self.data for dp values)
            temp_data = self.data[dist_feat_key][idx][0]
            data.append(temp_data)
            # # assign each sample its sequence and sample index so we can keep track of it
            # if idx <= seq_idx: #TODO zero indexed values are only for ref_seq array
            #     # reduce the index since indices <= seq_idx are zero indexed
            #     idx -= 1
            temp_seq = np.full((temp_data.shape[0],1), idx)
            temp_idx = np.arange(temp_data.shape[0]).reshape(temp_data.shape[0],1)
            output_idx.append(np.concatenate((temp_seq,temp_idx), axis=1))
        data = np.concatenate(data, axis=0)
        output_idx = np.concatenate(output_idx, axis=0)
        # get the neighbors
        dist = np.linalg.norm((position - data[:,-2:]), axis=1) # TODO no reason to index data?
        sorted_indexes = np.argsort(dist)
        return output_idx[sorted_indexes[:num_neighbors]], dist[sorted_indexes[:num_neighbors]]

    def get_time_neighbor_data(self, seq_idx, num_neighbors, future=False):
        """
        Return the indexes/keys of the samples in the data set that are the nearest neighbors
        in terms of their temporal position in the sequence.
        Inputs:
            seq_idx (int): the reference sequence index to get the neighbors for.
                i.e. for seq_idx=3, its 2 nearest neighbors will be 2 and 1.
            num_neighbors (int): the number of closest neighbors to return data for
            future (bool): whether to include future indexes (greater than seq_idx).
        Outputs:
            output_idx (dict): A (num_neighbors x 2) array, where each row is the
                reference sequence index and the sample index. So the correspoinding delta
                values can be obtained by:
                self.data[whatever_feature][sequence_index][reference_sequence_index][sample_index],
                where reference_sequence_idx is the the sequence index you want to measure the delta
                values from. This value doesn't matter for dp value.
        """
        assert seq_idx != 0
        # assuming all the features have same format
        feature = list(self.data.keys())[0]
        # import ipdb; ipdb.set_trace()
        if future:
            neighbors = list(self.data[feature][seq_idx].keys())
            neighbors = placing_utils.get_n_nearest(seq_idx, neighbors, num_neighbors, remove_value=False) # value has already been removed
        else:
            neighbors = np.arange(seq_idx)[-num_neighbors:]
            # neighbors = np.arange(1, seq_idx+1)[-num_neighbors:]
        output_idx = []
        for neighbor in neighbors:
            num_samples = self.data[feature][seq_idx][neighbor].shape[0]
            temp_seq = np.full((num_samples,1), neighbor)
            temp_idx = np.arange(num_samples).reshape(num_samples,1)
            output_idx.append(np.concatenate((temp_seq,temp_idx), axis=1))
        return np.concatenate(output_idx, axis=0)

    def fit_gaussian(self, feature, seq_idx, time_neighbor_idxs, pos_neighbor_idxs,
                     mode='kde', future=False, bandwidths=None, covariance_type="full",
                     time_neighbor_weights=None, pos_neighbor_weights=None,
                     kernel_type='gaussian', num_samples=100, n_jobs=4):
        """
        Inputs:
            feature: the key of the feature to make prediction with
            seq_idx (int): the index in the sequence to make the prediction for
            time_neighbor_idxs (np.array): (num_samples x 2) array of the indices for the
                relevant temporal neighbor data samples to use for fitting the gaussian.
                Each row is a single data point and should specify the reference_sequence_index
                and the sample_index (see self.get_time_neighbor_data output).
                e.g. data points are called as self.data[feature][seq_idx][reference_sequence_index][sample_index]
            pos_neighbor_idxs (np.array): (num_samples x 2) array of the indices for the
                relevant spatial neighbor data samples to use for fitting the gaussian.
                Each row is a single data point and should specify the sequence_index
                and the reference_sequence_index (see self.get_pos_neighbor_data output).
                e.g. data points are called as self.data[feature][seq_idx][reference_sequence_index][0]

            mode (str): what method to use to generate the gaussians.
                Can be 'meanshift' or 'kde'.
            future (bool): whether to include future data samples of the sequences 
                (i.e. samples that have a sequence index > greater than seq_idx).
            covariance_type (string): see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
            time_neighbor_weights (np.array): a (time_neighbor_idxs.shape[0]) size array
                of the weights for each sample in dataset. The weights should be
                in the same order as time_neighbor_idxs array. They are set to one by default.
            pos_neighbor_weights (np.array): a (pos_neighbor_idxs.shape[0]) size array
                of the weights for each sample in dataset. The weights should be
                in the same order as pos_neighbor_idxs array. They are set to one by default.
            kernel_type (string): see kernel arg here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
            num_samples (int): number of bandwidth values to test for KDE distribution
                (when setting bandwidth to None).
            n_jobs (int): The number of CPU processes to use if using opt_bandwidth
                to get the bandwidth values.

        Outputs:
            output (np.array): the weighted log probabilities of shape (n_samples,)
        """
        #TODO the below is pretty inefficient, maybe switch to using arrays instead of dictionaries or try using itemgetter
        train_data = []
        for idx in range(time_neighbor_idxs.shape[0]):
            ref_seq_idx = time_neighbor_idxs[idx,0]
            sample_idx = time_neighbor_idxs[idx,1]
            train_data.append(self.data[feature][seq_idx][ref_seq_idx][sample_idx,:])
        for idx in range(pos_neighbor_idxs.shape[0]):
            temp_seq_idx = pos_neighbor_idxs[idx,0]
            sample_idx = pos_neighbor_idxs[idx,1]
            # NOTE used [temp_seq_idx - 1] because the seq_idx might not always work. May want to consider using other values but -1 is the only one that will work in all cases.
            train_data.append(self.data[feature][temp_seq_idx][temp_seq_idx-1][sample_idx,:])
        train_data = np.array(train_data)
        assert (time_neighbor_idxs.shape[0] + pos_neighbor_idxs.shape[0]) == train_data.shape[0]

        if time_neighbor_weights is None:
            time_neighbor_weights = np.ones(time_neighbor_idxs.shape[0])
        assert time_neighbor_weights.shape[0] == time_neighbor_idxs.shape[0]
        if pos_neighbor_weights is None:
            pos_neighbor_weights = np.ones(pos_neighbor_idxs.shape[0])
        assert pos_neighbor_weights.shape[0] == pos_neighbor_idxs.shape[0]
        weights =  np.concatenate((time_neighbor_weights, pos_neighbor_weights), axis=0)

        if mode == 'meanshift':
            #TODO look into this, I think you might be making the code needlessly complicated here
            _, temp_centers = mean_shift_centers(train_data, bandwidths) #TODO need to decide on bandwidth format, its constant across features right now
            num_components = temp_centers.shape[0]
            gaussian = GaussianMixture(n_components=num_components,
                                       covariance_type=covariance_type)
            gaussian.fit(train_data)
        elif mode == 'kde':
            if train_data.ndim == 1:
                train_data = train_data.reshape(-1, 1)
            # instantiate and fit the KDE model
            gaussian = kernel_density.kde_func(train_data=train_data,
                                               bandwidth=bandwidths,
                                               kernel_type=kernel_type,
                                               num_samples=num_samples,
                                               return_dist=True,
                                               sample_weights=weights,
                                               n_jobs=n_jobs)
        else:
            raise ValueError('Invalid method type')
        
        return gaussian


class MultiGaussian:
    def __init__(self, train_data, centers, n, num_features=6,
                 covariance_type="full"):
        """
        **DEPRECATING, use SequenceGMMs**
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
    #**DEPRECATING** use sklearn GaussianMixture
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

def mean_shift_centers(data, bandwidth=None):
    """
    Cluster the data and return the mean shift centers and labels
    Inputs:
        data (np.array): data to fit mean shift to. of shape (n_samples, n_features)
        bandwidth (float): bandwidth of kernel. see reference.
    Outputs:
        cluster_centers (np.array): Coordinates of cluster centers.
            shape is [n_clusters, n_features]: 
        labels (np.array): Labels of each point. of shape (n_samples,)

    Ref:
     - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    """
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels, cluster_centers

def mean_shift(data, bandwidth=None):
    """
    **DEPRECATING** use mean_shift_centers
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
        ms = MeanShift(bandwidth=bandwidth[i])
        
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
