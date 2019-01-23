import numpy as np
import argparse

import h5py
import os
import pdb
import pickle

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge

from utils.data_utils import recursively_get_dict_from_group
from utils.trajectory_utils import truncate_expert_data

def load_data(h5_path):
    assert os.path.exists(h5_path) and os.path.isfile(h5_path), \
            "h5 file does not exist {}".format(h5_path)
    h5f = h5py.File(h5_path, 'r')
    data = recursively_get_dict_from_group(h5f)
    return data

class DMPTrajectory(object):
    def __init__(self, num_dims, num_basis, num_sensors):
        self.tau = 2.05
        self.alpha = 25.0
        self.beta = self.alpha / 4.0

        self.num_dims = num_dims
        self.num_basis = num_basis
        self.num_sensors = num_sensors

        self.mean = np.array([np.exp(-i * (0.5 / (self.num_basis - 1))) 
                for i in range(self.num_basis)])
        self.std = [0.5 / (0.65 * (self.mean[i+1]-self.mean[i])**2) 
                for i in range(self.num_basis - 1)]
        self.std += [self.std[-1]]
        self.std = np.array(self.std)
        # Get mu and h for all parameters separately
        self.mu_all = np.zeros((num_dims, num_sensors, num_basis))
        self.h_all = np.zeros((num_dims, num_sensors, num_basis))
        for i in range(num_dims):
            for j in range(num_sensors):
                self.mu_all[i, j] = self.mean
                self.h_all[i, j] = self.std

        self.phi_j = np.ones((self.num_sensors))

        print("Mean: {}".format(np.array_str(self.mean, precision=2,
            suppress_small=True, max_line_width=100)))
        print("Std: {}".format(np.array_str(self.std, precision=2,
            suppress_small=True, max_line_width=100)))
        self.print_stuff()


    def print_stuff(self):
        NUM_DIM,  NUM_BASIS = self.num_dims, self.num_basis
        self.c = np.zeros([NUM_DIM, NUM_BASIS])
        self.h = np.zeros([NUM_DIM, NUM_BASIS])
        for i in range(NUM_DIM):
            for j in range(1,NUM_BASIS+1):
                self.c[i,j-1] = np.exp(-(j - 1) * 0.5/(NUM_BASIS - 1))
            for j in range(1,NUM_BASIS):
                self.h[i,j-1] = 0.5 / (0.65 * (self.c[i,j] - self.c[i,j-1])**2)
            self.h[i,NUM_BASIS-1] = self.h[i,NUM_BASIS-2]
        print("Mean: {}".format(np.array_str(self.c[0], precision=2,
            suppress_small=True, max_line_width=100)))
        print("Std: {}".format(np.array_str(self.h[0], precision=2,
            suppress_small=True, max_line_width=100)))

    def get_x_values(self, dt, x_start=1.0):
        x_list = [x_start]
        for i in range(dt.shape[0] - 1):
            dx = -(self.tau * x_list[-1])
            x_list.append(x_list[-1] + dx * dt[i])
        return np.array(x_list)

    def convert_data_to_dmp_train_better(self, data_dict):
        time = data_dict['time']
        dt = time[1:] - time[:-1]
        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"
        q = data_dict['q']
        dq = data_dict['dq']
        ddq = (dq[1:, :] - dq[:-1, :]) / dt
        # Repeat the last x to get equal length array.
        ddq = np.vstack([ddq, ddq[-1:, :]])
        print("ddq: max: {:.3f}, min: {:.3f}, mean: {:.3f}".format(
            ddq.min(), ddq.max(), np.mean(ddq)))

        y0 = q[0]
        force_val = (self.tau**2)*ddq - self.alpha*(self.beta*(y0-q) - self.tau*dq)

        # Get the x values
        x_arr = self.get_x_values(dt)
        # Repeat the last x to get equal length array. Shape (T)
        x_arr = np.hstack([x_arr, x_arr[-1:]])
        # x_arr will be of shape (T, N, M, K)
        for _, axis_size in enumerate(
                [self.num_dims, self.num_sensors, self.num_basis]):
            x_arr = np.repeat(x_arr[..., None], axis_size, axis=-1)
        assert x_arr.shape == \
                (time.shape[0], self.num_dims, self.num_sensors, self.num_basis)
        psi_tijk = np.exp(-self.h_all * (x_arr - self.mu_all)**2)
        psi_tij_sum = np.sum(psi_tijk, axis=-1, keepdims=True)
        feat_tijk = (psi_tijk * x_arr) / (psi_tij_sum + 1e-6)

        # Get the minimum jerk features
        min_jerk_t = np.minimum(-np.log(x_arr[:,:,:,0:1])/self.tau,
                                        np.ones(x_arr[:,:,:,0:1].shape))
        feat_min_jerk_tij = (min_jerk_t**3)*(6*(min_jerk_t**2)-15*min_jerk_t+10)
        
        # Concatenate the basis function features and min jerk fetures
        feat_tijk = self.alpha*self.beta*np.concatenate(
                [feat_min_jerk_tij, feat_tijk], axis=-1)

        # This reshape happens according to "C" order, i.e., last axis change
        # first, this means that 1400 parameters are laid according to each
        # dim.
        X = feat_tijk.copy().reshape(time.shape[0], self.num_dims, -1)
        y = force_val.copy()
        assert X.shape[0] == y.shape[0], "X, y n_samples do not match"
        assert X.shape[1] == y.shape[1], "X, y n_dims do not match"

        # X^T\beta = y (where we have to find \beta)
        return {'X': X, 'y': y}
        

    def convert_data_to_dmp_train_format(self, data_dict):
        time = data_dict['time']
        dt = time[1:] - time[:-1]
        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"
        q = data_dict['q']
        dq = data_dict['dq']
        ddq = (dq[1:, :] - dq[:-1, :]) / dt
        # Repeat the last x to get equal length array.
        ddq = np.vstack([ddq, ddq[-1:, :]])

        # Let's first calculate sum_j(phi_j * f(x;w_j)) (forcing function)
        y0 = q[0]
        force_val = (self.tau**2)*ddq - self.alpha*(self.beta*(y0-q) - self.tau*dq)
        # Get the x values
        x_arr = self.get_x_values(dt)
        # Repeat the last x to get equal length array.
        x_arr = np.hstack([x_arr, x_arr[-1:]])[:, None]

        x_arr_rep = np.repeat(x_arr,  self.num_basis, axis=1)

        psi_k = np.exp(-self.std * (x_arr_rep - self.mean)**2)
        psi_k_sum = np.sum(psi_k, axis=1, keepdims=True)
        # feat = (psi_k * x_arr) / (psi_k_sum * self.mean)
        feat = (psi_k * x_arr) / (psi_k_sum + 1e-6)

        # Add min jerk trajectory to factor array
        min_jerk_t = np.minimum(-np.log(x_arr)/self.tau, np.ones(x_arr.shape))
        feat_min_jerk = (min_jerk_t**3)*(6*(min_jerk_t**2) - 15*min_jerk_t + 10)

        # feat is shaped (T, num_basis+1)
        feat = np.hstack([feat_min_jerk, feat])


        # TODO(Mohit): Premultiply here with phi_j's if they are not 1.
        psi_jk = np.tile(feat, (1, self.num_sensors))
        psi_ijk = np.tile(psi_jk, (self.num_dims, self.num_dims))

        X = psi_ijk.copy()
        y = (force_val.copy().flatten('F'))/(self.alpha * self.beta)
        assert X.shape[0] == y.shape[0], "X, y shape do not match"

        # X^T\beta = y (where we have to find \beta)
        return {'X': X, 'y': y[:, None]}

    def run_dmp_with_weights(self, weights, y0, dt, traj_time=100):
        '''Run DMP with given weights.
        
        weights: array of weights. size: (N*M*K, 1) i.e. 
            (num_dims*num_sensors*num_basis, 1)
        y0: Start location for dmps. Array of size (N,)
        dt: Time step to use. Float.
        traj_time: Time length to sample trajectories. Integer
        '''
        x = 1.0
        y  = np.zeros((traj_time, self.num_dims))
        dy = np.zeros((traj_time, self.num_dims))
        y[0] = y0
        # This reshape happens along the vector i.e. the first (M*K) values 
        # belong to dimension 0 (i.e. N = 0). Following (M*K) values belong to
        # dimension 1 (i.e. N = 1), and so forth.
        # NOTE: We add 1 for the weights of the jerk basis function
        for i in range(traj_time - 1):
            # psi_ijk is of shape (N, M, K)
            psi_ijk = np.exp(-self.h_all * (x-self.mu_all)**2)
            psi_ij_sum = np.sum(psi_ijk, axis=2, keepdims=True)
            f = (psi_ijk * weights[:, :, 1:] * x).sum(
                    axis=2, keepdims=True) / (psi_ij_sum + 1e-6)
            f_min_jerk = min(-np.log(x)/self.tau, 1)
            f_min_jerk = (f_min_jerk**3)*(6*(f_min_jerk**2) - 15*f_min_jerk+ 10)
            psi_ij_jerk = weights[:, :, 0:1] * f_min_jerk
            
            # calcualte f(x; w_j) -- shape (N, M)
            all_f_ij = self.alpha * self.beta * (f + psi_ij_jerk).squeeze()
            # Calculate sum_j(phi_j * f(x; w_j) -- shape (N,)
            all_f_i = np.dot(all_f_ij, self.phi_j)

            ddy = self.alpha*(self.beta*(y0 - y[i]) - self.tau*dy[i]) + all_f_i
            ddy = ddy / (self.tau ** 2)
            dy[i+1] = dy[i] + ddy * dt
            y[i+1] = y[i] + dy[i+1] * dt

            x += (-self.tau * x) * dt
            
        return y, dy


    def train(self, X_train, y_train, X_test, y_test, use_ridge=False,
              fit_intercept=True):
        if use_ridge:
            clf = Ridge(alpha=1.0,
                        fit_intercept=fit_intercept).fit(X_train, y_train)
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],
                          fit_intercept=fit_intercept).fit(X_train, y_train)
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_train)
        print("Score (max 1.0) Train: {:.3f}, Test: {:.3f}".format(
            train_score, test_score))
        return clf
    
    def save_weights(self, save_path, weights, **kwargs):
        mean = [0.0] + self.mean.tolist()
        std = [0.0] + self.std.tolist()
        phi_j_list = self.phi_j.tolist()
        weights_list = weights.tolist()

        data_dict = {
            'tau': self.tau,
            'alpha': self.alpha,
            'beta': self.beta,
            'num_dims': self.num_dims,
            'num_basis': self.num_basis+1, # +1 for the minimum jerk trajectory
            'num_sensors': self.num_sensors,
            'mu': mean,
            'h': std,
            'phi_j': phi_j_list,
            'weights': weights_list,
        }
        for k, v in kwargs.items():
            data_dict[k] = v

        with open(save_path, 'wb') as pkl_f:
            pickle.dump(data_dict, pkl_f, protocol=2)
            print("Did save dmp params: {}".format(save_path))

def shuffle_data(X, y):
    perm = np.random.permutation(np.arange(X.shape[0]))
    X, y = X[perm], y[perm]
    return X, y

def train1(dmp_traj, h5_path):
    expert_data = load_data(h5_path)
    truncated_expert_data = truncate_expert_data(expert_data)
    X, y = [], []
    for k in sorted(expert_data.keys()):
        data = dmp_traj.convert_data_to_dmp_train_format(
                truncated_expert_data[k])
        assert type(data['X']) is np.ndarray \
            and type(data['y']) is np.ndarray, "Incorrect data type returned"
            
        X.append(data['X'])
        y.append(data['y'])

    # Get train and test data?
    X, y = np.vstack(X), np.vstack(y)
    train_size = int(X.shape[0] * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    clf = dmp_traj.train(X_train, y_train, X_test, y_test,
                         use_ridge=True,
                         fit_intercept=False)
    weights = np.reshape(
            clf.coef_.copy().squeeze(),
            (dmp_traj.num_dims, dmp_traj.num_sensors, dmp_traj.num_basis+1))

    return weights

def train2(dmp_traj, h5_path):
    expert_data = load_data(h5_path)
    truncated_expert_data = truncate_expert_data(expert_data)
    X, y = [], []
    for k in sorted(truncated_expert_data.keys()):
        if k == '3':
            continue
        data = dmp_traj.convert_data_to_dmp_train_better(
                truncated_expert_data[k])
        assert type(data['X']) is np.ndarray \
            and type(data['y']) is np.ndarray, "Incorrect data type returned"
            
        X.append(data['X'])
        y.append(data['y'])

    # Get train and test data?
    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    train_size = int(X.shape[0] * 0.9)
    print("Train size: {} Test size: {}".format(train_size, 
                                                X.shape[0]-train_size))
    X, y = shuffle_data(X, y)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Train a classifier separately for each dimension.
    weights = []
    for i in range(dmp_traj.num_dims):
        clf = dmp_traj.train(X_train[:, i, :],
                             y_train[:, i],
                             X_test[:, i],
                             y_test[:, i], 
                             use_ridge=True,
                             fit_intercept=False)
        weights.append(clf.coef_.copy().squeeze())
        print("Got weights for dim: {}, min: {:.3f}, max: {:.3f}, avg: {:.3f}".
                format(i, weights[-1].min(),
                       weights[-1].max(), weights[-1].mean()))
    
    weights_ijk = np.array([
        np.reshape(W, (dmp_traj.num_sensors, 1+dmp_traj.num_basis))
        for W in weights])
    return weights_ijk

def main(args):
    num_dims, num_basis, num_sensors = 7, 19, 10
    dmp_traj = DMPTrajectory(num_dims, num_basis, num_sensors)
    # weights = train1(dmp_traj, args.h5_path)
    weights = train2(dmp_traj, args.h5_path)
    y, dy = dmp_traj.run_dmp_with_weights(weights,
                                          np.zeros((dmp_traj.num_dims)),
                                          0.05,
                                          traj_time=100)
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Convert data with csvs into h5 file')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to h5 file.')
    args = parser.parse_args()
    import time
    np.random.seed(int(time.time()))
    main(args)
