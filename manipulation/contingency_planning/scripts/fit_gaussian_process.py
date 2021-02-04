from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import joblib

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str, default='same_blocks/pick_up/franka_fingers_contingency_data.npz')
    args = parser.parse_args()
    
    file_path = args.file_path
    data_idx = file_path.rfind('contingency_data')
    prefix = file_path[:data_idx]

    success_params_file_path = prefix + 'success_contingency_gp_params.pkl'
    failure_params_file_path = prefix + 'failure_contingency_gp_params.pkl'

    data = np.load(args.file_path)
    X = data['X'][::100]
    input_dim = X.shape[1]
    success_Y = data['success_Y'][::100]
    failure_Y = data['failure_Y'][::100]
    print(X.shape)

    success_gp = GaussianProcessRegressor(n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    success_gp.fit(X, success_Y)

    joblib.dump(success_gp, success_params_file_path)
    
    failure_gp = GaussianProcessRegressor(n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    failure_gp.fit(X, failure_Y)

    joblib.dump(failure_gp, failure_params_file_path)
 