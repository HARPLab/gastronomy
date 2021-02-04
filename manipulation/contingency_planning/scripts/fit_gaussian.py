from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import joblib
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import multivariate_normal, gaussian_kde

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str, default='same_blocks/pick_up/franka_fingers_contingency_data.npz')
    args = parser.parse_args()
    
    file_path = args.file_path
    data_idx = file_path.rfind('contingency_data')
    prefix = file_path[:data_idx]

    success_success_params_file_path = prefix + 'success_success_contingency_gkde_params.pkl'
    success_failure_params_file_path = prefix + 'success_failure_contingency_gkde_params.pkl'
    failure_success_params_file_path = prefix + 'failure_success_contingency_gkde_params.pkl'
    failure_failure_params_file_path = prefix + 'failure_failure_contingency_gkde_params.pkl'
    failure_params_file_path = prefix + 'failure_contingency_gp_params.pkl'

    data = np.load(args.file_path)

    success_success_data = data['success_success_data']
    success_failure_data = data['success_failure_data']
    failure_success_data = data['failure_success_data']
    failure_failure_data = data['failure_failure_data']
    print(success_success_data.shape)
    # X = data['X']#[::13]
    # input_dim = X.shape[1]
    # success_Y = data['success_Y']#[::13]
    # failure_Y = data['failure_Y']#[::13]

    # normalized_success_Y = success_Y / np.sum(success_Y)
    # normalized_failure_Y = failure_Y / np.sum(failure_Y)
    # normalized_success_failure_Y = np.abs((1-success_Y) / np.sum(1-success_Y))
    # normalized_failure_failure_Y = np.abs((1-failure_Y) / np.sum(1-failure_Y))
    # print(np.min(normalized_success_failure_Y))
    # print(np.min(normalized_failure_failure_Y))

    # num_samples = 100
    # success_success_samples = np.zeros(X.shape)
    # success_failure_samples = np.zeros(X.shape)
    # failure_success_samples = np.zeros(X.shape)
    # failure_failure_samples = np.zeros(X.shape)

    # for i in range(num_samples):
    #     success_success_samples[i,:] = X[np.random.choice(np.arange(X.shape[0]), p=normalized_success_Y.flatten())]
    #     success_failure_samples[i,:] = X[np.random.choice(np.arange(X.shape[0]), p=normalized_success_failure_Y.flatten())]
    #     failure_success_samples[i,:] = X[np.random.choice(np.arange(X.shape[0]), p=normalized_failure_Y.flatten())]
    #     failure_failure_samples[i,:] = X[np.random.choice(np.arange(X.shape[0]), p=normalized_failure_failure_Y.flatten())]
    success_success_kernel = gaussian_kde(np.transpose(success_success_data))
    success_failure_kernel = gaussian_kde(np.transpose(success_failure_data))
    failure_success_kernel = gaussian_kde(np.transpose(failure_success_data))
    failure_failure_kernel = gaussian_kde(np.transpose(failure_failure_data))
 
    # success_success_mean = np.mean(success_success_data, axis=0)
    # success_success_cov = np.cov(success_success_data, rowvar=0)
    # success_failure_mean = np.mean(success_failure_data, axis=0)
    # success_failure_cov = np.cov(success_failure_data, rowvar=0)
    # failure_success_mean = np.mean(failure_success_data, axis=0)
    # failure_success_cov = np.cov(failure_success_data, rowvar=0)
    # failure_failure_mean = np.mean(failure_failure_data, axis=0)
    # failure_failure_cov = np.cov(failure_failure_data, rowvar=0)

    # np.savez('franka_fingers_gaussian.npz', success_success_mean=success_success_mean,
    #                                         success_success_cov=success_success_cov,
    #                                         success_failure_mean=success_failure_mean,
    #                                         success_failure_cov=success_failure_cov,
    #                                         failure_success_mean=failure_success_mean,
    #                                         failure_success_cov=failure_success_cov,
    #                                         failure_failure_mean=failure_failure_mean,
    #                                         failure_failure_cov=failure_failure_cov)

    # print(success_success_mean)
    # print(success_success_cov)

    # max_x = np.max(success_success_data,axis=0)
    # min_x = np.min(success_success_data,axis=0)

    # x_ = np.linspace(min_x[0], max_x[0], 100)
    # y_ = np.linspace(min_x[1], max_x[1], 100)
    # #z_ = np.linspace(min_x[2], max_x[2], 100)
    # z_ = [0]

    # x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    # print(x.shape)

    # pos = np.concatenate((x.reshape(100,100,1),y.reshape(100,100,1),z.reshape(100,100,1)),axis=-1)

    # nx, ny = (10, 10)
    # x = np.linspace(min_x[0], max_x[0], nx)
    # y = np.linspace(min_x[1], max_x[1], ny)

    # xv, yv, zv = np.meshgrid(x, y, [0])
    # pos = np.vstack((xv.flatten(),yv.flatten(),zv.flatten()))
    # print(pos.shape)
    
    # print(xv)
    # print(yv)
    # print(zv)
    # x, y = np.mgrid[min_x[0]:max_x[0]:.01, min_x[1]:max_x[1]:.01]
    # #print(x.shape)
    # pos = np.dstack((x, y))
    # new_pos = np.concatenate((pos,np.zeros((x.shape[0],x.shape[1],1))),axis=-1)
    # print(new_pos.shape)
    # print(new_pos.reshape())

    #for i in range()

    # success_success_rv = multivariate_normal(success_success_mean, success_success_cov)
    # success_failure_rv = multivariate_normal(success_failure_mean, success_failure_cov)
    # failure_rv = multivariate_normal(failure_mean, failure_cov)

    # success_success_pdf = success_success_rv.pdf(new_pos)
    # success_failure_pdf = success_failure_rv.pdf(new_pos)
    # success_success_pdf = success_success_kernel.pdf(pos)
    # success_failure_pdf = success_failure_kernel.pdf(pos)
    # failure_success_pdf = failure_success_kernel.pdf(pos)
    # failure_failure_pdf = failure_failure_kernel.pdf(pos)

    joblib.dump(success_success_kernel, success_success_params_file_path)
    joblib.dump(success_failure_kernel, success_failure_params_file_path)
    joblib.dump(failure_success_kernel, failure_success_params_file_path)
    joblib.dump(failure_failure_kernel, failure_failure_params_file_path)
    #print(success_failure_pdf.shape)
    #print(xv.shape)
    # success_pdf = success_success_pdf / (success_success_pdf+success_failure_pdf)
    # failure_pdf = failure_success_pdf / (failure_success_pdf+failure_failure_pdf)


    # fig2 = plt.figure()

    # ax2 = fig2.add_subplot(111)

    # p = ax2.contourf(xv.reshape(10,10), yv.reshape(10,10), failure_pdf.reshape(10,10))
    # fig2.colorbar(p)
    # plt.show()

    # print(success_mean)
    # print(success_cov)
    # print(failure_mean)
    # print(failure_cov)

    # success_gaussian = stats.gaussian_kde(np.transpose(success_samples))
    # failure_gaussian = stats.gaussian_kde(np.transpose(failure_samples))

    # success_reg = GradientBoostingRegressor(random_state=0)
    # success_reg.fit(X, success_Y.flatten())
    # success_clf = KernelRidge(alpha=1.0)
    # success_clf.fit(X, success_Y)
    # joblib.dump(success_clf, success_params_file_path)

    # failure_reg = GradientBoostingRegressor(random_state=0)
    # failure_reg.fit(X, failure_Y.flatten())
    # failure_clf = KernelRidge(alpha=1.0)
    # failure_clf.fit(X, success_Y)
    # joblib.dump(failure_clf, failure_params_file_path)

    # def gaus(x,mu,sigma):
    #     return np.exp(-0.5*np.transpose(x-mu)*np.linalg.inv(sigma)*(x-mu))/np.sqrt((2*np.pi)**input_dim * np.linalg.det(sigma))

    # popt,pcov = curve_fit(gaus,X,success_Y)#, p0=[np.ones(3), np.eye(3)])

    # print(X.shape)

    # success_gp = GaussianProcessRegressor(n_restarts_optimizer=9)
    # # Fit to data using Maximum Likelihood Estimation of the parameters
    # success_gp.fit(X, success_Y)

    
    
    # failure_gp = GaussianProcessRegressor(n_restarts_optimizer=9)
    # # Fit to data using Maximum Likelihood Estimation of the parameters
    # failure_gp.fit(X, failure_Y)

    # joblib.dump(failure_gp, failure_params_file_path)
 