from __future__ import division
import numpy as np
from scipy.stats import t

debug = False 

class GaussianPredictor(object):
  # You can't just ask someone what the probability is
  # https://en.wikipedia.org/wiki/Conjugate_prior
  def __init__(self, num_variables, apriori_n, apriori_mu0, apriori_alpha, apriori_beta):
    self.num_variables = num_variables 
    self.apriori_n = apriori_n
    self.apriori_mu0 = apriori_mu0
    self.apriori_alpha = apriori_alpha
    self.apriori_beta = apriori_beta

  def Fit(self, x_d):
    if (type(x_d) is not np.ndarray or x_d.ndim != 2):
      raise ValueError('x_d should be a two dimensional numpy array which is number_of_data_points by the number of variables measured')
    if (x_d.shape[1] == 0):
      raise ValueError('The second dimension of x_d should not be zero (you need to have variables, even if you have no measurements)')

    total_time = x_d.shape[0]
    num_variables = x_d.shape[1]

    if (total_time == 0):
      mean_val = np.zeros(num_variables)
      ssd = np.zeros(num_variables)
    else:
      mean_val = np.mean(x_d,0)
      ssd = np.sum((x_d - mean_val)**2,0)

    return self.FitToParams(mean_val, ssd, total_time, num_variables)
    
  def FitToParams(self, sample_mean, sample_ssd, sample_n, num_variables):
    if (np.any(np.isnan(sample_mean)) or np.any(np.isnan(sample_ssd)) or np.any(np.isnan(sample_n))):
      raise ValueError('There should be no nans in your data')

    if (type(sample_mean) is not np.ndarray or sample_mean.ndim != 1 
          or sample_mean.shape[0] != self.num_variables):
      raise ValueError('sample_mean should be a two dimensional numpy array that is length of number of variables measured')

    if (type(sample_ssd) is not np.ndarray or sample_ssd.ndim != 1 
          or sample_ssd.shape[0] != self.num_variables):
      print(sample_ssd)
      raise ValueError('sample_ssd should be a one dimensional numpy array that is length of number of variables measured')

    self.post_mean = (self.apriori_n * self.apriori_mu0 + sample_n * sample_mean)/(self.apriori_n + sample_n)
    self.post_n = self.apriori_n + sample_n
    self.post_alpha = self.apriori_alpha + sample_n/2.0
    if (sample_n == 0):
      self.post_beta = self.apriori_beta
    else:
      # compute the new sample ssd / 2, assuming that beta is twice the prior ssd
      self.post_beta = (sample_ssd + self.apriori_beta*2 + sample_n * (sample_mean-self.post_mean)**2 + self.apriori_n * (self.apriori_mu0-self.post_mean)**2)/2.0

    return self

  def GetProbability(self, x):
    if (type(x) is not np.ndarray or x.ndim != 1 
          or x.shape[0] != self.num_variables):
      raise ValueError('x should be a one dimensional numpy array that is length of number of variables measured')
    t_dof = self.post_alpha * 2 
    # copied from 
    # https://en.wikipedia.org/wiki/Conjugate_prior
    # some day I'd love to learn how to derive this
    scale_param = np.sqrt(self.post_beta * (self.post_n + 1)/(self.post_n * self.post_alpha))
    log_prob = 0
    for i in range(self.num_variables):
      if debug:
        print("Computing T dist pdf at %s, for mean %s, variance %s, and dof %s" % (x, self.post_mean, scale_param, t_dof))
      log_prob = log_prob + np.log(t.pdf(x[i], t_dof, loc=self.post_mean[i], scale=scale_param[i]))
    return np.exp(log_prob)

# given a d x vars dataset of observtions, return a GaussianPredictor model
def DefaultGaussianModelFactory(x_d):
  if (type(x_d) is not np.ndarray or x_d.ndim != 2):
    raise ValueError('x_d should be a two dimensional numpy array which is number_of_data_points by the number of variables measured')
  if (x_d.shape[1] == 0):
    raise ValueError('The second dimension of x_d should not be zero (you need to have variables, even if you have no measurements)')

  num_variables = x_d.shape[1]
  
  apriori_n = 3.0
  apriori_mu0 = np.zeros(num_variables)
  apriori_alpha = apriori_n/2
  apriori_beta = (np.ones(num_variables) * 2)/2

  gauss_pred = GaussianPredictor(num_variables, apriori_n, apriori_mu0, apriori_alpha, apriori_beta)
  return gauss_pred.Fit(x_d)

