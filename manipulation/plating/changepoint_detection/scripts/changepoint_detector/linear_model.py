from __future__ import division
import numpy as np
from scipy.stats import t
from sklearn import linear_model as lm

debug = False 

class LinearPredictor(object):
  # You can't just ask someone what their probability is
  # https://en.wikipedia.org/wiki/Conjugate_prior
  def __init__(self, num_variables, apriori_n, apriori_mu0, apriori_alpha, apriori_beta, prediction_gap = 0):
    self.num_variables = num_variables 
    # Oliver said I should just make up this math. Later I'll implement the ``correct'' version according to Bayesian statistics
    self.apriori_n = apriori_n
    self.apriori_mu0 = apriori_mu0
    self.apriori_alpha = apriori_alpha
    self.apriori_beta = apriori_beta
    self.prediction_gap = prediction_gap

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
    elif (total_time < 3):
      mean_val = np.mean(x_d,0)
      ssd = np.sum((x_d - mean_val)**2,0)
    else:
      mean_val = np.zeros(num_variables)
      ssd = np.zeros(num_variables)
      for variable_index in range(num_variables):
        model = lm.LinearRegression()
        x = np.arange(total_time)
        y = x_d[:,variable_index]
        x_val_ssd = np.sum(np.square(x - (total_time - 1)/2))
        x_y_sumprod = np.sum((x-(total_time - 1)/2) * (y - np.mean(y)))
        coeff = x_y_sumprod / x_val_ssd
        intercept = np.mean(y) - coeff * (total_time - 1)/2
        preds = x * coeff + intercept
        resids = preds - y
        raw_residual_ssd = np.sum(np.square(resids))
        # inflate the ssd to get what I think is an unbiased estimate of the residual variance 
        # we always have only one INDEPENDENT variable (time), and may have multiple dependent variables (counted in num_variables)
        residual_variance = raw_residual_ssd / (total_time - 1 - 1)
        # compute the variance of the predicted response (see https://en.wikipedia.org/wiki/Mean_and_predicted_response)
        predicted_response_variance = residual_variance * (1 + 1.0/total_time + ((total_time + self.prediction_gap) - (total_time-1)/2)**2/x_val_ssd)
        ssd[variable_index] = predicted_response_variance * total_time
        mean_val[variable_index] = np.array([[total_time + self.prediction_gap]]) * coeff + intercept

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
def DefaultLinearModelFactory(x_d):
  if (type(x_d) is not np.ndarray or x_d.ndim != 2):
    raise ValueError('x_d should be a two dimensional numpy array which is number_of_data_points by the number of variables measured')
  if (x_d.shape[1] == 0):
    raise ValueError('The second dimension of x_d should not be zero (you need to have variables, even if you have no measurements)')

  num_variables = x_d.shape[1]
  
  apriori_n = 3.0
  apriori_mu0 = np.zeros(num_variables)
  apriori_alpha = apriori_n/2
  apriori_beta = (np.ones(num_variables) * 2)/2

  lin_pred = LinearPredictor(num_variables, apriori_n, apriori_mu0, apriori_alpha, apriori_beta)
  return lin_pred.Fit(x_d)

