import unittest
from changepoint_detector import linear_model as gm
import numpy as np
from scipy.stats import t 

class TestLinearModel(unittest.TestCase):
  def test_factory(self):
    model_generator = gm.DefaultLinearModelFactory
    # requires time by datapoints data to be a numpy array
    self.assertRaises(ValueError, model_generator, [0])
    self.assertRaises(ValueError, model_generator, [[0]])
    self.assertRaises(ValueError, model_generator, np.array(0))
    self.assertRaises(ValueError, model_generator, np.array([0,1]))
    self.assertRaises(ValueError, model_generator, np.array([[]]))

    fake_priori_data = np.array([[-1],[0],[1]])
    empty_data = np.array([])
    empty_data.shape = (0,1)

    m_empty = model_generator(empty_data)
    # check no-input created correctly)
    # prior made up data is [-1,0,1]
    self.assertEqual(m_empty.post_mean,0)
    self.assertEqual(m_empty.post_n,3)
    self.assertEqual(m_empty.post_beta,np.sum(fake_priori_data**2)/2)

    sample_data = np.array([[1]]) 
    m_simple = model_generator(sample_data)
    # prior made up data is [-1,0,1], so total data is
    # [-1,0,1,1]
    self.assertEqual(m_simple.post_mean, 1/4)
    self.assertEqual(m_simple.post_n, 4)
    new_samp = np.concatenate((fake_priori_data,sample_data))
    self.assertEqual(m_simple.post_beta, sum((new_samp - np.mean(new_samp))**2)/2)
    
    sample_data = np.array([[1],[10],[10],[13]]) 
    regularized_data = np.concatenate((fake_priori_data,sample_data))
    m_bigger = model_generator(sample_data)
    # we're building a model on [1,10,10,13] using [0,1,2,3].
    # wolfram alpha tells us our model is 3.6x + 3.1, so our result for x=4 is 14.4 + 3.1 = 17.5.
    # we then compute the mean of (17.5 * 4) / 7
    self.assertAlmostEqual(m_bigger.post_mean[0], 17.5 * 4 / 7)
    self.assertEqual(m_bigger.post_n, np.size(regularized_data))
    #### Some day figure out if this is the ``right'' number
    self.assertAlmostEqual(m_bigger.post_beta[0], 304) 


    sample_data = np.array([[1,2],[10,3],[10,4],[13,5]]) 
    regularized_data = np.concatenate((np.concatenate((fake_priori_data,fake_priori_data),1),sample_data))
    m_twod = model_generator(sample_data)
    # prior made up data is [-1,0,1], so total data is
    # [-1,0,1,1]
    self.assertAlmostEqual(m_twod.post_mean[0], 10)
    self.assertEqual(m_twod.post_mean[1], (6*4)/7)
    self.assertEqual(m_twod.post_n, regularized_data.shape[0])
    fict_data = np.array([-1,0,1,6,6,6,6])
    # ssd of fake data / 2
    expect_post_beta_1 = np.sum(np.square((fict_data - (24/7)))) /2
    #### Some day figure out if this is the ``right'' number
    self.assertAlmostEqual(m_twod.post_beta[0], 304)
    self.assertAlmostEqual(m_twod.post_beta[1], expect_post_beta_1) 
    
  def test_probability(self):
    empty_data = np.array([])
    empty_data.shape = (0,1)
    model_generator = gm.DefaultLinearModelFactory
    m_empty = model_generator(empty_data)
    prob1 = t.pdf(1,3,scale=np.sqrt(2 * 4/(3 * 3)))
    prob2 = t.pdf(2,3,scale=np.sqrt(2 * 4/(3 * 3)))
    self.assertAlmostEqual(m_empty.GetProbability(np.array([1])), prob1)
    self.assertAlmostEqual(m_empty.GetProbability(np.array([2])), prob2) 
    empty_data.shape = (0,2)
    m_empty2 = model_generator(empty_data)
    self.assertAlmostEqual(m_empty2.GetProbability(np.array([1,2])), prob1 * prob2) 
    
    self.assertRaises(ValueError, m_empty.GetProbability, 0)
    self.assertRaises(ValueError, m_empty.GetProbability, [0])
    self.assertRaises(ValueError, m_empty.GetProbability, np.array([]))
    self.assertRaises(ValueError, m_empty2.GetProbability, 0)
    self.assertRaises(ValueError, m_empty2.GetProbability, [0])
    self.assertRaises(ValueError, m_empty2.GetProbability, np.array([0]))
  
  def test_lagged_factory(self):
    num_variables = 1
    apriori_n = 3.0
    apriori_mu0 = np.zeros(num_variables)
    apriori_alpha = apriori_n/2
    apriori_beta = (np.ones(num_variables) * 2)/2
    probability_lag = 1
    lin_pred = gm.LinearPredictor(num_variables, apriori_n, apriori_mu0, apriori_alpha, apriori_beta, probability_lag)
    data = np.array([0,1,2])
    # since lagged 1, prediction should be 4, with ssd of idkwat
    data.shape = (3,1)
    model_generator = lin_pred.Fit
    m = model_generator(data)
    self.assertAlmostEqual(m.post_mean[0], (4 * 3 + 0 * 3)/6)
    probability_lag = 10
    lin_pred = gm.LinearPredictor(num_variables, apriori_n, apriori_mu0, apriori_alpha, apriori_beta, probability_lag)
    model_generator = lin_pred.Fit
    m = model_generator(data)
    self.assertAlmostEqual(m.post_mean[0], 13/2)
    

if __name__ == '__main__':
    unittest.main()
