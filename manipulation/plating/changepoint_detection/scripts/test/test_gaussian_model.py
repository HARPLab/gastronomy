import unittest
from changepoint_detector import gaussian_model as gm
import numpy as np
from scipy.stats import t 

class TestGaussianModel(unittest.TestCase):
  def test_factory(self):
    model_generator = gm.DefaultGaussianModelFactory
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
    # prior made up data is [-1,0,1], so total data is
    # [-1,0,1,1]
    self.assertEqual(m_bigger.post_mean, np.mean(regularized_data))
    self.assertEqual(m_bigger.post_n, np.size(regularized_data))
    self.assertEqual(m_bigger.post_beta, sum((regularized_data - np.mean(regularized_data))**2)/2)


    sample_data = np.array([[1,2],[10,3],[10,4],[13,5]]) 
    regularized_data = np.concatenate((np.concatenate((fake_priori_data,fake_priori_data),1),sample_data))
    m_twod = model_generator(sample_data)
    # prior made up data is [-1,0,1], so total data is
    # [-1,0,1,1]
    self.assertEqual(m_twod.post_mean[0], np.mean(regularized_data,0)[0])
    self.assertEqual(m_twod.post_mean[1], np.mean(regularized_data,0)[1])
    self.assertEqual(m_twod.post_n, regularized_data.shape[0])
    expected_post_beta = sum(np.square(regularized_data - np.mean(regularized_data,0)))/2
    self.assertAlmostEqual(m_twod.post_beta[0], expected_post_beta[0])
    self.assertAlmostEqual(m_twod.post_beta[1], expected_post_beta[1]) 
    
  def test_probability(self):
    empty_data = np.array([])
    empty_data.shape = (0,1)
    model_generator = gm.DefaultGaussianModelFactory
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

if __name__ == '__main__':
    unittest.main()
