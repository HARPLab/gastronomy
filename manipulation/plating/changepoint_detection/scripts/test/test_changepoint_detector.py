from __future__ import division
import unittest
from changepoint_detector import changepoint_detector as cd
from changepoint_detector import gaussian_model as gm
import numpy as np
from scipy.stats import t

class MockedModel(object):
  def GetProbability(self, x):
    return 1

def MockedModelGenerator(data):
  return MockedModel()

def AssertAllAlmostEqual(utest, arraya, arrayb):
  eps = 1e-15
  utest.assertTrue(np.all(np.abs(arraya - arrayb) < eps), "%s and %s were not almost equal" % (arraya, arrayb))

class TestChangepointDetector(unittest.TestCase):
  def test_probability(self):
    data = np.array([[1],[0],[-1]])
    self.assertEqual(cd.ComputeProbability(data, 0, 0, 0, MockedModelGenerator), 1)
    self.assertRaises(ValueError, cd.ComputeProbability, data, 0, 1, 0, MockedModelGenerator)
    self.assertRaises(ValueError, cd.ComputeProbability, data, 3, 1, 0, MockedModelGenerator)
    self.assertEqual(cd.ComputeProbability(data, 1, 0, 0, MockedModelGenerator), 1)
    self.assertEqual(cd.ComputeProbability(data, 1, 1, 0, MockedModelGenerator), 1)
    data = np.array([[1,2],[0,1],[-1,2]])
    self.assertRaises(ValueError, cd.ComputeProbability, data, 0, 1, 0, MockedModelGenerator)
    self.assertRaises(ValueError, cd.ComputeProbability, data, 3, 1, 0, MockedModelGenerator)
    self.assertEqual(cd.ComputeProbability(data, 0, 0, 0, MockedModelGenerator), 1)
    self.assertEqual(cd.ComputeProbability(data, 0, 0, 0, MockedModelGenerator), 1)
    self.assertEqual(cd.ComputeProbability(data, 1, 0, 0, MockedModelGenerator), 1)
    self.assertEqual(cd.ComputeProbability(data, 1, 1, 0, MockedModelGenerator), 1)

  def test_get_next_probability_values(self):
    previous_probs = np.array([1,0,0,0,0])
    max_duration = 5
    model_generator = MockedModelGenerator
    reset_prob = 0.5
    data = np.array([[1],[0],[-1]])

    self.assertRaises(ValueError, cd.GetNextProbabilityValues, previous_probs, data, 1, 100, model_generator, reset_prob, 0)
    self.assertRaises(ValueError, cd.GetNextProbabilityValues, np.array([1,1,1,0,0]), data, 1, max_duration, model_generator, reset_prob, 0)
    # change: now we do allow computing first probability
    #self.assertRaises(ValueError, cd.GetNextProbabilityValues, previous_probs, data, 0, max_duration, model_generator, reset_prob, 0)
    self.assertRaises(ValueError, cd.GetNextProbabilityValues, np.array([0.5,0.25,0.125,0.125,0]), data, 1, max_duration, model_generator, reset_prob, 0) 
    self.assertRaises(ValueError, cd.GetNextProbabilityValues, np.array([0.5,0.25,0.125,0.0625,0.0625]), data, 1, max_duration, model_generator, reset_prob, 0) 
    
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(previous_probs, data, 1, max_duration, model_generator, reset_prob, 0), [0.5,0.5,0,0,0])
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(np.array([0.5,0.5,0,0,0]), data, 1, max_duration, model_generator, reset_prob, 0), [0.5,0.25,0.25,0,0])
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(np.array([0.5,0.25,0.25,0,0]), data, 1, max_duration, model_generator, reset_prob, 0), [0.5,0.25,0.125,0.125,0])
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(np.array([0.5,0.25,0.25-1e-10,1e-10,0]), data, 1, max_duration, model_generator, reset_prob, 0), [0.5,0.25,0.125,0.125-1e-10/2,1e-10/2])
    
    reset_prob = 1
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(previous_probs, data, 1, max_duration, model_generator, reset_prob, 0), previous_probs)
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(np.array([0.5,0.25,0.25-1e-10,1e-10,0]), data, 1, max_duration, model_generator, reset_prob, 0), previous_probs) 

    reset_prob = 0
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(previous_probs, data, 1, max_duration, model_generator, reset_prob, 0), [0,1,0,0,0])
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(np.array([0.5,0.25,0.25-1e-10,1e-10,0]), data, 1, max_duration, model_generator, reset_prob, 0), [0,0.5,0.25,0.25-1e-10,1e-10])

  def test_full_calculation(self):
    data = np.array([[1],[0],[-1]])
    reset_prob = 0 
    model_generator = MockedModelGenerator
    # causes overflow (max_duration too short)
    max_duration = 3
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    self.assertRaises(ValueError, change_detector.ModelChangepoints)

    max_duration = 5
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    AssertAllAlmostEqual(self, change_detector.ModelChangepoints(), [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])
    reset_prob = 1 
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    AssertAllAlmostEqual(self, change_detector.ModelChangepoints(), [[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])

class IntegrationTestChangepointDetectorGaussian(unittest.TestCase):
  def test_probability(self):
    model_generator = gm.DefaultGaussianModelFactory

    data = np.array([[1],[0],[-1]])
    self.assertRaises(ValueError, cd.ComputeProbability, data, 0, 1, 0, model_generator)
    self.assertRaises(ValueError, cd.ComputeProbability, data, 3, 1, 0, model_generator)
    self.assertAlmostEqual(cd.ComputeProbability(data, 0, 0, 0, model_generator), t.pdf(1,3,scale=np.sqrt(8/9)))
    self.assertAlmostEqual(cd.ComputeProbability(data, 1, 0, 0, model_generator), t.pdf(0,3,scale=np.sqrt(8/9)))
    sample_data = np.array([1,0,-1,1]);
    sample_var = np.var(sample_data);
    sample_mean = np.mean(sample_data);
    self.assertAlmostEqual(cd.ComputeProbability(data, 1, 1, 0, model_generator), t.pdf(sample_mean,4,scale=np.sqrt(sample_var * 5/4)))
    sample_data = np.array([1,0,-1,0]);
    sample_var = np.var(sample_data);
    sample_mean = np.mean(sample_data);
    self.assertAlmostEqual(cd.ComputeProbability(data, 2, 1, 0, model_generator), t.pdf((-1 - sample_mean),4, scale=np.sqrt(sample_var*5/4)))
  
  def test_get_next_probability_values(self):
    data = np.array([[1],[0],[-1]])
    max_duration = 5
    reset_prob = 0.5
    model_generator = gm.DefaultGaussianModelFactory 

    sample_data = np.array([1,0,-1,1]);
    sample_mean = np.mean(sample_data);
    sample_ssd = sum(np.square(sample_data-sample_mean))
    raw_continue_prob = 0.5 * t.pdf(0,3,scale=np.sqrt(8/9))
    raw_continue_prob_2 = 0.5 * t.pdf((0-sample_mean),4,scale=np.sqrt(sample_ssd * 5/(4*4)))
    raw_reset_prob = 0.5*raw_continue_prob + 0.5*raw_continue_prob_2
    big_initial = np.array([0.5, 0.25, 0.25, 0, 0])
    # because they're truncated, all the previous models have only one data point
    raw_next_prob = [raw_reset_prob, 0.5 * raw_continue_prob, 0.25 * raw_continue_prob_2, 0.25 * raw_continue_prob_2, 0]
    next_prob = raw_next_prob / sum(raw_next_prob)
    AssertAllAlmostEqual(self, cd.GetNextProbabilityValues(big_initial, data, 1, max_duration, model_generator, reset_prob, 0), next_prob)

  def test_full_calculation(self):
    model_generator = gm.DefaultGaussianModelFactory 
    data = np.array([[1],[0],[-1]])
    reset_prob = 0 
    # causes overflow (max_duration too short)
    max_duration = 3
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    self.assertRaises(ValueError, change_detector.ModelChangepoints)

    max_duration = 5
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    AssertAllAlmostEqual(self, change_detector.ModelChangepoints(), [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])
    reset_prob = 1 
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    AssertAllAlmostEqual(self, change_detector.ModelChangepoints(), [[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])

    data = np.array([[1],[0]])
    max_duration = 4
    reset_prob = 0.5
    sample_data = np.array([1,0,-1,1]);
    sample_mean = np.mean(sample_data);
    sample_ssd = sum(np.square(sample_data - sample_mean))
    raw_continue_prob = 0.5*0.5 * t.pdf((0-sample_mean),4,scale=np.sqrt(sample_ssd * (5)/(4 * 4)))
    raw_continue_prob_2 = 0.5*0.5 * t.pdf(0,3,scale=np.sqrt(2*4/(3*3)))
    raw_reset_prob = raw_continue_prob + raw_continue_prob_2
    raw_next_prob = [raw_reset_prob, raw_continue_prob_2, raw_continue_prob, 0]
    next_prob = raw_next_prob / sum(raw_next_prob)
    change_detector = cd.ChangepointDetector(data, reset_prob, model_generator, max_duration)
    AssertAllAlmostEqual(self, change_detector.ModelChangepoints(), [[1,0,0,0],[0.5,0.5,0,0],next_prob])

  def test_full_calculation_regression_varied_mean(self):
    # this isn't really a test, it just makes a pretty picture
    return
    # but we also note that (after t0) the probability of reset is always exactly the reset_prob
    num_variables = 1
    apriori_n = 3.0
    apriori_mean = np.zeros(num_variables)
    apriori_ssd = np.ones(num_variables)/1000000
    model_generator = gm.GaussianPredictor(num_variables, apriori_n, apriori_mean, apriori_ssd).Fit
    np.random.seed(0)
    chunk_length = 30
    part_1 = np.random.normal(0, size=(chunk_length,1))
    part_2 = np.random.normal(10,size=(chunk_length,1))
    part_3 = np.random.normal(0,size=(chunk_length,1))
    part_4 = np.random.normal(-10,size=(chunk_length,1))
    in_data = np.concatenate((part_1, part_2, part_3, part_4))/1000

    max_duration = chunk_length * 4 
    reset_prob = 1/(chunk_length)
    change_detector = cd.ChangepointDetector(in_data, reset_prob, model_generator, max_duration)
    result_probs = change_detector.ModelChangepoints()
    AssertAllAlmostEqual(self, result_probs[1:,0], reset_prob, 0)
    
    #import matplotlib.pyplot as plt
    #plt.plot(in_data)
    #plt.show()
    #plt.imshow(np.transpose(1-result_probs), cmap='gray', origin='lower')
    #plt.show() 
  
  def test_full_calculation_regression_varied_variance(self):
    # this isn't really a test, it just makes a pretty picture
    return
    # but we also note that (after t0) the probability of reset is always exactly the reset_prob
    num_variables = 1
    apriori_n = 3.0
    apriori_mean = np.zeros(num_variables)
    apriori_ssd = np.ones(num_variables)/1000000
    model_generator = gm.GaussianPredictor(num_variables, apriori_n, apriori_mean, apriori_ssd).Fit
    np.random.seed(0)
    chunk_length = 30
    part_1 = np.random.normal(0,1, size=(chunk_length,1))
    part_2 = np.random.normal(0,10,size=(chunk_length,1))
    part_3 = np.random.normal(0,1,size=(chunk_length,1))
    part_4 = np.random.normal(0,10,size=(chunk_length,1))
    part_5 = np.random.normal(0,1,size=(chunk_length,1))
    in_data = np.concatenate((part_1, part_2, part_3, part_4, part_5))/1000

    max_duration = chunk_length * 5  + 1
    reset_prob = 1/(chunk_length)
    change_detector = cd.ChangepointDetector(in_data, reset_prob, model_generator, max_duration)
    result_probs = change_detector.ModelChangepoints()
    AssertAllAlmostEqual(self, result_probs[1:,0], reset_prob, 0)
    
    import matplotlib.pyplot as plt
    plt.plot(in_data)
    plt.show()
    plt.imshow(np.transpose(1-result_probs), cmap='gray', origin='lower')
    plt.show() 
        

if __name__ == '__main__':
    unittest.main()
