import numpy as np

debug = False

# probability of value seen at cur_index, given model built on prior_runlength data (does not include cur_index)
# data should be of length time by number of variables
def ComputeProbability(data, cur_index, prior_runlength, prediction_gap, model_generator):
  if (prior_runlength > cur_index):
    raise(ValueError("You can't look back at data that happens before time 0"))
  if (cur_index >= data.shape[0]):
    raise(ValueError("You can't look past the last data point"))
  if (type(data) is not np.ndarray or data.ndim != 2):
    raise ValueError('data should be a numpy array of dimension 2')
  x = data[cur_index,:]
  prior_data = data[(cur_index-prior_runlength-prediction_gap):(cur_index-prediction_gap),:]
  model = model_generator(prior_data)
  prob = model.GetProbability(x)
  if (debug):
    print("Computing probability for %s modeled on %s. Answer: %s" % (x, prior_data, prob))
  return prob

def GetNextProbabilityValues(previous_probs, data, cur_index, max_duration, model_generator, reset_prob, prediction_gap):
  if (type(previous_probs) is not np.ndarray or previous_probs.ndim != 1 
        or previous_probs.shape[0] != max_duration):
    raise ValueError('previous_probs should be a numpy array of length max_duration')
  if (type(data) is not np.ndarray or data.ndim != 2):
    raise ValueError('data should be a numpy array of dimension 2')
  eps = 1e-10
  if abs(sum(previous_probs) - 1) > eps:
    raise ValueError('previous_probs must sum to one')
  cutoff_eps = 0.05
  if previous_probs[max_duration-1] > cutoff_eps:
    raise ValueError('Overflow error in previous_probs. Increase max_duration since we had a non-negligible probability mass of having a run_length longer than previous_probs can hold')

  new_probs = np.zeros(max_duration)
  # for computational simplicity, start out by assuming a constant reset probability
  # (in the Bayesian Online Changepoint Detection paper, set H(tau) = 1/lambda.
  inv_lambda = reset_prob 

  num_variables = data.shape[1];
  for run_length in range(1,max_duration):
    # you can't look past the first data point
    capped_run_length = max(min(cur_index-prediction_gap, run_length-1),0)
    model_fitting_prob = ComputeProbability(data, cur_index, capped_run_length, prediction_gap, model_generator)
    # keep cumulative track of the truncation probability
    new_probs[0] = new_probs[0] + previous_probs[run_length - 1] * model_fitting_prob * inv_lambda
    new_probs[run_length] = previous_probs[run_length-1] * model_fitting_prob * (1-inv_lambda)
    

  normalization_factor = sum(new_probs)
  new_probs = new_probs / normalization_factor
  
  if new_probs[max_duration-1] > cutoff_eps:
    raise ValueError('Overflow error in resulting previous_probs. Increase max_duration since next time this runs we will have a non-negligible probability mass of having a run_length longer than previous_probs can hold')
  
  return new_probs
     

class ChangepointDetector(object):
  def __init__(self, data, reset_prob, model_generator, max_duration = 1000, prediction_gap = 0):
    # time x vars dataset of observations
    self.data = data
    self.time = data.shape[0]
    self.max_duration = max_duration
    self.model_generator = model_generator
    self.reset_prob = reset_prob
    self.probability_lattice = np.zeros((self.time+1, self.max_duration))
    # say that the game starts at the beginning (all probability mass set to initial run length 0)
    self.probability_lattice[0,0] = 1
    self.prediction_gap = prediction_gap

  def ModelChangepoints(self):
    # informed by (for NIPS 2009 BOCPD cite):
    # @INPROCEEDINGS{Turner2009,
    # author = {Ryan Turner and Yunus Saat\c{c}i and Carl Edward Rasmussen},
    # title = {Adaptive Sequential {B}ayesian Change Point Detection},
    # booktitle = {Temporal Segmentation Workshop at NIPS 2009},
    # year = {2009},
    # editor = {Zaid Harchaoui},
    # address = {Whistler, BC, Canada},
    # month = {December},
    # url = {http://mlg.eng.cam.ac.uk/rdturner/BOCPD.pdf}
    # }
    # we pass start at 0 and pass cur_index-1 to GetNextProbabilityValues (i had thought we reset AT the first timestep, but we reset just before, according to that code)
    for cur_index in range(1,self.time+1):
      self.probability_lattice[cur_index,:] = GetNextProbabilityValues(self.probability_lattice[cur_index-1,:], self.data, cur_index-1, self.max_duration, self.model_generator, self.reset_prob, self.prediction_gap)
    return self.probability_lattice
