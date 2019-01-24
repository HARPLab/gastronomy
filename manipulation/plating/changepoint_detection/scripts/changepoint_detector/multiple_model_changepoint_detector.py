import numpy as np

debug = False

# probability of value seen at cur_index, given model built on prior_runlength data (does not include cur_index)
# data should be of length time by number of variables
def ComputeProbability(data, cur_index, prior_runlength, model_generator):
  if (prior_runlength > cur_index):
    raise(ValueError("You can't look back at data that happens before time 0"))
  if (cur_index >= data.shape[0]):
    raise(ValueError("You can't look past the last data point"))
  if (type(data) is not np.ndarray or data.ndim != 2):
    raise ValueError('data should be a numpy array of dimension 2')
  x = data[cur_index,:]
  prior_data = data[(cur_index-prior_runlength):cur_index,:]
  model = model_generator(prior_data)
  prob = model.GetProbability(x)
  if (debug):
    print("Computing probability for %s modeled on %s. Answer: %s" % (x, prior_data, prob))
  return prob

# certain parameters (previous_probs_tuple, data_tuple, cur_index, etc.) are now tuples
# each element in the tuple corresponds to the associated _FORM_ of the associated model.
# For example, the models could be "in world reference frame" and "in other fork reference frame"
# the first element of each tuple corresponds to "in world reference frame" and the second element to "in other fork"
def GetNextProbabilityValues(previous_probs_tuple, data_tuple, cur_index, max_duration, model_generator_tuple, reset_prob_tuple):
  if (len(previous_probs_tuple) != len(data_tuple) or len(data_tuple) != len(model_generator_tuple) or len(model_generator_tuple) != len(reset_prob_tuple)):
    raise ValueError('all tuples must be of the same length')
  num_tuple_elements = len(previous_probs_tuple)
  for tuple_index in range(num_tuple_elements):
    if (type(previous_probs_tuple[tuple_index]) is not np.ndarray or previous_probs_tuple[tuple_index].ndim != 1 
          or previous_probs_tuple[tuple_index].shape[0] != max_duration):
      raise ValueError('previous_probs_tuple[ind] should be a numpy array of length max_duration')
    if (type(data_tuple[tuple_index]) is not np.ndarray or data_tuple[tuple_index].ndim != 2):
      raise ValueError('data_tuple[indx] should be a numpy array of dimension 2')
    cutoff_eps = 0.05
    if previous_probs_tuple[tuple_index][max_duration-1] > cutoff_eps:
      raise ValueError('Overflow error in previous_probs_tuple[indx]. Increase max_duration since we had a non-negligible probability mass of having a run_length longer than previous_probs can hold')
  eps = 1e-10
  cum_tuple_prob_sum = 0
  for tuple_index in range(num_tuple_elements):
    cum_tuple_prob_sum += sum(previous_probs_tuple[tuple_index])
  if abs(cum_tuple_prob_sum - 1) > eps:
    raise ValueError('previous_probs_tuple must sum to one when summing all items in the tuple together')


  new_probs_tuple = tuple([np.zeros(max_duration) for i in range(num_tuple_elements)])

  # for computational simplicity, start out by assuming a constant reset probability
  # (in the Bayesian Online Changepoint Detection paper, set H(tau) = 1/lambda.
  # furthermore, when using multiple models, for simplicity we compute "total reset probability", and then later divide that value into
  # reset to FORM 1, reset to FORM 2 model
  # so, this is the probability of any kind of reset
  inv_lambda = 0
  for tuple_index in range(num_tuple_elements):
    inv_lambda += reset_prob_tuple[tuple_index]

  for tuple_index in range(num_tuple_elements):
    for run_length in range(1,max_duration):
      # you can't look past the first data point
      capped_run_length = min(cur_index, run_length-1)
      model_fitting_prob = ComputeProbability(data_tuple[tuple_index], cur_index, capped_run_length, model_generator_tuple[tuple_index])
      # keep cumulative track of the truncation probability
      # note that the new_probs_tuple[ind][0] will later need to be split into FORM 1 and FORM 2 reset probs 
      new_probs_tuple[tuple_index][0] = new_probs_tuple[tuple_index][0] + previous_probs_tuple[tuple_index][run_length - 1] * model_fitting_prob * inv_lambda
      new_probs_tuple[tuple_index][run_length] = previous_probs_tuple[tuple_index][run_length-1] * model_fitting_prob * (1-inv_lambda)
    
  normalization_factor = 0
  for tuple_index in range(num_tuple_elements):
    normalization_factor += sum(new_probs_tuple[tuple_index])
  
  new_probs_tuple = tuple([new_probs / normalization_factor for new_probs in new_probs_tuple])

  cum_reset_prob = 0
  for tuple_index in range(num_tuple_elements):
    cum_reset_prob += new_probs_tuple[tuple_index][0]  
  # divide the cross probability among the FORMS of models
  if (abs(cum_reset_prob - inv_lambda) > eps):
    raise RuntimeError("Based on my understanding of the algorithm, this sum should always equal the total reset probability") 
  for tuple_index in range(num_tuple_elements):
    new_probs_tuple[tuple_index][0] = reset_prob_tuple[tuple_index]
    
  for tuple_index in range(num_tuple_elements):
    if new_probs_tuple[tuple_index][max_duration-1] > cutoff_eps:
      raise ValueError('Overflow error in new_probs_tuple[indx]. Increase max_duration since we had a non-negligible probability mass of having a run_length longer than previous_probs can hold')
  
  return new_probs_tuple
     

class MultipleModelChangepointDetector(object):
  def __init__(self, data_tuple, reset_prob_tuple, model_generator_tuple, max_duration = 1000):
    num_tuple_elements = len(data_tuple)
    for tuple_index in range(num_tuple_elements):
      if (type(data_tuple[tuple_index]) is not np.ndarray or data_tuple[tuple_index].ndim != 2):
        raise ValueError('data_tuple[0] should be a numpy array of dimension 2')
      for tuple_index_two in range(tuple_index):
        if (data_tuple[tuple_index].shape[0] != data_tuple[tuple_index_two].shape[0]):
          raise ValueError('data_tuple items must have the same length. If they do not (eg: one is observations and the other is first differences), please skip values of one until they do')
    self.time = data_tuple[0].shape[0]
    # time x vars dataset of observations
    self.data_tuple = data_tuple
    self.reset_prob_tuple = reset_prob_tuple
    self.model_generator_tuple = model_generator_tuple
    self.max_duration = max_duration
    self.probability_lattice_tuple = tuple([np.zeros((self.time+1, self.max_duration)) for i in range(num_tuple_elements)])
    # say that the game starts at the beginning (all probability mass set to initial run length 0)
    # with equal probability for either model
    for tuple_index in range(num_tuple_elements):
      self.probability_lattice_tuple[tuple_index][0,0] = 1.0/num_tuple_elements

  def ModelChangepoints(self):
    # we pass start at 0 and pass cur_index-1 to GetNextProbabilityValues (i had thought we reset AT the first timestep, but we reset just before, according to Ryan Turner 
    # code I compared my implementation with)
    for cur_index in range(1,self.time+1):
      previous_probability_values_tuple = tuple([prob_lattice[cur_index-1,:] for prob_lattice in self.probability_lattice_tuple])
      next_prob_tuple = GetNextProbabilityValues(previous_probability_values_tuple, 
                                                 self.data_tuple, 
                                                 cur_index-1, 
                                                 self.max_duration, 
                                                 self.model_generator_tuple, 
                                                 self.reset_prob_tuple)
      for tuple_index in range(len(self.data_tuple)):
        self.probability_lattice_tuple[tuple_index][cur_index,:] = next_prob_tuple[tuple_index]
    return self.probability_lattice_tuple
