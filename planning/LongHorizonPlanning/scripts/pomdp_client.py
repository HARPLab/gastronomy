import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import pylab as plt
import math


from draw_env import *

GLOBAL_TIME = 0
print_status = False

class Action():
	def __init__(self, id, name, pomdp, a_type, time_steps, additional_info={}):
		self.id = id
		self.name = name
		self.pomdp = pomdp
		self.time_steps = time_steps
		self.type = a_type
		self.additional_info = additional_info

	def print(self):
		print ("id: ", self.id, " name: ", self.name, " pomdp: ", self.pomdp, " time_steps: ", self.time_steps, " type: ", self.type, "info: ", self.additional_info)


class ClientPOMDP(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op):
		self.transition_function = {}

	def get_state_index(self,state):
		new_state = tuple(state)
		new_state_index = np.ravel_multi_index(new_state,self.state_space_dim)
		return int(new_state_index)

	def get_state_tuple(self,new_state_index):
		state = np.unravel_index(new_state_index,self.state_space_dim)
		new_state = list(state)
		return new_state

	def get_observation_index(self,observation):
		new_obs = tuple(observation)
		new_obs_index = np.ravel_multi_index(new_obs,self.observation_space_dim)
		return int(new_obs_index)

	def get_observation_tuple(self,observation_index):
		obs = np.unravel_index(observation_index,self.observation_space_dim)
		new_obs = list(obs)
		return new_obs

	def compute_P(self):
		feature_array = [0] * len(self.state_space_dim)
		self.compute_P_elements(len(self.state_space_dim)-1,feature_array)

	def compute_P_elements(self,feature_index,feature_array):
		if feature_index < 0:
			self.transition_function[self.get_state_index(feature_array)] = {} 
			state_index = self.get_state_index(feature_array)
			for action in range(self.nA):
				self.transition_function[state_index][action], steps = self.simulate_action(state_index, action, all_poss_actions)
				### observation function
				for outcome in self.transition_function[state_index][action]:
					next_state_index = outcome[1]
					if next_state_index not in self.observation_function.keys():
						self.observation_function[next_state_index] = {} 

					obs = self.simulate_observation (next_state_index, action)
					if action not in self.observation_function[next_state_index].keys():
						self.observation_function[next_state_index][action] = obs
					else:
						self.observation_function[next_state_index][action].update(obs)

			return

		for i in range(self.state_space_dim[feature_index]):
			new_feature_array = feature_array.copy()
			new_feature_array[feature_index] = i
			self.compute_P_elements(feature_index-1,new_feature_array)

	def distance(self, a, b):
		(x1, y1) = a
		(x2, y2) = b
		return np.sqrt(np.power((x1 - x2),2) + np.power((y1 - y2),2))

	def is_part_of_action_space(self, action):
		return (action in self.valid_actions)

	def get_tuple (self, index, dim):
		state = np.unravel_index(index,dim)
		new_state = list(state)
		return new_state