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

class GlobalPOMDP(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_tasks, pomdp_solvers, tasks, robot):
		self.print_status = False
		self.pomdp_tasks = pomdp_tasks
		self.pomdp_solvers = pomdp_solvers
		self.tasks = tasks
		self.robot = robot

		self.name = "pomdp_agent" 

		self.nA = 0
		self.nS = 1
		self.nO = 1
		obs = []
		self.feature_indices = {}
		self.task_indices = []
		self.state_space_dim = ()
		self.observation_space_dim = ()
		self.navigation_goals_len = len(self.pomdp_tasks)
		self.one_task_actions_num = self.pomdp_tasks[0].nA - self.navigation_goals_len

		self.actions = np.ones((self.one_task_actions_num*len(self.pomdp_tasks)+self.navigation_goals_len,len(self.pomdp_tasks)))
		count = 0
		for l in range(len(self.pomdp_tasks)):
			for i in range(self.one_task_actions_num):
				self.actions[count,l] = i
				count += 1 

		for l in range(len(self.pomdp_tasks)):
			self.actions[count,:] = l + self.one_task_actions_num
			count += 1 

		self.nA = self.actions.shape[0]

		# feature_count = 0
		# task_count = 0
		# for task in self.tasks:
		# 	start = feature_count
		# 	for feature in task.get_features():
		# 		if feature.type == "discrete" and feature.name in ["cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise", \
		# 			"time_since_hand_raise","current_request","customer_satisfaction"]:

		# 			self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
		# 			self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
		# 			obs.append((feature.low, feature.high, feature.discretization, feature.name))
		# 			self.feature_indices[feature.name+str(task_count)] = feature_count
		# 			feature_count += 1
		# 			if feature.observable:
		# 				self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
		# 				self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
		# 	end = feature_count
		# 	self.task_indices.append((start,end))
		# 	self.nA += self.pomdp_tasks[task_count].nA - self.pomdp_tasks[task_count].navigation_goals_len
		# 	task_count += 1
			

		# self.nA += self.pomdp_tasks[task_count-1].navigation_goals_len
		# self.action_space = spaces.Discrete(self.nA)

		# # robot's features
		# for feature in self.robot.get_features():
		# 	if feature.type == "discrete" and feature.name in ["x","y"]:
		# 		self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
		# 		self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
		# 		obs.append((feature.low, feature.high, feature.discretization, feature.name))
		# 		self.feature_indices[feature.name] = feature_count
		# 		feature_count += 1
		# 		if feature.observable:
		# 			self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
		# 			self.nO *= int((feature.high - feature.low) / feature.discretization) + 1

		# print (self.nS)
		# self.state_space = obs
		# if self.print_status:
		# 	print ("state space: ", self.state_space)
		# 	print ("state space dim: ", self.state_space_dim)
		# self.reset()

		# self.dense_reward = True

		# self.state = None


	def get_belief(self):
		pass

	def step(self, env, action, start_state=None):
		return env.step(action,start_state)

	def simulate_action(self, env, start_state_index, action):
		return env.simulate_action(start_state_index,action)

	def simulate_observation (self, env, next_state_index, action):
		return env.simulate_observation(next_state_index,action)


	def get_possible_next_states (self,env,observations):
		return env.get_possible_next_states(observations)

	def get_possible_obss (self, env, belief_probs):
		return env.get_possible_obss (belief_probs)


	def reset(self):
		pass

	def get_state_index(self,env,state):
		new_state = tuple(state)
		new_state_index = np.ravel_multi_index(new_state,env.state_space_dim)
		return int(new_state_index)

	def get_state_tuple(self,env,new_state_index):
		state = np.unravel_index(new_state_index,env.state_space_dim)
		new_state = list(state)
		return new_state

	def get_observation_index(self,env,observation):
		new_obs = tuple(observation)
		new_obs_index = np.ravel_multi_index(new_obs,env.observation_space_dim)
		return int(new_obs_index)

	def get_observation_tuple(self,env,observation_index):
		obs = np.unravel_index(observation_index,env.observation_space_dim)
		new_obs = list(obs)
		return new_obs

	def render(self, start_state=None, mode='human'):
		pass