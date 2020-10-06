import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import pylab as plt
import math
import itertools


from draw_env import *
from pomdp_client import *

GLOBAL_TIME = 0

class AgentPOMDP(ClientPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_tasks, pomdp_solvers, tasks, robot, random):
		pass

	def print_actions(self):
		for a in self.actions:
			for l in range(len(self.pomdp_tasks)): 
				a[l].print()

	def get_belief(self, beliefs):
		belief = []
		pos_beliefs_len = 1
		pos_beliefs_dim = ()
		pos_beliefs = []

		for a in range(len(self.pomdp_tasks)):
			pos_beliefs.append([])
			for prob,ps in beliefs[a]: 
				pos_beliefs[a].append((prob,ps))

			pos_beliefs_len *= len(pos_beliefs[a])
			pos_beliefs_dim += (len(pos_beliefs[a]),)
		
		count = 0
		while count < pos_beliefs_len:
			poss = self.get_tuple(count,pos_beliefs_dim)
			new_state = np.zeros(len(self.state_space_dim),dtype=int)
			prob = 1.0
			for a in range(len(self.pomdp_tasks)):		
				indices = list(range(self.task_indices[a][0],self.task_indices[a][1]))
				indices.extend(self.robot_indices)
				new_state[indices] = self.pomdp_tasks[a].get_state_tuple(pos_beliefs[a][poss[a]][1])
				prob *= pos_beliefs[a][poss[a]][0]

			
			belief.append((prob,self.get_state_index(new_state)))
			count += 1
		
		return belief

	def get_pomdp (self, action, prev_pomdp):
		if np.array_equal(action,self.all_no_ops['1']) and prev_pomdp is not None:
			print ("get_pomdp in pomdp_agent")
			# set_trace()
			pomdp = prev_pomdp
		else:
			pomdp = self.pomdp_actions[action]
			
		p_action = self.actions[action][pomdp]

		return pomdp, p_action

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None):
		if self.is_part_of_action_space(action) and start_state_index in self.transition_function.keys() and action in self.transition_function[start_state_index].keys():
			return self.transition_function[start_state_index][action]

		actions = self.actions[action]
		start_state =  self.get_state_tuple(start_state_index)
		new_state = np.asarray(deepcopy(start_state))
		all_outcomes = []
		outcomes_len = 1
		outcomes_dim = ()
		outcomes = []
		k_steps = 0

		for a in range(len(self.pomdp_tasks)):
			indices = list(range(self.task_indices[a][0],self.task_indices[a][1])) + self.robot_indices
			state = new_state[indices]
			outcomes, steps = self.pomdp_tasks[a].simulate_action(self.pomdp_tasks[a].get_state_index(state.tolist()),actions[a],all_poss_actions,horizon)
			k_steps = max(steps,k_steps)
			outcomes.append(outcomes)
			outcomes_len *= len(outcomes[a])
			outcomes_dim += (len(outcomes[a]),)

		count = 0
		while count < outcomes_len:
			outs = self.get_tuple(count,outcomes_dim)
			new_outcome = [1.0,-1, 0.0, True, 1, False]
			for a in range(len(self.pomdp_tasks)):		
				indices = list(range(self.task_indices[a][0],self.task_indices[a][1])) + self.robot_indices
				outcome = outcomes[a][outs[a]]
				new_outcome[0] *= outcome[0]
				new_outcome[2] += outcome[2]
				new_outcome[3] = new_outcome[3] and outcome[3]
				new_state[indices] = self.pomdp_tasks[a].get_state_tuple(outcome[1])

			new_outcome[1] = self.get_state_index(new_state)
			all_outcomes.append(tuple(new_outcome))

			count += 1

		if self.is_part_of_action_space(action):
			if start_state_index not in self.transition_function.keys():
				self.transition_function[start_state_index] = {} 

			if action not in self.transition_function[start_state_index].keys():
				self.transition_function[start_state_index][action] = all_outcomes
			else:
				self.transition_function[start_state_index][action].extend(all_outcomes)

		# print (self.get_state_tuple(start_state_index), actions, all_outcomes, self.get_state_tuple(all_outcomes[0][1]))
		# set_trace()
		return all_outcomes, k_steps



