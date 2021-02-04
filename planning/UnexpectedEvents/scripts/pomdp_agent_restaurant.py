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
from pomdp_agent import *

GLOBAL_TIME = 0

class AgentPOMDPRestaurant(AgentPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_tasks, pomdp_solvers, tasks, robot, random, extra_pomdps=None):
		self.random = random
		self.print_status = False
		self.pomdp_tasks = pomdp_tasks
		self.pomdp_solvers = pomdp_solvers
		self.tasks = tasks
		self.robot = robot
		self.extra_pomdp_solvers = extra_pomdps

		self.name = "pomdp_agent_restaurant" 

		self.nA = 0
		self.nS = 1
		self.nO = 1
		obs = []
		self.feature_indices = {}
		self.obs_feature_indices = {}
		self.task_indices = []
		self.state_space_dim = ()
		self.observation_space_dim = ()
		self.navigation_goals_len = len(self.pomdp_tasks)
		self.one_task_actions_num = self.pomdp_tasks[0].non_navigation_actions_len

		self.pomdp_actions = []
		

		self.actions = np.full(((self.one_task_actions_num-1)*len(self.pomdp_tasks)+1+self.navigation_goals_len,len(self.pomdp_tasks)),None)
		self.pomdp_actions = np.full(((self.one_task_actions_num-1)*len(self.pomdp_tasks)+1+self.navigation_goals_len,),None)

		self.all_no_ops = {}
		
		for n in self.pomdp_tasks[0].noop_actions.keys():	
			all_no_op = np.full((len(self.pomdp_tasks),),None)
			for l in range(len(self.pomdp_tasks)):	
				all_no_op[l] = self.pomdp_tasks[l].noop_actions[n]

			self.all_no_ops[n] = all_no_op

		self.actions[0,:] =  self.all_no_ops['1']
		self.pomdp_actions[0] = 0

		# for a in self.all_no_ops.keys():
		# 	for l in range(len(self.pomdp_tasks)): 
		# 		self.all_no_ops[a][l].print()

		# set_trace()

		count = 1
		for l in range(len(self.pomdp_tasks)):	
			for i in range(self.one_task_actions_num):
				pomdp = None
				self.actions[count,l] = self.pomdp_tasks[l].actions[i]				
				if i != self.pomdp_tasks[l].noop_actions['1'].id: ## no_op
					pomdp = l
					for k in range(len(self.pomdp_tasks)):	
						if k != l:
							self.actions[count,k] = self.pomdp_tasks[k].noop_actions[str(self.actions[count,l].time_steps)]

				if pomdp is not None:
					self.pomdp_actions[count] = pomdp
					count += 1	

		for l in range(len(self.pomdp_tasks)):
			self.actions[count,:] = self.pomdp_tasks[l].actions[self.pomdp_tasks[l].task.table.id + self.one_task_actions_num]
			self.pomdp_actions[count] = l

			count += 1	
		
		self.all_actions =  self.actions
		self.all_pomdp_actions = self.pomdp_actions

		# set_trace()
		# for a in self.actions:
		# 	for l in range(len(self.pomdp_tasks)): 
		# 		a[l].print()

		# print (self.pomdp_actions)
		# set_trace()

		self.nA = len(self.actions)
		self.action_space = spaces.Discrete(self.nA)

		# self.feasible_actions = self.actions

		# self.feasible_actions = np.full(((self.one_task_actions_num-1-1)*len(self.pomdp_tasks)+1+self.navigation_goals_len,len(self.pomdp_tasks)),None)
		self.feasible_actions = np.array(self.actions)
		# self.feasible_actions_index = np.full(((self.one_task_actions_num-1-1)*len(self.pomdp_tasks)+1+self.navigation_goals_len,),-1)
		self.feasible_actions_index = np.array(range(0,len(self.actions)))
		# count = 0
		# for i in range(len(self.actions)):
		# 	action_0 = False
		# 	for j in range(len(self.pomdp_tasks)):
		# 		if self.actions[i,j].id == 0:
		# 			action_0 = True

		# 	if not action_0:
		# 		self.feasible_actions[count,:] = self.actions[i,:]
		# 		self.feasible_actions_index[count] = i				
		# 		count += 1


		# print (self.actions, self.feasible_actions)
		self.feasible_actions = list(self.feasible_actions)
		self.valid_actions = list(self.feasible_actions)

		# set_trace()

		feature_count = 0
		obs_feature_count = 0
		task_count = 0
		for task in self.tasks:
			start = feature_count
			for feature in task.get_features():
				if feature.type == "discrete" and feature.name in self.pomdp_tasks[0].non_robot_features:

					self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
					self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					obs.append((feature.low, feature.high, feature.discretization, feature.name))
					self.feature_indices[feature.name+str(task_count)] = feature_count
					feature_count += 1
					if feature.observable:
						self.obs_feature_indices[feature.name+str(task_count)] = obs_feature_count
						self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
						self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
						obs_feature_count += 1
			end = feature_count
			self.task_indices.append((start,end))

			task_count += 1
			

		
		self.robot_indices = []
		self.robot_obs_indices = []
		# robot's features
		for feature in self.robot.get_features():
			if feature.type == "discrete" and feature.name in ["x","y"]:
				self.robot_indices.append(feature_count)
				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
				obs.append((feature.low, feature.high, feature.discretization, feature.name))
				self.feature_indices[feature.name] = feature_count
				feature_count += 1
				if feature.observable:
					self.robot_obs_indices.append(obs_feature_count)
					self.obs_feature_indices[feature.name] = obs_feature_count
					self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
					obs_feature_count += 1

		# print (self.nS)
		self.state_space = obs
		if self.print_status:
			print ("state space: ", self.state_space)
			print ("state space dim: ", self.state_space_dim)

		self.transition_function = {}
		self.observation_function = {}


		self.dense_reward = True

		# if self.print_status:
		# 	print ("computing P ...")
		# self.compute_P()
		# if self.print_status:
		# 	print ("done computing P ...")

		self.state = None

	def step(self, actions, position=None, start_state=None, belief=None, observation=None):
		global GLOBAL_TIME
		if start_state is None:
			start_state = self.state
		print ("step action: ", self.get_state_tuple(start_state), actions)
		
		k = 1
		sum_prob = 0
		new_state = np.asarray(deepcopy(start_state))
		terminal = True

		for a in range(len(self.pomdp_tasks)):
			indices = list(range(self.task_indices[a][0],self.task_indices[a][1])) + self.robot_indices
			outcomes, steps = self.pomdp_tasks[a].simulate_action(self.pomdp_tasks[a].get_state_index(new_state[indices]),actions[a],all_poss_actions=True, horizon=None)
			for outcome in outcomes:
				rand_num = self.random.choice(100)
				sum_prob += outcome[0]*100
				if  rand_num < sum_prob:
					# print (rand_num, sum_prob, outcome)
					new_state[indices] = self.pomdp_tasks[a].get_state_tuple(outcome[1])
					reward += outcome[2]
					terminal = terminal and outcome[3]
					break

		position = (new_state[self.feature_indices['x']],new_state[self.feature_indices['y']])
		self.state = self.get_state_index(new_state)
		obs = self.get_state_tuple(self.state)

		for a in reversed(range(len(self.pomdp_tasks))):
			obs.pop(self.feature_indices['customer_satisfaction' + str(a)])

		debug_info = {}
		# debug_info['prob'] = prob
		# debug_info['steps'] = k
		# debug_info['action_highlevel'] = action_highlevel
		print ("new state", new_state, reward, terminal)
		return self.get_state_index(new_state), self.get_observation_index(obs), reward, terminal, debug_info, position

	def simulate_observation (self, next_state_index, action):
		if next_state_index in self.observation_function.keys() and action in self.observation_function[next_state_index].keys():
			return self.observation_function[next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	
		# print (next_state, action)
		for a in reversed(range(len(self.pomdp_tasks))):
			next_state.pop(self.feature_indices['customer_satisfaction' + str(a)])

		obs = {(self.get_observation_index(next_state),1.0)}
		# print (obs, next_state)
		# set_trace()
		if next_state_index not in self.observation_function.keys():
			self.observation_function[next_state_index] = {} 

		if action not in self.observation_function[next_state_index].keys():
			self.observation_function[next_state_index][action] = obs
		else:
			self.observation_function[next_state_index][action].update(obs)

		return obs

	def get_possible_next_states (self,observation, belief_prob=None):
		possible_states = list()
		
		obs_tuple = self.get_observation_tuple(observation)
		# print (obs_tuple)
		## generate everything in advance
		pos_len = 1
		pos_dim = ()
		for a in range(len(self.pomdp_tasks)): 
			sat_index = self.feature_indices["customer_satisfaction"+str(a)]
			obs_tuple.insert(sat_index,-1)
			pos_len *= self.state_space_dim[sat_index]
			pos_dim += (self.state_space_dim[sat_index],)

		count = 0
		while count < pos_len:
			pos_s = self.get_tuple(count,pos_dim)
			for a in range(len(self.pomdp_tasks)): 
				sat_index = self.feature_indices["customer_satisfaction"+str(a)]
				obs_tuple[sat_index] = pos_s[a]

			count += 1
			possible_states.append(self.get_state_index(obs_tuple))

		# 	print (obs_tuple)
		# set_trace()

		return possible_states

	def get_possible_obss (self, belief_prob, all_poss_actions, horizon):
		possible_obss = set()
		possible_states = set()

		# if all_poss_actions:
		actions = self.actions
		# else:
		# 	actions = self.feasible_actions
		
		for (prob,state) in belief_prob:
			for a in actions:
				# if a not in self.env.navigation_actions:
				outcomes, steps = self.simulate_action(state,a,all_poss_actions,horizon)
				for outcome in outcomes:
					possible_states.add(outcome[1])

		for ps in possible_states:
			st = self.get_state_tuple(ps)
			for a in reversed(range(len(self.pomdp_tasks))):  
				st.pop(self.feature_indices["customer_satisfaction"+str(a)])
			possible_obss.add(self.get_observation_index(st)) #self.env.get_observation_tuple(obs)
			# print (st)

		return possible_obss
