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

class AgentPOMDPPackage(AgentPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_tasks, pomdp_solvers, tasks, robot, random, extra_pomdps=None):
		self.random = random
		self.print_status = False
		self.pomdp_tasks = pomdp_tasks
		self.pomdp_solvers = pomdp_solvers
		self.tasks = tasks
		self.robot = robot
		self.extra_pomdp_solvers = extra_pomdps

		self.name = "pomdp_agent_package" 

		self.nA = 0
		self.nS = 1
		self.nO = 1
		obs = []
		self.feature_indices = {}
		self.obs_feature_indices = {}
		self.task_indices = []
		self.state_space_dim = ()
		self.observation_space_dim = ()
		self.navigation_goals_len = self.pomdp_tasks[0].navigation_goals_len
		self.one_task_actions_num = self.pomdp_tasks[0].non_navigation_actions_len

		self.pomdp_actions = []
		

		self.actions = np.full(((self.one_task_actions_num-1)*len(self.pomdp_tasks)+1+(self.pomdp_tasks[0].nA_valid-self.one_task_actions_num),len(self.pomdp_tasks)),None)
		self.pomdp_actions = np.full(((self.one_task_actions_num-1)*len(self.pomdp_tasks)+1+(self.pomdp_tasks[0].nA_valid-self.one_task_actions_num),),None)

		self.all_actions = np.full(((self.one_task_actions_num-1)*len(self.pomdp_tasks)+1+(self.pomdp_tasks[0].nA_valid-self.one_task_actions_num)*(len(self.pomdp_tasks)),len(self.pomdp_tasks)),None)
		self.all_pomdp_actions = np.full(((self.one_task_actions_num-1)*len(self.pomdp_tasks)+1+(self.pomdp_tasks[0].nA_valid-self.one_task_actions_num)*(len(self.pomdp_tasks)),),None)


		self.all_no_ops = {}
		
		for n in self.pomdp_tasks[0].noop_actions.keys():	
			all_no_op = np.full((len(self.pomdp_tasks),),None)
			for l in range(len(self.pomdp_tasks)):	
				all_no_op[l] = self.pomdp_tasks[l].noop_actions[n]

			self.all_no_ops[n] = all_no_op

		self.actions[0,:] =  self.all_no_ops['1']
		self.pomdp_actions[0] = 0


		self.all_actions[0,:] =  self.all_no_ops['1']
		self.all_pomdp_actions[0] = 0

		# for a in self.all_no_ops.keys():
		# 	for l in range(len(self.pomdp_tasks)): 
		# 		self.all_no_ops[a][l].print()

		count = 1
		for l in range(len(self.pomdp_tasks)):	
			for i in range(self.one_task_actions_num):
				pomdp = None
				self.actions[count,l] = self.pomdp_tasks[l].valid_actions[i]
				self.all_actions[count,l] = self.pomdp_tasks[l].valid_actions[i]				
				if i != self.pomdp_tasks[l].noop_actions['1'].id: ## no_op
					pomdp = l
					for k in range(len(self.pomdp_tasks)):	
						if k != l:
							self.actions[count,k] = self.pomdp_tasks[k].noop_actions[str(self.actions[count,l].time_steps)]
							self.all_actions[count,k] = self.pomdp_tasks[k].noop_actions[str(self.actions[count,l].time_steps)]

				if pomdp is not None:
					# self.actions[count,l].print()
					self.pomdp_actions[count] = pomdp
					self.all_pomdp_actions[count] = pomdp
					count += 1	
		
		### driving
		self.action_mapping = {}
		all_count = count
		for j in range(0, self.pomdp_tasks[0].task.table.restaurant.num_trucks):
			for i in range(j*(self.navigation_goals_len)+self.pomdp_tasks[0].non_navigation_actions_len,(j+1)*(self.navigation_goals_len)+self.pomdp_tasks[0].non_navigation_actions_len):
				for l in range(len(self.pomdp_tasks)):
					self.actions[count,l] = self.pomdp_tasks[0].valid_actions[i]
				self.pomdp_actions[count] = 0

				count += 1	

		## all driving
		const_count = all_count
		for l in range(len(self.pomdp_tasks)):
			temp_count = const_count
			for j in range(0, self.pomdp_tasks[0].task.table.restaurant.num_trucks):
				for i in range(j*(self.navigation_goals_len)+self.pomdp_tasks[0].non_navigation_actions_len,(j+1)*(self.navigation_goals_len)+self.pomdp_tasks[0].non_navigation_actions_len):
					self.all_actions[all_count,:] = self.pomdp_tasks[l].valid_actions[i]
					self.all_pomdp_actions[all_count] = l

					self.action_mapping[self.all_actions[all_count,0].id] = self.actions[temp_count,0].id
					all_count += 1	
					temp_count += 1

		const_count = count
		# pinging 
		for j in range(0, self.pomdp_tasks[0].task.table.restaurant.num_trucks):
			for l in range(len(self.pomdp_tasks)):
				self.actions[count,l] = self.pomdp_tasks[0].valid_actions[j + self.pomdp_tasks[l].nA_special_valid]
			self.pomdp_actions[count] = 0

			count += 1		

		## all pinging
		for l in range(len(self.pomdp_tasks)):
			temp_count = const_count
			for j in range(0, self.pomdp_tasks[0].task.table.restaurant.num_trucks):
				self.all_actions[all_count,:] = self.pomdp_tasks[l].valid_actions[j + self.pomdp_tasks[l].nA_special_valid]
				self.all_pomdp_actions[all_count] = l

				self.action_mapping[self.all_actions[all_count,0].id] = self.actions[temp_count,0].id	
				all_count += 1	
				temp_count += 1
		

		# # set_trace()
		# for a in self.actions:
		# 	for l in range(len(self.pomdp_tasks)): 
		# 		a[l].print()

		# # print (self.pomdp_actions)
		# # set_trace()

		# # set_trace()
		# for a in self.all_actions:
		# 	for l in range(len(self.pomdp_tasks)): 
		# 		a[l].print()

		# print (self.all_pomdp_actions)
		# print (self.action_mapping)
		# set_trace()

		self.nA = len(self.actions)
		self.action_space = spaces.Discrete(self.nA)

		self.feasible_actions = np.array(self.actions)
		self.feasible_actions_index = np.array(range(0,len(self.actions)))
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
					obs.append((feature.low, feature.high+1, feature.discretization, feature.name))
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
			if feature.type == "discrete":
				self.robot_indices.append(feature_count)
				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
				obs.append((feature.low, feature.high+1, feature.discretization, feature.name))
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

		if self.print_status:
			print ("state space: ", self.state_space, self.state_space_dim, self.nS, self.feature_indices)
		for a in range(len(self.pomdp_tasks)): 
			self.observation_space_dim += (self.pomdp_tasks[a].state_space[self.pomdp_tasks[a].feature_indices['current_location']][1]+2+1,) ## location, success/fail or NA
			self.nO *= self.pomdp_tasks[a].state_space[self.pomdp_tasks[a].feature_indices['current_location']][1]+2+1 ## or NA
			self.obs_feature_indices["ping_loc_success_"+str(a)] = obs_feature_count
			obs_feature_count += 1

		if self.print_status:
			print ("observation space: ", self.observation_space_dim, self.nO, self.obs_feature_indices)

		self.transition_function = {}
		self.observation_function = {}
		self.dense_reward = True

		self.state = None
		# set_trace()

	def get_random_observation (self, action):
		# set_trace()
		if action.id >= self.pomdp_tasks[0].ping_indices[0] and action.id < self.pomdp_tasks[0].ping_indices[1]:
			obs = [self.robot.get_feature('x_'+str(action.additional_info["truck"])).value]
		elif action.id >= self.pomdp_tasks[0].non_navigation_actions_len and action.id < self.pomdp_tasks[0].nA_special:
			dest = action.additional_info["city"]
			truck = action.additional_info["truck"]
			action_status = self.random.choice(a=[self.pomdp_tasks[0].success_ind,self.pomdp_tasks[0].fail_ind],p=[0.75,0.25])
			if action_status == self.pomdp_tasks[0].success_ind:
				self.robot.set_feature("x_"+str(truck),dest)
				obs = [self.random.choice(a=[self.pomdp_tasks[0].success_ind,self.pomdp_tasks[0].fail_ind],p=[0.9,0.1])]
			else:
				probs = np.full((self.robot.restaurant.num_cities,),1.0/(self.robot.restaurant.num_cities-1))
				probs[dest] = 0
				random_dest = self.random.choice(a=list(range(0,self.robot.restaurant.num_cities)),p=probs)
				self.robot.set_feature("x_"+str(truck),random_dest)
				obs = [self.random.choice(a=[self.pomdp_tasks[0].success_ind,self.pomdp_tasks[0].fail_ind],p=[0.1,0.9])]
		# set_trace()
		return obs

	def step(self, actions, position=None, start_state=None):
		global GLOBAL_TIME
		print("agent pomdp: step function")
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

		position = None
		obs = []
		self.state = self.get_state_index(new_state)
		for a in range(len(self.pomdp_tasks)):
			if a >= 1 and actions[a-1].id == actions[a].id:
				obs += obs[a-1]
			else:
				obs += self.pomdp_tasks[a].get_random_observation()

		debug_info = {}
		# debug_info['prob'] = prob
		# debug_info['steps'] = k
		# debug_info['action_highlevel'] = action_highlevel
		# print ("new state", new_state, reward, terminal)
		return self.get_state_index(new_state), self.get_observation_index(obs), reward, terminal, debug_info, position

	def simulate_observation (self, next_state_index, action):
		if next_state_index in self.observation_function.keys() and action in self.observation_function[next_state_index].keys():
			return self.observation_function[next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	
		# print (next_state, action)
		obs = []
		for a in range(len(self.pomdp_tasks)):
			act = action[a]
			indices = list(range(self.task_indices[a][0],self.task_indices[a][1])) + self.robot_indices
			next_state_i = next_state[indices] 
			obs += self.pomdp_tasks[a].simulate_observation(next_state_i,act)
		# print (obs, next_state)
		# set_trace()
		if next_state_index not in self.observation_function.keys():
			self.observation_function[next_state_index] = {} 

		if action not in self.observation_function[next_state_index].keys():
			self.observation_function[next_state_index][action] = obs
		else:
			self.observation_function[next_state_index][action].update(obs)

		return obs

	def get_possible_next_states (self, observation, belief_prob):
		new_time = np.zeros(len(self.pomdp_tasks))
		for a in range(len(self.pomdp_tasks)): 
			st = self.get_state_tuple(belief_prob[a][1])
			if st[self.feature_indices["time"+str(a)]] < self.state_space[self.feature_indices['time'+str(a)]][1]:
				new_time[a] = st[self.feature_indices["time"+str(a)]] + 1
			else:
				new_time[a] = st[self.feature_indices["time"+str(a)]]

		possible_states = set()
		count = 0
		while count < self.nS:
			obs_tuple = self.get_tuple(count,self.state_space_dim)
			for a in range(len(self.pomdp_tasks)): 
				time_index = self.feature_indices["time"+str(a)]
				obs_tuple[time_index] = new_time[a]
			print(obs_tuple)
			possible_states.add(self.get_state_index(obs_tuple))

		print ("get_possible_next_states in pomdp_agent_package")
		# set_trace()
		return possible_states

	def get_possible_obss (self, belief_prob,all_poss_actions,horizon):
		possible_obss = set()

		ping_loc_index = self.obs_feature_indices["ping_loc_success"]
		obs_tuple = self.get_observation_tuple(0)

		for p in range(0,self.observation_space_dim[ping_loc_index]):
			obs_tuple[ping_loc_index] = p
			possible_obss.add(self.get_observation_index(obs_tuple))

		return possible_obss

