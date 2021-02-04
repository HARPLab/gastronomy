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
from pomdp_client import *

GLOBAL_TIME = 0
print_status = False

class ClientPOMDPPackage(ClientPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op):
		global print_status
		self.deterministic = deterministic
		self.gamma = gamma
		self.random = random
		self.reset_random = reset_random
		self.task = task
		self.robot = robot
		self.no_op = no_op

		self.actions = []
		self.name = "pomdp_task"
		self.non_robot_features = ["current_location","time"]
		self.robot_features = False
		self.navigation_goals_len = len(navigation_goals) 		

		num_time_steps = 2 # load and pings

		self.actions.append(Action(0, "ping package - package "+str(self.task.table.id), self.task.table.id, True, num_time_steps))
		self.actions.append(Action(1, 'no op - package '+str(self.task.table.id), self.task.table.id, True, 1))
		self.actions.append(Action(2, 'unload - package '+str(self.task.table.id), self.task.table.id, True, 1))

		self.valid_actions = self.actions[0:len(self.actions)]

		self.load_indices = (len(self.actions),len(self.actions)+self.task.table.restaurant.num_trucks)
		count = 0
		for i in range(0, self.task.table.restaurant.num_trucks):
			additional_info = {}
			additional_info["truck"] = i
			self.actions.append(Action(self.load_indices[0]+i, 'load - package '+str(self.task.table.id) + " on truck " + str(i), self.task.table.id, True, num_time_steps, additional_info))
			self.valid_actions.append(self.actions[self.load_indices[0]+count])
			count += 1

		self.non_navigation_actions_len = len(self.actions)
		self.navigation_actions = []
		count = len(self.actions)
		for n in range(self.task.table.restaurant.num_tables):
			for j in range(0, self.task.table.restaurant.num_trucks):
				for i in range(self.non_navigation_actions_len, self.non_navigation_actions_len + self.navigation_goals_len):
					additional_info = {}
					additional_info["city"] = i-self.non_navigation_actions_len
					additional_info["truck"] = j
					act = Action(count, 'drive truck ' + str(j) + ' to city '+str((additional_info["city"])) + " - package " + str(n), n, False, 1, additional_info)
					self.actions.append(act)
					self.navigation_actions.append(act)
					if n == self.task.table.id:
						self.valid_actions.append(self.actions[count])
					count += 1
		
		self.nA_special = len(self.actions)
		self.nA_special_valid = len(self.valid_actions)

		self.ping_indices = (len(self.actions),len(self.actions)+self.task.table.restaurant.num_trucks*self.task.table.restaurant.num_tables)
		count = len(self.actions)
		for n in range(self.task.table.restaurant.num_tables):
			for i in range(self.nA_special, self.nA_special + self.task.table.restaurant.num_trucks):
				additional_info = {}
				additional_info["truck"] = (i-self.nA_special)
				act = Action(count, 'ping truck '+str((i-self.nA_special)) + " - package " + str(n), n, True, num_time_steps, additional_info)
				self.actions.append(act)
				if n == self.task.table.id:
					self.valid_actions.append(self.actions[count])
				count += 1		

		self.nA = len(self.actions)
		self.nA_valid = len(self.valid_actions)
		# self.valid_actions = self.actions

		# self.pomdps_actions = self.actions[0:self.nA_special] + [self.actions[self.task.table.id + self.nA_special]]
		self.pomdps_actions = self.actions[0:self.nA]
		self.feasible_actions = list(self.valid_actions)
		# self.feasible_actions.remove(self.feasible_actions[0])

		# print ("------------------------------------------")
		# for a in self.actions:
		# 	a.print()
		# print ("------------------------------------------")
		# print ("++++++++++++++++++++++++++++++++++++++++++")
		# for a in self.valid_actions:
		# 	a.print()
		# print ("++++++++++++++++++++++++++++++++++++++++++")

		self.noop_actions = {}
		self.noop_actions['1'] = self.actions[1]
		self.noop_actions['2'] = Action(1, "no op 2t - package "+str(self.task.table.id), self.task.table.id, True, 2)

		# for a in self.pomdps_actions:
		# 	a.print()
		# set_trace()

		self.navigation_goals = navigation_goals
		self.action_space = spaces.Discrete(self.nA)
		self.nS = 1
		self.nO = 1
		obs = []
		self.feature_indices = {}
		self.obs_feature_indices = {}
		self.state_space_dim = ()
		self.observation_space_dim = ()

		feature_count = 0
		obs_feature_count = 0

		for feature in self.task.get_features():
			if feature.type == "discrete" and feature.name in self.non_robot_features:
				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
				obs.append((feature.low, feature.high+1, feature.discretization, feature.name))
				self.feature_indices[feature.name] = feature_count
				feature_count += 1
				if feature.observable:
					self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
					self.obs_feature_indices[feature.name] = obs_feature_count
					obs_feature_count += 1

		# robot's features
		self.robot_indices = []
		self.robot_obs_indices = []
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
					self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
					self.obs_feature_indices[feature.name] = obs_feature_count
					obs_feature_count += 1

		# print ("# states: ", self.nS)
		self.state_space = obs
		if print_status:
			print ("# states: ", self.nS)
			print ("state space: ", self.state_space)
			print ("state space dim: ", self.state_space_dim)

		# print ("state space: ", self.state_space, self.state_space_dim, self.nS, self.feature_indices)

		self.observation_space_dim += (self.state_space[self.feature_indices['current_location']][1]+2+1,) ## location, success/fail or NA
		self.nO *= self.state_space[self.feature_indices['current_location']][1]+2+1 ## or NA
		self.success_ind = self.state_space[self.feature_indices['current_location']][1]
		self.fail_ind = self.state_space[self.feature_indices['current_location']][1]+1
		self.NA_ind = self.state_space[self.feature_indices['current_location']][1]+2
		self.obs_feature_indices["ping_loc_success"] = obs_feature_count
		obs_feature_count += 1

		# print ("observation space: ", self.observation_space_dim, self.nO, self.obs_feature_indices)

		self.transition_function = {}
		self.observation_function = {}
		self.belief_library = {}


		self.dense_reward = True


		# if self.print_status:
		# 	print ("computing P ...")
		# self.compute_P()
		# if self.print_status:
		# 	print ("done computing P ...")
		# set_trace()
		self.state = None

	def get_random_robot_features(self, outcomes):
		states = []
		probs = []
		for outcome in outcomes:
			st = outcome[1]
			pr = outcome[0]
			states.append(st)
			probs.append(pr)

		new_state_index = self.random.choice(a=states,p=probs)
		new_state = self.get_state_tuple(new_state_index)
		robot_features = {}
		# set_trace()
		for ri in self.robot_indices:
			robot_features[self.state_space[ri][3]]=new_state[ri]
		return robot_features

	def is_feasible_robot_state(self, state_index):
		state = self.get_state_tuple(state_index)
		# print ("state: ", state)
		# set_trace()
		for ri in self.robot_indices:
			if state[ri] != self.robot.get_feature(self.state_space[ri][3]).value:
				# print (self.state_space[ri][3],self.robot.get_feature(self.state_space[ri][3]).value)
				return False
		return True

	def step(self, action, start_state=None, simulate=False, belief=None, observation=None):
		global GLOBAL_TIME
		if start_state is None:
			start_state = self.state

		if print_status:
			print ("STEP: POMDP ", self.task.table.id ," step action: ", self.get_state_tuple(start_state))
			action.print()
		
		k = 1
		sum_prob = 0

		outcomes, steps = self.simulate_action(start_state,action,all_poss_actions=True,horizon=None)
		# if not self.robot.updated:
		# 	self.robot.updates_features(self.get_random_robot_features(outcomes))
			# self.robot.print_features()
			# set_trace()

		states = np.zeros((len(outcomes)),dtype=int)
		probs = np.zeros((len(outcomes)),dtype=float)
		c = 0

		for outcome in outcomes:
			st = outcome[1]
			pr = outcome[0]
			if self.is_feasible_robot_state(st):
				states[c] = st
				probs[c] = pr
			c += 1

		# set_trace()
		if sum(probs) == 0:
			print ("STEP: POMDP ", self.task.table.id ," step action: ", self.get_state_tuple(start_state))
			action.print()
			print ("Step: sum(probs) == 0")
			# set_trace()
		new_state_index = self.random.choice(a=states,p=probs/sum(probs))
		for outcome in outcomes:
			if  new_state_index == outcome[1]:				
				reward = outcome[2]
				terminal = outcome[3]
				break

		new_state = self.get_state_tuple(new_state_index)
		position = None
		if action.id >= self.non_navigation_actions_len and action.id < self.nA_special:
			position = ()
			for i in range(0,self.task.table.restaurant.num_trucks):
				position += (new_state[self.feature_indices['x_'+str(i)]],)
			# print ("POSITION: ", position)
		# new_state_index = self.get_state_index(new_state)
		if not simulate:
			self.state = new_state_index

		if observation is not None and action.id >= self.non_navigation_actions_len:
			obs = self.get_observation_tuple(observation)
		else:
			obs = self.get_random_observation(action, new_state_index)
		# if obs[0] == 5:
		# 	set_trace()
		debug_info = {}
		# debug_info['prob'] = prob
		# debug_info['steps'] = k
		# debug_info['action_highlevel'] = action_highlevel
		if print_status:
			print ("STEP: new state", new_state, reward, terminal, obs)
			# set_trace()

		return new_state_index, self.get_observation_index(obs), reward, terminal, debug_info, position

	def reset(self, random):
		# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
		if random:
			# "cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise","time_since_hand_raise","current_request","customer_satisfaction"
			loc = None
			while (loc is None or loc == self.task.table.goal_x):
				loc = self.reset_random.randint(0,self.state_space[self.feature_indices['x_0']][1])
		else:
			loc = 0

		state = [loc,0] # 

		for feature in self.robot.get_features():
			state.append(self.robot.get_feature(feature.name).value)

		self.state = self.get_state_tuple(state)
		return state

	def render(self, start_state=None, mode='human'):
		pass

	def get_reward(self,start_state,new_state,k_steps):
		reward = 0
		count = 0
		start_time = start_state [self.feature_indices['time']]
		new_time = new_state [self.feature_indices['time']]

		while (count < k_steps):			
			reward += -1 * math.pow(1.2,min(start_time+1,new_time))
			start_time += 1
			count += 1
		return reward

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None, remaining_time_steps=None):
		if remaining_time_steps is None:
			part_of_action_space = self.is_part_of_action_space(action)
			if part_of_action_space in self.transition_function.keys() and\
				start_state_index in self.transition_function[part_of_action_space].keys() \
					and action in self.transition_function[part_of_action_space][start_state_index].keys():

				if print_status:
					outcomes = self.transition_function[part_of_action_space][start_state_index][action][0]
					print ("action: ", action.name)
					print ("POMDP: ", self.task.table.id, outcomes, self.get_state_tuple(start_state_index), action.id)
					for out in outcomes:
						print (self.get_state_tuple(out[1]))

				return self.transition_function[part_of_action_space][start_state_index][action]

		k_steps = 1
		if action.time_steps is not None:
			k_steps = action.time_steps

		reward = 0
		terminal = False
		start_state =  self.get_state_tuple(start_state_index)

		outcomes = []		

		new_state = deepcopy(start_state)
		new_time = min (start_state[self.feature_indices["time"]] + k_steps, self.state_space[self.feature_indices['time']][1]-1)

		# if start_state[self.feature_indices["time"]] < self.state_space[self.feature_indices['time']][1]-1:
		# 	new_time = start_state[self.feature_indices["time"]] + 1
		# else:
		# 	new_time = start_state[self.feature_indices["time"]]

		new_state[self.feature_indices["time"]] = new_time

		evolve_reward = self.get_reward(start_state,new_state,k_steps)

		if (start_state[self.feature_indices['current_location']] == self.task.table.goal_x) and \
			 not (not action.type or action.id >= self.ping_indices[0] and action.id < self.ping_indices[1]):
			terminal = True
			# new_state[self.feature_indices["time"]] = start_state[self.feature_indices["time"]] 
			outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
		else:
			if not action.type and (action.additional_info["city"] != start_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]): # navigation action
				if self.is_part_of_action_space(action):
					reward += -1.0

				if (start_state[self.feature_indices['current_location']] == self.task.table.goal_x):
					terminal = True
					# new_state[self.feature_indices["time"]] = start_state[self.feature_indices["time"]] 
					new_position = action.additional_info["city"]
					new_state [self.feature_indices['x_'+str(action.additional_info["truck"])]] = new_position
					outcomes.append((0.75,self.get_state_index(new_state), reward, terminal, 1, False))
					
					prob = 0.25/(self.task.table.restaurant.num_cities-1)
					for i in range(0,self.task.table.restaurant.num_cities):
						if i != new_position:
							new_state [self.feature_indices['x_'+str(action.additional_info["truck"])]] = i
							outcomes.append((prob,self.get_state_index(new_state), reward, terminal, 1, False))
				else:
					# reward += -0.1
					reward += evolve_reward
					new_position = action.additional_info["city"]

					# if new_position == 0:
					# 	random_pos = 1
					# elif new_position == self.task.table.restaurant.num_cities-1:
					# 	random_pos = self.task.table.restaurant.num_cities - 2
					# else:
					# 	random_pos = new_position + 1
					# new_state [self.feature_indices['x_'+str(action.additional_info["truck"])]] = random_pos
					# outcomes.append((0.25,self.get_state_index(new_state), reward, False, 1, False))

					new_state [self.feature_indices['x_'+str(action.additional_info["truck"])]] = new_position
					outcomes.append((0.75,self.get_state_index(new_state), reward, False, 1, False))

					prob = 0.25/(self.task.table.restaurant.num_cities-1)
					for i in range(0,self.task.table.restaurant.num_cities):
						if i != new_position:
							new_state [self.feature_indices['x_'+str(action.additional_info["truck"])]] = i
							outcomes.append((prob,self.get_state_index(new_state), reward, False, 1, False))

			elif not action.type and not (action.additional_info["city"] != start_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]): # navigation action
				if self.is_part_of_action_space(action):
					reward += -1.0

				if (start_state[self.feature_indices['current_location']] == self.task.table.goal_x):
					terminal = True
					# new_state[self.feature_indices["time"]] = start_state[self.feature_indices["time"]] 
					outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
				else:
					# reward += -0.1
					reward += evolve_reward
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
			else:
				if action.id == 0:
					reward += -0.1
					reward += evolve_reward
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
				elif action.id == 1:
					# reward += -0.1
					reward += evolve_reward
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
				elif action.id >= self.load_indices[0] and action.id < self.load_indices[1] \
					 and (start_state [self.feature_indices['x_'+str(action.additional_info["truck"])]] == start_state[self.feature_indices['current_location']]):
					reward += -0.1
					reward += evolve_reward
					new_state [self.feature_indices['current_location']] = self.task.table.restaurant.num_cities + action.additional_info["truck"]
					outcomes.append((0.9,self.get_state_index(new_state), reward, False, 1, False))
					new_state [self.feature_indices['current_location']] = start_state[self.feature_indices['current_location']]
					outcomes.append((0.1,self.get_state_index(new_state), reward, False, 1, False))
				elif action.id == 2 and ((self.task.table.restaurant.num_cities) <= start_state[self.feature_indices['current_location']]):
					terminal = False
					reward1 = -0.1
					# reward1 = 0.0
					reward1 += evolve_reward
					if start_state[self.feature_indices['x_'+str(start_state[self.feature_indices['current_location']]-self.task.table.restaurant.num_cities)]] == self.task.table.goal_x:
						reward2 = 10.0
						reward2 += evolve_reward
						terminal = True
					else:
						reward2 = -0.1
						# reward2 = 0
						reward2 += evolve_reward
					new_state [self.feature_indices['current_location']] = start_state[self.feature_indices['current_location']]
					outcomes.append((0.1,self.get_state_index(new_state), reward1, False, 1, False))
					new_state [self.feature_indices['current_location']] = start_state[self.feature_indices['x_'+str(start_state[self.feature_indices['current_location']]-self.task.table.restaurant.num_cities)]]
					outcomes.append((0.9,self.get_state_index(new_state), reward2, terminal, 1, False))
				elif action.id == 2 and not ((self.task.table.restaurant.num_cities) <= start_state[self.feature_indices['current_location']]):
					reward += -0.1
					reward += evolve_reward
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
				elif action.id >= self.ping_indices[0] and action.id < self.ping_indices[1]:
					if self.is_part_of_action_space(action):
						reward += -0.1

					if (start_state[self.feature_indices['current_location']] == self.task.table.goal_x):
						terminal = True
						# new_state[self.feature_indices["time"]] = start_state[self.feature_indices["time"]] 
						outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
					else:
						# reward += -0.1
						reward += evolve_reward
						outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
				else:
					reward += -10
					# set_trace()
					reward += evolve_reward
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))

		if remaining_time_steps is None:
			if len(self.transition_function.keys()) == 0:
				self.transition_function[True] = {}
				self.transition_function[False] = {}

			part_of_action_space = self.is_part_of_action_space(action)
			if start_state_index not in self.transition_function[part_of_action_space].keys():
				self.transition_function[part_of_action_space][start_state_index] = {} 

			if action not in self.transition_function[part_of_action_space][start_state_index].keys():
				self.transition_function[part_of_action_space][start_state_index][action] = (outcomes,k_steps)
			else:
				self.transition_function[part_of_action_space][start_state_index][action].extend((outcomes,k_steps))

		if print_status:
			print ("action: ", action.name)
			print ("POMDP: ", self.task.table.id, outcomes, start_state, action.id, terminal)
			for out in outcomes:
				print (self.get_state_tuple(out[1]))
			# set_trace()

		return outcomes, k_steps

	def simulate_observation (self, next_state_index, action):
		if next_state_index in self.observation_function.keys() and action in self.observation_function[next_state_index].keys():
			return self.observation_function[next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	
		obs = set()
		# action.print()
		if action.type is False:
			if action.additional_info["city"] == next_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]:
				# obs.add((self.get_observation_index([self.success_ind]),1.0))
				obs.add((self.get_observation_index([self.success_ind]),0.9))
				obs.add((self.get_observation_index([self.fail_ind]),0.1))
			else:
				obs.add((self.get_observation_index([self.fail_ind]),0.9))
				obs.add((self.get_observation_index([self.success_ind]),0.1))
				# obs.add((self.get_observation_index([self.fail_ind]),1.0))
		elif action.id >= self.ping_indices[0] and action.id < self.ping_indices[1]:
			obs.add((self.get_observation_index([next_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]]),1.0))
			# prob = 0.2/(self.state_space[self.feature_indices['x_'+str(action.additional_info["truck"])]][1]-1)
			# for l in range(0,self.state_space[self.feature_indices['x_'+str(action.additional_info["truck"])]][1]):
			# 	if l != next_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]:
			# 		obs.add((self.get_observation_index([l]),prob))
		elif action.id == 0:
			obs.add((self.get_observation_index([next_state[self.feature_indices['current_location']]]),1.0))
			# prob = 0.2/(self.state_space[self.feature_indices['current_location']][1]-1)
			# for l in range(0,self.state_space[self.feature_indices['current_location']][1]):
			# 	if l != next_state[self.feature_indices['current_location']]:
			# 		obs.add((self.get_observation_index([l]),prob))
		elif action.id == self.noop_actions['1'].id:
			obs.add((self.get_observation_index([self.NA_ind]),1.0))
		elif action.id >= self.load_indices[0] and action.id < self.load_indices[1]:
			if (action.additional_info["truck"] + self.task.table.restaurant.num_cities) == next_state[self.feature_indices['current_location']]:
				obs.add((self.get_observation_index([self.success_ind]),0.9))
				obs.add((self.get_observation_index([self.fail_ind]),0.1))
			else:
				obs.add((self.get_observation_index([self.fail_ind]),0.9))
				obs.add((self.get_observation_index([self.success_ind]),0.1))
		elif action.id == 2:
			if next_state[self.feature_indices['current_location']] < self.task.table.restaurant.num_cities:
				obs.add((self.get_observation_index([self.success_ind]),0.9))
				obs.add((self.get_observation_index([self.fail_ind]),0.1))
			else:
				obs.add((self.get_observation_index([self.fail_ind]),0.9))
				obs.add((self.get_observation_index([self.success_ind]),0.1))

		if next_state_index not in self.observation_function.keys():
			self.observation_function[next_state_index] = {} 

		if action not in self.observation_function[next_state_index].keys():
			self.observation_function[next_state_index][action] = obs
		else:
			self.observation_function[next_state_index][action].update(obs)

		# if print_status:
		# 	print ("name: ", action.name)
		# 	print ("simulate_observation, POMDP: ", self.task.table.id, obs, action.id, action.name, self.get_state_tuple(next_state_index))

		return obs

	def get_random_observation (self, action, next_state_index):
		next_state = self.get_state_tuple(next_state_index)	
		obs = None
		if action.type is False:
			if action.additional_info["city"] == next_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]:
				obs = [self.success_ind]
			else:
				obs = [self.fail_ind]
		elif action.id >= self.ping_indices[0] and action.id < self.ping_indices[1]:
			obs = [next_state[self.feature_indices['x_'+str(action.additional_info["truck"])]]]
		elif action.id == 0:
			obs = [next_state[self.feature_indices['current_location']]]
		elif action.id == self.noop_actions['1'].id:
			obs = [self.NA_ind]
		elif action.id >= self.load_indices[0] and action.id < self.load_indices[1]:
			if (action.additional_info["truck"] + self.task.table.restaurant.num_cities) == next_state[self.feature_indices['current_location']]:
				obs = [self.success_ind]
			else:
				obs = [self.fail_ind]
		elif action.id == 2:
			if next_state[self.feature_indices['current_location']] < self.task.table.restaurant.num_cities:
				obs = [self.success_ind]
			else:
				obs = [self.fail_ind]

		return obs

	def get_possible_next_states (self, observation, belief_prob, action=None):
		st = self.get_state_tuple(belief_prob[0][1])
		k_steps = action.time_steps
		new_time = min (st[self.feature_indices["time"]] + k_steps, self.state_space[self.feature_indices['time']][1]-1)

		possible_states = set()
		count = 0
		while count < self.nS:
			obs_tuple = self.get_tuple(count,self.state_space_dim)
			obs_tuple[self.feature_indices["time"]] = new_time
			count += 1
			possible_states.add(self.get_state_index(obs_tuple))

		return possible_states

	def get_possible_obss (self, belief_prob,all_poss_actions,horizon):
		possible_obss = set()

		ping_loc_index = self.obs_feature_indices["ping_loc_success"]
		obs_tuple = self.get_observation_tuple(0)

		for p in range(0,self.observation_space_dim[ping_loc_index]):
			obs_tuple[ping_loc_index] = p
			possible_obss.add(self.get_observation_index(obs_tuple))

		return possible_obss