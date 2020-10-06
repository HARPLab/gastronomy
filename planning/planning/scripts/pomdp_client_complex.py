import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pdb import set_trace
import numpy as np
from copy import deepcopy
from copy import copy
from time import sleep
import pylab as plt
import math


from draw_env import *
from pomdp_client_restaurant import *

GLOBAL_TIME = 0
print_status = False

class ClientPOMDPComplex(ClientPOMDPRestaurant):
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
		self.non_robot_features = ["cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise", \
				"time_since_hand_raise","current_request","customer_satisfaction"]
		self.robot_features = False
		self.navigation_goals_len = len(navigation_goals) 		

		self.actions.append(Action(0, "I'll be back - table "+str(self.task.table.id), self.task.table.id, True, 1))
		self.actions.append(Action(1, 'no op - table '+str(self.task.table.id), self.task.table.id, True, 1))
		self.actions.append(Action(2, 'serve - table '+str(self.task.table.id), self.task.table.id, True, 1))
		self.actions.append(Action(3, 'food not ready - table '+str(self.task.table.id), self.task.table.id, True, 1))

		self.non_navigation_actions_len = len(self.actions)
		self.nA = self.non_navigation_actions_len + self.navigation_goals_len 
		self.navigation_actions = []

		for i in range(self.non_navigation_actions_len, self.nA):
			act = Action(i, 'go to table '+str((i-self.non_navigation_actions_len)), (i-self.non_navigation_actions_len), False, None)
			self.actions.append(act)
			self.navigation_actions.append(act)

		if no_op:
			self.valid_actions = self.actions[0:self.non_navigation_actions_len]
			self.valid_actions.append(self.actions[self.task.table.id + self.non_navigation_actions_len])
		else:
			self.valid_actions = self.actions

		self.pomdps_actions = self.actions[0:self.non_navigation_actions_len] + [self.actions[self.task.table.id + self.non_navigation_actions_len]]
		self.feasible_actions = list(self.valid_actions)
		# self.feasible_actions.remove(self.feasible_actions[0])

		self.noop_actions = {}
		self.noop_actions['1'] = self.actions[1]
		self.noop_actions['2'] = Action(1, "no op 2t - table "+str(self.task.table.id), self.task.table.id, True, 2)
		self.noop_actions['3'] = Action(1, "no op 3t - table "+str(self.task.table.id), self.task.table.id, True, 3)
		self.noop_actions['4'] = Action(1, "no op 4t - table "+str(self.task.table.id), self.task.table.id, True, 4)

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
				obs.append((feature.low, feature.high, feature.discretization, feature.name))
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
			if feature.type == "discrete" and feature.name in ["x","y"]:
				self.robot_indices.append(feature_count)
				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
				obs.append((feature.low, feature.high, feature.discretization, feature.name))
				self.feature_indices[feature.name] = feature_count
				feature_count += 1
				if feature.observable:
					self.robot_obs_indices.append(obs_feature_count)
					self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
					self.obs_feature_indices[feature.name] = obs_feature_count
					obs_feature_count += 1

		print ("# states: ", self.nS)
		self.state_space = obs
		if print_status:
			print ("state space: ", self.state_space)
			print ("state space dim: ", self.state_space_dim)

		self.transition_function = {}
		self.observation_function = {}
		self.belief_library = {}


		self.dense_reward = True


		# if self.print_status:
		# 	print ("computing P ...")
		# self.compute_P()
		# if self.print_status:
		# 	print ("done computing P ...")

		self.state = None


	def compute_satisfaction(self, max_time, max_sat, start_time, new_time, start_sat, threshold):
		new_sat = start_sat

		if new_time <= threshold:
			pass
		elif start_time <= threshold and new_time > threshold and new_time <= threshold*2:
			new_sat = max(new_sat-1,0)
		elif start_time <= threshold*2 and new_time > threshold*2 and new_time <= threshold*3:
			new_sat = max(new_sat-1,0)
		elif start_time <= threshold*3 and new_time > threshold*3 and new_time <= threshold*4:
			new_sat = max(new_sat-1,0)
		elif new_time > threshold*4:
			new_sat = 0

		return new_sat

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None, remaining_time_steps=None):
		global GLOBAL_TIME

		if remaining_time_steps is None:
			part_of_action_space = self.is_part_of_action_space(action) or all_poss_actions
			if part_of_action_space in self.transition_function.keys() and\
				start_state_index in self.transition_function[part_of_action_space].keys() \
					and action in self.transition_function[part_of_action_space][start_state_index].keys():
				return self.transition_function[part_of_action_space][start_state_index][action]

		k_steps = 1
		if action.time_steps is not None:
			k_steps = action.time_steps

		reward = 0
		start_state =  self.get_state_tuple(start_state_index)

		outcomes = []		

		new_state = deepcopy(start_state)

		distance_to_table = self.distance((start_state[self.feature_indices['x']],start_state[self.feature_indices['y']]),(self.task.table.goal_x,self.task.table.goal_y))
		
		nav_action = False
		random_cooking_time = False
		random_eating_time = False
		random_drinking_time = False

		drinking_prob = []
		eating_prob = []
		cooking_prob = []

		if not action.type: 
			current_position = (start_state[self.feature_indices['x']],start_state[self.feature_indices['y']])
			new_position, reward_nav, k_steps = self.simulate_navigation_action(action, current_position)

			if new_position != (self.task.table.goal_x,self.task.table.goal_y):
				reward_nav = 0.0

			if all_poss_actions or self.is_part_of_action_space(action):
				new_state [self.feature_indices['x']] = new_position[0]
				new_state [self.feature_indices['y']] = new_position[1]

			nav_action = True


		if start_state [self.feature_indices['hand_raise']] == 1:
			if (start_state [self.feature_indices['current_request']] != 4 and start_state [self.feature_indices['current_request']] != 5) or \
			(start_state [self.feature_indices['current_request']] == 4 and start_state [self.feature_indices['food']] == 3) or \
			(start_state [self.feature_indices['current_request']] == 5 and start_state [self.feature_indices['water']] == 3):
				new_state [self.feature_indices['time_since_hand_raise']] += 1
				if new_state [self.feature_indices['time_since_hand_raise']] >= self.state_space[self.feature_indices['time_since_hand_raise']][1]:
					new_state [self.feature_indices['time_since_hand_raise']] = self.state_space[self.feature_indices['time_since_hand_raise']][1]


		# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
		# ["cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise","time_since_hand_raise","current_request","customer_satisfaction"]

		t_hand = int(self.state_space[self.feature_indices['time_since_hand_raise']][1]/3)
		if start_state [self.feature_indices['current_request']] == 3: 
			if not random_cooking_time:
				if new_state [self.feature_indices['cooking_status']] != 2:
					if start_state [self.feature_indices['time_since_hand_raise']] <= t_hand \
						and (new_state [self.feature_indices['time_since_hand_raise']] > t_hand \
							and new_state [self.feature_indices['time_since_hand_raise']] <= t_hand*2):
						new_state [self.feature_indices['cooking_status']] += 1
					elif (start_state [self.feature_indices['time_since_hand_raise']] > t_hand \
						and start_state [self.feature_indices['time_since_hand_raise']] <= t_hand*2) \
						and (new_state [self.feature_indices['time_since_hand_raise']] > t_hand*2):
						new_state [self.feature_indices['cooking_status']] += 1
					elif new_state [self.feature_indices['time_since_hand_raise']] > t_hand*3:
						new_state [self.feature_indices['cooking_status']] += 1

			if random_cooking_time and new_state [self.feature_indices['cooking_status']] != 2:
				cooking_prob.append((0.3,new_state [self.feature_indices['cooking_status']]))
				cooking_prob.append((0.7,new_state [self.feature_indices['cooking_status']]+1))


		if start_state [self.feature_indices['current_request']] == 3 and start_state [self.feature_indices['cooking_status']] == 2:
			if start_state [self.feature_indices['food']] == 0:
				new_state [self.feature_indices['time_since_food_ready']] += 1
				if new_state [self.feature_indices['time_since_food_ready']] >= self.state_space[self.feature_indices['time_since_food_ready']][1]:
					new_state [self.feature_indices['time_since_food_ready']] = self.state_space[self.feature_indices['time_since_food_ready']][1]

		if start_state [self.feature_indices['current_request']] == 4 and start_state [self.feature_indices['food']] != 0:
			new_state [self.feature_indices['time_since_served']] += 1
			if new_state [self.feature_indices['time_since_served']] >= self.state_space[self.feature_indices['time_since_served']][1]:
				new_state [self.feature_indices['time_since_served']] = self.state_space[self.feature_indices['time_since_served']][1]

			if not random_eating_time:
				if new_state [self.feature_indices['food']] != 3:
					if start_state [self.feature_indices['time_since_served']] <= t_hand \
						and (new_state [self.feature_indices['time_since_served']] > t_hand \
							and new_state [self.feature_indices['time_since_served']] <= t_hand*2):
						new_state [self.feature_indices['food']] += 1
					elif (start_state [self.feature_indices['time_since_served']] > t_hand \
						and start_state [self.feature_indices['time_since_served']] <= t_hand*2) \
						and (new_state [self.feature_indices['time_since_served']] > t_hand*2):
						new_state [self.feature_indices['food']] += 1
					elif new_state [self.feature_indices['time_since_served']] > t_hand*3:
						new_state [self.feature_indices['food']] += 1

			if random_eating_time and new_state [self.feature_indices['food']] != 3:
				eating_prob.append((0.3,new_state [self.feature_indices['food']]))
				eating_prob.append((0.7,new_state [self.feature_indices['food']]+1))

		### water
		if start_state [self.feature_indices['current_request']] == 5:
			if start_state [self.feature_indices['water']] > 0:
				new_state [self.feature_indices['time_since_served']] += 1
				if new_state [self.feature_indices['time_since_served']] >= self.state_space[self.feature_indices['time_since_served']][1]:
					new_state [self.feature_indices['time_since_served']] = self.state_space[self.feature_indices['time_since_served']][1]

				if not random_drinking_time:
					if new_state [self.feature_indices['water']] != 3:
						if start_state [self.feature_indices['time_since_served']] <= t_hand \
							and (new_state [self.feature_indices['time_since_served']] > t_hand \
								and new_state [self.feature_indices['time_since_served']] <= t_hand*2):
							new_state [self.feature_indices['water']] += 1
						elif (start_state [self.feature_indices['time_since_served']] > t_hand \
							and start_state [self.feature_indices['time_since_served']] <= t_hand*2) \
							and (new_state [self.feature_indices['time_since_served']] > t_hand*2):
							new_state [self.feature_indices['water']] += 1
						elif new_state [self.feature_indices['time_since_served']] > t_hand*3:
							new_state [self.feature_indices['water']] += 1

				if random_drinking_time and new_state [self.feature_indices['water']] != 3:
					drinking_prob.append((0.3,new_state [self.feature_indices['water']]))
					drinking_prob.append((0.7,new_state [self.feature_indices['water']]+1))



		current_req = start_state [self.feature_indices['current_request']]
		max_time = self.state_space[self.feature_indices['time_since_hand_raise']][1]
		max_sat = self.state_space[self.feature_indices['customer_satisfaction']][1]
		if current_req == 3 and start_state [self.feature_indices['cooking_status']] == 2:			
			time_index = self.feature_indices['time_since_food_ready']
			start_time = start_state [time_index]
			new_time = new_state [time_index]
			threshold = int(max_time/(max_sat+1))
		else:
			time_index = self.feature_indices['time_since_hand_raise']
			start_time = start_state [time_index]
			new_time = new_state [time_index]
			threshold = int(max_time/(max_sat))
			

		start_sat = new_state [self.feature_indices['customer_satisfaction']]
		
		if remaining_time_steps is None:
			time_steps = k_steps
		else:
			time_steps = remaining_time_steps

		if new_state [self.feature_indices['hand_raise']] == 0:
			terminal = True
			reward = 0.0
			if (action.id == 3 or action.id == 2 or action.id == 0) and distance_to_table != 0:
				reward = -np.Inf
			outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
		else:
			if not action.type: 
				if (start_state [self.feature_indices['current_request']] != 4 and start_state [self.feature_indices['current_request']] != 5) or \
							(start_state [self.feature_indices['current_request']] == 4 and start_state [self.feature_indices['food']] == 3) or \
							(start_state [self.feature_indices['current_request']] == 5 and start_state [self.feature_indices['water']] == 3):
					sat = self.compute_satisfaction(max_time, max_sat, start_time, min(start_time+1,max_time), start_sat, threshold)
					new_state [time_index] = min(start_time+1,max_time)
				else:
					sat = self.compute_satisfaction(max_time, max_sat, start_time, start_time, start_sat, threshold)
					# new_state [time_index] = min(start_time+1,max_time)

				new_state [self.feature_indices['customer_satisfaction']] = sat					
				reward += (reward_nav/k_steps + self.get_reward(start_state,new_state,False))
				outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
			
			else:
				sat = self.compute_satisfaction(max_time, max_sat, start_time, new_time, start_sat, threshold)
				new_state [self.feature_indices['customer_satisfaction']] = sat

				new_req = 0
				if action.id == 2 and distance_to_table == 0 and current_req != 8:
					if current_req == 1 or current_req == 2 or (current_req == 3 and start_state [self.feature_indices['cooking_status']] == 2) or \
						(current_req == 4 and start_state [self.feature_indices['food']] == 3) or \
						(current_req == 5 and start_state [self.feature_indices['water']] == 3) or current_req == 6 or current_req == 7: 

						new_state [self.feature_indices['time_since_hand_raise']] = 0

						if (current_req == 3 and start_state [self.feature_indices['cooking_status']] == 2):
							new_state [self.feature_indices['food']] = 1

						if (current_req == 4 and start_state [self.feature_indices['food']] == 3):
							new_state [self.feature_indices['time_since_served']] = 0
							new_state [self.feature_indices['water']] = 1

						new_req = current_req + 1
						new_state [self.feature_indices['current_request']] = new_req
						if sat == self.state_space[self.feature_indices['customer_satisfaction']][0]: ## low
							new_state [self.feature_indices['customer_satisfaction']] = sat
							outcomes.append((0.7,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
							new_state [self.feature_indices['customer_satisfaction']] = sat + 1
							outcomes.append((0.3,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

						elif sat == self.state_space[self.feature_indices['customer_satisfaction']][1]: ## high
							new_state [self.feature_indices['customer_satisfaction']] = sat
							outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

						else: 
							new_state [self.feature_indices['customer_satisfaction']] = sat
							outcomes.append((0.4,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
							new_state [self.feature_indices['customer_satisfaction']] = sat + 1
							outcomes.append((0.6,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

					else:
						outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))


				elif action.id == 0 and distance_to_table == 0 and start_state [self.feature_indices['hand_raise']] == 1 and \
						start_state [self.feature_indices['time_since_hand_raise']] > 4 and start_state [self.feature_indices['time_since_hand_raise']] < 9:
					new_state [self.feature_indices['time_since_hand_raise']] = 0

					# if sat == self.state_space[self.feature_indices['customer_satisfaction']][0]: ## low
					# 	new_state [self.feature_indices['customer_satisfaction']] = sat
					# 	outcomes.append((0.9,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
					# 	new_state [self.feature_indices['customer_satisfaction']] = sat + 1
					# 	outcomes.append((0.1,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

					# elif sat == self.state_space[self.feature_indices['customer_satisfaction']][1]: ## high
					# 	new_state [self.feature_indices['customer_satisfaction']] = sat
					# 	outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

					# else: 
					# 	new_state [self.feature_indices['customer_satisfaction']] = sat
					# 	outcomes.append((0.95,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
					# 	new_state [self.feature_indices['customer_satisfaction']] = sat + 1
					# 	outcomes.append((0.05,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

					outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))	


				elif current_req == 3 and action.id == 3 and start_state [self.feature_indices['cooking_status']] < 2 and distance_to_table == 0:
					new_state [self.feature_indices['customer_satisfaction']] = self.state_space[self.feature_indices['customer_satisfaction']][1] - 1
					outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					
				elif action.id == 2 and current_req == 8 and new_state [self.feature_indices['hand_raise']] == 1 and distance_to_table == 0:
					# print (start_state, new_state)
					# set_trace()
					new_state [self.feature_indices['hand_raise']] = 0 
					terminal = True
					reward += 1.0
					outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))

				elif (action.id == 3 or action.id == 2 and action.id == 0 and action.id == 1) and distance_to_table != 0:
					reward += -np.Inf
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))

				else:
					terminal = False 
					if new_state [self.feature_indices['hand_raise']] == 0:
						terminal = True
					outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), terminal, 1, False))
			

		GLOBAL_TIME += k_steps
		##outcomes.append((prob,self.get_state_index(new_state), reward=0, terminal, k=1, action_highlevel=False))
		# for outcome in outcomes:
		# 	print (start_state, action, (outcome[0],self.get_state_tuple(outcome[1]),outcome[2],outcome[3],outcome[4],outcome[5]))

		## stochastic cooking and eating time
		if (random_eating_time and len(eating_prob) != 0) or (random_cooking_time and len(cooking_prob) != 0) or \
			(random_drinking_time and len(drinking_prob) != 0):
			new_outcomes = []
			# print (start_state [self.feature_indices['current_request']], outcomes, start_state, new_state)
			for out in outcomes:
				# print ("outcome: ", out)
				prob = out[0];new_state = self.get_state_tuple(out[1]);rew = out[2];terminal = out[3];steps = out[4];info = out[5]
				if random_eating_time and len(eating_prob) != 0:
					for status in eating_prob:
						# print ("eating status: ", status)
						new_state[self.feature_indices['food']] = status[1]
						new_outcome = (prob*status[0],self.get_state_index(new_state),rew,terminal,steps,info)
						new_outcomes.append(new_outcome)
						# print ("new_outcome: ", new_outcome)
				elif random_drinking_time and len(drinking_prob) != 0:
					for status in drinking_prob:
						# print ("drinking status: ", status)
						new_state[self.feature_indices['water']] = status[1]
						new_outcome = (prob*status[0],self.get_state_index(new_state),rew,terminal,steps,info)
						new_outcomes.append(new_outcome)
						# print ("new_outcome: ", new_outcome)
				elif random_cooking_time and len(cooking_prob) != 0:
					# print (cooking_prob)
					for status in cooking_prob:
						# print ("cooking status: ", status)
						new_state[self.feature_indices['cooking_status']] = status[1]
						new_outcome = (prob*status[0],self.get_state_index(new_state),rew,terminal,steps,info)
						new_outcomes.append(new_outcome)
						# print ("new_outcome: ", new_outcome)
					# set_trace()

			outcomes = new_outcomes
			# print (outcomes)
			# set_trace()

		new_outcomes = self.compute_outcomes(k_steps, time_steps, start_state, new_state, action, all_poss_actions, horizon, outcomes)

		if print_status:
			print ("POMDP: ", self.task.table.id, new_outcomes, start_state, action.id, self.get_state_tuple(new_outcomes[0][1]))
			# set_trace()

		if remaining_time_steps is None:
			if len(self.transition_function.keys()) == 0:
				self.transition_function[True] = {}
				self.transition_function[False] = {}

			part_of_action_space = self.is_part_of_action_space(action) or all_poss_actions
			if start_state_index not in self.transition_function[part_of_action_space].keys():
				self.transition_function[part_of_action_space][start_state_index] = {} 

			if action not in self.transition_function[part_of_action_space][start_state_index].keys():
				self.transition_function[part_of_action_space][start_state_index][action] = (new_outcomes,k_steps)
			else:
				self.transition_function[part_of_action_space][start_state_index][action].extend((new_outcomes,k_steps))

		return new_outcomes, k_steps

	def compute_outcomes(self, k_steps, time_steps, start_state, new_state, action, all_poss_actions, horizon, outcomes):
		if k_steps > 1:
			# set_trace()
			if time_steps > 1:
				start_state_copy = deepcopy(start_state)				

				new_outcomes = []
				for out in outcomes:
					new_state_copy = deepcopy(start_state)
					prob = out[0];temp_state = self.get_state_tuple(out[1]);rew = out[2];terminal = out[3];steps = out[4];info = out[5]
					new_state_copy [self.feature_indices['customer_satisfaction']] = temp_state [self.feature_indices['customer_satisfaction']]
					new_state_copy [self.feature_indices['time_since_hand_raise']] = temp_state [self.feature_indices['time_since_hand_raise']]
					new_state_copy [self.feature_indices['cooking_status']] = temp_state [self.feature_indices['cooking_status']]
					new_state_copy [self.feature_indices['time_since_food_ready']] = temp_state [self.feature_indices['time_since_food_ready']]
					new_state_copy [self.feature_indices['food']] = temp_state [self.feature_indices['food']]
					new_state_copy [self.feature_indices['water']] = temp_state [self.feature_indices['water']]
					new_state_copy [self.feature_indices['time_since_served']] = temp_state [self.feature_indices['time_since_served']]
					# print ("out: ", out)
					# print ("new: ", new_state_copy)
					partial_outcomes, partial_k_steps = self.simulate_action(self.get_state_index(new_state_copy), action, \
							all_poss_actions, horizon, time_steps-1)

					for part_out in partial_outcomes:
						# print ("part out: ", part_out)
						new_state_copy = self.get_state_tuple(part_out[1])
						# print (new_state_copy)
						# print ("new: ", new_state_copy)
						rew += self.gamma * part_out[0] * part_out[2]
						terminal = terminal or part_out[3]
						new_outcome = (prob*part_out[0],self.get_state_index(new_state_copy),rew,terminal,steps,info)
						new_outcomes.append(new_outcome)

				return new_outcomes

			else:
				new_outcomes = []
				for out in outcomes:
					prob = out[0];new_state_copy = self.get_state_tuple(out[1]);rew = out[2];terminal = out[3];steps = out[4];info = out[5]
					new_state_copy [self.feature_indices['x']] = new_state [self.feature_indices['x']]
					new_state_copy [self.feature_indices['y']] = new_state [self.feature_indices['y']]
					new_state_copy [self.feature_indices['current_request']] = new_state [self.feature_indices['current_request']]
					new_outcomes.append((prob,self.get_state_index(new_state_copy), rew, terminal, 1, False))

				return new_outcomes

		return outcomes
