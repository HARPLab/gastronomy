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
from pomdp_client import *

GLOBAL_TIME = 0
print_status = False

class ClientPOMDPComplex(ClientPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op, run_on_robot):
		global print_status
		ClientPOMDP.__init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op, run_on_robot)

		self.name = "pomdp_task"
		if not self.KITCHEN:
			self.non_robot_features = ["have_bread","cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise", \
					"time_since_hand_raise", "current_request","customer_satisfaction"]
			# self.non_robot_features = ["next_request","cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise", \
			# 		"time_since_hand_raise", "current_request","customer_satisfaction"]
		else:
			self.non_robot_features = ["have_bread","cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise", \
					"time_since_hand_raise", "food_picked_up","current_request","customer_satisfaction"]
			# self.non_robot_features = ["next_request","cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise", \
			# 		"time_since_hand_raise", "food_picked_up","current_request","customer_satisfaction"]

		self.robot_features = False
		self.navigation_goals = navigation_goals

		self.set_actions()

		self.obs_feature_indices = [{},{}]
		self.unobs_feature_indices = [[],[]]
		self.observation_space_dim = [(),()]
		self.observation_function = [{},{}]
		self.observation_function_costs = [{},{}]
		self.nO = [1,0]

		self.set_states(obs_type=Observation_Type.ORIGINAL)

		
		self.dense_reward = True
		self.state = None

	def set_states (self, obs_type=Observation_Type.ORIGINAL, hidden_vars_names = [], model_vars=[], parameter_vars={}):
		self.belief_library = {}
		self.observation_function[obs_type] = {}
		self.observation_function_costs[obs_type] = {}
		self.nO[obs_type] = 1
		self.obs_feature_indices[obs_type] = {}
		self.unobs_feature_indices[obs_type] = []
		self.observation_space_dim[obs_type] = ()
		self.nS = 1
		obs = []
		if obs_type == Observation_Type.ORIGINAL:	 
			self.feature_indices = {}
			self.state_space_dim = ()

		self.transition_function = {}
		

		feature_count = 0
		obs_feature_count = 0

		self.obs_feature_indices[obs_type]["explicit"] = 0
		self.observation_space_dim[obs_type] += (2,)
		self.nO[obs_type] *= 2
		obs_feature_count += 1

		## unhappy, neutral, happy, unknown
		self.observation_space_dim[obs_type] += (4,) ## yes or no
		self.nO[obs_type] *= 4
		self.obs_feature_indices[obs_type]["emotion"] = obs_feature_count
		obs_feature_count += 1

		for feature in self.task.get_features():
			if feature.type == "discrete" and feature.name in self.non_robot_features:	
				if obs_type == Observation_Type.ORIGINAL:	
					if feature.type == "discrete" and feature.name in self.non_robot_features:
						self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
						self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
						obs.append((feature.low, feature.high, feature.discretization, feature.name))
						self.feature_indices[feature.name] = feature_count

				if feature.observable and not feature.dependent and feature.name not in hidden_vars_names:
					self.observation_space_dim[obs_type] += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO[obs_type] *= int((feature.high - feature.low) / feature.discretization) + 1
					self.obs_feature_indices[obs_type][feature.name] = obs_feature_count
					obs_feature_count += 1
				else:
					self.unobs_feature_indices[obs_type].append(feature_count)

				feature_count += 1


		## answer to the question
		if obs_type == Observation_Type.HUMAN_INPUT:
			self.observation_space_dim[obs_type] += (2,) ## yes or no
			self.nO[obs_type] *= 2
			self.obs_feature_indices[obs_type]["answer"] = obs_feature_count
			obs_feature_count += 1
			# feature_count += 1

		# model features
		for m_name in model_vars:
			if obs_type == Observation_Type.ORIGINAL:
				self.nS *= 2
				self.state_space_dim += (2,)
				obs.append((0, 1, 1, m_name))
				self.feature_indices[m_name] = feature_count				

				self.unobs_feature_indices[Observation_Type.ORIGINAL].append(feature_count)
				feature_count += 1
				## added
				self.nS *= 2
				self.state_space_dim += (2,)
				obs.append((0, 1, 1, "Q"+m_name))
				self.feature_indices["Q"+m_name] = feature_count				

				self.unobs_feature_indices[Observation_Type.ORIGINAL].append(feature_count)
				feature_count += 1
				##
				for par_name in parameter_vars[m_name]:
					self.nS *= 2
					self.state_space_dim += (2,)
					obs.append((0, 1, 1, par_name))
					self.feature_indices[par_name] = feature_count				

					self.unobs_feature_indices[Observation_Type.ORIGINAL].append(feature_count)
					feature_count += 1
					## added
					self.nS *= 2
					self.state_space_dim += (2,)
					obs.append((0, 1, 1, "Q"+par_name))
					self.feature_indices["Q"+par_name] = feature_count				

					self.unobs_feature_indices[Observation_Type.ORIGINAL].append(feature_count)
					feature_count += 1

			elif obs_type == Observation_Type.HUMAN_INPUT:
				self.unobs_feature_indices[Observation_Type.HUMAN_INPUT].append(feature_count)
				feature_count += 1
				## added
				self.unobs_feature_indices[Observation_Type.HUMAN_INPUT].append(feature_count)
				feature_count += 1
				##
				for par_name in parameter_vars[m_name]:
					self.unobs_feature_indices[Observation_Type.HUMAN_INPUT].append(feature_count)
					feature_count += 1	
					## added
					self.unobs_feature_indices[Observation_Type.HUMAN_INPUT].append(feature_count)
					feature_count += 1		


		# robot's features
		for feature in self.robot.get_features():
			if feature.type == "discrete" and feature.name in ["x","y"]:
				if obs_type == Observation_Type.ORIGINAL:
					self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
					self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					obs.append((feature.low, feature.high, feature.discretization, feature.name))
					self.feature_indices[feature.name] = feature_count
				
				if feature.observable and not feature.dependent:
					self.observation_space_dim[obs_type] += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO[obs_type] *= int((feature.high - feature.low) / feature.discretization) + 1
					self.obs_feature_indices[obs_type][feature.name] = obs_feature_count
					obs_feature_count += 1
				else:
					self.unobs_feature_indices[obs_type].append(feature_count)

				feature_count += 1

		self.unobs_feature_indices[obs_type].sort()

		if obs_type == Observation_Type.ORIGINAL:	
			self.state_space = obs
		if print_status:
			print ("# states: ", self.nS)
			print ("state space: ", self.state_space)
			print ("state space dim: ", self.state_space_dim)
			print ("states:")
			for o in obs:
				print (o)

	def set_actions(self, clarification_actions=[]):
		self.actions = []
		self.navigation_goals_len = len(self.navigation_goals) 	
		self.len_clarification_actions = 0	

		#### ADD ACTIONS
		self.actions.append(Action(0, "I'll be back - table "+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 1))
		self.actions.append(Action(1, 'no op - table '+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 1))
		self.actions.append(Action(2, 'food not ready - table '+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 1))
		self.actions.append(Action(3, 'please take the menu - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(4, 'what is your order? - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(5, 'please take your food - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(6, 'please take your drink - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(7, 'here is your bill - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(8, 'please place cash in my basket - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(9, 'here is your receipt, good bye! - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		self.actions.append(Action(10, 'table is clean - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))
		
		self.actions.append(Action(11, 'pick up food for table '+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 1))
		self.actions.append(Action(12, 'please take the bread - table '+str(self.task.table.id), self.task.table.id, Action_Type.SERVE, 1))

		if self.ACTIVE_PLANNING: # changing this
			len_actions = len(self.actions)
			action_index = len(self.actions)
			for a in clarification_actions:
				a.set_id(action_index)
				self.actions.append(a)
				action_index += 1
			self.len_clarification_actions = len(self.actions) - len_actions

		#### ADD ACTIONS

		self.non_navigation_actions_len = len(self.actions)
		self.nA = self.non_navigation_actions_len + self.navigation_goals_len 
		self.navigation_actions = []

		for i in range(self.non_navigation_actions_len, self.nA):
			act = Action(i, 'go to table '+str((i-self.non_navigation_actions_len)), (i-self.non_navigation_actions_len), Action_Type.NAVIGATION, None)
			self.actions.append(act)
			self.navigation_actions.append(act)

		################## go to kitchen + pick up actions
		if self.KITCHEN:
			nA_prev = self.nA
			self.nA += self.navigation_goals_len 
			for i in range(nA_prev, self.nA):
				act = Action(i, 'go to kitchen', (i-nA_prev), Action_Type.NAVIGATION, None, True)
				self.actions.append(act)
				self.navigation_actions.append(act)

		###################################################

		self.valid_actions = self.actions[0:self.non_navigation_actions_len]
		# self.valid_actions = self.actions[0:self.non_navigation_actions_len]
		self.valid_actions.append(self.actions[self.task.table.id + self.non_navigation_actions_len])
		if self.KITCHEN:
			self.valid_actions.append(self.actions[self.task.table.id + nA_prev])

		self.feasible_actions = list(self.valid_actions)
		# self.feasible_actions.pop(0)
		# for i in reversed(range(self.non_navigation_actions_len-self.len_clarification_actions,self.non_navigation_actions_len)):
		# 	self.feasible_actions.pop(i)


		self.pomdps_actions = self.actions[0:self.non_navigation_actions_len] + [self.actions[self.task.table.id + self.non_navigation_actions_len]]
		if self.KITCHEN:
			self.pomdps_actions += [self.actions[self.task.table.id + nA_prev]]

		self.noop_actions = {}
		self.noop_actions['1'] = self.actions[self.no_op_action_id]
		self.noop_actions['2'] = Action(self.no_op_action_id, "no op 2t - table "+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 2)
		self.noop_actions['3'] = Action(self.no_op_action_id, "no op 3t - table "+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 3)
		self.noop_actions['4'] = Action(self.no_op_action_id, "no op 4t - table "+str(self.task.table.id), self.task.table.id, Action_Type.INFORMATIVE, 4)
		
		self.action_space = spaces.Discrete(self.nA)

		if print_status:
			for a in self.feasible_actions:
				a.print()

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

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None, remaining_time_steps=None, modified=True):
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
		reward += action.cost
		start_state =  self.get_state_tuple(start_state_index)

		outcomes = []		

		new_state = deepcopy(start_state)

		distance_to_table = self.distance((start_state[self.feature_indices['x']],start_state[self.feature_indices['y']]),(self.task.table.goal_x,self.task.table.goal_y))

		if self.KITCHEN:
			distance_to_kitchen = self.distance((start_state[self.feature_indices['x']],start_state[self.feature_indices['y']]),(self.kitchen_pos[0],self.kitchen_pos[1]))
		
		nav_action = False
		random_cooking_time = False
		random_eating_time = False
		random_drinking_time = False

		drinking_prob = []
		eating_prob = []
		cooking_prob = []

		if action.type == Action_Type.NAVIGATION: 
			current_position = (start_state[self.feature_indices['x']],start_state[self.feature_indices['y']])

			if not action.kitchen:
				new_position, reward_nav, k_steps = self.simulate_navigation_action(action, current_position)

				if new_position != (self.task.table.goal_x,self.task.table.goal_y):
					reward_nav = 0.0
				else:
					if reward_nav == 0:
						reward_nav = 1

			elif action.kitchen:
				new_position, reward_nav, k_steps = self.simulate_go_to_kitchen_action(action, current_position)

				if not self.is_part_of_action_space(action): ## be careful, this won't work if pomdps share actions
					reward_nav = 0.0
				else:
					if reward_nav == 0:
						reward_nav = 1


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

		if start_state [self.feature_indices['current_request']] == 3 and start_state[self.feature_indices['cooking_status']] < 2:
			if start_state[self.feature_indices['have_bread']] == 1:
				new_state[self.feature_indices['have_bread']] = 0

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
			if (action.id == 3 or action.type == Action_Type.SERVE or action.id == 0) and distance_to_table != 0:
				reward = self.goal_pomdp_reward(-np.Inf)
			outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
		else:
			if action.type == Action_Type.NAVIGATION: 
				if (start_state [self.feature_indices['current_request']] != 4 and start_state [self.feature_indices['current_request']] != 5) or \
							(start_state [self.feature_indices['current_request']] == 4 and start_state [self.feature_indices['food']] == 3) or \
							(start_state [self.feature_indices['current_request']] == 5 and start_state [self.feature_indices['water']] == 3):
					sat = self.compute_satisfaction(max_time, max_sat, start_time, min(start_time+1,max_time), start_sat, threshold)
					new_state [time_index] = min(start_time+1,max_time)
				else:
					# do not change satisfaction if people are eating or drinking, this is needed for actions of length > 1
					sat = self.compute_satisfaction(max_time, max_sat, start_time, start_time, start_sat, threshold)
					# new_state [time_index] = min(start_time+1,max_time)

				new_state [self.feature_indices['customer_satisfaction']] = sat					
				reward += (reward_nav/k_steps + self.get_reward(start_state,new_state,False))
				outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
			
			else:
				sat = self.compute_satisfaction(max_time, max_sat, start_time, new_time, start_sat, threshold)
				new_state [self.feature_indices['customer_satisfaction']] = sat

				new_req = 0
				if action.type == Action_Type.SERVE and distance_to_table == 0 and current_req != 8:
					if (current_req == 1 and action.id == 3) or (current_req == 2 and action.id == 4) or \
					(action.id == 5 and current_req == 3 and start_state [self.feature_indices['cooking_status']] == 2 and (not self.KITCHEN or start_state[self.feature_indices['food_picked_up']] == 1)) or \
						(action.id == 6 and current_req == 4 and start_state [self.feature_indices['food']] == 3 and (not self.KITCHEN or start_state[self.feature_indices['food_picked_up']] == 1)) or \
						(action.id == 7 and current_req == 5 and start_state [self.feature_indices['water']] == 3) or (action.id == 8 and current_req == 6) or (action.id == 9 and current_req == 7): 

						new_state [self.feature_indices['time_since_hand_raise']] = 0

						if (current_req == 3 and start_state [self.feature_indices['cooking_status']] == 2 and (not self.KITCHEN or start_state[self.feature_indices['food_picked_up']] == 1)):
							new_state [self.feature_indices['food']] = 1
							if self.KITCHEN: 
								new_state[self.feature_indices['food_picked_up']] = 0

						if (current_req == 4 and start_state [self.feature_indices['food']] == 3 and (not self.KITCHEN or start_state[self.feature_indices['food_picked_up']] == 1)):
							new_state [self.feature_indices['time_since_served']] = 0
							new_state [self.feature_indices['water']] = 1
							if self.KITCHEN:
								new_state[self.feature_indices['food_picked_up']] = 0

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


						# if action.id == 4:
						# 	new_outcomes = []
						# 	# set_trace()
						# 	for outc in outcomes:
						# 		out_s = self.get_state_tuple(outc[1])
						# 		new_outcomes.append((outc[0]*0.8, outc[1], outc[2], outc[3], outc[4], outc[5]))
						# 		out_s [self.feature_indices['current_request']] = 4
						# 		out_s [self.feature_indices['food']] = 3
						# 		new_outcomes.append((outc[0]*0.2,self.get_state_index(out_s), outc[2], outc[3], outc[4], outc[5]))

						# 	outcomes = new_outcomes
					elif action.id == 12 and current_req == 3 and start_state[self.feature_indices['cooking_status']] < 2 and distance_to_table == 0:
						# reward += 1
						if start_state[self.feature_indices['have_bread']] == 0:
							# set_trace()
							new_state [self.feature_indices['customer_satisfaction']] = self.state_space[self.feature_indices['customer_satisfaction']][1]
							new_state [self.feature_indices['have_bread']] = 1
							outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
						else:
							reward = self.goal_pomdp_reward(-np.Inf)
							outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))

					else:
					# 	next_req = start_state [self.feature_indices['next_request']]
					# 	if (next_req == 1 and action.id == 3) or (next_req == 2 and action.id == 4) or \
					# (action.id == 5 and next_req == 3 and start_state [self.feature_indices['cooking_status']] == 2 and (not self.KITCHEN or start_state[self.feature_indices['food_picked_up']] == 1)) or \
					# 	(action.id == 6 and next_req == 4 and start_state [self.feature_indices['food']] == 3 and (not self.KITCHEN or start_state[self.feature_indices['food_picked_up']] == 1)) or \
					# 	(action.id == 7 and next_req == 5 and start_state [self.feature_indices['water']] == 3) or (action.id == 8 and next_req == 6) or (action.id == 9 and next_req == 7): 
					# 		reward += 0
					# 		outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					# 	else:
						reward += self.goal_pomdp_reward(-np.Inf) # negative reward for executing the wrong serve action given the current state 
						outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))


				elif action.id == 0 and distance_to_table == 0 and start_state [self.feature_indices['hand_raise']] == 1 and \
						start_state [self.feature_indices['time_since_hand_raise']] > 4 and start_state [self.feature_indices['time_since_hand_raise']] < 9:
					reward += self.goal_pomdp_reward(-np.Inf)
					# new_state [self.feature_indices['time_since_hand_raise']] = 0

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


				elif action.id == 2 and current_req == 3 and start_state [self.feature_indices['cooking_status']] < 2 and distance_to_table == 0:
					reward += self.goal_pomdp_reward(-np.Inf) ## do not take food is not ready for now
					if sat == self.state_space[self.feature_indices['customer_satisfaction']][1]: ## high
						new_state [self.feature_indices['customer_satisfaction']] = sat
						outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
					else: 
						new_state [self.feature_indices['customer_satisfaction']] = sat
						outcomes.append((0.9,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
						new_state [self.feature_indices['customer_satisfaction']] = sat + 1
						outcomes.append((0.1,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

				elif action.id == 10 and current_req == 8 and new_state [self.feature_indices['hand_raise']] == 1 and distance_to_table == 0:
					# print (start_state, new_state)
					# set_trace()
					new_state [self.feature_indices['hand_raise']] = 0 
					terminal = True
					reward += self.goal_pomdp_reward(100.0)
					outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))

				elif action.type == Action_Type.CLARIFICATION:
					if action.state["cat"] == 1:
						if start_state [self.feature_indices["Q"+"m" + str(action.state["model_num"])]] == 1:
							reward = np.Inf
						new_state [self.feature_indices["Q"+"m" + str(action.state["model_num"])]] = 1 

						outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					elif action.state["cat"] == 2:
						if "observation" in action.state.keys():
							name = "_o" + str(action.state["observation"][1])
						elif "start_state" in action.state.keys():
							name = "_s" + str(action.state["next_state"])

						if start_state[self.feature_indices["Q"+"m" + str(action.state["model_num"]) + name]] == 1:
							reward = np.Inf
							outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
						else:
							new_state [self.feature_indices["Q"+"m" + str(action.state["model_num"]) + name]] = 1 

							if start_state[self.feature_indices["m" + str(action.state["model_num"])]] == 1 and \
								start_state[self.feature_indices["m" + str(action.state["model_num"]) + name]] == 1:
								outcomes.append((0.5,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
								new_state [self.feature_indices["m" + str(action.state["model_num"]) + name]] = 0
								outcomes.append((0.5,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
							elif start_state[self.feature_indices["m" + str(action.state["model_num"])]] == 1 and \
								start_state[self.feature_indices["m" + str(action.state["model_num"]) + name]] == 0:
								outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
							elif start_state[self.feature_indices["m" + str(action.state["model_num"])]] == 0:
								# reward += 5
								outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))


						# if start_state [self.feature_indices["m" + str(action.state["model_num"]) + name]] == 1:
						# 	outcomes.append((0.5,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
						# 	new_state [self.feature_indices["m" + str(action.state["model_num"]) + name]] = 0
						# 	outcomes.append((0.5,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
						# else:
						# 	# new_state [self.feature_indices["m" + str(action.state["model_num"]) + name]] = 1
						# 	outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))

				elif self.KITCHEN and distance_to_kitchen == 0 and action.id == 11 and (current_req == 3 and start_state [self.feature_indices['cooking_status']] == 2)\
					and start_state [self.feature_indices['food_picked_up']] == 0:
					new_state [self.feature_indices['food_picked_up']] = 1
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))

				elif self.KITCHEN and distance_to_kitchen == 0 and action.id == 11 and (current_req == 4 and start_state [self.feature_indices['food']] == 3)\
					and start_state [self.feature_indices['food_picked_up']] == 0:
					new_state [self.feature_indices['food_picked_up']] = 1
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))

				elif (action.id == 2 or action.type == Action_Type.SERVE or action.id == 0) and distance_to_table != 0:
					reward += self.goal_pomdp_reward(-np.Inf)
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
				elif (action.id == 2):
					reward += self.goal_pomdp_reward(-np.Inf)
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))

				elif self.KITCHEN and action.id == 11 and distance_to_kitchen != 0:
					reward += self.goal_pomdp_reward(-np.Inf)
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))

				else:
					terminal = False 
					if new_state [self.feature_indices['hand_raise']] == 0:
						terminal = True
					outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), terminal, 1, False))
			

		GLOBAL_TIME += k_steps
		##outcomes.append((prob,self.get_state_index(new_state), reward=0, terminal, k=1, action_highlevel=False))
		# if self.task.table.id == 1:
		# 	for outcome in outcomes:
		# 		print (start_state, action.name, (outcome[0],self.get_state_tuple(outcome[1]),outcome[2],outcome[3],outcome[4],outcome[5]))
		# 		if action.id == 12:
		# 			set_trace()

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

		new_outcomes = self.compute_outcomes(k_steps, time_steps, start_state, new_state, action, all_poss_actions, horizon, outcomes, modified)
		new_outcomes = self.recompute_outcomes_based_on_modified_transitions(start_state_index, action, new_outcomes, modified)
		new_outcomes = self.compute_observation_cost(action, new_outcomes, modified)

		if print_status:
			print ("POMDP: ", self.task.table.id, start_state, action.id, action.name)
			strs = ""
			for n in new_outcomes:
				strs += "(" + str(n[0]) + ","  + str(self.get_state_tuple(n[1])) + "," + str(n[2]) + ")"
			print ("new outcomes: ", strs)
			# set_trace()

		if remaining_time_steps is None:
			# print (new_outcomes)
			# if (len(new_outcomes) == 1 and new_outcomes[0][0] == 0.5):
			# 	set_trace()

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

	def compute_outcomes(self, k_steps, time_steps, start_state, new_state, action, all_poss_actions, horizon, outcomes, modified):
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
					new_state_copy [self.feature_indices['have_bread']] = temp_state [self.feature_indices['have_bread']]
					new_state_copy [self.feature_indices['time_since_food_ready']] = temp_state [self.feature_indices['time_since_food_ready']]
					new_state_copy [self.feature_indices['food']] = temp_state [self.feature_indices['food']]
					new_state_copy [self.feature_indices['water']] = temp_state [self.feature_indices['water']]
					new_state_copy [self.feature_indices['time_since_served']] = temp_state [self.feature_indices['time_since_served']]
					# print ("out: ", out)
					# print ("new: ", new_state_copy)
					partial_outcomes, partial_k_steps = self.simulate_action(self.get_state_index(new_state_copy), action, \
							all_poss_actions, horizon, time_steps-1, modified)

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
					# new_state_copy [self.feature_indices['current_request']] = new_state [self.feature_indices['current_request']]
					new_outcomes.append((prob,self.get_state_index(new_state_copy), rew, terminal, 1, False))

				return new_outcomes

		return outcomes

	def recompute_outcomes_based_on_modified_transitions(self, start_state_index, action, outcomes, modified):
		# if len(self.modified_transition_function) > 0:
		# 	set_trace()
		if not modified:
			return outcomes
		modified_transition_function = self.modified_transition_function
		old_state_index = start_state_index
		start_state = self.get_state_tuple(old_state_index)
		selected_model = -1
		if self.models is not None:
			for model in range(0,len(self.models)):
				if start_state[self.feature_indices["m"+str(model)]] == 1:
					modified_transition_function = self.models[model].modified_transition_function
					selected_model = model
					break
		else:
			if start_state_index in modified_transition_function.keys() and action.id in modified_transition_function[start_state_index].keys():
				# set_trace()
				new_outcomes = []
				non_zero_transitions = len(modified_transition_function[old_state_index][action.id]) + len(outcomes)
				for (n_state,prob) in modified_transition_function[old_state_index][action.id]:
					# cost = self.get_reward(self.get_state_tuple(start_state_index),self.get_state_tuple(n_state),False)
					cost = 0
					new_outcomes.append((1.0/non_zero_transitions, n_state, cost, False, 1, False))

				for out in outcomes:
					new_outcomes.append((1.0/non_zero_transitions,out[1],out[2],out[3],out[4],out[5]))
				return new_outcomes
			else:
				return outcomes

		
		if len(modified_transition_function) > 0:
			# set_trace()
			new_outcomes = set()
			for tr in modified_transition_function.keys():
				start_modified_state = self.models[selected_model].get_state_tuple(tr)
				if action.id in modified_transition_function[tr].keys():
					if start_modified_state[self.models[selected_model].feature_indices["customer_satisfaction"]] >= \
						start_state[self.feature_indices["customer_satisfaction"]]:					
						for out in outcomes:
							next_state = self.get_state_tuple(out[1])
							next_modified_state = self.get_state_tuple(modified_transition_function[tr][action.id][0][0])
							next_state[self.feature_indices["customer_satisfaction"]] = next_modified_state[self.models[selected_model].feature_indices["customer_satisfaction"]]
							# cost = self.get_reward(start_state,next_state,False)
							cost = action.cost + self.get_reward(start_state,next_state,False)
							new_outcomes.add((1.0, self.get_state_index(next_state), cost))

							selected_par = None
							# set_trace()
							for var in self.model_pars["m"+str(selected_model)]:
								if var[0] == "state":
									if var[2] == action.id:
										s_m = self.models[selected_model].get_state_tuple(var[1])
										n_m = self.models[selected_model].get_state_tuple(var[3])
										if start_state[self.feature_indices["customer_satisfaction"]] <= s_m[self.models[selected_model].feature_indices["customer_satisfaction"]] \
										and next_state[self.feature_indices["customer_satisfaction"]] <= n_m[self.models[selected_model].feature_indices["customer_satisfaction"]]:
											selected_par = var[3]
											break

							if next_state[self.feature_indices["m"+str(selected_model)+"_s"+str(selected_par)]] != 0:
								new_outcomes.add((1.0, out[1], out[2]))

							# if start_modified_state[self.models[selected_model].feature_indices["customer_satisfaction"]] > \
							# 	start_state[self.feature_indices["customer_satisfaction"]] and start_state[self.feature_indices["Qm1_s2168023255"]] == 1:
							# 	print ("start state: ", start_state, start_modified_state)
							# 	print ("next state: ", next_state, next_modified_state)

			new_outcomes_tpl = []
			cost = None
			if len(new_outcomes) == 0:
				new_outcomes_tpl = outcomes
			elif len(new_outcomes) == 1:
				new_outcomes_tpl.append((1.0,out[1],out[2],self.is_goal_state(out[1]), 1, False))
			else:
				cost = UNRELIABLE_PARAMETER_COST*len(new_outcomes)
				for out in new_outcomes:		
					new_outcomes_tpl.append((1.0/len(new_outcomes),out[1],cost + out[2],self.is_goal_state(out[1]), 1, False))
			
			return new_outcomes_tpl
		else:
			return outcomes

	def compute_observation_cost (self, action, outcomes, modified):
		if not modified:
			return outcomes
		elif self.models is None:
			return outcomes
		else:
			new_outcomes = []
			for out in outcomes:
				obs_cost = self.simulate_observation_cost (out[1], action, modified=True)
				if obs_cost == 0:
					new_outcomes.append(out)
				else:
					# if self.num_step == 3:
					# 	set_trace()
					new_outcomes.append((out[0],out[1],out[2]+obs_cost,out[3],out[4],out[5]))

		return new_outcomes