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

print_status = False

class ClientPOMDPSimple(ClientPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op):
		global print_status
		ClientPOMDP.__init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op, run_on_robot)

		self.name = "pomdp_task"
		self.non_robot_features = ["hand_raise", "time_since_hand_raise","current_request","customer_satisfaction"]
		self.robot_features = False
		self.navigation_goals_len = len(navigation_goals) 
		
		self.actions = []
		self.actions.append(Action(0, "I'll be back - table "+str(self.task.table.id), self.task.table.id, 1, 1))
		self.actions.append(Action(1, 'no op - table '+str(self.task.table.id), self.task.table.id, 1, 1))
		self.actions.append(Action(2, 'serve - table '+str(self.task.table.id), self.task.table.id, 1, 2))


		self.non_navigation_actions_len = len(self.actions)
		self.nA = self.non_navigation_actions_len + self.navigation_goals_len ## I'll be back, no-action, serve
		self.navigation_actions = []


		for i in range(self.non_navigation_actions_len, self.nA):
			act = Action(i, 'go to table '+str((i-self.non_navigation_actions_len)), (i-self.non_navigation_actions_len), 0, None)
			self.actions.append(act)
			self.navigation_actions.append(act)

		if no_op:
			self.valid_actions = self.actions[0:self.non_navigation_actions_len]
			self.valid_actions += [self.actions[self.task.table.id + self.non_navigation_actions_len]]
		else:
			self.valid_actions = self.actions

		self.pomdps_actions = self.actions[0:self.non_navigation_actions_len] + [self.actions[self.task.table.id + self.non_navigation_actions_len]]
		self.feasible_actions = list(self.valid_actions)
		# self.feasible_actions.remove(self.feasible_actions[0])

		self.noop_actions = {}
		self.noop_actions['1'] = self.actions[self.no_op_action_id]
		self.noop_actions['2'] = Action(self.no_op_action_id, "no op 2t - table "+str(self.task.table.id), self.task.table.id, 1, 2)
		self.noop_actions['3'] = Action(self.no_op_action_id, "no op 3t - table "+str(self.task.table.id), self.task.table.id, 1, 3)
		self.noop_actions['4'] = Action(self.no_op_action_id, "no op 4t - table "+str(self.task.table.id), self.task.table.id, 1, 4)

		# for a in self.actions:
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
		for feature in self.robot.get_features():
			if feature.type == "discrete" and feature.name in ["x","y"]:
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

		print ("# states: ", self.nS)
		self.state_space = obs
		if self.print_status:
			print ("state space: ", self.state_space)
			print ("state space dim: ", self.state_space_dim)
		# self.reset()

		self.transition_function = {}
		self.observation_function = {}


		self.dense_reward = True


		# if self.print_status:
		# 	print ("computing P ...")
		# self.compute_P()
		# if self.print_status:
		# 	print ("done computing P ...")

	def reset(self, random):
		
		# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
		if random:
			current_req = self.reset_random.randint(1,self.state_space[self.feature_indices['current_request']][1])
			customer_satisfaction = self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 
		else:
			current_req = 1
			customer_satisfaction = 1

		state = [1,0,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value]
		return state

	def compute_satisfaction(self, max_time, max_sat, start_time, new_time, start_sat):
		t_hand_sat = int(max_time/(max_sat))

		new_sat = start_sat
		if new_time <= t_hand_sat:
			pass
		elif start_time <= t_hand_sat and new_time > t_hand_sat and new_time <= t_hand_sat*2:
			new_sat = max(new_sat-1,0)
		elif start_time <= t_hand_sat*2 and new_time > t_hand_sat*2 and new_time <= t_hand_sat*3:
			new_sat = max(new_sat-1,0)
		elif start_time <= t_hand_sat*3 and new_time > t_hand_sat*3 and new_time <= t_hand_sat*4:
			new_sat = max(new_sat-1,0)
		elif new_time > t_hand_sat*4:
			new_sat = 0

		return new_sat

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None, remaining_time_steps=None):
		
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
		nav_action = False

		distance_to_table = self.distance((start_state[self.feature_indices['x']],start_state[self.feature_indices['y']]),(self.task.table.goal_x,self.task.table.goal_y))

		if action.type == 0: # navigation action
			current_position = (start_state[self.feature_indices['x']],start_state[self.feature_indices['y']])
			new_position, reward_nav, k_steps = self.simulate_navigation_action(action, current_position)
			# print (action,current_position,new_position,reward_nav,k_steps)

			if new_position != (self.task.table.goal_x,self.task.table.goal_y):
				reward_nav = 0.0

			if all_poss_actions or self.is_part_of_action_space(action):
				new_state [self.feature_indices['x']] = new_position[0]
				new_state [self.feature_indices['y']] = new_position[1]

			nav_action = True

		t_hand = int(self.state_space[self.feature_indices['time_since_hand_raise']][1]/3)
		
		if start_state [self.feature_indices['hand_raise']] == 1:
			new_state [self.feature_indices['time_since_hand_raise']] += 1
			if new_state [self.feature_indices['time_since_hand_raise']] >= self.state_space[self.feature_indices['time_since_hand_raise']][1]:
				new_state [self.feature_indices['time_since_hand_raise']] = self.state_space[self.feature_indices['time_since_hand_raise']][1]


		time_index = self.feature_indices['time_since_hand_raise']
		start_time = start_state [time_index]
		new_time = new_state [time_index]
		start_sat = new_state [self.feature_indices['customer_satisfaction']]
		max_time = self.state_space[self.feature_indices['time_since_hand_raise']][1]
		max_sat = self.state_space[self.feature_indices['customer_satisfaction']][1]		

		if remaining_time_steps is None:
			time_steps = k_steps
		else:
			time_steps = remaining_time_steps
		
		if new_state [self.feature_indices['hand_raise']] == 0:
			terminal = True
			reward = 0.0
			if (action.id == 2 or action.id == 0) and distance_to_table != 0:
				reward = -np.Inf
			outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
		else:
			sat = self.compute_satisfaction(max_time, max_sat, start_time, min(start_time+1,max_time), start_sat)
			new_state [self.feature_indices['customer_satisfaction']] = sat
			new_state [time_index] = min(start_time+1,max_time)

			if action.type == 0:
				reward += (reward_nav/k_steps + self.get_reward(start_state,new_state,False))
				outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))			
			else:
				current_req = start_state [self.feature_indices['current_request']]
				new_req = 0

				if action.id == 2 and distance_to_table == 0 and current_req != 8:
					if current_req == 1 or current_req == 2 or (current_req == 3) or \
						(current_req == 4) or \
						current_req == 5 or current_req == 6 or current_req == 7: 

						new_state [self.feature_indices['time_since_hand_raise']] = 0

						new_req = current_req + 1
						new_state [self.feature_indices['current_request']] = new_req
						if sat == self.state_space[self.feature_indices['customer_satisfaction']][0]: 
							if not self.deterministic:
								new_state [self.feature_indices['customer_satisfaction']] = 0
								outcomes.append((0.7,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
								new_state [self.feature_indices['customer_satisfaction']] = sat + 1
								outcomes.append((0.3,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
							else:
								new_state [self.feature_indices['customer_satisfaction']] = sat + 1
								outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

						elif sat == self.state_space[self.feature_indices['customer_satisfaction']][1]:
							new_state [self.feature_indices['customer_satisfaction']] = sat
							outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

						else: 
							if not self.deterministic:
								new_state [self.feature_indices['customer_satisfaction']] = sat
								outcomes.append((0.4,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
								new_state [self.feature_indices['customer_satisfaction']] = sat + 1
								outcomes.append((0.6,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))
							else:
								new_state [self.feature_indices['customer_satisfaction']] = sat + 1
								outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,True), False, 1, False))

					else:
						outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))


				elif action.id == 0 and distance_to_table == 0 and start_state [self.feature_indices['hand_raise']] == 1 and \
					start_state [self.feature_indices['time_since_hand_raise']] > 4 and start_state [self.feature_indices['time_since_hand_raise']] < 9:
					new_state [self.feature_indices['time_since_hand_raise']] = 0
					# outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					# if sat == self.state_space[self.feature_indices['customer_satisfaction']][0]: 
					# 		if not self.deterministic:
					# 			new_state [self.feature_indices['customer_satisfaction']] = 0
					# 			outcomes.append((0.9,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					# 			new_state [self.feature_indices['customer_satisfaction']] = sat + 1
					# 			outcomes.append((0.1,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					# 		else:
					# 			new_state [self.feature_indices['customer_satisfaction']] = sat + 1
					# 			outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))

					# elif sat == self.state_space[self.feature_indices['customer_satisfaction']][1]:
					# 	new_state [self.feature_indices['customer_satisfaction']] = sat
					# 	outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))

					# else: 
					# 	if not self.deterministic:
					# 		new_state [self.feature_indices['customer_satisfaction']] = sat
					# 		outcomes.append((0.6,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					# 		new_state [self.feature_indices['customer_satisfaction']] = sat + 1
					# 		outcomes.append((0.4,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))
					# 	else:
					# 		new_state [self.feature_indices['customer_satisfaction']] = sat + 1
					# 		outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))

					outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), False, 1, False))	

					
				elif action.id == 2 and current_req == 8 and new_state [self.feature_indices['hand_raise']] == 1 and distance_to_table == 0:
					new_state [self.feature_indices['hand_raise']] = 0 
					terminal = True
					reward += 1.0
					outcomes.append((1.0,self.get_state_index(new_state), reward, terminal, 1, False))
				elif (action.id == 2 or action.id == 0) and distance_to_table != 0:
					reward += -np.Inf
					outcomes.append((1.0,self.get_state_index(new_state), reward, False, 1, False))
				else:
					terminal = False
					if new_state [self.feature_indices['hand_raise']] == 0:
						terminal = True
					outcomes.append((1.0,self.get_state_index(new_state), reward+self.get_reward(start_state,new_state,False), terminal, 1, False))
		


		new_outcomes = self.compute_outcomes(k_steps, time_steps, start_state, new_state, action, all_poss_actions, horizon, outcomes)
		# print (new_outcomes)
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

		if print_status:
			print ("POMDP: ", self.task.table.id, new_outcomes, start_state, action.id, self.get_state_tuple(new_outcomes[0][1]))
				# set_trace()
			# set_trace()

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

