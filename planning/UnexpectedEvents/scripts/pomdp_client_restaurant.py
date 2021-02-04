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

class ClientPOMDPRestaurant(ClientPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op):
		pass

	def step(self, action, start_state=None, simulate=False, belief=None, observation=None):
		global GLOBAL_TIME
		if start_state is None:
			start_state = self.state

		if print_status:
			print ("STEP: POMDP ", self.task.table.id ," step action: ", self.get_state_tuple(start_state), action)
		
		k = 1
		sum_prob = 0

		outcomes, steps = self.simulate_action(start_state,action,all_poss_actions=True,horizon=None)

		for outcome in outcomes:
			rand_num = self.random.choice(100)
			sum_prob += outcome[0]*100
			# print (rand_num, sum_prob, outcome)
			if  rand_num < sum_prob:				
				new_state_index = outcome[1]
				reward = outcome[2]
				terminal = outcome[3]
				break

		new_state = self.get_state_tuple(new_state_index)
		position = (new_state[self.feature_indices['x']],new_state[self.feature_indices['y']])
		# new_state_index = self.get_state_index(new_state)
		if not simulate:
			self.state = new_state_index
		obs = new_state ## be careful, we are not doing deep copy!!
		obs.pop(self.feature_indices['customer_satisfaction'])
		debug_info = {}
		# debug_info['prob'] = prob
		# debug_info['steps'] = k
		# debug_info['action_highlevel'] = action_highlevel
		if print_status:
			print ("STEP: new state", new_state, reward, terminal)
			set_trace()

		return new_state_index, self.get_observation_index(obs), reward, terminal, debug_info, position

	def simulate_navigation_action (self,action,position):
		goal = self.navigation_goals[action.id-(self.nA-len(self.navigation_goals))]

		# dist = self.distance(position,goal) + self.distance(goal,(self.task.table.goal_x,self.task.table.goal_y))
		dist = self.distance(position,goal)
		reward = -dist/3 ## this was -dist
		steps = math.ceil(max(math.ceil(dist)/4,1))

		return goal, reward, steps

	def reset(self, random):
		# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
		if random:
			# "cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise","time_since_hand_raise","current_request","customer_satisfaction"
			current_req = self.reset_random.randint(1,self.state_space[self.feature_indices['current_request']][1])
			food = 0
			water = 0
			if current_req == 4:
				food = 3
			if current_req == 5:
				food = 3
				water = 3
			customer_satisfaction = self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 
		else:
			food = 0
			water = 0
			current_req = 1
			customer_satisfaction = 1

		state = [0,0,water,food,0,1,0,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 
		return state

	def render(self, start_state=None, mode='human'):
		pass

	def compute_satisfaction(self, max_time, max_sat, start_time, new_time, start_sat, threshold):
		pass

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None, remaining_time_steps=None):
		pass

	def simulate_observation (self, next_state_index, action):
		if next_state_index in self.observation_function.keys() and action in self.observation_function[next_state_index].keys():
			return self.observation_function[next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	
		next_state.pop(self.feature_indices["customer_satisfaction"])

		obs = {(self.get_observation_index(next_state),1.0)}

		if next_state_index not in self.observation_function.keys():
			self.observation_function[next_state_index] = {} 

		if action not in self.observation_function[next_state_index].keys():
			self.observation_function[next_state_index][action] = obs
		else:
			self.observation_function[next_state_index][action].update(obs)

		return obs

	def get_reward(self,start_state,new_state,high):
		reward = 0
		if new_state [self.feature_indices['hand_raise']] == 1:
			start_sat = start_state [self.feature_indices['customer_satisfaction']]
			new_sat = new_state [self.feature_indices['customer_satisfaction']]
			if high:
				reward = 10.0 * (self.state_space[self.feature_indices['customer_satisfaction']][1]-new_sat) + 10.0
			else:
				# for i in range(start_state [self.feature_indices['time_since_hand_raise']]+1, \
				# 	new_state [self.feature_indices['time_since_hand_raise']]+1):
				i = min(new_state [self.feature_indices['time_since_hand_raise']],10)
				if new_sat == 0:
					reward = -1 * math.pow(2,i)
				elif new_sat == 1:
					reward = -1 * math.pow(1.7,i)
				elif new_sat == 2:
					reward = -1 * math.pow(1.4,i)
				elif new_sat > start_sat  and new_sat > 2:
					reward = 1.0
		return reward

	def get_possible_next_states (self,observation, belief_prob=None, action=None):
		possible_states = list()
		
		obs_tuple = self.get_observation_tuple(observation)
		sat_index = self.feature_indices["customer_satisfaction"]
		obs_tuple.insert(sat_index,-1)

		for i in range(self.state_space_dim[sat_index]):
			obs_tuple[sat_index] = i
			possible_states.append(self.get_state_index(obs_tuple))

		# set_trace()
		# return range(new_belief_prob.shape[0])
		return possible_states

	def get_possible_obss (self, belief_prob,all_poss_actions,horizon):
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

		# print (possible_states)

		for ps in possible_states:
			st = self.get_state_tuple(ps)
			st.pop(self.feature_indices["customer_satisfaction"])
			possible_obss.add(self.get_observation_index(st)) #self.env.get_observation_tuple(obs)

		return possible_obss