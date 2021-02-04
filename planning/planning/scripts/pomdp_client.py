import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import pylab as plt
import math
from enum import IntEnum

from draw_env import *

import rospy
from std_msgs.msg import String
import json 
# from cobot_msgs.srv import *


GLOBAL_TIME = 0
print_status = False

class History():
	def __init__(self, pre, mismatch, post):
		self.pre = pre
		self.mismatch = mismatch
		self.post = post


class Action_Type(IntEnum):
	CLARIFICATION = 2
	NAVIGATION = 0
	INFORMATIVE = 1
	SERVE = 3

class Observation_Type(IntEnum):
	ORIGINAL = 0
	HUMAN_INPUT = 1

class Action():
	def __init__(self, id, name, pomdp, a_type, time_steps, kitchen=False, state=None):
		self.id = id
		self.name = name
		self.pomdp = pomdp
		self.time_steps = time_steps
		self.type = a_type
		self.kitchen = kitchen
		if a_type == Action_Type.CLARIFICATION:
			self.cost = -2
			self.state = state

	def set_id (self, id):
		self.id = id
	def print(self):
		print ("id: ", self.id, " name: ", self.name, " pomdp: ", self.pomdp, " time_steps: ", self.time_steps, " type: ", self.type)


class ClientPOMDP(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, task, robot, navigation_goals, gamma, random, reset_random, deterministic, no_op, run_on_robot):
		print ("deterministic: ", deterministic, " no op: ", no_op)
		self.deterministic = deterministic
		self.gamma = gamma
		self.random = random
		self.reset_random = reset_random
		self.task = task
		self.robot = robot
		self.no_op = no_op
		self.navigation_goals = navigation_goals
		self.run_on_robot= run_on_robot

		self.no_op_action_id = 1
		self.KITCHEN = True
		self.ACTIVE_PLANNING = True
		self.kitchen_pos = (3,4)
		self.epsilon = 0.1
		print ("Epsilon: ", self.epsilon)
		self.history = None

		self.transition_function = {}
		self.modified_transition_function = {}
		self.modified_observation_function = {}
		if run_on_robot:
			try:
				self.executeAction = rospy.Publisher('CobotExecuteAction4Table', String, queue_size=10)
				# rostopic pub /ActionStatus std_msgs/String "'0'"
				rospy.Subscriber("ActionStatus", String, self.action_status)
				# rostopic pub /StateDataTable0 std_msgs/String "'{ \"food\":1,\"water\":1,\"cooking_status\":1,\"current_request":1}'" -- does not work
				rospy.Subscriber("StateDataTable" + str(self.task.table.id), String, self.state_data)
				# self.executeAction = rospy.ServiceProxy('CobotExecuteAction4Table', ExecuteAction4Table)
				self.pub_cur_state = rospy.Publisher('CurrentStateTable' + str(self.task.table.id), String, queue_size=10)
			except:
				print("Failed to contact action executor service.")
				set_trace()

	def step(self, action, start_state=None, simulate=False, robot=False, selected_pomdp=None):
		global GLOBAL_TIME
		if start_state is None:
			start_state = self.state

		if print_status:
			print ("STEP: POMDP ", self.task.table.id ," step action: ", self.get_state_tuple(start_state), action)
		
		k = 1
		sum_prob = 0

		outcomes, steps, observation = self.execute_action(start_state,action,simulate,robot, selected_pomdp)

		if not robot:
			for outcome in outcomes:
				rand_num = self.random.choice(100)
				sum_prob += outcome[0]*100
				# print (rand_num, sum_prob, outcome)
				if  rand_num < sum_prob:				
					new_state_index = outcome[1]
					reward = outcome[2]
					terminal = outcome[3]
					break
		else:
			new_state_index = outcomes[0][1]
			reward = outcomes[0][2]
			terminal = outcomes[0][3]

		new_state = self.get_state_tuple(new_state_index)

		if "x" in self.feature_indices:
			position = (new_state[self.feature_indices['x']],new_state[self.feature_indices['y']])
		else:
			position = None

		obs = observation


		if not simulate:
			self.prev_state = self.state
			self.state = new_state_index

		debug_info = {}
		# debug_info['prob'] = prob
		# debug_info['steps'] = k
		# debug_info['action_highlevel'] = action_highlevel
		if print_status:
			print ("STEP: new state", new_state_index, reward, terminal)
			set_trace()

		return new_state_index, self.get_observation_index(obs), reward, terminal, debug_info, position

	def simulate_navigation_action (self,action,position):
		goal = self.navigation_goals[action.id-(self.non_navigation_actions_len)]

		# dist = self.distance(position,goal) + self.distance(goal,(self.task.table.goal_x,self.task.table.goal_y))
		dist = self.distance(position,goal)
		reward = -dist/3 ## this was -dist
		steps = math.ceil(max(math.ceil(dist)/4,1))

		return goal, reward, steps

	def simulate_go_to_kitchen_action (self,action,position):
		goal = self.kitchen_pos

		# dist = self.distance(position,goal) + self.distance(goal,(self.task.table.goal_x,self.task.table.goal_y))
		dist = self.distance(position,goal)
		reward = -dist/3 ## this was -dist
		steps = math.ceil(max(math.ceil(dist)/4,1))

		return goal, reward, steps

	def get_uniform_belief (self, state):
		new_belief_state = []
		feature = self.task.get_feature("customer_satisfaction")
		prob = 1.0/(feature.high - feature.low + 1)
		for b in range(feature.low,feature.high+1,feature.discretization):
			state[self.feature_indices[feature.name]] = b
			new_belief_state.append((prob,self.get_state_index(state)))

		return new_belief_state

	def reset(self, random):
		# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
		cooking_status = 0
		food_picked_up = 0
		time_since_hand_raise = 0
		hand_raise = 1
		random = False
		if random:
			# "cooking_status","time_since_food_ready","water","food","time_since_served","hand_raise","time_since_hand_raise","current_request","customer_satisfaction"
			if self.task.table.id != 2:
				current_req = self.reset_random.randint(1,self.state_space[self.feature_indices['current_request']][1])
				food = 0
				water = 0
				if current_req == 4:
					food = 3
				if current_req == 5:
					food = 3
					water = 3
				if current_req > 5:
					food = 3
					water = 3
				customer_satisfaction = self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 
			else:
				current_req = 5
				cooking_status = 0
				food = 3
				water = 3
				food_picked_up = 0
				time_since_hand_raise = 4
				customer_satisfaction = self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 
		else:
			# food = 0
			# water = 0
			# current_req = 1
			if self.task.table.id == 2:
				current_req = 5
				cooking_status = 0
				food = 0
				water = 3
				food_picked_up = 0
				time_since_hand_raise = 4

				# current_req = 4
				# cooking_status = 0
				# food = 2
				# water = 0
				# food_picked_up = 0
				# time_since_hand_raise = 4

				customer_satisfaction = self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 
			else:
				customer_satisfaction = 1
				current_req = 8
				cooking_status = 0
				food = 0
				water = 0
				food_picked_up = 0
				hand_raise = 0
				# customer_satisfaction = self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 

		if not self.KITCHEN:
			state = [cooking_status,0,water,food,0,hand_raise,time_since_hand_raise,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 
		else:
			state = [cooking_status,0,water,food,0,hand_raise,time_since_hand_raise,food_picked_up,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 

		# if not self.KITCHEN:
		# 	state = [0,cooking_status,0,water,food,0,hand_raise,time_since_hand_raise,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 
		# else:
		# 	state = [0,cooking_status,0,water,food,0,hand_raise,time_since_hand_raise,food_picked_up,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 


		return state

	def get_state_index(self,state):
		new_state = tuple(state)
		new_state_index = np.ravel_multi_index(new_state,self.state_space_dim)
		return int(new_state_index)

	def get_state_tuple(self,new_state_index):
		state = np.unravel_index(new_state_index,self.state_space_dim)
		new_state = list(state)
		return new_state

	def get_observation_index(self,observation):
		obs_type = observation[0]
		new_obs = tuple(observation[1])
		new_obs_index = np.ravel_multi_index(new_obs,self.observation_space_dim[obs_type])
		return (obs_type,int(new_obs_index))

	def get_observation_tuple(self, observation):
		obs_type = observation[0]
		observation_index = observation[1]
		obs = np.unravel_index(observation_index,self.observation_space_dim[obs_type])
		new_obs = list(obs)
		return (obs_type,new_obs)

	def render(self, start_state=None, mode='human'):
		pass

	def distance(self, a, b):
		(x1, y1) = a
		(x2, y2) = b
		return np.sqrt(np.power((x1 - x2),2) + np.power((y1 - y2),2))

	def is_part_of_action_space(self, action):
		return (action in self.valid_actions)

	def compute_satisfaction(self, max_time, max_sat, start_time, new_time, start_sat, threshold):
		pass

	def action_status (self, data):
		action = json.loads(data.data)
		if (action["status"] == 0):
			self.success = True
		elif (action["status"] == 1):
			self.success = True
			print ("ACTION EXECUTION INTERRUPTED")

	def state_data (self, data):
		self.state_data = json.loads(data.data)

	def get_action_msg (self, state_index, action):
		req_ack = False
		if self.KITCHEN:
			if action.id == 0:
				msg = "I'll  be back to service your table"
				req_ack = True
			elif action.id == 1:
				msg = "just waiting"
			elif action.id == 2:
				msg = "your food is not ready"
				req_ack = True
			elif action.id == 11:
				msg = "pick up food for table " + str(action.pomdp)
				req_ack = True
			elif action.type == Action_Type.CLARIFICATION:
				msg = action.name[:-10]
				req_ack = True
			elif action.type == Action_Type.SERVE:
				state = self.get_state_tuple(state_index)
				current_request = state [self.feature_indices['current_request']]
				food = state [self.feature_indices['food']]
				water = state [self.feature_indices['water']]
				msg = action.name[:-10]
			else:
				if action.kitchen:
					msg = "navigating to the kitchen"
				else:
					msg = "navigating to T" + str(action.pomdp)
		else:
			if action.id == 0:
				msg = "I'll  be back to service your table"
				req_ack = True
			elif action.id == 1:
				msg = "just waiting"
			elif action.id == 2:
				msg = "your food is not ready"
				req_ack = True
			elif action.type == Action_Type.CLARIFICATION:
				msg = action.name[:-10]
				req_ack = True
			elif action.type == Action_Type.SERVE:
				state = self.get_state_tuple(state_index)
				current_request = state [self.feature_indices['current_request']]
				food = state [self.feature_indices['food']]
				water = state [self.feature_indices['water']]
				msg = action.name
			else:
				msg = "going to T" + str(action.pomdp)

		return msg, req_ack

	def execute_action(self, start_state_index, action, simulate, robot, selected_pomdp=None):
		print("to be executed: ")
		action.print()
		unexpected_observation = False
		if simulate or not robot:
			observation = []
			obs_type = Observation_Type.ORIGINAL
			outcomes, steps = self.simulate_action(start_state_index,action,all_poss_actions=True,horizon=None)
			state = self.get_state_tuple(outcomes[0][1])
			observation = state ## be careful, we are not doing deep copy!!
			observation.pop(self.feature_indices['customer_satisfaction'])
		else:
			outcomes, steps = self.simulate_action(start_state_index,action,all_poss_actions=True,horizon=None)
			if self.is_part_of_action_space(action) and selected_pomdp: # and action.id != self.no_op_action_id:
				msg, req_ack = self.get_action_msg (start_state_index, action)
				action_command = json.dumps({"table": self.task.table.id, "type": int(action.type), 
					"name": msg, "id": int(action.id), "req_ack": req_ack, "kitchen":action.kitchen})
				self.success = False
				self.state_data = None
				self.executeAction.publish(action_command)
				# success = self.executeAction(self.id, action.type, action.name)
				try:
					while not self.success:
						print ("action execution...", msg)
						sleep(1)
				except KeyboardInterrupt:
					print ("keyboard interrupt")
				except:
					print ("action execution exception")
					set_trace()

				action.print()


			try:
				new_state = {}
				outcome = self.get_state_tuple(outcomes[0][1])
				for f in self.obs_feature_indices[Observation_Type.ORIGINAL].keys():
					if f in self.feature_indices:
						new_state[f] = int(outcome[self.feature_indices[f]])

				new_state['table'] = self.task.table.id
				self.state_data = None
				try:
					while self.state_data is None:
						print ("waiting for observation..." + " table " + str(self.task.table.id))
						self.pub_cur_state.publish(json.dumps(new_state))
						sleep(1)
				except KeyboardInterrupt:
					print ("keyboard interrupt")


				if (action.type == Action_Type.CLARIFICATION):
					# set_trace()
					observation_tpl = self.get_observation_tuple((Observation_Type.HUMAN_INPUT,0))
					obs_type = observation_tpl[0]
					observation = observation_tpl[1]
					indices = self.obs_feature_indices[obs_type]
				else:
					observation_tpl = self.get_observation_tuple((Observation_Type.ORIGINAL,0))
					obs_type = observation_tpl[0]
					observation = observation_tpl[1]
					indices = self.obs_feature_indices[obs_type]

				# print (self.state_data,self.feature_indices.keys())
				for s in self.state_data.keys():
					if s in indices.keys():
						observation[indices[s]] = self.state_data[s]

				# set_trace()
			except:
				print ("observation exception: ", "table: ", self.task.table.id)
				set_trace()


		return outcomes, steps, (obs_type,observation)

	def simulate_observation (self, next_state_index, action): ## here
		if action.type != Action_Type.CLARIFICATION:
			obs_type = Observation_Type.ORIGINAL
			# set_trace()
		else:
			# set_trace()
			obs_type = Observation_Type.HUMAN_INPUT

		if next_state_index in self.observation_function[obs_type].keys() and action in self.observation_function[obs_type][next_state_index].keys():
			return self.observation_function[obs_type][next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	

		if obs_type == Observation_Type.ORIGINAL:
			for index in reversed(self.unobs_feature_indices[obs_type]):
				next_state.pop(index)

			next_state.insert(self.obs_feature_indices[obs_type]["explicit"],0)

			obs = {(self.get_observation_index((obs_type,next_state)),1.0)}

		elif obs_type == Observation_Type.HUMAN_INPUT:
			answer = self.get_clarification(action, next_state)
			# set_trace()
			for index in reversed(self.unobs_feature_indices[obs_type]):
				next_state.pop(index)
			index = self.obs_feature_indices[obs_type]["answer"]

			next_state.insert(self.obs_feature_indices[obs_type]["explicit"],0)
			next_state.insert(index,answer)
			obs = {(self.get_observation_index((obs_type,next_state)),1.0)}

		if len(self.modified_observation_function) > 0:

			n_obs = []
			new_obs = set()
			for mo in self.modified_observation_function.keys():
				if action.id in self.modified_observation_function[mo]:
					for (n_state,prob) in self.modified_observation_function[mo][action.id]:
						if n_state == next_state_index:
							n_obs.append(mo)
			for no in n_obs:
				new_obs.add((no,self.epsilon))

			new_obs.add((list(obs)[0][0],1.0-len(n_obs)*self.epsilon))
			# set_trace()
		else:
			new_obs = obs


		if next_state_index not in self.observation_function[obs_type].keys():
			self.observation_function[obs_type][next_state_index] = {} 

		if action not in self.observation_function[obs_type][next_state_index].keys():
			self.observation_function[obs_type][next_state_index][action] = new_obs
		else:
			self.observation_function[obs_type][next_state_index][action].update(new_obs)

		return new_obs

	def get_clarification (self, action, state):
		if state[self.feature_indices["m" + str(action.state["model_num"])]] == 1:
			return 1
		else:
			return 0

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

	def get_possible_next_states (self, belief_prob, action, all_poss_actions, horizon):
		possible_states = set()
		
		for (prob,state) in belief_prob:
			outcomes, steps = self.simulate_action(state,action,all_poss_actions=True,horizon=horizon)
			for outcome in outcomes:
				possible_states.add(outcome[1])

		return possible_states

	def get_possible_next_states_by_obs (self, belief_prob, action, obs_index): # here
		possible_states = set()

		(obs_type, obs_tpl) = self.get_observation_tuple(obs_index)
		if obs_type == Observation_Type.HUMAN_INPUT: 
			index = self.obs_feature_indices[obs_type]["answer"]
			obs_tpl.pop(index)

		obs_tpl.pop(self.obs_feature_indices[obs_type]["explicit"])
		for index in (self.unobs_feature_indices[obs_type]):
			obs_tpl.insert(index,-1)

		for (prob,state) in belief_prob:
			outcomes, steps = self.simulate_action(state,action,all_poss_actions=True, horizon=None)
			for outcome in outcomes:
				state_tpl = self.get_state_tuple(outcome[1])
				for s_i in range(len(state_tpl)):
					if s_i not in self.unobs_feature_indices[obs_type]:
						state_tpl[s_i] = obs_tpl[s_i]
				# set_trace()
				possible_states.add(self.get_state_index(state_tpl))

		if len(self.modified_observation_function) > 0:
			set_trace()
			if obs_index in self.modified_observation_function.keys():
				if action in self.modified_observation_function[obs_index].keys():
					for s in self.modified_observation_function[obs_index][action]:
						possible_states.append(s)

		return possible_states

	def get_possible_obss (self, belief_prob,all_poss_actions,horizon): ## here
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
					possible_states.add((a,outcome[1]))

		# print (possible_states)

		for ps_tpl in possible_states:
			obs_type = Observation_Type.ORIGINAL
			action = ps_tpl[0]
			if action.type == Action_Type.CLARIFICATION and action in self.feasible_actions:
				obs_type = Observation_Type.HUMAN_INPUT

			ps = ps_tpl[1]
			st = self.get_state_tuple(ps)				

			for index in reversed(self.unobs_feature_indices[obs_type]):
				st.pop(index)

			st.insert(self.obs_feature_indices[obs_type]["explicit"],0)

			if obs_type == Observation_Type.ORIGINAL:
				possible_obss.add(self.get_observation_index((obs_type,st)))

			elif obs_type == Observation_Type.HUMAN_INPUT: 
				index = self.obs_feature_indices[obs_type]["answer"]
				st.insert(index,0)
				possible_obss.add(self.get_observation_index((obs_type,st)))
				st[index] = 1
				possible_obss.add(self.get_observation_index((obs_type,st)))

		if len(self.modified_observation_function) > 0:
			set_trace()
			for ps_tpl in possible_states: 
				for mo in self.modified_observation_function.keys():
					for action in self.modified_observation_function[mo]:
						for n_state in self.modified_observation_function[mo][action]:
							if n_state == ps_tpl[1]:
								possible_obss.add(mo)

		return possible_obss

	def add_transition(self, start_state_index, action, next_state_index, epsilon):
		if not start_state_index in self.modified_transition_function.keys():
			self.modified_transition_function[start_state_index] = {}
			self.modified_transition_function[start_state_index][action.id] = [(next_state_index,epsilon)]
		elif not action.id in self.modified_transition_function[start_state_index].keys():
				self.modified_transition_function[start_state_index][action.id] = [(next_state_index,epsilon)]
		else:
			self.modified_transition_function[start_state_index][action.id].append((next_state_index,epsilon))

	def add_observation(self, obs, action, next_state_index, epsilon):
		if not obs in self.modified_observation_function.keys():
			self.modified_observation_function[obs] = {}
			self.modified_observation_function[obs][action.id] = [(next_state_index,epsilon)]
		elif not action.id in self.modified_observation_function[obs].keys():
				self.modified_observation_function[obs][action.id] = [(next_state_index,epsilon)]
		else:
			self.modified_observation_function[obs][action.id].append((next_state_index,epsilon))

	# def add_hidden_variables(self, hidden_vars_names, obs_type):
	# 	self.observation_function[obs_type] = {}
	# 	self.nO[obs_type] = 1
	# 	obs = []
	# 	self.obs_feature_indices[obs_type] = {}
	# 	self.unobs_feature_indices[obs_type] = []
	# 	self.observation_space_dim[obs_type] = ()

	# 	feature_count = 0
	# 	obs_feature_count = 0

	# 	self.obs_feature_indices[obs_type]["explicit"] = 0
	# 	self.observation_space_dim[obs_type] += (2,)
	# 	self.nO[Observation_Type.ORIGINAL] *= 2
	# 	obs_feature_count += 1

	# 	for feature in self.task.get_features():
	# 		if feature.type == "discrete" and feature.name in self.non_robot_features:		
	# 			if feature.type == "discrete" and feature.name in self.non_robot_features:
	# 				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
	# 				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
	# 				obs.append((feature.low, feature.high, feature.discretization, feature.name))
	# 				self.feature_indices[feature.name] = feature_count

	# 			if feature.observable and not feature.dependent and feature.name not in hidden_vars_names:
	# 				self.observation_space_dim[obs_type] += (int((feature.high - feature.low) / feature.discretization) + 1,)
	# 				self.nO[obs_type] *= int((feature.high - feature.low) / feature.discretization) + 1
	# 				self.obs_feature_indices[obs_type][feature.name] = obs_feature_count
	# 				obs_feature_count += 1
	# 			else:
	# 				self.unobs_feature_indices[obs_type].append(feature_count)

	# 			feature_count += 1

	# 	## answer to the question
	# 	if obs_type == Observation_Type.HUMAN_INPUT:
	# 		self.observation_space_dim[obs_type] += (2,) ## yes or no
	# 		self.nO[obs_type] *= 2
	# 		self.obs_feature_indices[obs_type]["answer"] = obs_feature_count
	# 		obs_feature_count += 1
	# 		feature_count += 1

	# 	# robot's features
	# 	for feature in self.robot.get_features():
	# 		if feature.type == "discrete" and feature.name in ["x","y"]:			
	# 			if feature.observable and not feature.dependent:
	# 				self.observation_space_dim[obs_type] += (int((feature.high - feature.low) / feature.discretization) + 1,)
	# 				self.nO[obs_type] *= int((feature.high - feature.low) / feature.discretization) + 1
	# 				self.obs_feature_indices[obs_type][feature.name] = obs_feature_count
	# 				obs_feature_count += 1
	# 			else:
	# 				self.unobs_feature_indices[obs_type].append(feature_count)

	# 			feature_count += 1

	# 	self.unobs_feature_indices[obs_type].sort()

	# def add_hidden_model_variables(self, model_vars):
	# 	# feature_count = len(self.unobs_feature_indices) + len(self.obs_feature_indices[obs_type])
	# 	state_count = len(self.state_space)
	# 	for m_name in model_vars:
	# 		self.nS *= 2
	# 		self.state_space_dim += (2,)
	# 		self.state_space.append((0, 1, 1, m_name))
	# 		self.feature_indices[m_name] = state_count				

	# 		self.unobs_feature_indices[Observation_Type.ORIGINAL].append(state_count)
	# 		self.unobs_feature_indices[Observation_Type.HUMAN_INPUT].append(state_count)
	# 		state_count += 1

	# 	self.unobs_feature_indices[Observation_Type.ORIGINAL].sort()
	# 	self.unobs_feature_indices[Observation_Type.HUMAN_INPUT].sort()

	def remove_hidden_variables (self, vars_names, obs_type, model_vars):
		self.nS = 1
		obs = []
		self.feature_indices = {}
		self.state_space_dim = ()

		self.observation_function[obs_type] = {}
		self.nO[obs_type] = 1
		self.obs_feature_indices[obs_type] = {}
		self.unobs_feature_indices[obs_type] = []
		self.observation_space_dim[obs_type] = ()

		if obs_type == Observation_Type.ORIGINAL:
			feature_count = 0
			obs_feature_count = 0

			self.obs_feature_indices[obs_type]["explicit"] = 0
			self.observation_space_dim[obs_type] += (2,)
			self.nO[obs_type] *= 2
			obs_feature_count += 1

			for feature in self.task.get_features():
				if feature.type == "discrete" and feature.name in self.non_robot_features:		
					self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
					self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					obs.append((feature.low, feature.high, feature.discretization, feature.name))
					self.feature_indices[feature.name] = feature_count
					# set_trace()		
					if (feature.observable and not feature.dependent) or feature.name in vars_names:
						self.observation_space_dim[obs_type] += (int((feature.high - feature.low) / feature.discretization) + 1,)
						self.nO[obs_type] *= int((feature.high - feature.low) / feature.discretization) + 1
						self.obs_feature_indices[obs_type][feature.name] = obs_feature_count
						obs_feature_count += 1
					else:
						self.unobs_feature_indices[obs_type].append(feature_count)

					feature_count += 1

			# robot's features
			for feature in self.robot.get_features():
				if feature.type == "discrete" and feature.name in ["x","y"]:
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


			# print ("# states: ", self.nS)
			self.state_space = obs
			self.unobs_feature_indices[obs_type].sort()
		# set_trace()	

	def reset_belief(self, obs, action, belief):
		new_belief_state = []
		
		explicit = 0
		if obs[0] == Observation_Type.ORIGINAL:
			explicit = obs[1][self.obs_feature_indices[Observation_Type.ORIGINAL]["explicit"]]


		if self.history is None:
			self.create_update_history(obs, action, belief)

		prob_obs_vs_state = 1
		if not explicit:
			prob_obs_vs_state = 1.0/3

		for m in self.history.mismatch:
			new_belief_state.append((prob_obs_vs_state*m[0],m[1]))

		if not explicit:
			for m in self.history.pre:
				new_belief_state.append((prob_obs_vs_state*m[0],m[1]))

			for m in self.history.post:
				new_belief_state.append((prob_obs_vs_state*m[0],m[1]))

		return new_belief_state

	def create_update_history(self, obs_index, action, belief, update=False):
		pre = []
		mismatch = []
		post = []
		#########################################################################
		sum_pr_obs = 0
		possible_next_states_obs = []
		if update:
			belief_prob = belief.prob
		for s_p in self.get_possible_next_states_by_obs(belief.prob, action, obs_index): #############
			for o_p in self.simulate_observation(s_p,action):
				# if explicit:
				# 	st = self.get_state_tuple(s_p)
				# 	next_req_value = st[self.feature_indices["current_request"]]
				# 	st[self.feature_indices["next_request"]] = next_req_value
				# 	possible_next_states_obs.append((o_p[1],self.get_state_index(st)))
				# else:
				if (o_p[0] == obs_index):
					possible_next_states_obs.append((o_p[1],s_p))
					sum_pr_obs += o_p[1]

		for s in range(len(possible_next_states_obs)):
			pre.append(((possible_next_states_obs[s][0])/sum_pr_obs,possible_next_states_obs[s][1]))

		#########################################################################
		sum_pr_state = 0
		possible_next_states_state = []
		## previous belief state
		for (prob,state) in belief.prob:
			outcomes, steps = self.simulate_action(state,action,all_poss_actions=True,horizon=None)
			for outcome in outcomes:
				# if explicit:
				# 	st = self.get_state_tuple(outcome[1])
				# 	st[self.feature_indices["next_request"]] = next_req_value
				# 	possible_next_states_state.append((prob * outcome[0],self.get_state_index(st)))
				# else:
				possible_next_states_state.append((prob * outcome[0],outcome[1]))
				sum_pr_state += prob * outcome[0]

		for s in range(len(possible_next_states_state)):
			mismatch.append(((possible_next_states_state[s][0])/sum_pr_state,possible_next_states_state[s][1]))

		#########################################################################
		## uniform belief state
		sum_pr_uniform = 0
		possible_next_states_uniform = set()
		## previous belief state
		for s in range(len(possible_next_states_state)):
			state_index = possible_next_states_state[s][1]
			st = self.get_state_tuple(outcome[1])
			for cur_req in range(1,self.state_space[self.feature_indices['current_request']][1]+1):
				if cur_req == 4:
					for fo in range(self.state_space[self.feature_indices['food']][0],self.state_space[self.feature_indices['food']][1]+1):
						st[self.feature_indices["current_request"]] = cur_req
						st[self.feature_indices["food"]] = fo
						st[self.feature_indices["water"]] = 0
						possible_next_states_uniform.add((1,self.get_state_index(st)))
				elif cur_req == 5:
					for wa in range(self.state_space[self.feature_indices['water']][0],self.state_space[self.feature_indices['water']][1]+1):
						st[self.feature_indices["current_request"]] = cur_req
						st[self.feature_indices["water"]] = wa
						st[self.feature_indices["food"]] = 3
						possible_next_states_uniform.add((1,self.get_state_index(st)))
				elif cur_req == 3:
					for c_st in range(self.state_space[self.feature_indices['cooking_status']][0],self.state_space[self.feature_indices['cooking_status']][1]+1):
						st[self.feature_indices["current_request"]] = cur_req
						st[self.feature_indices["water"]] = 0
						st[self.feature_indices["food"]] = 0
						st[self.feature_indices["cooking_status"]] = c_st
						possible_next_states_uniform.add((1,self.get_state_index(st)))
				elif cur_req < 3:
					st[self.feature_indices["current_request"]] = cur_req
					st[self.feature_indices["water"]] = 0
					st[self.feature_indices["food"]] = 0
					st[self.feature_indices["cooking_status"]] = 0
					possible_next_states_uniform.add((1,self.get_state_index(st)))
				else:
					st[self.feature_indices["current_request"]] = cur_req
					possible_next_states_uniform.add((1,self.get_state_index(st)))

			sum_pr_uniform = len(possible_next_states_uniform)
			possible_next_states_uniform = list(possible_next_states_uniform)
			for s in range(len(possible_next_states_uniform)):
				post.append(((possible_next_states_uniform[s][0])/sum_pr_uniform,possible_next_states_uniform[s][1]))
		#########################################################################
		self.history = History(pre,mismatch,post)

	def get_current_request_text (self, task, st):
		req_text = ""
		req = st[task.feature_indices['current_request']]
		if req == 1:
			req_text += "want menu"
		elif req == 2:
			req_text += "ready to order"
		elif req == 3 and st[task.feature_indices['cooking_status']] != 2:
			req_text += "want food"
		elif req == 3 and st[task.feature_indices['cooking_status']] == 2:
			req_text += "food ready"
		elif req == 4 and st[task.feature_indices['food']] != 3:
			req_text += "eating"
		elif req == 4 and st[task.feature_indices['food']] == 3:
			req_text += "want dessert"
		elif req == 5 and st[task.feature_indices['water']] != 3:
			req_text += "eating dessert"
		elif req == 5 and st[task.feature_indices['water']] == 3:
			req_text += "want bill"
		elif req == 6:
			req_text += "cash ready" ##payment ready
		elif req == 7:
			req_text += "cash collected"
		elif req == 8:
			req_text += "clean table"
		return req_text

	def explain_it (self, task, x, y, o):
		if x is not None:
			state = self.get_state_tuple(x)
			next_state = self.get_state_tuple(y)
			x_exp = self.get_current_request_text(task, state)
			y_exp = self.get_current_request_text(task, next_state)
			return x_exp, y_exp
		elif o is not None:
			next_state = self.get_state_tuple(y)
			o_exp = "this"
			y_exp = self.get_current_request_text(task, next_state)
			return y_exp, o_exp

	def add_model_transitions (self, valid_pomdp_tasks, hidden_vars_names):
		# set_trace()
		#### ADD ACTIONS
		model_vars = []
		self.models = valid_pomdp_tasks
		len_actions = len(self.actions)
		clarification_actions = []
		action_index = 0
		for task in valid_pomdp_tasks:
			if len(task.modified_transition_function) > 0:
				x = list(task.modified_transition_function.keys())[0]
				a = list(task.modified_transition_function[x].keys())[0]
				y = task.modified_transition_function[x][a][0][0]
				x_exp, y_exp = self.explain_it(task, x, y, None)
				clarification_actions.append(Action(None, 'is it possible to go from *' + x_exp + '* to *' + y_exp + '* state with *' + self.actions[a].name + '* action - table '+str(self.task.table.id), self.task.table.id, Action_Type.CLARIFICATION, 1, \
					state={"start_state":x,"next_state":y,"action":a,"model_num":action_index, "model":task}))

			if len(task.modified_observation_function) > 0:
				o = list(task.modified_observation_function.keys())[0]
				a = list(task.modified_observation_function[o].keys())[0]
				y = task.modified_observation_function[o][a][0][0]
				y_exp, o_exp = self.explain_it(task, None, y, o[1])
				o_exp = "money on table"
				clarification_actions.append(Action(None, 'is it possible to observe *' + o_exp + '* in *' + y_exp + '* state with *' + self.actions[a].name + '*- table '+str(self.task.table.id), self.task.table.id, Action_Type.CLARIFICATION, 1, \
					state={"observation":o,"next_state":y,"action":a,"model_num":action_index, "model":task}))

			model_vars.append("m"+str(action_index))
			action_index += 1

		self.set_actions(clarification_actions)
		# for cl in clarification_actions:
		# 	self.actions.append(cl)
		# 	self.valid_actions.append(cl)

		# self.len_clarification_actions = len(clarification_actions)

		# self.non_navigation_actions_len = self.non_navigation_actions_len + self.len_clarification_actions
		# self.nA = len(self.actions)
		# self.feasible_actions = list(self.valid_actions)
		# self.pomdps_actions += self.actions[len_actions:len(self.actions)]
		# self.action_space = spaces.Discrete(self.nA)

		###### back to original agent POMDP
		# agent_pomdp.feasible_actions = []
		# agent_pomdp.feasible_actions_index = []
		# for i in range(agent_pomdp.actions.shape[0]):
		# 	if agent_pomdp.actions[i,agent_pomdp.pomdp_actions[i]] in agent_pomdp.pomdp_tasks[agent_pomdp.pomdp_actions[i]].feasible_actions:
		# 		agent_pomdp.feasible_actions.append(agent_pomdp.actions[i,:])
		# 		agent_pomdp.feasible_actions_index.append(i)
				
		# agent_pomdp.feasible_actions_index = np.array(agent_pomdp.feasible_actions_index)
		# agent_pomdp.valid_actions = list(self.feasible_actions)

		# ############################
		#### ADD STATES
		self.set_states(obs_type=Observation_Type.ORIGINAL, hidden_vars_names=hidden_vars_names, model_vars=model_vars)
		self.set_states(obs_type=Observation_Type.HUMAN_INPUT, hidden_vars_names=hidden_vars_names, model_vars=model_vars)

	def get_all_possible_models (self, belief, action, obs):
		from pomdp_client_complex import ClientPOMDPComplex
		from pomdp_solver import POMDPSolver
		explicit = 0
		obs_tuple  = self.get_observation_tuple(obs)
		if obs_tuple[0] == Observation_Type.ORIGINAL:
			explicit = obs_tuple[1][self.obs_feature_indices[Observation_Type.ORIGINAL]["explicit"]]

		if self.history is None:
			self.create_update_history(obs, action, belief)

		
		possible_pomdp_tasks = []
		for m in self.history.mismatch:
			next_state = m[1]
			pomdp_task = ClientPOMDPComplex(self.task, self.robot, self.navigation_goals, self.gamma, \
				self.random, self.reset_random, self.deterministic, self.no_op, self.run_on_robot)
			pomdp_task.add_observation(obs,action,next_state,self.epsilon)
			possible_pomdp_tasks.append(pomdp_task)

		if not explicit:
			for cur_state in belief.prob:
				for m in self.history.pre:
					next_state = m[1]
					pomdp_task = ClientPOMDPComplex(self.task, self.robot, self.navigation_goals, self.gamma, \
						self.random, self.reset_random, self.deterministic, self.no_op, self.run_on_robot)
					pomdp_task.add_transition(cur_state[1],action,next_state,self.epsilon)
					possible_pomdp_tasks.append(pomdp_task)

		valid_pomdp_tasks = []
		valid_new_beliefs = []
		for pomdp_task in possible_pomdp_tasks:
			pomdp_solver = POMDPSolver(pomdp_task,belief)
			# set_trace()
			prob = pomdp_solver.compute_1_over_eta (belief.prob, action, obs, all_poss_actions=True,horizon=None)
			if prob > 0:
				valid_pomdp_tasks.append(pomdp_task)
				new_belief = pomdp_solver.update_belief(belief.prob, action, obs, all_poss_actions=True, horizon=None)		
				valid_new_beliefs.append(new_belief)

		return reversed(valid_pomdp_tasks), reversed(valid_new_beliefs)

	def adapt_pomdp(self, pomdp_solver, agent_pomdp, initial_belief, action, obs):
		add_variables = ["current_request","food","water","cooking_status"] ## ,"time_since_served"
		# add_variables = []
		self.transition_function = {}
		possible_pomdp_tasks, valid_new_beliefs = self.get_all_possible_models (initial_belief, action, obs)
		belief_mappings = []
		model = 0
		for belief in valid_new_beliefs:
			state_mappings = []
			for b in belief:
				state_mapping = {}
				state = self.get_state_tuple(b[1])
				for s in self.feature_indices.keys():
					state_mapping[s] = state[self.feature_indices[s]]
				print ("state: ", (b[0],state_mapping), "model: ", model)
				state_mappings.append((b[0], state_mapping))
			model += 1

			belief_mappings.append(state_mappings)


		# self.add_hidden_variables(add_variables, Observation_Type.ORIGINAL)
		# self.add_hidden_variables(add_variables, Observation_Type.HUMAN_INPUT)
		self.add_model_transitions (possible_pomdp_tasks, add_variables)
		for a in self.actions:
			if a.name == action.name:
				new_action = a
				break
		agent_pomdp.set_actions()		
		agent_pomdp.set_states()
		new_belief = self.get_new_belief (belief_mappings)

		# pick a random state
		sum_prob = 0
		for (prob,outcome) in new_belief:
			rand_num = self.random.choice(100)
			sum_prob += prob*100
			# print (rand_num, sum_prob, outcome)
			if  rand_num < sum_prob:				
				self.state = outcome
				break

		return new_belief

	def get_new_belief (self, belief_mappings):
		new_belief = []
		num_models = len(belief_mappings)
		model = 0
		for state_mappings in belief_mappings:
			for s_m in state_mappings:
				prob = s_m[0]
				state_mapping = s_m[1]
				new_state_tuple  = self.get_state_tuple(0)
				for f in self.feature_indices.keys():
					if f in state_mapping:
						new_state_tuple[self.feature_indices[f]] = state_mapping[f]

				prob *= (1.0/num_models)
				new_state_tuple[self.feature_indices["m"+str(model)]] = 1
				new_belief.append((prob,self.get_state_index(new_state_tuple)))
				print ("state: ", (prob,new_state_tuple), "model: ", model)
				new_state_tuple[self.feature_indices["m"+str(model)]] = 0
			model += 1

		print ("new_belief", new_belief)
		# set_trace()

		return new_belief

	def back_to_original_model(self, agent_pomdp):
		set_trace()
		rmv_variables = ["current_request"] ## ,"time_since_served"
		self.remove_hidden_variables(rmv_variables, Observation_Type.ORIGINAL)
		self.remove_hidden_variables(rmv_variables, Observation_Type.HUMAN_INPUT)

		self.feasible_actions = list(self.valid_actions)
		for i in reversed(range(self.non_navigation_actions_len-self.len_clarification_actions,self.non_navigation_actions_len)):
			self.feasible_actions.pop(i)

		###### back to original agent POMDP
		agent_pomdp.feasible_actions = []
		agent_pomdp.feasible_actions_index = []
		for i in range(agent_pomdp.actions.shape[0]):
			if agent_pomdp.actions[i,agent_pomdp.pomdp_actions[i]] in agent_pomdp.pomdp_tasks[agent_pomdp.pomdp_actions[i]].feasible_actions:
				agent_pomdp.feasible_actions.append(agent_pomdp.actions[i,:])
				agent_pomdp.feasible_actions_index.append(i)
				
		agent_pomdp.feasible_actions_index = np.array(agent_pomdp.feasible_actions_index)
		agent_pomdp.valid_actions = list(self.feasible_actions)


	def get_tuple (self, index, dim):
		state = np.unravel_index(index,dim)
		new_state = list(state)
		return new_state
