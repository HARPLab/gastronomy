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

import rospy
from std_msgs.msg import String
import json 
# from cobot_msgs.srv import *


GLOBAL_TIME = 0
print_status = False

class Action():
	def __init__(self, id, name, pomdp, a_type, time_steps, kitchen=False):
		self.id = id
		self.name = name
		self.pomdp = pomdp
		self.time_steps = time_steps
		self.type = a_type
		self.kitchen = kitchen

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

		self.no_op_action_id = 1
		self.KITCHEN = False
		self.kitchen_pos = (3,4)

		self.transition_function = {}
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

		outcomes, steps, unexpected_observation = self.execute_action(start_state,action,simulate,robot, selected_pomdp)

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
		position = (new_state[self.feature_indices['x']],new_state[self.feature_indices['y']])
		# new_state_index = self.get_state_index(new_state)
		if not simulate:
			self.prev_state = self.state
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

		return new_state_index, self.get_observation_index(obs), reward, terminal, debug_info, position, unexpected_observation

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

		if not self.KITCHEN:
			state = [0,0,water,food,0,1,0,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 
		else:
			state = [0,0,water,food,0,1,0,0,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 

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
		new_obs = tuple(observation)
		new_obs_index = np.ravel_multi_index(new_obs,self.observation_space_dim)
		return int(new_obs_index)

	def get_observation_tuple(self,observation_index):
		obs = np.unravel_index(observation_index,self.observation_space_dim)
		new_obs = list(obs)
		return new_obs

	def render(self, start_state=None, mode='human'):
		pass

	def compute_P(self):
		feature_array = [0] * len(self.state_space_dim)
		self.compute_P_elements(len(self.state_space_dim)-1,feature_array)

	def compute_P_elements(self,feature_index,feature_array):
		if feature_index < 0:
			self.transition_function[self.get_state_index(feature_array)] = {} 
			state_index = self.get_state_index(feature_array)
			for action in range(self.nA):
				self.transition_function[state_index][action], steps = self.simulate_action(state_index, action, all_poss_actions)
				### observation function
				for outcome in self.transition_function[state_index][action]:
					next_state_index = outcome[1]
					if next_state_index not in self.observation_function.keys():
						self.observation_function[next_state_index] = {} 

					obs = self.simulate_observation (next_state_index, action)
					if action not in self.observation_function[next_state_index].keys():
						self.observation_function[next_state_index][action] = obs
					else:
						self.observation_function[next_state_index][action].update(obs)

			return

		for i in range(self.state_space_dim[feature_index]):
			new_feature_array = feature_array.copy()
			new_feature_array[feature_index] = i
			self.compute_P_elements(feature_index-1,new_feature_array)

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
			elif action.id == 3:
				msg = "your food is not ready"
				req_ack = True
			elif action.id == 4:
				msg = "pick up food for table " + str(action.pomdp)
				req_ack = True
			elif action.id == 2:
				state = self.get_state_tuple(state_index)
				current_request = state [self.feature_indices['current_request']]
				food = state [self.feature_indices['food']]
				water = state [self.feature_indices['water']]
				if current_request == 1:
					msg = "please take the menu"
					req_ack = True
				if current_request == 2:
					msg = "what is your order?"
					req_ack = True
				if current_request == 3 and food == 0:
					msg = "please take your food"
					req_ack = True
				if current_request == 4 and water == 0:
					msg = "please take your drink"
					req_ack = True
				if current_request == 5 and water == 3:
					msg = "here is your bill"
					req_ack = True
				if current_request == 6:
					msg = "please place cash in my basket"
					req_ack = True
				if current_request == 7:
					msg = "here is your reciept, good bye!"
				if current_request == 8:
					msg = "table is clean"
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
			elif action.id == 3:
				msg = "your food is not ready"
				req_ack = True
			elif action.id == 2:
				state = self.get_state_tuple(state_index)
				current_request = state [self.feature_indices['current_request']]
				food = state [self.feature_indices['food']]
				water = state [self.feature_indices['water']]
				if current_request == 1:
					msg = "please take the menu"
					req_ack = True
				if current_request == 2:
					msg = "what is your order?"
					req_ack = True
				if current_request == 3 and food == 0:
					msg = "please take your food"
					req_ack = True
				if current_request == 4 and water == 0:
					msg = "please take your drink"
					req_ack = True
				if current_request == 5 and water == 3:
					msg = "here is your bill"
					req_ack = True
				if current_request == 6:
					msg = "can I have the cash?"
					req_ack = True
				if current_request == 7:
					msg = "here is your receipt"
				if current_request == 8:
					msg = "table is clean"
			else:
				msg = "going to T" + str(action.pomdp)

		return msg, req_ack

	def execute_action(self, start_state_index, action, simulate, robot, selected_pomdp=None):
		print("to be executed: ")
		action.print()
		unexpected_observation = False
		if simulate or not robot:
			outcomes, steps = self.simulate_action(start_state_index,action,all_poss_actions=True,horizon=None)
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
				for f in self.obs_feature_indices.keys():
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

				new_outcomes = []

				# obs = new_state ## be careful, we are not doing deep copy!!
				# obs.pop(self.feature_indices['customer_satisfaction'])

				# set_trace()
				print (outcomes)
				for outcome in outcomes:
					new_state = self.get_state_tuple(outcome[1])
					for s in self.state_data.keys():
						if s in self.feature_indices.keys():
							new_state[self.feature_indices[s]] = self.state_data[s]
					new_outcome = (outcome[0],self.get_state_index(new_state), outcome[2], outcome[3], outcome[4], outcome[5])
					new_outcomes.append(new_outcome)

				if outcomes != new_outcomes:
					unexpected_observation = True
					print ("observation do not match the model")
					# set_trace() 
				outcomes = new_outcomes
				print (outcomes)
				# set_trace()
			except:
				print ("observation exception: ", "table: ", self.task.table.id)
				set_trace()


		return outcomes, steps, unexpected_observation

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

	def get_possible_next_states (self,observation):
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