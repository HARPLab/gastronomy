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
ACTION_COST = 2
UNRELIABLE_PARAMETER_COST = 100

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
		self.cost = ACTION_COST + 2
		self.name = name
		self.pomdp = pomdp
		self.time_steps = time_steps
		self.type = a_type
		self.kitchen = kitchen
		self.state = {}
		# if a_type != Action_Type.NAVIGATION:
		# 	self.cost = ACTION_COST
		if self.id == 1:
			self.cost = ACTION_COST
		elif a_type == Action_Type.CLARIFICATION:
			# self.cost = -2
			self.cost = ACTION_COST + 5
			self.state = state

	def set_id (self, id):
		self.id = id
	def print(self):
		print ("*****")
		print ("id: ", self.id, " name: ", self.name, " pomdp: ", self.pomdp, " time_steps: ", self.time_steps, " type: ", self.type)
		print (self.state)


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
		self.models = None
		self.model_pars = None
		self.goal_pomdp = True

		self.num_step = 0

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

		if self.goal_pomdp:
			reward = -reward
			if dist == 0.0:
				reward = np.Inf
		return goal, reward, steps

	def simulate_go_to_kitchen_action (self,action,position):
		goal = self.kitchen_pos

		# dist = self.distance(position,goal) + self.distance(goal,(self.task.table.goal_x,self.task.table.goal_y))
		dist = self.distance(position,goal)
		reward = -dist/3 ## this was -dist
		steps = math.ceil(max(math.ceil(dist)/4,1))
		if self.goal_pomdp:
			reward = -reward
			if dist == 0.0:
				reward = np.Inf
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
		have_bread = 0
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
			if self.task.table.id == 0:
				current_req = 3
				cooking_status = 0
				food = 0
				water = 0
				food_picked_up = 0
				time_since_hand_raise = 0

				# current_req = 5
				# cooking_status = 0
				# food = 0
				# water = 3
				# food_picked_up = 0
				# time_since_hand_raise = 4

				customer_satisfaction = 4#self.reset_random.randint(0,self.state_space[self.feature_indices['customer_satisfaction']][1]+1) 
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
			state = [have_bread, cooking_status,0,water,food,0,hand_raise,time_since_hand_raise,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 
		else:
			state = [have_bread, cooking_status,0,water,food,0,hand_raise,time_since_hand_raise,food_picked_up,current_req,customer_satisfaction,self.robot.get_feature('x').value,self.robot.get_feature('y').value] # 

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
			print ("does not work")
			set_trace()
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

				emotion = self.get_dominant_emotion(int(outcome[self.feature_indices['customer_satisfaction']]), action)
				new_state['emotion'] = emotion
				new_state['table'] = self.task.table.id
				self.state_data = None
				try:
					while self.state_data is None:
						print ("waiting for observation..." + " table " + str(self.task.table.id))
						self.pub_cur_state.publish(json.dumps(new_state))
						sleep(1)
				except KeyboardInterrupt:
					print ("keyboard interrupt")

				obs_type = Observation_Type.ORIGINAL
				if (action.type == Action_Type.CLARIFICATION):
					# set_trace()
					observation_tpl = self.get_observation_tuple((Observation_Type.HUMAN_INPUT,0))
					obs_type = Observation_Type.HUMAN_INPUT
					observation = observation_tpl[1]
					indices = self.obs_feature_indices[obs_type]
				else:
					observation_tpl = self.get_observation_tuple((Observation_Type.ORIGINAL,0))
					obs_type = Observation_Type.ORIGINAL
					observation = observation_tpl[1]
					indices = self.obs_feature_indices[obs_type]

				for s in self.state_data.keys():
					if s in indices.keys():
						observation[indices[s]] = self.state_data[s]

				# set_trace()
				print (self.state_data,self.feature_indices.keys())
			except:
				print ("observation exception: ", "table: ", self.task.table.id)
				set_trace()


		return outcomes, steps, (obs_type,observation)

	
	def simulate_observation_cost (self, next_state_index, action, modified=True): ## here
		if not modified:
			return 0

		if action.type != Action_Type.CLARIFICATION:
			obs_type = Observation_Type.ORIGINAL
		else:
			obs_type = Observation_Type.HUMAN_INPUT
		
		if next_state_index in self.observation_function_costs[obs_type].keys() and action in self.observation_function_costs[obs_type][next_state_index].keys():
			return self.observation_function_costs[obs_type][next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	
		sat = next_state[self.feature_indices['customer_satisfaction']]
		########################
		modified_observation_function = self.modified_observation_function
		old_state_index = next_state_index
		selected_model = -1
		if self.models is not None:
			for model in range(0,len(self.models)):
				if next_state[self.feature_indices["m"+str(model)]] == 1:
					modified_observation_function = self.models[model].modified_observation_function
					for i in range(len(next_state)-3,self.feature_indices["m"+str(0)]-1,-1):
						next_state.pop(i)
					# 	print (next_state)
					# set_trace()
					old_state = deepcopy(next_state)
					old_state_index = self.models[model].get_state_index(next_state)
					selected_model = model
					# print ("next_state: ", next_state)
					break
		###########################
		if selected_model == -1:
			return 0

		next_state = self.get_state_tuple(next_state_index)	
		if obs_type == Observation_Type.ORIGINAL:
			for index in reversed(self.unobs_feature_indices[obs_type]):
				next_state.pop(index)

			next_state.insert(self.obs_feature_indices[obs_type]["explicit"],0)
			next_state.insert(self.obs_feature_indices[obs_type]["emotion"],0)

			obs = self.get_emotion(obs_type, next_state, sat, action)
			

		elif obs_type == Observation_Type.HUMAN_INPUT:
			answer = self.get_clarification(action, next_state)
			# set_trace()
			for index in reversed(self.unobs_feature_indices[obs_type]):
				next_state.pop(index)
			index = self.obs_feature_indices[obs_type]["answer"]

			next_state.insert(self.obs_feature_indices[obs_type]["explicit"],0)
			next_state.insert(self.obs_feature_indices[obs_type]["emotion"],0)
			next_state.insert(index,answer)
			
			obs = self.get_emotion(obs_type, next_state, sat, action)

		obs_cost = 0
		if selected_model != -1:
			temp_new_obs = set()
			new_outcomes = []
			# print (modified_observation_function)
			for om in modified_observation_function.keys():
				if action.id in modified_observation_function[om].keys():
					for (ot,pr) in obs:
						# print (om, action.id, ot, pr)
						modified_observation = self.models[selected_model].get_observation_tuple(om)
						observation = self.get_observation_tuple(ot)[1]						
						next_state = self.get_state_tuple(next_state_index)
						next_modified_state = old_state
						# print (next_state, next_modified_state)
						# set_trace()
						if next_state[self.feature_indices["customer_satisfaction"]] <= next_modified_state[self.models[selected_model].feature_indices["customer_satisfaction"]]:							
							##################
							selected_par = None
							# set_trace()
							for var in self.model_pars["m"+str(selected_model)]:
								if var[0] == "observation":
									if var[2] == action.id:
										obs_tpl = self.models[selected_model].get_observation_tuple(var[1])
										n_m = self.models[selected_model].get_state_tuple(var[3])
										# set_trace()
										if observation[self.obs_feature_indices[obs_type]["emotion"]] == obs_tpl[1][self.models[selected_model].obs_feature_indices[obs_type]["emotion"]] and \
										next_state[self.feature_indices["customer_satisfaction"]] <= n_m[self.models[selected_model].feature_indices["customer_satisfaction"]]:
											selected_par = var[1][1]
											break
										# else: # not sure about this part
										# 	set_trace()

							if selected_par is not None and next_state[self.feature_indices["m"+str(selected_model)+"_o"+str(selected_par)]] != 0:
								temp_new_obs.add(ot)
							if selected_par is None:
								temp_new_obs.add(ot)	
							#################
							observation[self.obs_feature_indices[obs_type]["emotion"]] = modified_observation[1][self.models[selected_model].obs_feature_indices[obs_type]["emotion"]]
							temp_new_obs.add(self.get_observation_index((obs_type,observation)))
			# set_trace()
			if len(temp_new_obs) == 0 or len(temp_new_obs) == 1:
				obs_cost = 0
			else:
				obs_cost = UNRELIABLE_PARAMETER_COST * len(temp_new_obs)

			# if (self.num_step is not None and self.num_step >= 4):
			# 	if obs_cost != 0:
			# 		set_trace()

			if next_state_index not in self.observation_function_costs[obs_type].keys():
				self.observation_function_costs[obs_type][next_state_index] = {} 

			if action not in self.observation_function_costs[obs_type][next_state_index].keys():
				self.observation_function_costs[obs_type][next_state_index][action] = obs_cost
			else:
				self.observation_function_costs[obs_type][next_state_index][action].update(obs_cost)

		return obs_cost

	def simulate_observation (self, next_state_index, action, modified=True): ## here
		if action.type != Action_Type.CLARIFICATION:
			obs_type = Observation_Type.ORIGINAL
			# set_trace()
		else:
			# set_trace()
			obs_type = Observation_Type.HUMAN_INPUT

		if next_state_index in self.observation_function[obs_type].keys() and action in self.observation_function[obs_type][next_state_index].keys():
			return self.observation_function[obs_type][next_state_index][action]

		next_state = self.get_state_tuple(next_state_index)	
		sat = next_state[self.feature_indices['customer_satisfaction']]

		########################
		modified_observation_function = self.modified_observation_function
		old_state_index = next_state_index
		selected_model = -1
		if self.models is not None:
			for model in range(0,len(self.models)):
				if next_state[self.feature_indices["m"+str(model)]] == 1:
					modified_observation_function = self.models[model].modified_observation_function
					for i in range(len(next_state)-3,self.feature_indices["m"+str(0)]-1,-1):
						next_state.pop(i)
					# 	print (next_state)
					# set_trace()
					old_state = deepcopy(next_state)
					old_state_index = self.models[model].get_state_index(next_state)
					selected_model = model
					# print ("next_state: ", next_state)
					break
		###########################

		next_state = self.get_state_tuple(next_state_index)	
		if obs_type == Observation_Type.ORIGINAL:
			for index in reversed(self.unobs_feature_indices[obs_type]):
				next_state.pop(index)

			next_state.insert(self.obs_feature_indices[obs_type]["explicit"],0)
			next_state.insert(self.obs_feature_indices[obs_type]["emotion"],0)

			obs = self.get_emotion(obs_type, next_state, sat, action)
			

		elif obs_type == Observation_Type.HUMAN_INPUT:
			answer = self.get_clarification(action, next_state)
			# set_trace()
			for index in reversed(self.unobs_feature_indices[obs_type]):
				next_state.pop(index)
			index = self.obs_feature_indices[obs_type]["answer"]

			next_state.insert(self.obs_feature_indices[obs_type]["explicit"],0)
			next_state.insert(self.obs_feature_indices[obs_type]["emotion"],0)
			next_state.insert(index,answer)
			
			obs = self.get_emotion(obs_type, next_state, sat, action)


		new_obs = obs
		if not modified:
			new_obs = obs
		else:
			if selected_model == -1 and len(modified_observation_function) > 0: #############?????????????? fix later
				n_obs = []
				new_obs = set()
				for mo in modified_observation_function.keys():
					if action.id in modified_observation_function[mo]:
						for (n_state,prob) in modified_observation_function[mo][action.id]:
							if n_state == old_state_index:
								n_obs.append(mo)

				non_zero_obs = len(n_obs)+len(obs)
				for new_o in n_obs:
					new_obs.add((new_o,1.0/non_zero_obs))

				for prev_o in obs:
					new_obs.add((prev_o,1.0/non_zero_obs))
			elif selected_model != -1:
				temp_new_obs = set()
				new_outcomes = []
				# print (modified_observation_function)
				for om in modified_observation_function.keys():
					if action.id in modified_observation_function[om].keys():
						for (ot,pr) in obs:
							# print (om, action.id, ot, pr)
							modified_observation = self.models[selected_model].get_observation_tuple(om)
							observation = self.get_observation_tuple(ot)[1]						
							next_state = self.get_state_tuple(next_state_index)
							next_modified_state = old_state
							# print (next_state, next_modified_state)
							# set_trace()
							if next_state[self.feature_indices["customer_satisfaction"]] <= next_modified_state[self.models[selected_model].feature_indices["customer_satisfaction"]]:							
								##################
								selected_par = None
								# set_trace()
								for var in self.model_pars["m"+str(selected_model)]:
									if var[0] == "observation":
										if var[2] == action.id:
											obs_tpl = self.models[selected_model].get_observation_tuple(var[1])
											n_m = self.models[selected_model].get_state_tuple(var[3])
											# set_trace()
											if observation[self.obs_feature_indices[obs_type]["emotion"]] == obs_tpl[1][self.models[selected_model].obs_feature_indices[obs_type]["emotion"]] and \
											next_state[self.feature_indices["customer_satisfaction"]] <= n_m[self.models[selected_model].feature_indices["customer_satisfaction"]]:
												selected_par = var[1][1]
												break
											# else: # not sure about this part
											# 	set_trace()

								if selected_par is not None and next_state[self.feature_indices["m"+str(selected_model)+"_o"+str(selected_par)]] != 0:
									temp_new_obs.add(ot)
								if selected_par is None:
									temp_new_obs.add(ot)	
								#################
								observation[self.obs_feature_indices[obs_type]["emotion"]] = modified_observation[1][self.models[selected_model].obs_feature_indices[obs_type]["emotion"]]
								temp_new_obs.add(self.get_observation_index((obs_type,observation)))

				# set_trace()
				# if (self.num_step is not None and self.num_step >= 4):
				# 	if len(temp_new_obs) != 0:
				# 		set_trace()

				if len(temp_new_obs) != 0:
					obs_tpl = set()
					for out in temp_new_obs:
						obs_tpl.add((out,1.0/len(temp_new_obs)))
					new_obs = obs_tpl	
				else:
					new_obs = obs
				# if len(new_obs) != 0:
				# 	set_trace()			


		if next_state_index not in self.observation_function[obs_type].keys():
			self.observation_function[obs_type][next_state_index] = {} 

		if action not in self.observation_function[obs_type][next_state_index].keys():
			self.observation_function[obs_type][next_state_index][action] = new_obs
		else:
			self.observation_function[obs_type][next_state_index][action].update(new_obs)

		return new_obs

	def get_clarification (self, action, state):
		if action.state["cat"] == 1:
			if state[self.feature_indices["m" + str(action.state["model_num"])]] == 1:
				return 1
			else:
				return 0
		elif action.state["cat"] == 2:
			if "observation" in action.state.keys():
				name = "_o" + str(action.state["observation"][1])
			elif "start_state" in action.state.keys():
				name = "_s" + str(action.state["next_state"])

			if state[self.feature_indices["m" + str(action.state["model_num"])]] == 1 and \
				state[self.feature_indices["m" + str(action.state["model_num"]) + name]] == 1:
				return 1
			elif state[self.feature_indices["m" + str(action.state["model_num"])]] == 1 and \
				state[self.feature_indices["m" + str(action.state["model_num"]) + name]] == 0:
				return 0
			elif state[self.feature_indices["m" + str(action.state["model_num"])]] == 0 and \
				state[self.feature_indices["m" + str(action.state["model_num"]) + name]] == 1:
				return 1
			elif state[self.feature_indices["m" + str(action.state["model_num"])]] == 0 and \
				state[self.feature_indices["m" + str(action.state["model_num"]) + name]] == 0:
				return 0


	def get_emotion (self, obs_type, next_state, sat, action):
		obs = set()
		emotion_index = self.obs_feature_indices[obs_type]["emotion"]
		if (action.id == 2 or action.type == Action_Type.SERVE):
			if sat == 0:
				next_state[emotion_index] = 0 # unhappy
				obs = {(self.get_observation_index((obs_type,next_state)),1.0)}
			elif sat == 1:
				# next_state[emotion_index] = 0
				# obs.add((self.get_observation_index((obs_type,next_state)),0.9))
				next_state[emotion_index] = 1
				obs.add((self.get_observation_index((obs_type,next_state)),1.0))
			elif sat == 2:
				# next_state[emotion_index] = 0
				# obs.add((self.get_observation_index((obs_type,next_state)),0.7))
				next_state[emotion_index] = 1
				obs.add((self.get_observation_index((obs_type,next_state)),1.0))
			elif sat == 3:
				# next_state[emotion_index] = 0
				# obs.add((self.get_observation_index((obs_type,next_state)),0.3))
				next_state[emotion_index] = 1
				obs.add((self.get_observation_index((obs_type,next_state)),0.7))
				next_state[emotion_index] = 2
				obs.add((self.get_observation_index((obs_type,next_state)),0.3))
			elif sat == 4:
				next_state[emotion_index] = 1
				obs.add((self.get_observation_index((obs_type,next_state)),0.3))
				next_state[emotion_index] = 2
				obs.add((self.get_observation_index((obs_type,next_state)),0.7))
			elif sat == 5:
				next_state[emotion_index] = 2 ## happy
				obs.add((self.get_observation_index((obs_type,next_state)),1.0))
		else:
			next_state[emotion_index] = 3 # unknown
			obs.add((self.get_observation_index((obs_type,next_state)),1.0))

		return obs

	def get_dominant_emotion (self, sat, action):
		if (action.id == 2 or action.type == Action_Type.SERVE):
			if sat == 0:
				return 0
			elif sat == 1:
				return 1
			elif sat == 2:
				return 1
			elif sat == 3:
				return 1
			elif sat == 4:
				return 2
			elif sat == 5:
				return 2
		else:
			return 3

	def get_reward(self,start_state,new_state,high):
		if not self.goal_pomdp:
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
						reward = 1.0 * new_sat ## was 1.0
			return reward
		else:
			cost = 0
			if new_state [self.feature_indices['hand_raise']] == 1:
				start_sat = start_state [self.feature_indices['customer_satisfaction']]
				new_sat = new_state [self.feature_indices['customer_satisfaction']]
				cost = 5.0 * (self.state_space[self.feature_indices['customer_satisfaction']][1]-new_sat)
				if high:
					cost = 2.0 * (self.state_space[self.feature_indices['customer_satisfaction']][1]-new_sat)
				else:
					i = min(new_state [self.feature_indices['time_since_hand_raise']],10)
					if new_sat < start_sat  and new_sat <= 2:
						if new_sat == 0:
							cost += math.pow(2,i)
						elif new_sat == 1:
							cost += math.pow(1.7,i)
						elif new_sat == 2:
							cost += math.pow(1.4,i)	
					elif new_sat < start_sat and new_sat > 2:
						cost += math.pow(1.3,i)	
					elif new_sat == start_sat and new_sat <= 2:
						cost += math.pow(1.1,i)	
					elif new_sat == start_sat and new_sat > 2:
						cost += math.pow(1.05,i)	
			return cost

	def get_possible_next_states (self, belief_prob, action, all_poss_actions, horizon):
		possible_states = set()
		# if horizon is None:
		# 	print ("get possible next states: ")
		for (prob,state) in belief_prob:
			outcomes, steps = self.simulate_action(state,action,all_poss_actions=True,horizon=horizon)
			for outcome in outcomes:
				possible_states.add(outcome[1])
				# if horizon is None:
				# 	print (self.get_state_tuple(outcome[1]))

		return possible_states

	def get_possible_next_states_by_obs (self, belief_prob, action, obs_index): # here
		possible_states = set()

		(obs_type, obs_tpl) = self.get_observation_tuple(obs_index)
		if obs_type == Observation_Type.HUMAN_INPUT: 
			index = self.obs_feature_indices[obs_type]["answer"]
			obs_tpl.pop(index)

		obs_tpl.pop(self.obs_feature_indices[obs_type]["emotion"])
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
				for sat in range(self.state_space[self.feature_indices['customer_satisfaction']][0],self.state_space[self.feature_indices['customer_satisfaction']][1]+1):
					state_tpl[self.feature_indices["customer_satisfaction"]] = sat
					possible_states.add(self.get_state_index(state_tpl))

		### what about the observation function
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

			emotion_index = self.obs_feature_indices[obs_type]["emotion"]
			st.insert(emotion_index,0)

			if obs_type == Observation_Type.HUMAN_INPUT: 
				answer_index = self.obs_feature_indices[obs_type]["answer"]
				st.insert(answer_index,0)

			for o in range(0,self.observation_space_dim[obs_type][emotion_index]):
				st[emotion_index] = o
				if obs_type == Observation_Type.ORIGINAL:
					possible_obss.add(self.get_observation_index((obs_type,st)))

				elif obs_type == Observation_Type.HUMAN_INPUT: 
					st[answer_index] = 0
					possible_obss.add(self.get_observation_index((obs_type,st)))
					st[answer_index] = 1
					possible_obss.add(self.get_observation_index((obs_type,st)))

		### what about the observation function
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
		# set_trace()
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
		'''for s in range(len(possible_next_states_state)):
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
				post.append(((possible_next_states_uniform[s][0])/sum_pr_uniform,possible_next_states_uniform[s][1]))'''
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

	def get_satisfaction_text (self, task, st):
		sat = st[task.feature_indices["customer_satisfaction"]]
		if sat == 0:
			sat_text = "very unsatisfied"
		if sat == 1:
			sat_text = "unsatisfied"
		if sat == 2:
			sat_text = "a bit unsatisfied"
		if sat == 3:
			sat_text = "neutral"
		if sat == 4:
			sat_text = "satisfied"
		if sat == 5:
			sat_text = "very satisfied"
		return sat_text

	def explain_it (self, task, x, y, o):
		if x is not None:
			state = self.get_state_tuple(x)
			next_state = self.get_state_tuple(y)
			x_exp = self.get_satisfaction_text(task, state)
			y_exp = self.get_satisfaction_text(task, next_state)
			return x_exp, y_exp
		elif o is not None:
			next_state = self.get_state_tuple(y)
			o_sat = self.get_observation_tuple(o)[1][self.obs_feature_indices[Observation_Type.ORIGINAL]["emotion"]]
			if o_sat == 0:
				o_exp = "unhappy"
			if o_sat == 1:
				o_exp = "neutral"
			if o_sat == 2:
				o_exp = "happy"
			if o_sat == 3:
				o_exp = "unknown"
			y_exp = self.get_satisfaction_text(task, next_state)
			return y_exp, o_exp

	def add_model_transitions (self, valid_pomdp_tasks, hidden_vars_names):
		# set_trace()
		#### ADD ACTIONS
		model_vars = []
		self.models = valid_pomdp_tasks
		self.model_pars = {}
		len_actions = len(self.actions)
		clarification_actions = []
		action_index = 0
		parameter_vars = {}
		m_name = ""
		for task in valid_pomdp_tasks:
			m_name = "m"+str(action_index)
			vars_tuples = []
			if len(task.modified_transition_function) > 0:
				if m_name not in parameter_vars.keys():
					parameter_vars[m_name] = []

				x = list(task.modified_transition_function.keys())[0]
				a = list(task.modified_transition_function[x].keys())[0]
				y = task.modified_transition_function[x][a][0][0]
				x_exp, y_exp = self.explain_it(task, x, y, None)
				clarification_actions.append(Action(None, 'is it possible to go from *' + x_exp + '* to *' + y_exp + '* state with *' + self.actions[a].name + '* action - table '+str(self.task.table.id), self.task.table.id, Action_Type.CLARIFICATION, 1, \
					state={"start_state":x,"next_state":y,"action":a,"model_num":action_index, "model":task, "cat":1}))
				outcomes, steps = self.simulate_action(x, self.actions[a], all_poss_actions=False, horizon=None, remaining_time_steps=None, modified=False)
				for outcome in outcomes:
					new_state_index = outcome[1]
					x_exp, y_exp = self.explain_it(task, x, new_state_index, None)
					clarification_actions.append(Action(None, 'is it still possible to go from *' + x_exp + '* to *' + y_exp + '* state with *' + self.actions[a].name + '* action - table '+str(self.task.table.id), self.task.table.id, Action_Type.CLARIFICATION, 1, \
					state={"start_state":x,"next_state":new_state_index,"action":a,"model_num":action_index, "model":task, "cat":2}))
					parameter_vars[m_name].append(m_name+"_s"+str(new_state_index))
					vars_tuples.append(("state", x, a, new_state_index))

			if len(task.modified_observation_function) > 0:
				if m_name not in parameter_vars.keys():
					parameter_vars[m_name] = []

				o = list(task.modified_observation_function.keys())[0]
				a = list(task.modified_observation_function[o].keys())[0]
				y = task.modified_observation_function[o][a][0][0]
				y_exp, o_exp = self.explain_it(task, None, y, o)
				# o_exp = "money on table"
				clarification_actions.append(Action(None, 'is it possible to observe *' + o_exp + '* in *' + y_exp + '* state with *' + self.actions[a].name + '*- table '+str(self.task.table.id), self.task.table.id, Action_Type.CLARIFICATION, 1, \
					state={"observation":o,"next_state":y,"action":a,"model_num":action_index, "model":task, "cat":1}))
				outcomes = self.simulate_observation (y, self.actions[a], modified=False)
				for outcome in outcomes:
					obs_index = outcome[0]
					y_exp, o_exp = self.explain_it(task, None, y, obs_index)
					clarification_actions.append(Action(None, 'is it still possible to observe *' + o_exp + '* in *' + y_exp + '* state with *' + self.actions[a].name + '*- table '+str(self.task.table.id), self.task.table.id, Action_Type.CLARIFICATION, 1, \
					state={"observation":obs_index,"next_state":y,"action":a,"model_num":action_index, "model":task, "cat":2}))
					parameter_vars[m_name].append(m_name+"_o"+str(obs_index[1]))
					vars_tuples.append(("observation", obs_index, a, y))
			
			self.model_pars[m_name] = vars_tuples

			model_vars.append(m_name)
			action_index += 1

		self.set_actions(clarification_actions)

		self.len_clarification_actions = len(clarification_actions)

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
		self.set_states(obs_type=Observation_Type.ORIGINAL, hidden_vars_names=hidden_vars_names, model_vars=model_vars, parameter_vars=parameter_vars)
		self.set_states(obs_type=Observation_Type.HUMAN_INPUT, hidden_vars_names=hidden_vars_names, model_vars=model_vars, parameter_vars=parameter_vars)
		return parameter_vars

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
			pomdp_task.add_observation(obs, action, next_state, self.epsilon)
			possible_pomdp_tasks.append(pomdp_task)

		if not explicit:
			for cur_state in belief.prob:
				for m in self.history.pre:
					next_state = m[1]
					pomdp_task = ClientPOMDPComplex(self.task, self.robot, self.navigation_goals, self.gamma, \
						self.random, self.reset_random, self.deterministic, self.no_op, self.run_on_robot)
					pomdp_task.add_transition(cur_state[1], action, next_state, self.epsilon)
					possible_pomdp_tasks.append(pomdp_task)

		valid_pomdp_tasks = []
		valid_new_beliefs = []
		for pomdp_task in possible_pomdp_tasks:
			pomdp_solver = POMDPSolver(pomdp_task,belief)
			prob = pomdp_solver.compute_1_over_eta (belief.prob, action, obs, all_poss_actions=True,horizon=None)
			if prob > 0:
				valid_pomdp_tasks.append(pomdp_task)
				new_belief = pomdp_solver.update_belief(belief.prob, action, obs, all_poss_actions=True, horizon=None)		
				valid_new_beliefs.append(new_belief)

		# print (len(valid_pomdp_tasks),len(valid_new_beliefs))
		return valid_pomdp_tasks, valid_new_beliefs

	def adapt_pomdp(self, pomdp_solver, agent_pomdp, initial_belief, action, obs):
		print ("augment POMDP")
		# add_variables = ["current_request","food","water","cooking_status"] ## ,"time_since_served"
		add_variables = []
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
				# print ("state: ", (b[0],state_mapping), "model: ", model)
				state_mappings.append((b[0], state_mapping))
			model += 1
			belief_mappings.append(state_mappings)


		parameter_vars = self.add_model_transitions (possible_pomdp_tasks, add_variables)
		
		for a in self.actions:
			if a.name == action.name:
				new_action = a
				break
		agent_pomdp.set_actions()		
		agent_pomdp.set_states()
		new_belief = self.get_new_belief (belief_mappings, parameter_vars)

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

	def get_new_belief (self, belief_mappings, parameter_vars):
		new_belief = []
		num_models = len(belief_mappings)
		
		new_state_tuple  = self.get_state_tuple(0)

		for model in range(0,num_models):
			start_index = self.feature_indices["m"+str(model)]+1
			end_index = len(new_state_tuple)-2 ## excluding x and y
			if model+1 < num_models:
				end_index = self.feature_indices["m"+str(model+1)]

			for m_f in range(start_index,end_index):
				if (m_f-start_index)%2 == 0:
					new_state_tuple[m_f] = 0
				else:
					new_state_tuple[m_f] = 1

		initial_model = self.get_state_index(new_state_tuple)

		model = 0
		for state_mappings in belief_mappings:
			for s_m in state_mappings:
				prob = s_m[0]
				state_mapping = s_m[1]
				new_state_tuple  = self.get_state_tuple(initial_model)
				for f in self.feature_indices.keys():
					if f in state_mapping:
						new_state_tuple[self.feature_indices[f]] = state_mapping[f]

				prob *= (1.0/num_models)
				new_state_tuple[self.feature_indices["m"+str(model)]] = 1

				possible_pars = []
				possible_pars_len = 1
				possible_pars_dim = ()
				for p in parameter_vars["m"+str(model)]:
					possible_pars.append(p)
					possible_pars_len *= 2
					possible_pars_dim += (2,)

				count = 0
				prob *= (1.0/possible_pars_len)
				while count < possible_pars_len:
					pars = self.get_tuple(count,possible_pars_dim)
					for var in range(len(possible_pars)):
						new_state_tuple[self.feature_indices[possible_pars[var]]] = pars[var]

					new_belief.append((prob,self.get_state_index(new_state_tuple)))
					# print ("state: ", (prob,new_state_tuple), "model: ", model)
					count += 1
			model += 1

		# print ("new_belief", new_belief)
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

	def is_goal (self, belief):
		for (prob,state_index) in belief.prob:
			state = self.get_state_tuple(state_index)
			if state [self.feature_indices['hand_raise']] == 1:
				return False
		return True
	def is_goal_state (self, state_index):
		state = self.get_state_tuple(state_index)
		if state [self.feature_indices['hand_raise']] == 1:
			return False
		return True

	def goal_pomdp_reward (self, reward):
		if self.goal_pomdp:
			if reward == -np.Inf:
				return np.Inf
			if reward == 100:
				return 0
		else:
			return reward

	def get_heuristic (self, belief):
		rew = np.Inf
		for (prob,state) in belief.prob:
			start_state = self.get_state_tuple(state)
			current_req = start_state [self.feature_indices['current_request']]
			t_hand = int(self.state_space[self.feature_indices['time_since_hand_raise']][1]/3)
			if current_req == 2:
				steps = 8 - current_req + 2*t_hand + 2*t_hand + 2*t_hand
			elif current_req == 3:
				cooking_status = start_state [self.feature_indices['cooking_status']]
				steps = 8 - current_req + (2-cooking_status)*t_hand + 2*t_hand + 2*t_hand
			elif current_req == 4:
				food = start_state [self.feature_indices['food']]
				steps = 8 - current_req + (3-food)*t_hand + 2*t_hand
			elif current_req == 5:
				water = start_state [self.feature_indices['water']]
				steps = 8 - current_req + (3-water)*t_hand
			else:
				steps = 8 - current_req
				
			rew = min(steps*ACTION_COST, rew)

		return rew

	def get_from_belief_library(self, belief, action, all_poss_actions):
		rew_t_t = None
		part_of_action_space = all_poss_actions
		if part_of_action_space in self.belief_library.keys():
			if belief in self.belief_library[part_of_action_space].keys():
				rew_t_t = self.belief_library[part_of_action_space][belief]

		if rew_t_t is None:
			rew_t_t = self.get_heuristic(belief)

		# if rew_found is not None and np.round(rew_found,2) != np.round(rew_t_t,2):
		# 	if "kitchen" in action.name:
		# 		print ("euqal: ")
		# 		print (action.name)
		# 		print (belief.get_string())
		# 		print (self.get_state_tuple(belief.prob[0][1]))
		# 		print (rew_found,rew_t_t)
		# 		set_trace()
		# 	# set_trace() 
		# 	rew_t_t = rew_found
		return rew_t_t

	def add_to_belief_library(self, belief, cost, all_poss_actions):
		part_of_action_space = all_poss_actions
		if len(self.belief_library.keys()) == 0:
			self.belief_library[True] = {}
			self.belief_library[False] = {}

		# if part_of_action_space in self.belief_library.keys():
		# 	if belief in self.belief_library[part_of_action_space].keys():
		# 		prev_cost = self.belief_library[part_of_action_space][belief]
		# 		cost = np.min(prev_cost, cost)
		self.belief_library[part_of_action_space][belief] = cost
