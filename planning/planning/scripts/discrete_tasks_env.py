import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import pylab as plt


GLOBAL_TIME = 0

class DiscreteTasks(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, tasks, robot, max_time, VI = False):
		self.tasks = tasks
		self.robot = robot

		self.include_time = False

		self.name = "discrete_tasks"

		self.nA = len(self.tasks)
		self.action_space = spaces.Discrete(self.nA)
		self.nS = 1
		obs = []
		self.task_start_end_index = []
		self.state_space_dim = ()

		## time
		if self.include_time:
			self.max_time = max_time
			obs.append((0, self.max_time, 1, "time"))
			self.state_space_dim += (self.max_time+1,)
			self.nS *= (self.max_time+1) ## time

		## tasks		
		self.nS *= np.power(2,len(self.tasks)) ## each task completed or not
		for task in self.tasks:
			obs.append((0, 1, 1, "task_id"))
			self.state_space_dim += (2,)
		
		
		for task in self.tasks:
			task_start_feature_index = len(obs)

			for feature in task.get_features():
				if feature.type == "discrete":
					self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
					self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					obs.append((feature.low, feature.high, feature.discretization, feature.name))

			self.task_start_end_index.append((task_start_feature_index,len(obs)))

		for feature in self.robot.get_features():
			if feature.type == "discrete":
				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
				obs.append((feature.low, feature.high, feature.discretization, feature.name))

		self.state_space = obs
		print ("state space: ", self.state_space)
		print ("state space dim: ", self.state_space_dim)
		self.reset()
		# set_trace()

		if VI:
			self.P = {}
			print ("computing P ...")
			self.compute_P()
			print ("done computing P ...")


	def step(self, task, start_state=None, simulate=False):
		global GLOBAL_TIME
		# print ("step task: ", self.state, task)
		if start_state == None:
			start_state = self.state
		task_num = task
		feature_start_index = self.task_start_end_index [task_num][0]
		feature_end_index = self.task_start_end_index [task_num][1]
		task = self.tasks[task_num]
		feature_names = []
		for i in range(0,len(start_state)):
			feature_names.append(self.state_space[i][3])

		robot_feature_start_index = self.task_start_end_index [len(self.tasks)-1][1]
		robot_feature_end_index = len(start_state)

		task_features = start_state[feature_start_index:feature_end_index]
		robot_features = start_state[robot_feature_start_index:robot_feature_end_index]

		# new_task_state, new_robot_state, num_steps = task.execute(start_state[0], task_features, \
		# 	feature_names[feature_start_index:feature_start_index + len(task.get_features())], \
		# 	robot_features, feature_names[robot_feature_start_index:robot_feature_end_index])

		if self.include_time:
			if not simulate:
				new_task_state, new_robot_state, num_steps = task.execute(start_state[0])
			else:
				new_task_state, new_robot_state, num_steps = task.simulate(task_features, \
				feature_names[feature_start_index:feature_start_index + len(task.get_features())], \
				robot_features, feature_names[robot_feature_start_index:robot_feature_end_index], start_state[0])
		else:
			if not simulate:
				new_task_state, new_robot_state, num_steps = task.execute()
			else:
				new_task_state, new_robot_state, num_steps = task.simulate(task_features, \
				feature_names[feature_start_index:feature_start_index + len(task.get_features())], \
				robot_features, feature_names[robot_feature_start_index:robot_feature_end_index])
		

		new_state_all = deepcopy(start_state)
		new_state_all [feature_start_index:feature_end_index] = new_task_state[0:len(new_task_state)]
		new_state_all [robot_feature_start_index:robot_feature_end_index] = new_robot_state[0:len(new_robot_state)]
		if self.include_time and new_state_all [task_num + 1] == 1:
			reward = 0.0
		elif not self.include_time and new_state_all [task_num] == 1:
			reward = -0.1*num_steps
		else:
			reward = 2.0 - 0.1 * num_steps
		k = num_steps
		action_highlevel = True
		prob = 1.0
		if self.include_time:
			new_state_all[0] = new_state_all[0] + num_steps
			new_state_all [task_num + 1] = 1 ## the task is finished
			terminal = all(x == 1 for x in new_state_all [1:len(self.tasks)+1])
		else:
			new_state_all [task_num] = 1 ## the task is finished
			terminal = all(x == 1 for x in new_state_all [0:len(self.tasks)])

		if not simulate:
			self.state = new_state_all

		# if GLOBAL_TIME % 101 == 0:
		# 	print (start_state, task_num)
		# 	print (self.get_state_index(new_state_all), new_state_all, reward, terminal, prob, k, action_highlevel)
		# 	set_trace()
			
		# print (self.get_state_index(new_state_all), new_state_all, reward, terminal, prob, k, action_highlevel)

		debug_info = {}
		debug_info['prob'] = 1.0
		debug_info['steps'] = k
		debug_info['action_highlevel'] = action_highlevel
		return self.get_state_index(new_state_all), reward, terminal, debug_info


	def reset_all(self):
		self.state = []
		if self.include_time:
			self.state.append(0) #time			
		self.state.extend([0 for i in range(len(self.tasks))])

		print ("reset all: ")
		for task in self.tasks:
			task.reset()
			for feature in task.get_features():
				print (feature.name, feature.value)
				self.state.append(feature.value)

		self.robot.reset()
		for feature in self.robot.get_features():
			print (feature.name, feature.value)
			self.state.append(feature.value)
		return self.get_state_index(self.state)

	def reset(self):
		self.state = []
		if self.include_time:
			self.state.append(0) #time
		# self.state.extend([np.random.randint(low=0,high=2) for i in range(len(self.tasks))])
		self.state.extend([0 for i in range(len(self.tasks))])

		# print ("reset: ")
		for task in self.tasks:
			for feature in task.get_features():
				# print ("task ", task.table.id, feature.name, feature.value)
				self.state.append(feature.value)

		self.robot.reset()
		for feature in self.robot.get_features():
			# print ("robot ", feature.name, feature.value)
			self.state.append(feature.value)
		return self.get_state_index(self.state)

	def get_state_index(self,state):
		new_state = tuple(state)
		new_state_index = np.ravel_multi_index(new_state,self.state_space_dim)
		return int(new_state_index)

	def get_state_tuple(self,new_state_index):
		state = np.unravel_index(new_state_index,self.state_space_dim)
		new_state = list(state)
		return new_state

	def render(self, start_state=None, mode='human'):
		if start_state == None:
			start_state = self.state

		grid_size = ()
		r_x_feature = self.robot.get_feature("x")
		r_y_feature = self.robot.get_feature("y")
		grid_size += (int((r_x_feature.high - r_x_feature.low) / r_x_feature.discretization) + 1,)
		grid_size += (int((r_y_feature.high - r_y_feature.low) / r_y_feature.discretization) + 1,)

		feature_names = []
		for i in range(0,len(start_state)):
			feature_names.append(self.state_space[i][3])

		robot_feature_start_index = self.task_start_end_index [len(self.tasks)-1][1]
		robot_feature_end_index = len(start_state)
		robot_features = start_state[robot_feature_start_index:robot_feature_end_index]

		en = np.ones(grid_size,dtype=np.float)
		x = 0
		y = 0
		for i in range(len(start_state)):
			if feature_names[i] == "x" and i < robot_feature_start_index:
				x = start_state[i]
			if feature_names[i] == "y" and i < robot_feature_start_index:
				y = start_state[i]
				en[x,y] = 0.1 ## tasks
			if feature_names[i] == "x" and i >= robot_feature_start_index:
				x = start_state[i]
			if feature_names[i] == "y" and i >= robot_feature_start_index:
				y = start_state[i]
				en[x,y] = 0.5 ## robot

		# en = np.reshape(en,self.grid_size*self.grid_size)
		# en = np.reshape(en,(self.grid_size,self.grid_size),order='F')
		im = plt.imshow(en,cmap='nipy_spectral')
		plt.colorbar(im,orientation='vertical')
		plt.ion()

		plt.show()
		plt.pause(.5)

		# plt.close()

	def compute_P(self):
		feature_array = [0] * len(self.state_space_dim)
		self.compute_P_elements(len(self.state_space_dim)-1,feature_array)

	def compute_P_elements(self,feature_index,feature_array):
		if feature_index < 0:
			self.P[self.get_state_index(feature_array)] = {} 
			state_index = self.get_state_index(feature_array)
			for a in range(self.nA):
				self.P[state_index][a] = self.query_model(state_index, a)
			return

		for i in range(self.state_space_dim[feature_index]):
			new_feature_array = feature_array.copy()
			new_feature_array[feature_index] = i
			self.compute_P_elements(feature_index-1,new_feature_array)

	def query_model(self, start_state_index, task_num):
		"""Return the possible transition outcomes for a state-action pair.
		This should be in the same format at the provided environments
		in section 2.
		Parameters
		----------
		state
		  State used in query. Should be in the same format at
		  the states returned by reset and step.
		action: int
		  The action used in query.
		Returns
		-------
		[(prob, nextstate, reward, is_terminal), ...]
		  List of possible outcomes
		"""	
		global GLOBAL_TIME

		start_state =  self.get_state_tuple(start_state_index)

		outcomes = []
		# if GLOBAL_TIME % 101 == 0:
		# 	self.render(start_state)

		new_state, reward, terminal, debug_info = self.step(task_num, start_state, simulate=True)

		# if GLOBAL_TIME % 101 == 0:
		# 	self.render(self.get_state_tuple(new_state))
		# 	set_trace()
		# 	plt.close()

		GLOBAL_TIME += 1 

		prob = debug_info['prob']
		k = debug_info['steps']
		action_highlevel = debug_info['action_highlevel']

		new_state_index = new_state
		outcomes.append((prob,new_state_index, reward, terminal, k, action_highlevel))

		return outcomes

