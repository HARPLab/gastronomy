import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pudb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import pylab as plt
import math
import itertools


from draw_env import *
from pomdp_agent import *
from pomdp_solver import *
from pomdp_client import *
from pomdp_client_simple import *
from pomdp_client_complex import *

GLOBAL_TIME = 0

class AgentHPOMDP(AgentPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_tasks, pomdp_solvers, tasks, robot, random, horizon, gamma, run_on_robot):
		self.random = random
		self.print_status = False
		self.pomdp_tasks = []
		pomdp_id = 0
		for pomdp_task in pomdp_tasks:
			if isinstance(pomdp_task,ClientPOMDPSimple):
				self.pomdp_tasks.append(HPOMDPSimple(pomdp_id, pomdp_task, pomdp_solvers[pomdp_id], tasks[pomdp_id], robot, random, horizon, gamma, run_on_robot))
			if isinstance(pomdp_task,ClientPOMDPComplex):
				self.pomdp_tasks.append(HPOMDPComplex(pomdp_id, pomdp_task, pomdp_solvers[pomdp_id], tasks[pomdp_id], robot, random, horizon, gamma, run_on_robot))

			pomdp_id += 1

		self.pomdp_solvers = pomdp_solvers
		self.tasks = tasks
		self.robot = robot
		self.gamma = gamma

		self.name = "agent_hpomdp" 

		self.nA = 0
		self.nS = 1
		self.nO = 1
		obs = []
		self.feature_indices = {}
		self.obs_feature_indices = {}
		self.task_indices = []
		self.state_space_dim = ()
		self.observation_space_dim = ()

		self.actions = np.full((len(self.pomdp_tasks),len(self.pomdp_tasks)),None)
		self.pomdps_actions = np.array(range(0,len(self.pomdp_tasks)))

		for l in range(len(self.pomdp_tasks)):
			act = Action(l, "task "+str(l), l, False, None)
			self.actions[l,:] = act


		self.feasible_actions = list(self.actions)
		self.feasible_actions_index = np.array(range(0,len(self.pomdp_tasks)))

		self.nA = self.actions.shape[0]

		self.action_space = spaces.Discrete(self.nA)

		# print (self.actions, self.feasible_actions)
		self.valid_actions = np.array(self.feasible_actions)

		feature_count = 0
		obs_feature_count = 0
		task_count = 0
		for task in self.tasks:
			start = feature_count
			for feature in task.get_features():
				if feature.type == "discrete" and feature.name in self.pomdp_tasks[0].non_robot_features:

					self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
					self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					obs.append((feature.low, feature.high, feature.discretization, feature.name))
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
		# robot's features
		for feature in self.robot.get_features():
			if feature.type == "discrete" and feature.name in ["x","y"]:
				self.robot_indices.append(feature_count)
				self.nS *= int((feature.high - feature.low) / feature.discretization) + 1
				self.state_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
				obs.append((feature.low, feature.high, feature.discretization, feature.name))
				self.feature_indices[feature.name] = feature_count
				feature_count += 1
				if feature.observable:
					self.obs_feature_indices[feature.name] = obs_feature_count
					self.observation_space_dim += (int((feature.high - feature.low) / feature.discretization) + 1,)
					self.nO *= int((feature.high - feature.low) / feature.discretization) + 1
					obs_feature_count += 1

		# print (self.nS)
		self.state_space = obs
		if self.print_status:
			print ("state space: ", self.state_space)
			print ("state space dim: ", self.state_space_dim)

		self.transition_function = {}
		self.observation_function = {}


		self.dense_reward = True

		# if self.print_status:
		# 	print ("computing P ...")
		# self.compute_P()
		# if self.print_status:
		# 	print ("done computing P ...")

		self.state = None


class HPOMDP(ClientPOMDP):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_id, pomdp_task, pomdp_solver, task, robot, random, horizon, gamma, run_on_robot):
		ClientPOMDP.__init__(self, pomdp_task.task, pomdp_task.robot, pomdp_task.navigation_goals, pomdp_task.gamma, pomdp_task.random, \
			pomdp_task.reset_random, pomdp_task.deterministic, pomdp_task.no_op, run_on_robot)

		self.random = random
		self.print_status = False
		self.pomdp_task = pomdp_task
		self.pomdp_solver = pomdp_solver
		self.task = task
		self.robot = robot
		self.gamma = gamma
		self.id = pomdp_id
		self.name = "hpomdp" 

	def simulate_action(self, start_state_index, action, all_poss_actions=False, horizon=None):
		if start_state_index in self.transition_function.keys() \
		and horizon in self.transition_function[start_state_index].keys() and action in self.transition_function[start_state_index][horizon].keys():
			return self.transition_function[start_state_index][horizon][action]

		# print ("****** action", action)
		start_state =  self.pomdp_task.get_state_tuple(start_state_index)
		new_state = np.asarray(deepcopy(start_state))
		outcomes = []
		k_steps = horizon

		belief = Belief([(1.0,start_state_index)])
		if action.id == self.id:	
			max_Q, max_a, max_time, Q_a, tree_size, time_steps, leaf_beliefs = self.pomdp_solver.compute_V \
									(belief, horizon, horizon, False, self.gamma, 0, all_poss_actions, HPOMDP=True)
			outcomes = leaf_beliefs ## do I need to aggregate duplicate states
			k_steps = max_time
		else:	
			belief = Belief([(1.0,start_state_index)])
			rew,_,_,_,leaf_beliefs_o = self.pomdp_solver.compute_Q_one_action(belief,self.pomdp_task.noop_actions['1'],k_steps,k_steps, \
									False,self.gamma, 0, all_poss_actions, HPOMDP=True)
			outcomes = leaf_beliefs_o		

		if start_state_index not in self.transition_function.keys():
			self.transition_function[start_state_index] = {} 

		if horizon not in self.transition_function[start_state_index].keys():
			self.transition_function[start_state_index][horizon] = {}

		if action not in self.transition_function[start_state_index][horizon].keys():
			self.transition_function[start_state_index][horizon][action] = (outcomes,k_steps)
		else:
			self.transition_function[start_state_index][horizon][action].extend((outcomes,k_steps))
		# print (self.get_state_tuple(start_state_index), actions, outcomes, self.get_state_tuple(outcomes[0][1]))

		# if horizon != max_time:
		# 	set_trace()

		return outcomes, k_steps


class HPOMDPSimple(HPOMDP, ClientPOMDPSimple):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_id, pomdp_task, pomdp_solver, task, robot, random, horizon, gamma, run_on_robot):
		HPOMDP.__init__(self, pomdp_id, pomdp_task, pomdp_solver, task, robot, random, horizon, gamma, run_on_robot)
		ClientPOMDPSimple.__init__(self, pomdp_task.task, pomdp_task.robot, pomdp_task.navigation_goals, pomdp_task.gamma, pomdp_task.random, \
			pomdp_task.reset_random, pomdp_task.deterministic, pomdp_task.no_op, run_on_robot)

class HPOMDPComplex(HPOMDP, ClientPOMDPComplex):
	metadata = {'render.modes': ['human']}

	def __init__(self, pomdp_id, pomdp_task, pomdp_solver, task, robot, random, horizon, gamma, run_on_robot):
		HPOMDP.__init__(self, pomdp_id, pomdp_task, pomdp_solver, task, robot, random, horizon, gamma, run_on_robot)
		ClientPOMDPComplex.__init__(self, pomdp_task.task, pomdp_task.robot, pomdp_task.navigation_goals, pomdp_task.gamma, pomdp_task.random, \
			pomdp_task.reset_random, pomdp_task.deterministic, pomdp_task.no_op, run_on_robot)

