from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import pylab as plt
import matplotlib.pyplot as matplt
import math
import sys
import _pickle as cPickle
import seaborn as sns
import pandas as pd
import os
from matplotlib.cbook import boxplot_stats
import time
from plot import Plot
from plot_all import Plot_All
from tabulate import tabulate

from draw_env import *
from pomdp_solver import *
from pomdp_client_complex import *
from pomdp_client_simple import *
from pomdp_agent_restaurant import *
from pomdp_agent_package import *
from pomdp_client_package import *
import multi_ind_pomdp_solver
from pomdp_hierarchical import *


print_status = False	
print_status_short = False

class POMDPTuple():
	def __init__(self, pair, lb, ub, extra_pomdps=None):
		self.LB = lb
		self.UB = ub
		self.LB_tpl = None
		self.UB_tpl = None
		self.tuple = pair
		self.extra_pomdps = extra_pomdps

class POMDPTasks():
	metadata = {'render.modes': ['human']}

	def __init__(self, restaurant, tasks, robot, seed, random, reset_random, horizon, greedy, simple, model, no_op, hybrid, deterministic, hybrid_3T, shani_baseline, hierarchical_baseline, package, folder):
		global print_status
		# print("seed: ", random.get_state()[1][0])

		
		# self.test_folder = '../tests_fixedH/'
		# self.test_folder = '../tests_fixedH_3T/'

		# self.test_folder = '../tests_nocash_3T/'
		# self.test_folder = '../tests_beliefLib_3T/'

		# self.test_folder = '../tests_belief_mix_4/'		
		# self.test_folder = '../tests_nocash_mix_2/'

		cluster = False
		if cluster:
			self.test_folder = "./"
		else:
			self.test_folder = "../"
			# self.test_folder = "../"


		# folder = "belief_mix"

		self.package = package
		self.gamma = 1.0 #0.95
		self.seed = seed
		self.restaurant = restaurant
		self.random = random
		self.greedy_hybrid = hybrid
		
		self.tasks = tasks
		self.robot = robot

		if self.package:
			self.precision = -0.1
		else:
			self.precision = -0.001

		self.num_random_executions = 30
		just_plot = False

		self.horizon = horizon
		self.max_horizon = self.horizon
		self.hybrid_3T = hybrid_3T
		self.hybrid_4T = False
		self.four_tables = False
		multi_ind_pomdp_solver.precision = self.precision
		self.greedy = greedy
		self.shani_baseline = shani_baseline
		self.hierarchical_baseline = hierarchical_baseline

		if self.greedy:
			if folder == "belief_2T":
				self.test_folder += 'tests_beliefLib/'
				self.hybrid_3T = False
				self.three_tables = False
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = True
			elif folder == "nocash_2T":
				self.test_folder += 'tests_nocash/'
				self.hybrid_3T = False
				self.three_tables = False
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = False
			elif folder == "belief_3T":
				self.test_folder += 'tests_beliefLib_3T/'
				self.hybrid_3T = True
				if not ("hybrid_3T" in model):
					model = model.replace("hybrid", "hybrid_3T")
				# set_trace()
				self.three_tables = True
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = True
			elif folder == "nocash_3T":
				self.test_folder += 'tests_nocash_3T/'
				self.hybrid_3T = True
				if not ("hybrid_3T" in model):
					model = model.replace("hybrid", "hybrid_3T")
				self.three_tables = True
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = False
			elif folder == "belief_mix":
				self.test_folder += 'tests_belief_mix_nobug/'
				self.hybrid_3T = False
				self.three_tables = True
				self.LB_UB = True
				self.MIX = True
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = True
			elif folder == "nocash_mix":
				self.test_folder += 'tests_nocash_mix_nobug/'
				self.hybrid_3T = False
				self.three_tables = True
				self.LB_UB = True
				self.MIX = True
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = False
			elif folder == "fixed_2T":
				self.test_folder += 'tests_fixedH/'
				self.hybrid_3T = False
				self.three_tables = False
				self.LB_UB = False
				self.MIX = False
				self.horizon = horizon
				multi_ind_pomdp_solver.cashing = False
			elif folder == "fixed_3T":
				self.test_folder += 'tests_fixedH_3T/'
				self.hybrid_3T = True
				if not ("hybrid_3T" in model):
					model = model.replace("hybrid", "hybrid_3T")
				self.three_tables = False
				self.LB_UB = False
				self.MIX = False
				self.horizon = horizon
				multi_ind_pomdp_solver.cashing = False
			elif folder == "fixed_4T":
				self.test_folder += 'tests_fixedH_4T/'
				self.hybrid_4T = True
				if not ("hybrid_4T" in model):
					model = model.replace("hybrid", "hybrid_4T")
				self.three_tables = False
				self.four_tables = False
				self.LB_UB = False
				self.MIX = False
				self.horizon = horizon
				multi_ind_pomdp_solver.cashing = False
			elif folder == "belief_4T":
				self.test_folder += 'tests_beliefLib_4T/'
				self.hybrid_4T = True
				if not ("hybrid_4T" in model):
					model = model.replace("hybrid", "hybrid_4T")
				# set_trace()
				self.three_tables = False
				self.four_tables = True
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = True
			elif folder == "nocash_4T":
				self.test_folder += 'tests_nocash_4T/'
				self.hybrid_4T = True
				if not ("hybrid_4T" in model):
					model = model.replace("hybrid", "hybrid_4T")
				self.three_tables = False
				self.four_tables = True
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = False
			elif folder == "shani_fixed":
				self.test_folder += 'tests_shani_fixed/'
				self.hybrid_3T = False
				self.hybrid_4T = False
				self.three_tables = False
				self.LB_UB = False
				self.MIX = False
				self.horizon = horizon
				multi_ind_pomdp_solver.cashing = False
				self.shani_baseline = True
				if self.horizon >= 7:
					self.hybrid_4T = True
				elif self.horizon >= 5:
					self.hybrid_3T = True
		elif folder == "HPOMDP":
				self.test_folder += 'tests_hpomdp_fixed/'
				self.hybrid_3T = False
				self.hybrid_4T = False
				self.three_tables = False
				self.LB_UB = False
				self.MIX = False
				self.horizon = horizon
				multi_ind_pomdp_solver.cashing = False
				self.hierarchical_baseline = True
		else:
			if folder == "agent_belief":
				self.test_folder += 'tests_agent_belief/'
				self.hybrid_3T = False
				self.three_tables = False
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = True
			elif folder == "agent_nocash":
				self.test_folder += 'tests_agent_nocash/'
				self.hybrid_3T = False
				self.three_tables = False
				self.LB_UB = True
				self.MIX = False
				self.horizon = 2
				multi_ind_pomdp_solver.cashing = False
			elif folder == "agent_fixed":
				self.test_folder += 'tests_agent_fixed/'
				self.hybrid_3T = False
				self.three_tables = False
				self.LB_UB = False
				self.MIX = False
				self.horizon = horizon
				multi_ind_pomdp_solver.cashing = False



		print ("-- horizon: ", self.horizon)
		print ("-- max horizon: ", self.max_horizon)
		print ("-- num executions: ", self.num_random_executions)
		random_initial_state = True
		self.max_steps = 21 # was 21
		self.LB = None
		self.UB = None

		self.render_belief = False
		self.no_op = no_op
		

		if self.hierarchical_baseline:
			self.no_op = True
			self.greedy_hybrid = True

		self.optimal_vs_greedy = False
		self.hybrid_vs_hybrid_3T = False
		self.model_folder = model + "_model"
		self.save_example = False
		if self.save_example:
			self.no_execution = True
			# self.num_random_executions = 16
			# self.exec_list = [15,16]##list(range(0,self.num_random_executions))
			self.num_random_executions = 10
			self.exec_list = [4]
			# self.exec_list = list(range(0,self.num_random_executions))
			# self.exec_list = [1]

		self.example_num = 0

		self.name = "pomdp_tasks"
		self.final_total_reward = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_time_steps = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_horizon = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_UB_minus_LB = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_num_pairs = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_belief_reward = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_max_Q = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_satisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_satisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_unsatisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_unsatisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_num_steps = np.zeros(self.num_random_executions)
		self.planning_time = np.zeros((self.num_random_executions,self.max_steps))
		self.tree_sizes = np.zeros((self.num_random_executions,self.max_steps))
		self.simple = simple

		if self.optimal_vs_greedy or self.hybrid_vs_hybrid_3T:
			self.final_total_reward_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_time_steps_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_horizon_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_num_pairs_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_UB_minus_LB_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_belief_reward_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_max_Q_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_satisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_satisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_unsatisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_unsatisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.tree_sizes_other = np.zeros((self.num_random_executions,self.max_steps))
			self.planning_time_other = np.zeros((self.num_random_executions,self.max_steps))

		if not just_plot:
			for random_exec in range(self.num_random_executions):

				self.pomdp_tasks = []
				self.pomdp_solvers = []
				count = 0

				navigation_goals = []

				if self.package:
					navigation_goals = list(range(0,self.restaurant.num_cities))
				else:
					for task in self.tasks:
						navigation_goals.append((task.table.goal_x,task.table.goal_y)) 
				
				beliefs = []
				if self.package:
					self.robot.reset()

				for task in self.tasks:
					if self.package:
						pomdp_task = ClientPOMDPPackage(task, robot, navigation_goals, self.gamma, random, reset_random, deterministic, self.no_op)
					else:
						if simple:
							pomdp_task = ClientPOMDPSimple(task, robot, navigation_goals, self.gamma, random, reset_random, deterministic, self.no_op)
						else:
							pomdp_task = ClientPOMDPComplex(task, robot, navigation_goals, self.gamma, random, reset_random, deterministic, self.no_op)

					self.pomdp_tasks.append(pomdp_task)

					if random_initial_state:
						start_state2 = pomdp_task.reset(random=True)
					else:
						start_state2 = pomdp_task.reset(random=False)
					
					print ("Table ", count, " start state: ", start_state2)
					pomdp_task.state = pomdp_task.get_state_index(start_state2)

					initial_belief = []
					initial_belief.append((1.0,pomdp_task.state))

					beliefs.append(initial_belief)

					self.pomdp_solvers.append(POMDPSolver(pomdp_task,initial_belief,random))
					count += 1


				## CODE
				self.multi_ind_pomdp = True
				print ("multiple ind POMDP: ", self.multi_ind_pomdp)
				if self.package:
					self.agent_pomdp = AgentPOMDPPackage(self.pomdp_tasks, self.pomdp_solvers, self.tasks, self.robot, random)
				else:
					self.agent_pomdp = AgentPOMDPRestaurant(self.pomdp_tasks, self.pomdp_solvers, self.tasks, self.robot, random)
				
				if self.hierarchical_baseline:
					self.agent_hpomdp = AgentHPOMDP(self.pomdp_tasks, self.pomdp_solvers, self.tasks, self.robot, random, self.horizon, self.gamma)
					self.agent_hpomdp_solver = multi_ind_pomdp_solver.MultiIndPOMDPSolver(self.agent_hpomdp,beliefs,random)
					# self.agent_hpomdp_solver = POMDPSolver(self.agent_hpomdp,self.agent_hpomdp.get_belief(beliefs),random)

				if not self.multi_ind_pomdp:
					self.agent_pomdp_solver = POMDPSolver(self.agent_pomdp,self.agent_pomdp.get_belief(beliefs),random)
				else:
					self.agent_pomdp_solver = multi_ind_pomdp_solver.MultiIndPOMDPSolver(self.agent_pomdp,beliefs,random)

				step = 0
				total_reward = np.zeros(self.max_steps)
				total_time_steps = np.zeros(self.max_steps)
				total_horizon = np.zeros(self.max_steps)
				total_num_pairs = np.zeros(self.max_steps)
				total_UB_minus_LB = np.zeros(self.max_steps)
				total_belief_reward = np.zeros(self.max_steps)
				total_max_Q = np.zeros(self.max_steps)
				satisfaction = np.zeros(self.max_steps)
				unsatisfaction = np.zeros(self.max_steps)
				total_satisfaction = np.zeros(self.max_steps)
				total_unsatisfaction = np.zeros(self.max_steps)
				tree_size = np.zeros(self.max_steps)
				duration = np.zeros(self.max_steps)

				if self.optimal_vs_greedy or self.hybrid_vs_hybrid_3T:
					total_reward_other = np.zeros(self.max_steps)
					total_time_steps_other = np.zeros(self.max_steps)
					total_horizon_other = np.zeros(self.max_steps)
					total_num_pairs_other = np.zeros(self.max_steps)
					total_UB_minus_LB_other = np.zeros(self.max_steps)
					total_belief_reward_other = np.zeros(self.max_steps)
					total_max_Q_other = np.zeros(self.max_steps)
					satisfaction_other = np.zeros(self.max_steps)
					unsatisfaction_other = np.zeros(self.max_steps)
					tree_size_other = np.zeros(self.max_steps)
					duration_other = np.zeros(self.max_steps)

				num_steps = 0
				self.step = 0
				self.example_num = random_exec
				if self.save_example and random_exec in self.exec_list:
					if not self.package:
						self.render(self.pomdp_tasks,str(step)+": ",num=num_steps,final_rew=None, exec_num=random_exec, render_belief=self.render_belief)
						# self.render(self.pomdp_tasks,str(step)+": ",num=num_steps,final_rew=None, exec_num=random_exec, render_belief=not self.render_belief)
					

				if print_status or print_status_short:
					if not self.package:
						self.render(self.pomdp_tasks,str(step)+": ",num=num_steps,final_rew=None, exec_num=random_exec, render_belief=self.render_belief)

				# set_trace()
				# 	print_status = True
				# else:
				# 	print_status = False

				terminal_prev = np.empty(len(self.tasks),dtype=bool)
				terminal_prev.fill(False)
				self.current_pomdp = None
				while (True):
					self.test_LB = -np.Inf
					# if random_exec not in self.exec_list and self.no_execution: 
					# 	break
					if print_status:
						print ("-------------------------------------------------------------------")

					if self.optimal_vs_greedy:
						start_states = np.zeros(len(self.tasks),dtype=int)
						
						for st in range(len(self.tasks)):
							start_states[st] = deepcopy(self.pomdp_tasks[st].state)
							# print (start_states[st])
							# print (self.pomdp_solvers[st].belief.prob, self.pomdp_tasks[st].get_state_tuple(self.pomdp_solvers[st].belief.prob[0][1]))

						all_terminal_other, position_other, sat_other, unsat_other, t_rew_other, t_b_rew_other, t_sat_other, t_unsat_other, action_other, time_steps_other, obss_other, obs_index_other, max_Q_other, tree_s_other, dur_other, max_pomdp_other, final_horizon_other, UB_minus_LB_other, num_pairs_other = \
						self.execute_episode(terminal_prev,not self.greedy, self.hierarchical_baseline, simulate_step=True, no_op=self.no_op, start_states=start_states)

						if print_status:
							print ("belief reward: ", t_b_rew_other)


						all_terminal, position, sat, unsat, t_rew, t_b_rew, t_sat, t_unsat, action, time_steps, obss, obs_index, max_Q, tree_s, dur, max_pomdp, final_horizon, UB_minus_LB, num_pairs = self.execute_episode(terminal_prev,self.greedy, self.hierarchical_baseline, simulate_step=False, no_op=self.no_op)
						self.current_pomdp = max_pomdp

						if print_status:
							print ("belief reward: ", t_b_rew)
							if action != action_other:
								set_trace()

						if action != action_other:
							total_reward_other[num_steps] = t_rew_other
							total_time_steps_other[num_steps] = time_steps_other
							total_horizon_other[num_steps] = horizon_other
							total_belief_reward_other[num_steps] = t_b_rew_other
							total_max_Q_other[num_steps] = max_Q_other
							satisfaction_other[num_steps] = sat_other
							unsatisfaction_other[num_steps] = unsat_other
							tree_size_other[num_steps] = tree_s_other
							duration_other[num_steps] = dur_other
						else:
							total_reward_other[num_steps] = t_rew
							total_time_steps_other[num_steps] = time_steps_other
							total_horizon_other[num_steps] = horizon_other
							total_belief_reward_other[num_steps] = t_b_rew_other
							total_max_Q_other[num_steps] = max_Q_other
							satisfaction_other[num_steps] = sat
							unsatisfaction_other[num_steps] = unsat
							tree_size_other[num_steps] = tree_s_other
							duration_other[num_steps] = dur_other


						if self.greedy:
							if not self.multi_ind_pomdp:
								self.agent_pomdp_solver.update_current_belief(action,obs_index,all_poss_actions=True, horizon=None)
							else:
								self.agent_pomdp_solver.update_current_belief(self.agent_pomdp.actions[action,:],obss,all_poss_actions=True, horizon=None)

							# print (self.agent_pomdp_solver.beliefs)
							# for b in self.agent_pomdp_solver.beliefs:
							# 	print (b.prob)
					elif self.hybrid_vs_hybrid_3T:
						start_states = np.zeros(len(self.tasks),dtype=int)
						
						for st in range(len(self.tasks)):
							start_states[st] = deepcopy(self.pomdp_tasks[st].state)
							
						self.hybrid_3T = True
						all_terminal_other, position_other, sat_other, unsat_other, t_rew_other, t_b_rew_other, t_sat_other, t_unsat_other, action_other, time_steps_other, obss_other, obs_index_other, max_Q_other, tree_s_other, dur_other, max_pomdp_other, final_horizon_other, UB_minus_LB_other, num_pairs_other = \
						self.execute_episode(terminal_prev,self.greedy, self.hierarchical_baseline, simulate_step=True, no_op=self.no_op, start_states=start_states)
						self.hybrid_3T = False

						if print_status:
							print ("belief reward: ", t_b_rew_other)


						all_terminal, position, sat, unsat, t_rew, t_b_rew, t_sat, t_unsat, action, time_steps, obss, obs_index, max_Q, tree_s, dur, max_pomdp, final_horizon, UB_minus_LB, num_pairs = self.execute_episode(terminal_prev,self.greedy, self.hierarchical_baseline, simulate_step=False, no_op=self.no_op)
						self.current_pomdp = max_pomdp

						if print_status:
							print ("belief reward: ", t_b_rew)
							if action != action_other:
								set_trace()

						if action != action_other:
							total_reward_other[num_steps] = t_rew_other
							total_time_steps_other[num_steps] = time_steps_other
							total_horizon_other[num_steps] = horizon_other
							total_belief_reward_other[num_steps] = t_b_rew_other
							total_max_Q_other[num_steps] = max_Q_other
							satisfaction_other[num_steps] = sat_other
							unsatisfaction_other[num_steps] = unsat_other
							tree_size_other[num_steps] = tree_s_other
							duration_other[num_steps] = dur_other
						else:
							total_reward_other[num_steps] = t_rew
							total_time_steps_other[num_steps] = time_steps_other
							total_horizon_other[num_steps] = horizon_other
							total_belief_reward_other[num_steps] = t_b_rew_other
							total_max_Q_other[num_steps] = max_Q_other
							satisfaction_other[num_steps] = sat
							unsatisfaction_other[num_steps] = unsat
							tree_size_other[num_steps] = tree_s_other
							duration_other[num_steps] = dur_other


						if self.greedy:
							if not self.multi_ind_pomdp:
								self.agent_pomdp_solver.update_current_belief(action,obs_index,all_poss_actions=True, horizon=None)
							else:
								self.agent_pomdp_solver.update_current_belief(self.agent_pomdp.actions[action,:],obss,all_poss_actions=True, horizon=None)

					else:
						all_terminal, position, sat, unsat, t_rew, t_b_rew, t_sat, t_unsat, action, time_steps, obss, obs_index, max_Q, tree_s, dur, max_pomdp, final_horizon, UB_minus_LB, num_pairs = \
						self.execute_episode(terminal_prev,self.greedy, self.hierarchical_baseline,simulate_step=False, no_op=self.no_op)
						self.current_pomdp = max_pomdp

					total_reward[num_steps] = t_rew
					total_time_steps[num_steps] = time_steps
					total_horizon[num_steps] = final_horizon
					total_num_pairs[num_steps] = num_pairs
					total_UB_minus_LB[num_steps] = UB_minus_LB
					total_belief_reward[num_steps] = t_b_rew
					total_max_Q[num_steps] = max_Q
					satisfaction[num_steps] = sat
					unsatisfaction[num_steps] = unsat
					total_satisfaction[num_steps] = t_sat
					total_unsatisfaction[num_steps] = t_unsat
					tree_size[num_steps] = tree_s
					duration[num_steps] = dur

					if not self.package:
						self.robot.set_feature('x',position[0])
						self.robot.set_feature('y',position[1])
					else:
						if position is not None:
							if print_status:
								print ("---------------- Robot position: ", position)

					num_steps += 1
					self.step += 1

					if print_status or print_status_short:
						print ("Step: ", num_steps)
						print ("Time steps: ", time_steps)
						print ("Max horizon, UB_minus_LB, num_pairs: ", final_horizon, UB_minus_LB, num_pairs)
						if not self.package:
							self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions[action,max_pomdp].name,num=num_steps,final_rew=t_b_rew,exec_num=random_exec, render_belief=self.render_belief, horizon=final_horizon)
							print ("max_Q:", max_Q)
							print ("t_b_rew:", t_b_rew)
							# set_trace()
						else:
							set_trace()

					if self.save_example and random_exec in self.exec_list:
						if not self.package:
							self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions[action,max_pomdp].name,num=num_steps,final_rew=t_b_rew,exec_num=random_exec, render_belief = self.render_belief, horizon=final_horizon)
							# self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions[action,max_pomdp].name,num=num_steps,final_rew=np.sum(total_belief_reward),exec_num=random_exec, render_belief = not self.render_belief)

					if all_terminal or num_steps>=self.max_steps:
						break
					step += 1
					
					# sleep(1)

					# print ("Step: ", num_steps)
					# self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions_names[action],num=num_steps,final_rew=np.sum(total_belief_reward),exec_num=random_exec)
					self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions[action,max_pomdp].name,num=num_steps,final_rew=t_b_rew,exec_num=random_exec, render_belief=self.render_belief, horizon=final_horizon)

					# if print_status:
						# if "I'll be back" in self.agent_pomdp.actions[action,max_pomdp].name:
					# set_trace()				

				print ("Done " + str(random_exec))
				self.final_total_reward[random_exec,:] = total_reward
				self.final_total_time_steps[random_exec,:] = total_time_steps
				self.final_total_horizon[random_exec,:] = total_horizon
				self.final_total_UB_minus_LB[random_exec,:] = total_UB_minus_LB
				self.final_total_num_pairs[random_exec,:] = total_num_pairs
				self.final_total_belief_reward[random_exec,:] = total_belief_reward
				self.final_total_max_Q[random_exec,:] = total_max_Q
				self.final_num_steps[random_exec] = num_steps
				self.final_satisfaction[random_exec,:] = satisfaction
				self.final_total_satisfaction[random_exec,:] = total_satisfaction
				self.final_unsatisfaction[random_exec,:] = unsatisfaction
				self.final_total_unsatisfaction[random_exec,:] = total_unsatisfaction
				self.planning_time[random_exec,:] = duration
				self.tree_sizes[random_exec,:] = tree_size

				if self.optimal_vs_greedy or self.hybrid_vs_hybrid_3T:
					self.final_total_reward_other[random_exec,:] = total_reward_other
					self.final_total_time_steps_other[random_exec,:] = total_time_steps_other
					self.final_total_horizon_other[random_exec,:] = total_horizon_other
					self.final_total_UB_minus_LB_other[random_exec,:] = total_UB_minus_LB_other
					self.final_total_num_pairs_other[random_exec,:] = total_num_pairs_other
					self.final_total_belief_reward_other[random_exec,:] = total_belief_reward_other
					self.final_total_max_Q_other[random_exec,:] = total_max_Q_other
					self.final_satisfaction_other[random_exec,:] = satisfaction_other
					self.final_total_satisfaction_other[random_exec,:] = total_satisfaction
					self.final_unsatisfaction_other[random_exec,:] = unsatisfaction_other
					self.final_total_unsatisfaction_other[random_exec,:] = total_unsatisfaction
					self.tree_sizes_other[random_exec,:] = tree_size_other
					self.planning_time_other[random_exec,:] = duration_other

					print ("total reward other: ", np.sum(self.final_total_reward_other[random_exec,:]))
					print ("total time steps other: ", np.sum(self.total_time_steps_other[random_exec,:]))
					print ("total max horizon other: ", np.sum(self.total_horizon_other[random_exec,:]))
					print ("total num pairs other: ", np.sum(self.total_num_pairs_other[random_exec,:]))
					print ("total UB minus LB other: ", np.sum(self.total_UB_minus_LB_other[random_exec,:]))
					print ("total belief reward other: ", np.sum(self.final_total_belief_reward_other[random_exec,:]))
					print ("total max Q other: ", np.sum(self.final_total_max_Q_other[random_exec,:]))
					print ("total satisfaction other: ", np.sum(self.final_satisfaction_other[random_exec,:])/np.sum(self.final_total_satisfaction_other[random_exec,:]))
					print ("total unsatisfaction other: ", np.sum(self.final_unsatisfaction_other[random_exec,:])/np.sum(self.final_total_unsatisfaction_other[random_exec,:]))
					print ("total tree size other: ", np.sum(self.tree_sizes_other[random_exec,:]))
					print ("planning time other: ", np.sum(self.planning_time_other[random_exec,:]))

				print ("num steps: ", np.sum(self.final_num_steps[random_exec]))
				print ("total reward: ", np.sum(self.final_total_reward[random_exec,:]))
				print ("total time steps: ", np.sum(self.final_total_time_steps[random_exec,:]))
				print ("total max horizon: ", np.sum(self.final_total_horizon[random_exec,:]))
				print ("total UB minus LB: ", np.sum(self.final_total_UB_minus_LB[random_exec,:]))
				print ("total num pairs: ", np.sum(self.final_total_num_pairs[random_exec,:]))
				print ("total belief reward: ", np.sum(self.final_total_belief_reward[random_exec,:]))
				print ("total max Q: ", np.sum(self.final_total_max_Q[random_exec,:]))
				print ("total satisfaction: ", np.sum(self.final_satisfaction[random_exec,:])) ## /np.sum(self.final_total_satisfaction[random_exec,:])
				print ("total unsatisfaction: ", np.sum(self.final_unsatisfaction[random_exec,:])) ## /np.sum(self.final_total_unsatisfaction[random_exec,:])
				print ("planning time: ", np.sum(self.planning_time[random_exec,:]))
				print ("tree size: ", np.sum(self.tree_sizes[random_exec,:]))
				print ("_________________________________________________________________________")
				if num_steps>=self.max_steps:
					print ("EXCEEDED MAX STEPS")
					# pass
					# set_trace()
				# set_trace()
				self.robot.reset()

			self.write_to_file(True)
			
		else:
			Plot_All(self).plot_statistics()
			# if "all_greedy" not in self.model_folder:
			# 	Plot(self).plot_statistics()
			# else:
			# 	if "shani" in self.model_folder:
			# 		Plot(self).plot_statistics_no_op(shani=True,hybrid=False,hybrid_3T=False)	
			# 	elif "hybrid_3T" in self.model_folder:
			# 		Plot(self).plot_statistics_no_op(shani=False,hybrid=False,hybrid_3T=True)
			# 	elif "hybrid" in self.model_folder:
			# 		Plot(self).plot_statistics_no_op(shani=False,hybrid=True,hybrid_3T=False)	

				
		# set_trace()

	def execute_episode(self, terminal_prev, greedy, hierarchical, simulate_step, no_op=False, start_states=None):
		tree_size = 0
		planning_time = 0
		start_time = time.clock() 
		if not hierarchical:
			if greedy:			
				pomdp_task, max_pomdp, action, max_Q, tree_size, time_step, final_horizon, UB_minus_LB, num_pairs = self.greedy_selection_of_individual_pomdps(self.horizon, no_op)
			else:
				if self.LB_UB:
					pomdp_task, max_pomdp, action, max_Q, tree_size, time_step, final_horizon, UB_minus_LB, num_pairs = self.adaptive_horizon_optimal_agent_pomdp(self.horizon, self.pomdp_tasks, self.agent_pomdp, self.agent_pomdp_solver)
				else:
					final_horizon = self.max_horizon
					pomdp_task, max_pomdp, action, max_Q, tree_size, time_step = self.optimal_agent_pomdp(self.horizon, self.pomdp_tasks, self.agent_pomdp, self.agent_pomdp_solver)	
					UB_minus_LB = max_Q.UB - max_Q.LB
					num_pairs = 1
					max_Q = max_Q.UB
		else:
			final_horizon = self.max_horizon
			pomdp_task, max_pomdp, action, max_Q, tree_size, time_step = self.hierarchical_agent_pomdp(self.horizon, self.pomdp_tasks, self.agent_hpomdp, self.agent_hpomdp_solver, no_op)	
			UB_minus_LB = 0
			num_pairs = 1
			# max_Q = max_Q.UB

		end_time = time.clock()	
		planning_time += end_time - start_time

		obss = []
		observations = []
		all_terminal = True

		total_reward = 0
		total_belief_reward = 0		
		satisfaction = 0
		unsatisfaction = 0
		total_satisfaction = 0
		total_unsatisfaction = 0
		position = None

		if greedy:
			beliefs = []
			for i in range(len(self.pomdp_solvers)):
				beliefs.append(self.pomdp_solvers[i].belief)
			if not isinstance(action,int):
				print ("action is not integer - pomdp_tasks_env")
				# set_trace()
			rew_out, tree_s, exp_time,_,_ = self.agent_pomdp_solver.compute_Q (beliefs=beliefs, actions=self.agent_pomdp.actions[action,:], horizon=time_step, max_horizon=self.max_horizon, one_action=True, gamma=self.gamma, tree_s=1, all_poss_actions=True, LB_UB=False)
			rew_out = rew_out.UB; exp_time = exp_time.UB; 

			total_belief_reward += rew_out
		elif not greedy:
			if not self.multi_ind_pomdp:
				rew_out, tree_s, exp_time,_,_ = self.agent_pomdp_solver.compute_Q (belief=self.agent_pomdp_solver.belief, action=action, horizon=time_step, max_horizon=self.max_horizon, one_action=True, gamma=self.gamma, tree_s=1, all_poss_actions=True, LB_UB=False)
				rew_out = rew_out.UB; exp_time = exp_time.UB;
				total_belief_reward += rew_out
			else:
				# set_trace()
				# print ("pomdp solver multi", self.agent_pomdp_solver.beliefs[0].prob, self.agent_pomdp_solver.beliefs[1].prob)
				rew_out, tree_s, exp_time,_,_ = self.agent_pomdp_solver.compute_Q (beliefs=self.agent_pomdp_solver.beliefs, actions=self.agent_pomdp.actions[action,:], horizon=time_step, max_horizon=self.max_horizon, one_action=True, gamma=self.gamma, tree_s=1,all_poss_actions=True, LB_UB=False)
				rew_out = rew_out.UB; exp_time = exp_time.UB; 
				total_belief_reward += rew_out

		obs = None
		if self.package:
			# self.robot.updated = False
			if self.agent_pomdp.actions[action,0].id >= self.pomdp_tasks[0].non_navigation_actions_len:
				obs = self.agent_pomdp.get_random_observation(self.agent_pomdp.actions[action,0])

		for i in range(len(self.pomdp_solvers)):
			if print_status or print_status_short:
				solver_prob = self.pomdp_solvers[i].belief.prob
				if not self.package:
					state_str = "-- initial belief: " + str(self.pomdp_tasks[i].get_state_tuple(self.pomdp_solvers[i].belief.prob[0][1]))
					for p in solver_prob:
						st = self.pomdp_tasks[i].get_state_tuple(p[1])			
						state_str += ", " + str(st[self.pomdp_tasks[i].feature_indices['customer_satisfaction']]) + ":" + str(round(p[0],2))
					print (state_str)
				else:
					belief_str = self.pomdp_solvers[i].get_readable_string()
					print ("-- initial belief: ", belief_str)


			if not simulate_step:
				new_state_index, obs, reward, terminal, debug_info, position = self.pomdp_tasks[i].step(self.agent_pomdp.actions[action,i],start_state=None,simulate=False,belief=self.pomdp_solvers[i].belief.prob,observation=obs)
			else:
				new_state_index, obs, reward, terminal, debug_info, position = self.pomdp_tasks[i].step(self.agent_pomdp.actions[action,i],start_state=start_states[i],simulate=True,belief=self.pomdp_solvers[i].belief.prob,observation=obs)

			all_terminal = all_terminal and terminal
			new_state = self.pomdp_tasks[i].get_state_tuple(new_state_index)


			if not terminal_prev[i]:
				if not self.package:
					satisfaction += new_state [self.pomdp_tasks[i].feature_indices['customer_satisfaction']]
					if new_state [self.pomdp_tasks[i].feature_indices['customer_satisfaction']] == 0:
						unsatisfaction += 1
				
					total_satisfaction += 2
					total_unsatisfaction += 1
				else:
					pass

				total_reward += reward

			if not simulate_step:
				terminal_prev[i] = terminal

			obss.append(obs)
			obs_tuple = self.pomdp_tasks[i].get_observation_tuple(obs)
			if len(self.pomdp_tasks[i].robot_obs_indices) == 0:
				observations.extend(obs_tuple)
			else:
				observations.extend(obs_tuple[0:self.pomdp_tasks[i].robot_obs_indices[0]]) ## hack
			new_state = self.pomdp_tasks[i].get_state_tuple(new_state_index)

			pomdp_solver = self.pomdp_solvers[i]
			
			if not simulate_step:
				start_time = time.clock() 
				pomdp_solver.update_current_belief(self.agent_pomdp.actions[action,i],obs,all_poss_actions=True, horizon=None)
				end_time = time.clock()	
				if greedy:
					planning_time += end_time - start_time

		if len(self.pomdp_tasks[i].robot_obs_indices) != 0:
			observations.extend(obs_tuple[self.pomdp_tasks[i].robot_obs_indices[0]:len(obs_tuple)]) #hack
		if print_status:
			print ("observations: ", observations)
		# set_trace()
		obs_index = None
		if not greedy:
			if not simulate_step: 
				start_time = time.clock() 				
				if not self.multi_ind_pomdp:
					obs_index = self.agent_pomdp.get_observation_index(observations)
					self.agent_pomdp_solver.update_current_belief(action,obs_index,all_poss_actions=True, horizon=None)
				else:
					self.agent_pomdp_solver.update_current_belief(self.agent_pomdp.actions[action,:],obss,all_poss_actions=True, horizon=None)
				end_time = time.clock()	
				planning_time += end_time - start_time

				if hierarchical:
					start_time = time.clock() 
					beliefs = []

					# for b in self.agent_pomdp_solver.beliefs:
					# 	beliefs.append(b.prob)
					# self.agent_hpomdp_solver.belief.prob = self.agent_hpomdp.get_belief(beliefs)

					self.agent_hpomdp_solver.beliefs = self.agent_pomdp_solver.beliefs
					end_time = time.clock()	
					planning_time += end_time - start_time

		if print_status:
			for i in range(len(self.pomdp_solvers)):			
				solver_prob = self.pomdp_solvers[i].belief.prob
				if not self.package:
					state_str = "-- updated belief: " + str(self.pomdp_tasks[i].get_state_tuple(self.pomdp_solvers[i].belief.prob[0][1]))
					for p in solver_prob:
						st = self.pomdp_tasks[i].get_state_tuple(p[1])			
						state_str += ", " + str(st[self.pomdp_tasks[i].feature_indices['customer_satisfaction']]) + ":" + str(round(p[0],2))
					print (state_str)
				else:
					belief_str = self.pomdp_solvers[i].get_readable_string()
					print ("-- updated belief: ", belief_str)
					if len (belief_str) == 0:
						set_trace()

		return all_terminal, position, satisfaction, unsatisfaction, total_reward, total_belief_reward, total_satisfaction, total_unsatisfaction, action, time_step, obss, obs_index, max_Q, tree_size, planning_time, max_pomdp, final_horizon, UB_minus_LB, num_pairs
		


	def write_to_file(self,file):
		if file:
			filename = self.test_folder + self.model_folder + '/' + 'tables-' + str(len(self.tasks)) + '_simple-' + str(self.simple) \
			 + '_greedy-' + str(self.greedy) +  '_horizon-' + str(self.max_horizon+1000) + '_#execs-' + \
			 str(self.num_random_executions) + '_seed-' + str(self.seed);

			# filename = self.test_folder + self.model_folder + '/' + 'tables-' + str(len(self.tasks)) + '_simple-' + str(self.simple) \
			#  + '_greedy-' + str(self.greedy) +  '_horizon-' + str(1000) + '_#execs-' + \
			#  str(self.num_random_executions) + '_seed-' + str(self.seed);

			if self.greedy:
				filename += "_noop-" + str(self.no_op)
				if self.hybrid_3T:
					filename += "_hybrid_3T-" + str(True)
				elif self.hybrid_4T:
					filename += "_hybrid_4T-" + str(True)
				else:
					filename += "_hybrid-" + str(self.greedy_hybrid)

				if self.shani_baseline:
					filename += "_shani-" + str(self.shani_baseline)

			if self.hierarchical_baseline:
				filename += "_H_POMDP-" + str(self.hierarchical_baseline)

			txt_filename = filename + '.txt'
			pickle_filename = filename + '.pkl'
			f = open(txt_filename,'a+')
			data = {}
			data['final_num_steps'] = self.final_num_steps
			data['final_total_time_steps'] = self.final_total_time_steps
			data['final_total_horizon'] = self.final_total_horizon
			data['final_total_UB_minus_LB'] = self.final_total_UB_minus_LB
			data['final_total_num_pairs'] = self.final_total_num_pairs
			data['final_total_reward'] = self.final_total_reward
			data['final_total_belief_reward'] = self.final_total_belief_reward
			data['final_total_max_Q'] = self.final_total_max_Q
			data['final_total_satisfaction'] = self.final_total_satisfaction
			data['final_satisfaction'] = self.final_satisfaction
			data['final_total_unsatisfaction'] = self.final_total_unsatisfaction
			data['final_unsatisfaction'] = self.final_unsatisfaction
			data['planning_time'] = self.planning_time
			data['tree_sizes'] = self.tree_sizes

			if self.optimal_vs_greedy or self.hybrid_vs_hybrid_3T:
				data['final_total_reward_other'] = self.final_total_reward_other
				data['final_total_time_steps_other'] = self.final_total_time_steps_other
				data['final_total_horizon_other'] = self.final_total_horizon_other
				data['final_total_UB_minus_LB_other'] = self.final_total_UB_minus_LB_other
				data['final_total_num_pairs_other'] = self.final_total_num_pairs_other
				data['final_total_belief_reward_other'] = self.final_total_belief_reward_other
				data['final_total_max_Q_other'] = self.final_total_max_Q_other
				data['final_total_satisfaction_other'] = self.final_total_satisfaction_other
				data['final_total_unsatisfaction_other'] = self.final_total_unsatisfaction_other
				data['final_satisfaction_other'] = self.final_satisfaction_other
				data['final_unsatisfaction_other'] = self.final_unsatisfaction_other
				data['tree_sizes_other'] = self.tree_sizes_other

			tables = np.empty(self.num_random_executions,dtype=int)
			tables.fill(len(self.tasks))
			greedy = np.empty(self.num_random_executions,dtype=bool)
			greedy.fill(self.greedy)
			simple = np.empty(self.num_random_executions,dtype=bool)
			simple.fill(self.simple)
			horizon = np.empty(self.num_random_executions,dtype=int)
			horizon.fill(self.horizon)
			max_horizon = np.empty(self.num_random_executions,dtype=int)
			max_horizon.fill(self.max_horizon)
			no_op = np.empty(self.num_random_executions,dtype=int)
			no_op.fill(self.no_op)
			hybrid = np.empty(self.num_random_executions,dtype=int)
			hybrid.fill(self.greedy_hybrid)
			shani = np.empty(self.num_random_executions,dtype=int)
			shani.fill(self.shani_baseline)
			H_POMDP = np.empty(self.num_random_executions,dtype=int)
			H_POMDP.fill(self.hierarchical_baseline)


			data['tables'] = tables
			data['greedy'] = greedy
			data['simple'] = simple
			data['horizon'] = horizon
			data['max_horizon'] = max_horizon
			data['no_op'] = no_op
			data['hybrid'] = hybrid
			data['shani'] = shani
			data['H_POMDP'] = H_POMDP

			# df = pd.DataFrame(data=data)
			# df.to_pickle(path=pickle_filename)
			cPickle.dump(data,open(pickle_filename,'wb+'))
		else:
			f = sys.stdout
		# set_trace()
		print ("final num steps: ", np.mean(self.final_num_steps), file=f)
		print ("final time steps: ", np.mean(self.final_total_time_steps), file=f)
		print ("final max horizon: ", np.mean(self.final_total_horizon), file=f)
		print ("final UB minus LB: ", np.mean(self.final_total_UB_minus_LB), file=f)
		print ("final num pairs: ", np.mean(self.final_total_num_pairs), file=f)
		print ("final total reward: ",np.mean(np.sum(self.final_total_reward,axis=1)), file=f)
		print ("final total belief reward: ",np.mean(np.sum(self.final_total_belief_reward,axis=1)), file=f)
		print ("final total max Q: ",np.mean(np.sum(self.final_total_max_Q,axis=1)), file=f)
		print ("final total satisfaction: ", np.mean(np.sum(self.final_satisfaction,axis=1)/np.sum(self.final_total_satisfaction,axis=1)), file=f)
		print ("final total unsatisfaction: ", np.mean(np.sum(self.final_unsatisfaction,axis=1)/np.sum(self.final_total_unsatisfaction,axis=1)), file=f)
		print ("final mean planning time: ", np.mean(np.sum(self.planning_time,axis=1)), file=f)
		print ("final mean tree sizes: ", np.mean(np.sum(self.tree_sizes,axis=1)), file=f)
		print ("# of random executions: ", self.num_random_executions, file=f)
		print ("# of tables: ", len(self.tasks), file=f)
		print ("horizon: ", self.horizon, file=f)
		print ("max horizon: ", self.max_horizon, file=f)
		print ("greedy: ", self.greedy, file=f)
		print ("simple: ", self.simple, file=f)
		print ("seed: ", self.seed, file=f)
		print ("no_op: ", self.no_op, file=f)
		print ("hybrid: ", self.greedy_hybrid, file=f)
		print ("shani: ", self.shani_baseline, file=f)
		print ("H_POMDP: ", self.hierarchical_baseline, file=f)

		if self.optimal_vs_greedy or self.hybrid_vs_hybrid_3T:
			print ("final total reward other: ",np.mean(np.sum(self.final_total_reward_other,axis=1)), file=f)
			print ("final total time steps other: ",np.mean(np.sum(self.final_total_time_steps_other,axis=1)), file=f)
			print ("final total max horizon other: ",np.mean(np.sum(self.final_total_horizon_other,axis=1)), file=f)
			print ("final total num pairs other: ",np.mean(np.sum(self.final_total_num_pairs_other,axis=1)), file=f)
			print ("final total UB minus LB other: ",np.mean(np.sum(self.final_total_UB_minus_LB_other,axis=1)), file=f)
			print ("final total belief reward other: ",np.mean(np.sum(self.final_total_belief_reward_other,axis=1)), file=f)
			print ("final total max_Q other: ",np.mean(np.sum(self.final_total_max_Q_other,axis=1)), file=f)
			print ("final total satisfaction other: ", np.mean(np.sum(self.final_satisfaction_other,axis=1)/np.sum(self.final_total_satisfaction_other,axis=1)), file=f)
			print ("final total unsatisfaction other: ", np.mean(np.sum(self.final_unsatisfaction_other,axis=1)/np.sum(self.final_total_unsatisfaction_other,axis=1)), file=f)
			print ("tree sizes other: ", np.mean(np.sum(self.tree_sizes_other,axis=1)), file=f)

		print ("DONEEEEE!", file=f)

		if file:
			f.close()

	def adaptive_horizon_optimal_agent_pomdp (self, horizon, pomdp_tasks, agent_pomdp, agent_pomdp_solver):
		self.UB = None
		self.LB = None
		increasing_horizon = horizon
		num_pairs = 0
		while increasing_horizon <= self.max_horizon and not (self.UB is not None and self.LB is not None and \
			(self.UB - self.LB < -self.precision)):
			pomdp_task, max_pomdp, max_action, max_Q, tree_size, max_time_step = self.optimal_agent_pomdp (increasing_horizon, pomdp_tasks, agent_pomdp, agent_pomdp_solver)
			UB = max_Q.UB
			LB = max_Q.LB

			if (self.UB is not None and self.LB is not None) and (self.UB - UB < self.precision or self.LB - LB > -self.precision):
				print ("num_tables", len(self.agent_pomdp.pomdp_tasks),"horizon", self.horizon, self.max_horizon, " step: ", self.step)
				print ("new UB, LB ", UB, LB)
				print ("prev UB, LB ", self.UB, self.LB)
				print ("self.UB < UB: ", self.UB < UB, " self.LB > LB: ", self.LB > LB)
				# set_trace()

			self.UB = UB
			self.LB = LB
			num_pairs += 1

			increasing_horizon += 1

		increasing_horizon -= 1
		UB_minus_LB = self.UB - self.LB
		return pomdp_task, max_pomdp, max_action, max_Q.UB, tree_size, max_time_step, increasing_horizon, UB_minus_LB, num_pairs

	def optimal_agent_pomdp (self, horizon, pomdp_tasks, agent_pomdp, agent_pomdp_solver):
		global print_status
		action_values = np.zeros(horizon*2)
		Q, a, max_time, Q_a, tree_size, exp_time_steps,_ = agent_pomdp_solver.compute_V(None,horizon,self.max_horizon,one_action=False,gamma=self.gamma,tree_s=1,all_poss_actions=True, LB_UB=self.LB_UB)
		# Q = Q.UB; a = a.UB; max_time = max_time.UB; Q_a = Q_a.UB; 

		if print_status:
			if not self.package:
				print ("robot: ", (self.robot.get_feature('x').value, self.robot.get_feature('y').value))
			else:
				belief_str = ""
				for t in range(0,len(self.pomdp_solvers)):
					belief_str += self.pomdp_solvers[t].get_readable_string()
				print ("agent pomdp belief: ", belief_str)
			print ("reward, selected action: ",(Q.LB,Q.UB,a.UB[0].name))
			print ("action values: ", action_values)

			# if horizon == 2:
			# 	print (self.agent_pomdp.actions_names[int(action_values[0])], " , ", \
			# 		self.agent_pomdp.actions_names[int(action_values[1])])
			# if horizon == 3:
			# 	print (self.agent_pomdp.actions_names[int(action_values[0])], " , ", \
			# 		self.agent_pomdp.actions_names[int(action_values[1])], " , ", \
			# 		self.agent_pomdp.actions_names[int(action_values[2])])
			print ("-- Optimal Max Q --")
			print (Q_a.UB)

		
		max_action = np.argmax(Q_a.UB) ## was a
		max_Q = Q
		max_pomdp, action = agent_pomdp.get_pomdp(max_action, self.current_pomdp)
		pomdp_task = pomdp_tasks[max_pomdp]	
		max_time_step = exp_time_steps[max_action][0]
		
		if print_status:
			print ("time steps: ", max_time_step)
			print ("POMDP ", max_pomdp)
			print ("final selected action: ",action.name)
			if len(pomdp_tasks) == len(self.pomdp_tasks):
				if self.LB is not None and self.LB - max_Q.LB > -self.precision:
					print ("WRONG LB")
					# set_trace()
				if (self.UB is not None and self.UB - max_Q.UB < self.precision) and not self.shani_baseline:
					print ("WRONG UB")
					# set_trace()

		return pomdp_task, max_pomdp, max_action, max_Q, tree_size, max_time_step


	def hierarchical_agent_pomdp (self, horizon, pomdp_tasks, agent_pomdp, agent_pomdp_solver, no_op):
		global print_status
		Q, a, max_time, Q_a, tree_size, exp_time_steps,_ = agent_pomdp_solver.compute_V(None,horizon,self.max_horizon,one_action=False, \
			gamma=self.gamma,tree_s=1, all_poss_actions=not no_op, HPOMDP=True, LB_UB=False)
		Q = Q.UB; a = a.UB; max_time = max_time.UB; Q_a = Q_a.UB; 

		action = a[0]
		pomdp_solver = self.pomdp_solvers[action.id]
		max_pomdp = action.id
		pomdp_task = pomdp_tasks[max_pomdp]	
		max_time_step = max_time

		if print_status:
			print ("---")
			if not self.package:
				print ("robot: ", (self.robot.get_feature('x').value, self.robot.get_feature('y').value))
			else:
				belief_str = ""
				for t in range(0,len(self.pomdp_solvers)):
					belief_str += self.pomdp_solvers[t].get_readable_string()
				print ("beliefs: ", belief_str)

			print ("reward, selected pomdp: ",(Q, action.name))
			print ("-- Hierarchical Max Q --")
			print (Q_a)

		max_Q, max_a, max_time, Q_a, tree_s, exp_time_steps,_ = pomdp_solver.compute_V(pomdp_solver.belief,horizon,self.max_horizon,one_action=False, \
			gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
		action = max_a		
		max_time_step = exp_time_steps[max_a.id]

		if self.package:
			if max_a.id < self.pomdp_tasks[max_pomdp].non_navigation_actions_len:
				for i in range(self.agent_pomdp.actions.shape[0]):
					if self.agent_pomdp.actions[i,max_pomdp] == action:
						max_action = i
						break
			else:
				for i in range(self.agent_pomdp.actions.shape[0]):
					if self.agent_pomdp.actions[i,max_pomdp] == self.pomdp_tasks[0].actions[self.agent_pomdp.action_mapping[action.id]]:
						max_action = i
						break
		else:
			for i in range(self.agent_pomdp.actions.shape[0]):
				if self.agent_pomdp.actions[i,max_pomdp] == action:
					max_action = i
					break

		if print_status:
			print ("reward, selected action: ",(max_Q, max_a))
			print ("-- One task Max Q --")
			print (Q_a)
		
		if print_status:
			print ("time steps: ", max_time_step)
			print ("POMDP ", max_pomdp)
			print ("final selected action: ",action.id, " - ", action.name)
			print ("---")
			# set_trace()

		return pomdp_task, max_pomdp, max_action, max_Q, tree_size, max_time_step


	def solve_ind_pomdps (self, selected_pomdp_solvers, horizon, no_op):
		global print_status
		time_start = time.clock()	
		Vs = np.zeros((len(self.pomdp_solvers),1))
		all_actions_time_steps = []
		pomdp_Q = []
		tree_size = 0
		upper_bound = 0
		
		for i in selected_pomdp_solvers:
			pomdp_solver = self.pomdp_solvers[i]

			if print_status:
				if not self.package:
					print ("goal: ", (pomdp_solver.env.task.table.goal_x, pomdp_solver.env.task.table.goal_y))
					print ("initial belief: ", self.pomdp_tasks[i].get_state_tuple(pomdp_solver.belief.prob[0][1]))
					print ("robot: ", (self.robot.get_feature('x').value, self.robot.get_feature('y').value))
				else:
					print ("goal: ", (pomdp_solver.env.task.table.goal_x))
					print ("initial belief: ", pomdp_solver.get_readable_string())
			
			max_Q, max_a, max_time, Q_a, tree_s, exp_time_steps,_ = pomdp_solver.compute_V(pomdp_solver.belief,horizon,self.max_horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
			all_actions_time_steps.append(exp_time_steps)
			Vs[i] = max_Q
			tree_size += tree_s
			pomdp_Q.append(Q_a)
			upper_bound += max_Q
			if print_status:
				print ("reward, selected action: ",(max_Q,max_a), Q_a)
		time_end = time.clock()
		# print ("time: ",time_end-time_start)
		return upper_bound, tree_size, all_actions_time_steps, Vs, pomdp_Q

	def compute_lower_bound (self, selected_pomdp_solvers, Vs, V_traj, horizon, no_op):
		global print_status
		lower_bound = -np.Inf
		for i in selected_pomdp_solvers:
			min_Vs = 0
			min_V_str = ""
			for j in selected_pomdp_solvers:
				if i != j:
					# pomdp_solver = self.pomdp_solvers[j]
					# min_V,_,_,_,_ = pomdp_solver.compute_Q_one_action(pomdp_solver.belief,self.pomdp_tasks[j].no_action,horizon,self.max_horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
					min_V = V_traj[j]
					min_Vs += min_V
					min_V_str += str(min_V)
				else:
					min_Vs += Vs[i]
					min_V_str += str(Vs[i])
			lower_bound = max(lower_bound,min_Vs)
			if print_status:
				print ("itemized: ", min_V_str, min_Vs)

		lower_bound = lower_bound[0]

		return lower_bound

	def compute_max_Qs_greedily (self, selected_pomdp_solvers, pomdp_Q, V_traj, all_actions_time_steps, horizon, no_op):
		# selected pomdps are the ones that we want to sum
		global print_status
		# print_status = True
		max_Qs = np.zeros((self.agent_pomdp.all_actions.shape[0],1))
		max_time_steps = np.zeros((self.agent_pomdp.all_actions.shape[0],1))	
		count = 0
		tree_size = 0

		# if print_status:
		# 	print ("--- POMDPs: ", selected_pomdp_solvers) 

		for action in range(self.agent_pomdp.all_actions.shape[0]):
			if print_status:
				print_str = "action: " + str(action) + " "
			for i in range(self.agent_pomdp.all_actions.shape[1]):
				pomdp_solver = self.pomdp_solvers[i]
				if self.agent_pomdp.all_pomdp_actions[action] in selected_pomdp_solvers:
					if i in selected_pomdp_solvers:
						ind_act = self.agent_pomdp.all_actions[action,i]					
						if not no_op or (ind_act in self.pomdp_tasks[i].valid_actions):
							max_Qs[count] += pomdp_Q[i][ind_act.id][0]
							max_time_steps[count] =  all_actions_time_steps[i][ind_act.id]
							if print_status:
								print_str += str(pomdp_Q[i][ind_act.id][0]) + " "
						else:
							if not self.package:
								rew, tree_s, exp_time,_,_ = pomdp_solver.compute_Q (self.pomdp_solvers[i].belief, action=ind_act, horizon=horizon, max_horizon=self.max_horizon, one_action=False, gamma=self.gamma, tree_s=0,all_poss_actions=not no_op)
							else:
								rew, tree_s, exp_time,_,_ = pomdp_solver.compute_Q (self.pomdp_solvers[i].belief, action=ind_act, horizon=horizon, max_horizon=self.max_horizon, \
									one_action=False, gamma=self.gamma, tree_s=0,all_poss_actions=True)
							tree_size += tree_s
							max_Qs[count] += rew
							if print_status:
								print_str += str(rew) + " "

					else:
						# rew, tree_s, exp_time,_,_ = pomdp_solver.compute_Q_one_action(pomdp_solver.belief,self.pomdp_tasks[i].no_action,horizon,self.max_horizon, \
						# 	one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
						# tree_size += tree_s
						rew = V_traj[i]
						max_Qs[count] += rew
						if print_status:
							print_str += str(rew) + " "
				else:
					max_Qs[count] = -np.Inf
					if print_status:
						print_str += "-inf "

			# if print_status:
			# 	print (print_str)
				# if self.test_LB != np.Inf:
				# 	set_trace()
				# set_trace()

			count += 1

		return max_Qs, tree_size, max_time_steps

	def select_max_Q_pomdp (self, max_Qs, max_time_steps):
		max_action = np.argmax(max_Qs)
		max_time_step = max_time_steps[max_action][0]
		max_Q = np.max(max_Qs)
		max_pomdp, pomdp_action = self.agent_pomdp.get_pomdp(max_action, self.current_pomdp)
		pomdp_task = self.pomdp_tasks[max_pomdp]	

		return pomdp_task, max_pomdp, max_action, max_Q, max_time_step, pomdp_action


	def select_best_pomdps (self, max_Qs, max_time_steps):
		global print_status
		pomdp_task, max_pomdp, max_action, max_Q, max_time_step, pomdp_action = self.select_max_Q_pomdp (max_Qs, max_time_steps)

		if print_status:
			print ("-- Greedy Max Q --")
			print (max_Qs)

			print ("Selected POMDP ", max_pomdp)
			print ("selected action: ",pomdp_action, self.agent_pomdp.actions_names[max_action])
			print ("time steps: ", max_time_step)

		if self.greedy_hybrid:
			max_2nd_Qs = deepcopy(max_Qs)
			max_2nd_pomdp = max_pomdp
			max_2nd_action = max_action
			count = 0
			while max_pomdp == max_2nd_pomdp and count < len(self.agent_pomdp.actions)+1:
				max_2nd_Qs[max_2nd_action] = -np.Inf
				max_2nd_pomdp, max_2nd_pomdp, max_2nd_action, max_2nd_Q, max_time_step, pomdp_action_2nd = self.select_max_Q_pomdp (max_2nd_Qs, max_time_steps)
				# print ("max_2nd_action", max_2nd_action)
				if max_2nd_action == self.agent_pomdp.all_no_ops['1'][max_2nd_pomdp]:
					print ("max_2nd_pomdp = max_pomdp in pomdp_tasks_env")
					# set_trace()
					max_2nd_pomdp = max_pomdp
				count += 1

			if count == len(self.agent_pomdp.actions)+1:
				# set_trace()
				if max_pomdp == 0:
					max_2nd_pomdp = len(self.pomdp_tasks) - 1
				else:
					max_2nd_pomdp = max_pomdp - 1
			
			selected_pomdps = [max_pomdp,max_2nd_pomdp]


			if print_status:

				print ("---------------------------------")
				print ("Selected 2nd best POMDP ", max_2nd_pomdp)
				

			# if self.hybrid_num_tables == 3 and len(self.pomdp_tasks) >= 3:
			# 	max_3nd_Qs = deepcopy(max_2nd_Qs)
			# 	max_3nd_pomdp = max_2nd_pomdp
			# 	max_3nd_action = max_2nd_action
			# 	count = 0
			# 	while (max_pomdp == max_3nd_pomdp or max_2nd_pomdp == max_3nd_pomdp) and count < len(self.agent_pomdp.actions)+1:
			# 		max_3nd_Qs[max_3nd_action] = -np.Inf
			# 		_, max_3nd_pomdp, max_3nd_action, max_3nd_Qs, max_time_step, pomdp_action_3nd = self.select_max_Q_pomdp (max_3nd_Qs, max_time_steps)

			# 		if max_3nd_action == self.agent_pomdp.all_no_op:
			# 			max_3nd_pomdp = max_pomdp
			# 		count += 1

			# 	if count == len(self.agent_pomdp.actions)+1:
			# 		pomdps_indices = list(range(0,len(self.pomdp_tasks)))
			# 		pomdps_indices.remove(max_pomdp)
			# 		pomdps_indices.remove(max_2nd_pomdp)
			# 		max_3nd_pomdp = pomdps_indices[0]
			# 		# set_trace()

			# 	selected_pomdps.append(max_3nd_pomdp)

			# 	if print_status:
			# 		print ("---------------------------------")
			# 		print ("Selected 3rd best POMDP ", max_3nd_pomdp)

		return selected_pomdps

	def run_optimal_planner_on_k(self, selected_pomdps, horizon, extra_pomdps=None):
		sub_pomdp_tasks = [self.pomdp_tasks[i] for i in selected_pomdps]
		sub_pomdp_solvers = [self.pomdp_solvers[i] for i in selected_pomdps]
		sub_tasks = [self.tasks[i] for i in selected_pomdps]

		beliefs = []

		for solver in sub_pomdp_solvers:
			beliefs.append(solver.belief.prob)

		if print_status:
			print ("** selected POMDPS: ", selected_pomdps)
			if extra_pomdps is not None:
				if len(extra_pomdps) == 1:
					print ([extra_pomdps[0].env.task.table.id])
				if len(extra_pomdps) == 2:
					print ([extra_pomdps[0].env.task.table.id,extra_pomdps[1].env.task.table.id])

		self.current_pomdp = None
		if self.package:
			sub_agent_pomdp = AgentPOMDPPackage(sub_pomdp_tasks, sub_pomdp_solvers, sub_tasks, self.robot, self.random, extra_pomdps)
		else:
			sub_agent_pomdp = AgentPOMDPRestaurant(sub_pomdp_tasks, sub_pomdp_solvers, sub_tasks, self.robot, self.random, extra_pomdps)
		sub_agent_pomdp_solver = multi_ind_pomdp_solver.MultiIndPOMDPSolver(sub_agent_pomdp,beliefs,self.random)

		if print_status:
			print ("beliefs: ", beliefs)
			print ("======= running optimal planner on " + str(len(selected_pomdps)) + " POMDP =======")
		sub_pomdp_task, sub_max_pomdp, sub_action, sub_max_Q, sub_tree_size, sub_time_step = \
			self.optimal_agent_pomdp (horizon, sub_pomdp_tasks, sub_agent_pomdp, sub_agent_pomdp_solver)
		max_time_step = sub_time_step
		if print_status:
			print ("==================================================")
			print ("action: ", sub_action)
			print ("time steps: ", max_time_step)

		return sub_agent_pomdp, sub_pomdp_task, sub_max_pomdp, sub_action, sub_max_Q, sub_tree_size, sub_time_step


	def greedy_selection_of_individual_pomdps (self, horizon, no_op=False):
		global print_status
		tree_size = 0			
		
		# upper_bound, tree_s, all_actions_time_steps, Vs, pomdp_Q = self.solve_ind_pomdps (list(range(0,len(self.pomdp_solvers))), horizon, no_op)
		# lower_bound = self.compute_lower_bound(list(range(0,len(self.pomdp_solvers))), Vs, horizon, no_op)
		# tree_size += tree_s

		# max_Qs, tree_s, max_time_steps = self.compute_max_Qs_greedily (list(range(0,len(self.pomdp_solvers))), pomdp_Q, all_actions_time_steps, horizon, no_op)
		# tree_size += tree_s
		# max_Q = np.max(max_Qs)

		# rew, tree_s, exp_time,_,_ = pomdp_solver.compute_Q_one_action(pomdp_solver.belief,self.pomdp_tasks[i].no_action,horizon,self.max_horizon, \
						# 	one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
						# tree_size += tree_s
		pomdp_Q = None
		Vs = None
		all_actions_time_steps = None
		V_traj = np.full((len(self.pomdp_solvers),),-np.Inf)
		for j in range(0,len(self.pomdp_solvers)):
			pomdp_solver = self.pomdp_solvers[j]
			min_V,_,_,_,_ = pomdp_solver.compute_Q_one_action(pomdp_solver.belief,self.pomdp_tasks[j].noop_actions['1'],self.max_horizon,self.max_horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
			V_traj[j] = min_V


		if not self.greedy_hybrid:
			if print_status:
				print(" ----- greedy -----")
			lower_bound = -np.Inf
			upper_bound, tree_s, all_actions_time_steps, Vs, pomdp_Q = self.solve_ind_pomdps (list(range(0,len(self.pomdp_solvers))), self.max_horizon, no_op)
			tree_size += tree_s

			max_Qs, tree_s, max_time_steps = self.compute_max_Qs_greedily (list(range(0,len(self.pomdp_solvers))), pomdp_Q, V_traj, all_actions_time_steps, self.max_horizon, no_op)
			tree_size += tree_s
			max_Q = np.max(max_Qs)

			pomdp_task, max_pomdp, max_action, max_Q, max_time_step, pomdp_action = self.select_max_Q_pomdp (max_Qs, max_time_steps)
			
		elif self.greedy_hybrid:

			if self.shani_baseline:
				if print_status:
					print(" ----- shani_baseline -----")

				pomdp_pairs = []
				upper_bound = np.Inf
				lower_bound = -np.Inf
				selected_pomdp_pairs = []
				
				if self.hybrid_4T:
					if len(self.pomdp_tasks) == 2:
						selected_pomdp_pairs.append((0,1,None))
					elif len(self.pomdp_tasks) == 3:
						selected_pomdp_pairs.append((0,1,2,None))
					elif len(self.pomdp_tasks) == 4:
						selected_pomdp_pairs.append((0,1,2,3,None))
					elif len(self.pomdp_tasks) > 4:
						pair = None
						for i in range(len(self.pomdp_solvers)):
							count = 0
							while (pair is None or pair in pomdp_pairs or len(pair) < 4):
								j = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								k = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								l = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								pair_invs = []
								pair = set((i,j,k,l))

								count += 1

								if count > len(self.pomdp_solvers)*4:
									pair = None
									break

							if pair is not None:
								pomdp_pairs.append(pair)

						if len(pomdp_pairs) != len(self.pomdp_solvers):
							while len(pomdp_pairs) != len(self.pomdp_solvers):
								i = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								j = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								k = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								l = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								pair = set((i,j,k,l))

								if not (pair is None or pair in pomdp_pairs or len(pair) < 4):
									pomdp_pairs.append(pair)

						for pp in pomdp_pairs:
							p = list(pp)
							selected_pomdp_pairs.append((p[0],p[1],p[2],p[3],None))
					else:
						print ("less than 4 tasks!  in pomdp_tasks_env")
						# set_trace()
					

				elif self.hybrid_3T: ## and len(self.pomdp_tasks) >= 3:
					if len(self.pomdp_tasks) == 2:
						selected_pomdp_pairs.append((0,1,None))
					elif len(self.pomdp_tasks) == 3:
						selected_pomdp_pairs.append((0,1,2,None))
					elif len(self.pomdp_tasks) > 3:
						pair = None
						for i in range(len(self.pomdp_solvers)):
							count = 0
							while (pair is None or pair in pomdp_pairs or len(pair) < 3):
								j = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								k = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								pair = set((i,j,k))

								count += 1
								# print (pair, pomdp_pairs)
								if count > len(self.pomdp_solvers)*3:
									pair = None
									break

							if pair is not None:
								pomdp_pairs.append(pair)

						if len(pomdp_pairs) != len(self.pomdp_solvers):
							while len(pomdp_pairs) != len(self.pomdp_solvers):
								i = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								j = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								k = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								pair = set((i,j,k))

								if not (pair is None or pair in pomdp_pairs or len(pair) < 3):
									pomdp_pairs.append(pair)
									# print (pomdp_pairs)
									# set_trace()		
						for pp in pomdp_pairs:
							p = list(pp)
							selected_pomdp_pairs.append((p[0],p[1],p[2],None))
					else:
						print ("less than 3 tasks! in pomdp_tasks_env")
						# set_trace()
					
				else:
					pair = None
					if len(self.pomdp_tasks) != 2:
						for i in range(len(self.pomdp_solvers)):
							count = 0
							while (pair is None or pair in pomdp_pairs or len(pair) < 2):
								j = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								pair = set((i,j))
								count += 1
								if count > len(self.pomdp_solvers)*2:
									pair = None
									break

							if pair is not None:
								pomdp_pairs.append(pair)

						if len(pomdp_pairs) != len(self.pomdp_solvers):
							while len(pomdp_pairs) != len(self.pomdp_solvers):
								i = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								j = self.random.randint(0,len(self.pomdp_solvers),size=1)[0]
								pair = set((i,j))

								if not (pair is None or pair in pomdp_pairs or len(pair) < 2):
									pomdp_pairs.append(pair)

						for pp in pomdp_pairs:
							p = list(pp)
							selected_pomdp_pairs.append((p[0],p[1],None))

					else:
						selected_pomdp_pairs.append((0,1,None))


				pomdp_pairs = selected_pomdp_pairs

				if print_status:
					print (pomdp_pairs)
				# print ( pomdp_pairs)
				# set_trace()

			else:
				if print_status:
					print(" ----- hybrid_optimal -----")
					
				pomdp_pairs = []

				## compute upper and lower bounds
				correct_size_tuples = self.get_correct_size_tuples(self.max_horizon)
				if self.hybrid_4T or (self.MIX and correct_size_tuples == 4):
					if len(self.pomdp_tasks) == 2:
						upper_bound = np.Inf
						lower_bound = -np.Inf
						pomdp_pairs.append((0,1,None))
					elif len(self.pomdp_tasks) == 3:
						upper_bound = np.Inf
						lower_bound = -np.Inf
						pomdp_pairs.append((0,1,2,None))
					elif len(self.pomdp_tasks) == 4:
						upper_bound = np.Inf
						lower_bound = -np.Inf
						pomdp_pairs.append((0,1,2,3,None))
					elif len(self.pomdp_tasks) > 4:
						upper_bound, tree_s, all_actions_time_steps, Vs, pomdp_Q = self.solve_ind_pomdps (list(range(0,len(self.pomdp_solvers))), self.max_horizon, no_op)
						lower_bound = self.compute_lower_bound(list(range(0,len(self.pomdp_solvers))), Vs, V_traj, self.max_horizon, no_op)
						tree_size += tree_s

						# upper_bound = min(np.max(max_Qs),upper_bound)						
						max_pomdp_pairs = -np.Inf
						max_pomdp_pairs_index = None
						for i in range(len(self.pomdp_solvers)):
							for j in range(len(self.pomdp_solvers)):
								if j > i:
									for k in range(len(self.pomdp_solvers)):
										if k > j:
											for l in range(len(self.pomdp_solvers)):
												if l > k:
													max_Qs_4_pomdps, tree_s_temp, max_time_steps_temp = self.compute_max_Qs_greedily ([i,j,k,l], pomdp_Q, V_traj, all_actions_time_steps, self.max_horizon, no_op)
													if print_status:
														print ("POMDP ", i, j, k, l, ":", np.max(max_Qs_4_pomdps), " ---- ", lower_bound)
													max_Q_4_pomdps = np.max(max_Qs_4_pomdps)
													if max_Q_4_pomdps - lower_bound > self.precision:
														pomdp_pairs.append((i,j,k,l,max_Q_4_pomdps))
														max_pomdp_pairs = max(max_Q_4_pomdps,max_pomdp_pairs)
														if max_pomdp_pairs == max_Q_4_pomdps:
															max_pomdp_pairs_index = len(pomdp_pairs)-1

						if print_status:
							print (pomdp_pairs, max_pomdp_pairs_index, max_pomdp_pairs, upper_bound)

						upper_bound = min(max_pomdp_pairs,upper_bound)

						if max_pomdp_pairs_index is None:
							print ("max_pomdp_pairs_index is None in pomdp_tasks_env")
							# set_trace()

					else:
						print ("less than 3 tasks!  in pomdp_tasks_env")
						# set_trace()
				elif self.hybrid_3T or (self.MIX and correct_size_tuples == 3): ## and len(self.pomdp_tasks) >= 3:
					if len(self.pomdp_tasks) == 2:
						upper_bound = np.Inf
						lower_bound = -np.Inf
						pomdp_pairs.append((0,1,None))
					elif len(self.pomdp_tasks) == 3:
						upper_bound = np.Inf
						lower_bound = -np.Inf
						pomdp_pairs.append((0,1,2,None))
					elif len(self.pomdp_tasks) > 3:
						upper_bound, tree_s, all_actions_time_steps, Vs, pomdp_Q = self.solve_ind_pomdps (list(range(0,len(self.pomdp_solvers))), self.max_horizon, no_op)
						lower_bound = self.compute_lower_bound(list(range(0,len(self.pomdp_solvers))), Vs, V_traj, self.max_horizon, no_op)
						tree_size += tree_s

						# upper_bound = min(np.max(max_Qs),upper_bound)						
						max_pomdp_pairs = -np.Inf
						max_pomdp_pairs_index = None
						for i in range(len(self.pomdp_solvers)):
							for j in range(len(self.pomdp_solvers)):
								if j > i:
									for k in range(len(self.pomdp_solvers)):
										if k > j:
											max_Qs_3_pomdps, tree_s_temp, max_time_steps_temp = self.compute_max_Qs_greedily ([i,j,k], pomdp_Q, V_traj, all_actions_time_steps, self.max_horizon, no_op)
											if print_status:
												print ("POMDP ", i, j, k, ":", np.max(max_Qs_3_pomdps), " ---- ", lower_bound)
											max_Q_3_pomdps = np.max(max_Qs_3_pomdps)
											if max_Q_3_pomdps - lower_bound > self.precision:
												pomdp_pairs.append((i,j,k,max_Q_3_pomdps))
												max_pomdp_pairs = max(max_Q_3_pomdps,max_pomdp_pairs)
												if max_pomdp_pairs == max_Q_3_pomdps:
													max_pomdp_pairs_index = len(pomdp_pairs)-1

						if print_status:
							print (pomdp_pairs, max_pomdp_pairs_index, max_pomdp_pairs, upper_bound)

						upper_bound = min(max_pomdp_pairs,upper_bound)

						if max_pomdp_pairs_index is None:
							print ("max_pomdp_pairs_index is None in pomdp_tasks_env")
							# set_trace()

					else:
						print ("less than 3 tasks!  in pomdp_tasks_env")
						# set_trace()
				else:
					if len(self.pomdp_tasks) != 2:
						upper_bound, tree_s, all_actions_time_steps, Vs, pomdp_Q = self.solve_ind_pomdps (list(range(0,len(self.pomdp_solvers))), self.max_horizon, no_op)
						lower_bound = self.compute_lower_bound(list(range(0,len(self.pomdp_solvers))), Vs, V_traj, self.max_horizon, no_op)
						tree_size += tree_s

						# upper_bound = min(np.max(max_Qs),upper_bound)						
						max_pomdp_pairs = -np.Inf
						max_pomdp_pairs_index = None
						for i in range(len(self.pomdp_solvers)):
							for j in range(len(self.pomdp_solvers)):
								if j > i:
									max_Qs_2_pomdps, tree_s_temp, max_time_steps_temp = self.compute_max_Qs_greedily ([i,j], pomdp_Q, V_traj, all_actions_time_steps, self.max_horizon, no_op)
									if print_status:
										print ("POMDP ", i, j, ":", np.max(max_Qs_2_pomdps), " ---- ", lower_bound)
									max_Q_2_pomdps = np.max(max_Qs_2_pomdps)
									if max_Q_2_pomdps - lower_bound > self.precision:
										pomdp_pairs.append((i,j,max_Q_2_pomdps))
										max_pomdp_pairs = max(max_Q_2_pomdps,max_pomdp_pairs)
										if max_pomdp_pairs == max_Q_2_pomdps:
											max_pomdp_pairs_index = len(pomdp_pairs)-1

						if print_status:
							print (pomdp_pairs, max_pomdp_pairs_index, max_pomdp_pairs, upper_bound)

						upper_bound = min(max_pomdp_pairs,upper_bound)

						if max_pomdp_pairs_index is None:
							print ("max_pomdp_pairs_index is None in pomdp_tasks_env")
							# set_trace()

					else:
						upper_bound = np.Inf
						lower_bound = -np.Inf
						pomdp_pairs.append((0,1,None))


					
					if print_status:
						print ("new UB: ",upper_bound)
						print ("new LB: ",lower_bound)
						self.UB = upper_bound
						self.LB = lower_bound
						self.test_LB = lower_bound

					selected_pomdps = []					

				
			self.LB = lower_bound
			self.UB = upper_bound

			pomdp_pairs_w_bounds = []
			for pp in pomdp_pairs:
				new_tpl = list(pp)
				pomdp_pairs_w_bounds.append(POMDPTuple(new_tpl[0:len(new_tpl)-1],lower_bound,upper_bound))

			pomdp_task, max_pomdp, max_action, max_Q, final_sub_tree_size, max_time_step, final_horizon, num_pairs = self.select_best_k_tuple(pomdp_pairs_w_bounds, horizon, no_op, tree_size, pomdp_Q, Vs, V_traj, all_actions_time_steps)
			tree_size += final_sub_tree_size

			if print_status or print_status_short:			
				print ("---------------------------------")
				print ("final UB: ",self.UB)
				print ("final LB: ",self.LB)
				print ("final UB - LB: ", self.UB-self.LB)
				# set_trace()
				
				if (np.round(upper_bound,3)-np.round(lower_bound,3)) < 0:
					print ("(np.round(upper_bound,3)-np.round(lower_bound,3)) < 0 in pomdp_tasks_env")
					# set_trace()

		# print ("max_action", max_action)
		return pomdp_task, max_pomdp, max_action, max_Q, tree_size, max_time_step, final_horizon, self.UB-self.LB, num_pairs
		

	def valid_termination(self, size_tuples, correct_size_tuples):
		if self.UB is None or self.LB is None:
			return False
		elif self.UB - self.LB < -self.precision:
			if len(self.pomdp_tasks) == size_tuples:
				return True
			elif self.hybrid_3T and size_tuples == 3:
				return True
			elif self.hybrid_4T and size_tuples == 4:
				return True
			elif not (self.three_tables or self.four_tables):
				return True			
			elif size_tuples == correct_size_tuples:
				return True
			elif self.MIX:
				return True
			else:
				return False
		else:
			return False

	def get_correct_size_tuples(self, horizon):
		if horizon <= 4:
			correct_size_tuples = 2
		elif horizon > 4 and horizon <= 6:
			correct_size_tuples = min(3,len(self.pomdp_tasks))
		else:
			correct_size_tuples = min(4,len(self.pomdp_tasks))
		return correct_size_tuples

	def select_best_k_tuple(self, selected_pomdp_pairs, horizon, no_op, tree_size, pomdp_Q, Vs, V_traj, all_actions_time_steps):
		# print_status = True
		num_pairs = 0
		increasing_horizon = horizon
		self.UB = None
		self.LB = None
		if self.hybrid_4T:
			size_tuples = 4
		elif self.hybrid_3T:
			size_tuples = 3
		else:
			size_tuples = 2

		# set_trace()
		# for pa in selected_pomdp_pairs:
		# 	print (pa.tuple, pa.extra_pomdps)
		correct_size_tuples = self.get_correct_size_tuples(self.max_horizon)
		if self.MIX:
			selected_pomdp_pairs = self.initialize_tuples(selected_pomdp_pairs, increasing_horizon, correct_size_tuples)
		# for pa in selected_pomdp_pairs:
		# 	print (pa.tuple, pa.extra_pomdps)
		# set_trace()
		# switched = False
		while increasing_horizon <= self.max_horizon and not self.valid_termination(size_tuples, correct_size_tuples):
			final_sub_pomdp_task = None
			final_sub_max_pomdp = None
			final_sub_action = None
			final_sub_max_Q = None
			final_sub_tree_size = None
			final_sub_time_step = None
			final_selected_pomdps = None
			final_sub_agent_pomdp = None
			# set_trace()
			UB = None
			LB = None
			to_be_removed = []
			for pp in selected_pomdp_pairs:
				pair = pp.tuple
				selected_pomdps = []
				for p in range(0,len(pair)):
					selected_pomdps.append(pair[p])

				extra_pomdps = None
				if pp.extra_pomdps is not None:
					extra_pomdps = []
					for e in pp.extra_pomdps:
						extra_pomdps.append(self.pomdp_solvers[e])

				## agent POMDP solution
				sub_agent_pomdp, sub_pomdp_task, sub_max_pomdp, sub_action, sub_max_Q, sub_tree_size, sub_time_step = self.run_optimal_planner_on_k(selected_pomdps, increasing_horizon, extra_pomdps)

				num_pairs += 1
				pp.UB_tpl = sub_max_Q.UB
				pp.LB_tpl = sub_max_Q.LB
				pp.UB = sub_max_Q.UB
				pp.LB = sub_max_Q.LB

				## adding with noops
				sub_max_Q_noop = 0
				for o in range(len(self.pomdp_solvers)):
					if o not in selected_pomdps and (extra_pomdps is None or o not in pp.extra_pomdps):
						sub_max_Q_noop += V_traj[o]

				pp.UB += sub_max_Q_noop
				pp.LB += sub_max_Q_noop

				# if len(selected_pomdps) == 3 and pp.LB_tpl - sub_max_Q.LB < self.precision:
				# 	set_trace()

				if print_status:
					print ("considered pairs: ", pair,pp.UB)

				if LB is None:
					LB = pp.LB
				else:
					LB = max(pp.LB,LB)

				if UB is None:
					UB = pp.UB
				else:
					UB = max(pp.UB,UB)

				# print ("considered pairs: ", pair, sub_max_Q.LB, sub_max_Q.UB, pp.LB,pp.UB, sub_max_Q_noop)
				if print_status:
					print ("pp.UB: ",pp.UB)
					print ("pp.LB: ",pp.LB)
					print ("UB: ",UB)
					print ("LB: ",LB)
					print ("UB - LB: ", UB-LB)

				if final_sub_max_Q is None or final_sub_max_Q < pp.UB:
					final_sub_pomdp_task = sub_pomdp_task
					final_sub_max_pomdp = sub_max_pomdp
					final_sub_action = sub_action
					final_sub_max_Q = pp.UB
					final_sub_tree_size = sub_tree_size
					final_sub_time_step = sub_time_step
					final_selected_pomdps = selected_pomdps
					final_sub_agent_pomdp = sub_agent_pomdp
			

			if (UB is None or LB is None):
				print ("num_tables", len(self.agent_pomdp.pomdp_tasks),"horizon", self.horizon, self.max_horizon,"UB or LB is None? ", UB, LB)
				# set_trace()

			if (self.UB is not None and self.LB is not None) and (self.UB - UB < self.precision or self.LB - LB > -self.precision):
				print ("num_tables", len(self.agent_pomdp.pomdp_tasks),"horizon", self.horizon, self.max_horizon, " step: ", self.step)
				print ("new UB, LB ", UB, LB)
				print ("prev UB, LB ", self.UB, self.LB)
				print ("self.UB < UB: ", self.UB < UB, " self.LB > LB: ", self.LB > LB)
				# set_trace()

			self.UB = UB
			self.LB = LB
			if print_status:
				if self.test_LB is not None and self.LB is not None and self.test_LB - self.LB > -self.precision:
					# set_trace()
					print ("num_tables", len(self.agent_pomdp.pomdp_tasks),"horizon", self.horizon, self.max_horizon, " step: ", self.step)
					pass


			if print_status:
				print ("-- selected pomdps: ",final_selected_pomdps, "max_Q: ",final_sub_max_Q)
				print ("considered pairs: ", pair,sub_max_Q)
				print ("---------------------------------")
				print ("new UB: ",self.UB)
				print ("new LB: ",self.LB)
				print ("new UB - LB: ", self.UB-self.LB)
				print ("length pomdp pairs: ", len(selected_pomdp_pairs))
				print ("current horizon: ", increasing_horizon)
				# sleep(3)
				# set_trace()

			if len(selected_pomdp_pairs) > 0:
				pomdp_pairs = []
				for pp in selected_pomdp_pairs:
					if pp.UB - self.LB > self.precision:
						pomdp_pairs.append(pp)

				selected_pomdp_pairs = pomdp_pairs

			increasing_horizon += 1

			###### move to 3 tuples
			# switched = False
			if self.MIX and size_tuples != self.get_correct_size_tuples(increasing_horizon):
				if print_status:
					print ("switch from ", size_tuples, " to ", self.get_correct_size_tuples(increasing_horizon))
				size_tuples = self.get_correct_size_tuples(increasing_horizon)
				
				selected_pomdp_pairs = self.extend_tuples(selected_pomdp_pairs, no_op, tree_size, pomdp_Q, Vs, V_traj, all_actions_time_steps)
				# switched = True

		increasing_horizon -= 1
		max_Q = final_sub_max_Q
		pomdp_task = final_sub_pomdp_task
		max_time_step = final_sub_time_step
		max_pomdp = final_selected_pomdps[final_sub_max_pomdp]
		
		max_action = final_sub_agent_pomdp.actions[final_sub_action,final_sub_max_pomdp]
		if not self.package:
			if max_action.type:			
				action = np.array(self.agent_pomdp.all_no_ops[str(max_action.time_steps)])	
				action[max_pomdp] = max_action
			else:
				action = np.full((len(self.agent_pomdp.pomdp_tasks),),max_action)			

			for i in range(self.agent_pomdp.actions.shape[0]):
				if np.array_equal(self.agent_pomdp.actions[i],action):
					max_action = i
					break

		else:
			if max_action.id < self.pomdp_tasks[max_pomdp].non_navigation_actions_len:
				action = np.array(self.agent_pomdp.all_no_ops[str(max_action.time_steps)])	
				action[max_pomdp] = max_action

				for i in range(self.agent_pomdp.actions.shape[0]):
					if np.array_equal(self.agent_pomdp.actions[i],action):
						max_action = i
						break

			else:
				action = np.full((len(self.agent_pomdp.pomdp_tasks),), self.pomdp_tasks[0].actions[self.agent_pomdp.action_mapping[max_action.id]])

				for i in range(self.agent_pomdp.actions.shape[0]):
					if np.array_equal(self.agent_pomdp.actions[i],action):
						max_action = i
						break
		
		if print_status or print_status_short:
			# set_trace()
			print ("diff: ",self.UB-self.LB)
			print ("Sub POMDP ", final_selected_pomdps[final_sub_max_pomdp])
			print ("Sub final selected action: ", final_sub_agent_pomdp.actions[final_sub_action,final_sub_max_pomdp].id, \
				final_sub_agent_pomdp.actions[final_sub_action,final_sub_max_pomdp].name)
			# set_trace()

		return pomdp_task, max_pomdp, max_action, max_Q, final_sub_tree_size, max_time_step, increasing_horizon, num_pairs
	
	def initialize_tuples (self, pairs, increasing_horizon, correct_size_tuples):
		# if len(pairs[0].tuple) == correct_size_tuples:
		# 	return pairs

		if print_status:
			for pa in pairs:
				print (pa.tuple, pa.extra_pomdps, pa.LB, pa.UB)

		pomdp_pairs = []
		for pa in pairs:
			if correct_size_tuples == 4:			
				for i in range(0,correct_size_tuples):
					for j in range(0,correct_size_tuples): 
						if j > i:
							for k in range(0,correct_size_tuples):
								if k != i and k != j:
									for l in range(0,correct_size_tuples):
										if l > k and l != i and l != j:
											new_pa = POMDPTuple([pa.tuple[i],pa.tuple[j]], pa.LB, pa.UB, [pa.tuple[k],pa.tuple[l]])
											pomdp_pairs.append(new_pa)
			elif correct_size_tuples == 3:
				for i in range(0,correct_size_tuples):
					for j in range(0,correct_size_tuples): 
						if j > i:
							for k in range(0,correct_size_tuples):
								if k != i and k != j:
									new_pa = POMDPTuple([pa.tuple[i],pa.tuple[j]], pa.LB, pa.UB, [pa.tuple[k]])
									pomdp_pairs.append(new_pa)
			elif correct_size_tuples == 2:
				pomdp_pairs = pairs
						
		if print_status:
			for pa in pomdp_pairs:
				print (pa.tuple, pa.extra_pomdps, pa.LB, pa.UB)
			# set_trace()
		return pomdp_pairs

	def extend_tuples (self, pairs, no_op, tree_size, pomdp_Q, Vs, V_traj, all_actions_time_steps):
		# set_trace()
		if pairs[0].extra_pomdps == None or len(pairs[0].extra_pomdps) == 0:
			return pairs

		if print_status:
			for pa in pairs:
				print (pa.tuple, pa.extra_pomdps, pa.LB, pa.UB)

		ppps = {}
		pomdp_pairs = []
		tuples_extra_mappings = {}
		for pa in pairs:
			for e in pa.extra_pomdps:
				pmdps = pa.tuple + [e]
				pmdps.sort()

				extra_pmdps = deepcopy(pa.extra_pomdps)
				extra_pmdps.remove(e)
				extra_pmdps.sort()

				## compute the UB of the the pair and the extra POMDP
				UB_tpl = pa.UB
				LB_tpl = pa.LB
				if print_status:
					print ("pmdps: ", (tuple(pmdps),tuple(extra_pmdps)), "LB: ", LB_tpl, "UB: ", UB_tpl)

				ttt = (tuple(pmdps),tuple(extra_pmdps))
				if ttt not in ppps:
					ppps[ttt] = (LB_tpl,UB_tpl)
				else:
					# set_trace()
					ppps[ttt] = (max(LB_tpl,ppps[ttt][0]),max(UB_tpl,ppps[ttt][1]))

				if print_status:
					print ("POMDP ", pmdps, extra_pmdps, ":", " ---- ", self.LB, LB_tpl, " ---- ", UB_tpl)		
					print ("diff: ", UB_tpl - self.LB)

				
		UB = None
		LB = None
		for pp in ppps.keys():
			UB_V = ppps[pp][1]
			LB_V = ppps[pp][0]
			if LB is None:
				LB = LB_V
			else:
				LB = max(LB_V,LB)

			if UB is None:
				UB = UB_V
			else:
				UB = max(UB_V,UB)

		if self.LB - LB > -self.precision:
			print ("LBs: ", self.LB,LB)
			# set_trace()

		self.UB = UB
		self.LB = LB
		if print_status:
			print ("LB, UB ---- ", self.LB, self.UB)
			for pp in ppps.keys():
				print (pp,ppps[pp])

		for pp in ppps.keys():
			UB_V = ppps[pp][1]
			LB_V = ppps[pp][0]
			if UB_V - self.LB > self.precision:
				if len(list(pp[1])) != 0:
					pomdp_pairs.append(POMDPTuple(list(pp[0]),LB_V,UB_V,list(pp[1])))
				else:
					# set_trace()
					pomdp_pairs.append(POMDPTuple(list(pp[0]),LB_V,UB_V,None))
				if print_status:
					print (pomdp_pairs)

		return pomdp_pairs


	def render(self, pomdp_tasks,actions_names, num=None, final_rew=None, exec_num=0, render_belief=False, horizon=None):
		# return
		if render_belief:
			self.render_beliefs(pomdp_tasks,actions_names, num, final_rew, exec_num)
		else:
			# return
			render_trace = False
			plt.close()
			# plt.clf()
			margin = 0.5
			x_low = self.robot.get_feature("x").low - margin
			x_high = self.robot.get_feature("x").high + margin
			y_low = self.robot.get_feature("y").low - margin
			y_high = self.robot.get_feature("y").high + margin
			coords = [x_low,x_high,y_low,y_high]
			drawTables(coords, self.restaurant.tables,pomdp_tasks,self.pomdp_solvers,actions_names,final_rew,horizon)
			drawRobot(self.restaurant.tables, self.robot, 0.5)
			# draw_POMDP_Tasks(pomdp_tasks)


			plt.axis('equal')
			plt.axis(coords)

			# drawPath(self.poses, self.counter)
			if not self.save_example:
				plt.tight_layout()
				plt.gcf().canvas.set_window_title('example #: ' + str(exec_num))
				plt.show(block=False)
				plt.pause(0.5)
			else:
				plt.tight_layout()
				t = str(num)
				t.rjust(2, '0')
				dirct = self.test_folder + self.model_folder + '/simple-' + str(self.simple) + '-horizon-' + str(self.horizon) + '-Hmax-' + str(self.max_horizon) + '-tables-' + str(len(self.tasks)) + '_example_' + str(self.example_num) + '/greedy-' + str(self.greedy) + '/'
				if not os.path.exists(dirct):
					os.makedirs(dirct)
				plt.savefig(dirct + str(exec_num) + "_" + t + '-greedy-' + str(self.greedy) + ".png",bbox_inches="tight")
				# set_trace()
				plt.close()
				# set_trace()

			if render_trace:
				set_trace() 
				render_trace = False
			## plt.close()

			# if self.finish_render:
			#     plt.show(block=False)
			#     plt.pause(0.0000000001)

			# if close:
			#     plt.close()

	def render_beliefs(self, pomdp_tasks,actions_names, num=None, final_rew=None, exec_num=0):
		render_trace = False
		plt.clf()
		plt.close()
		margin = 0.5
		x_low = self.robot.get_feature("x").low - margin
		x_high = self.robot.get_feature("x").high + margin
		y_low = self.robot.get_feature("y").low - margin
		y_high = self.robot.get_feature("y").high + margin
		coords = [x_low,x_high,y_low,y_high]
		drawTables_beliefs(coords, self.restaurant.tables,pomdp_tasks, actions_names,self.pomdp_solvers,final_rew)
		drawRobot_beliefs(self.robot, self.restaurant.tables, actions_names)
		# draw_POMDP_Tasks(pomdp_tasks)


		# plt.axis('equal')
		# plt.axis(coords)

		# drawPath(self.poses, self.counter)
		if not self.save_example:
			plt.tight_layout()
			plt.gcf().canvas.set_window_title('example #: ' + str(exec_num))
			plt.show(block=False)
			plt.pause(0.5)
		else:
			plt.tight_layout()
			t = str(num)
			t.rjust(2, '0')
			dirct = '../tests/' + self.model_folder + '/simple-' + str(self.simple) + '-horizon-' + str(self.horizon) + '-tables-' + str(len(self.tasks)) + '_example_' + str(self.example_num) + '/greedy-' + str(self.greedy) + '/'
			if not os.path.exists(dirct):
				os.makedirs(dirct)
			plt.savefig(dirct + str(exec_num) + "_" + t + '-greedy-' + str(self.greedy) + "_belief.png")

		if render_trace:
			set_trace() 
			render_trace = False
		## plt.close()

		# if self.finish_render:
		#     plt.show(block=False)
		#     plt.pause(0.0000000001)

		# if close:
		#     plt.close()





