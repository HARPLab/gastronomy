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
# from pomdp_solver import *
from goal_pomdp_solver import *
from pomdp_client_complex import *
from pomdp_client_simple import *
from pomdp_agent import *
# from multi_ind_pomdp_solver import *
from multi_ind_goal_pomdp_solver import *


print_status = True
print_status_short = False
PAUSE_TIME = 5


class POMDPTasks():
	metadata = {'render.modes': ['human']}

	def __init__(self, restaurant, tasks, robot, seed, random, reset_random, horizon, greedy, simple, model, no_op, run_on_cobot, hybrid, deterministic, hybrid_3T, shani_baseline, hierarchical_baseline):
		global print_status
		# print("seed: ", random.get_state()[1][0])
		horizon = 300
		self.tasks =[tasks[0]]
		# self.test_folder = '../tests_new_env/'
		self.test_folder = '../tests/'
		self.run_on_cobot = run_on_cobot
		# self.run_on_cobot = True

		if self.run_on_cobot:
			import rospy
			rospy.init_node('task', anonymous=True)
		
		self.gamma = 0.99
		self.seed = seed
		self.restaurant = restaurant
		self.random = random
		
		# self.tasks = tasks
		self.robot = robot
		self.num_random_executions = 10 # was 100
		print ("num executions: ", self.num_random_executions)
		random_initial_state = True
		self.max_steps = 41 # was 21
		self.LB = None
		self.UB = None
		just_plot_graphs = False
		self.render_belief = False
		self.no_op = no_op
		self.greedy_hybrid = hybrid
		self.hybrid_3T = hybrid_3T
		self.shani_baseline = shani_baseline
		self.hierarchical_baseline = hierarchical_baseline
		self.mismatch = False

		if self.hierarchical_baseline:
			self.no_op = True
			self.greedy_hybrid = True

		self.agentpomdp_vs_greedy = False
		self.hybrid_vs_greedy = False
		self.model_folder = model + "_model"
		self.save_example = True
		self.no_execution = False
		if self.save_example:
			self.no_execution = False
			self.num_random_executions = 1
			self.exec_list = [0]##list(range(0,self.num_random_executions))
			# self.exec_list = [6] ## execution
			# self.exec_list = list(range(0,self.num_random_executions))

		self.example_num = 0

		self.name = "pomdp_tasks"
		self.final_total_reward = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_belief_reward = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_max_Q = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_satisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_satisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_total_unsatisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_unsatisfaction = np.zeros((self.num_random_executions,self.max_steps))
		self.final_num_steps = np.zeros(self.num_random_executions)
		self.planning_time = np.zeros((self.num_random_executions,self.max_steps))
		self.tree_sizes = np.zeros((self.num_random_executions,self.max_steps))
		self.horizon = horizon
		self.simple = simple

		if self.agentpomdp_vs_greedy or self.hybrid_vs_greedy:
			self.final_total_reward_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_belief_reward_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_max_Q_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_satisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_satisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_total_unsatisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.final_unsatisfaction_other = np.zeros((self.num_random_executions,self.max_steps))
			self.tree_sizes_other = np.zeros((self.num_random_executions,self.max_steps))
			self.planning_time_other = np.zeros((self.num_random_executions,self.max_steps))

		if not just_plot_graphs:
			for random_exec in range(self.num_random_executions):

				self.pomdp_tasks = []
				self.pomdp_solvers = []
				count = 0

				navigation_goals = []

				for task in self.tasks:
					navigation_goals.append((task.table.goal_x,task.table.goal_y)) 
				
				beliefs = []
				for task in self.tasks:
					if simple:
						pomdp_task = ClientPOMDPSimple(task, robot, navigation_goals, self.gamma,random, reset_random, deterministic, self.no_op, self.run_on_cobot)
					else:
						pomdp_task = ClientPOMDPComplex(task, robot, navigation_goals, self.gamma,random, reset_random, deterministic, self.no_op, self.run_on_cobot)

					self.pomdp_tasks.append(pomdp_task)

					if random_initial_state:
						start_state2 = pomdp_task.reset(random=True)
					else:
						start_state2 = pomdp_task.reset(random=False)
					
					print ("Table ", count, " start state: ", start_state2)
					pomdp_task.state = pomdp_task.get_state_index(start_state2)

					initial_belief = []
					# uniform_belief = pomdp_task.get_uniform_belief(start_state2)
					# initial_belief = uniform_belief
					initial_belief.append((1.0,pomdp_task.state))

					beliefs.append(initial_belief)

					self.pomdp_solvers.append(GoalPOMDPSolver(pomdp_task,initial_belief,random))
					count += 1


				## CODE
				self.greedy = greedy
				self.multi_ind_pomdp = True
				print ("multiple ind POMDP: ", self.multi_ind_pomdp)
				self.agent_pomdp = AgentPOMDP(self.pomdp_tasks, self.pomdp_solvers, self.tasks, self.robot, random)
				
				self.agent_pomdp_solver = MultiIndGoalPOMDPSolver(self.agent_pomdp,beliefs,random)

				# while True:
				# 	min_Q, min_a, min_time, Q_a, tree_size, time_steps, _ = self.pomdp_solvers[0].run_RTDP_bel(self.pomdp_solvers[0].belief,300,300,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
				# 	solver_prob = self.pomdp_solvers[0].belief.prob
				# 	state_str = "---------------- initial belief POMDP " + str(0) + ": " + str(self.pomdp_tasks[0].get_state_tuple(self.pomdp_solvers[0].belief.prob[0][1]))
				# 	for p in solver_prob:
				# 		st = self.pomdp_tasks[0].get_state_tuple(p[1])			
				# 		state_str += ", " + str(st[self.pomdp_tasks[0].feature_indices['customer_satisfaction']]) + "-" + \
				# 		str(st[self.pomdp_tasks[0].feature_indices['current_request']]) + ":" + str(round(p[0],2))
				# 	print (state_str)

				# 	new_state_index, obs, reward, terminal, debug_info, position = self.pomdp_tasks[0].step(min_a,start_state=None,simulate=False, robot=self.run_on_cobot, selected_pomdp=True)
				# 	self.pomdp_solvers[0].update_current_belief(min_a,obs,all_poss_actions=True, horizon=None)
					# set_trace()
				# min_Q, min_a, min_time, Q_a, tree_size, time_steps, _ = self.agent_pomdp_solver.run_RTDP_bel(None,horizon,max_horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=True)

				# if print_status:
				# 	for e in range(len(self.tasks)):
				# 		print ("reward, selected action: ",(min_Q,min_a[e]), Q_a)
				# set_trace()


				step = 0
				self.step = step
				total_reward = np.zeros(self.max_steps)
				total_belief_reward = np.zeros(self.max_steps)
				total_max_Q = np.zeros(self.max_steps)
				satisfaction = np.zeros(self.max_steps)
				unsatisfaction = np.zeros(self.max_steps)
				total_satisfaction = np.zeros(self.max_steps)
				total_unsatisfaction = np.zeros(self.max_steps)
				tree_size = np.zeros(self.max_steps)
				duration = np.zeros(self.max_steps)

				if self.agentpomdp_vs_greedy or self.hybrid_vs_greedy:
					total_reward_other = np.zeros(self.max_steps)
					total_belief_reward_other = np.zeros(self.max_steps)
					total_max_Q_other = np.zeros(self.max_steps)
					satisfaction_other = np.zeros(self.max_steps)
					unsatisfaction_other = np.zeros(self.max_steps)
					tree_size_other = np.zeros(self.max_steps)
					duration_other = np.zeros(self.max_steps)

				num_steps = 0
				self.example_num = random_exec
				if self.save_example and random_exec in self.exec_list:
					pass
					# self.render(self.pomdp_tasks,str(step)+": ",num=num_steps,final_rew=None, exec_num=random_exec, render_belief=self.render_belief)
					

				if print_status or print_status_short:
					self.render(self.pomdp_tasks,str(step)+": ",num=num_steps,final_rew=None, exec_num=random_exec, render_belief=self.render_belief)

				# set_trace()
				# 	print_status = True
				# else:
				# 	print_status = False

				terminal_prev = np.empty(len(self.tasks),dtype=bool)
				terminal_prev.fill(False)
				self.current_pomdp = None
				while (True):
					if self.no_execution and random_exec not in self.exec_list: 
						break
					if print_status:
						print ("-------------------------------------------------------------------")
						print ("--- STEP --- " + str(step)+": ")

					
					all_terminal, position, sat, unsat, t_rew, t_b_rew, t_sat, t_unsat, action, obss, obs_index, max_Q, tree_s, dur, max_pomdp = \
					self.execute_episode(terminal_prev,self.greedy, self.hierarchical_baseline,simulate_step=False, no_op=self.no_op)
					self.current_pomdp = max_pomdp

					total_reward[num_steps] = t_rew
					total_belief_reward[num_steps] = t_b_rew
					total_max_Q[num_steps] = max_Q
					satisfaction[num_steps] = sat
					unsatisfaction[num_steps] = unsat
					total_satisfaction[num_steps] = t_sat
					total_unsatisfaction[num_steps] = t_unsat
					tree_size[num_steps] = tree_s
					duration[num_steps] = dur

					self.robot.set_feature('x',position[0])
					self.robot.set_feature('y',position[1])
					num_steps += 1

					if print_status or print_status_short:
						print ("Step: ", num_steps)
						msg = self.pomdp_tasks[0].get_action_msg (self.pomdp_tasks[max_pomdp].prev_state,self.agent_pomdp.actions[action,max_pomdp])
						self.render(self.pomdp_tasks,str(step)+": "+msg[0],num=num_steps,final_rew=np.sum(total_belief_reward),exec_num=random_exec, render_belief=self.render_belief)
						# set_trace()

					# if self.save_example and random_exec in self.exec_list:
					# 	msg = self.pomdp_tasks[0].get_action_msg (self.pomdp_tasks[max_pomdp].state,self.agent_pomdp.actions[action,max_pomdp])
					# 	self.render(self.pomdp_tasks,str(step)+": "+msg[0],num=num_steps,final_rew=np.sum(total_belief_reward),exec_num=random_exec, render_belief = self.render_belief)
					# 	self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions[action,max_pomdp].name,num=num_steps,final_rew=np.sum(total_belief_reward),exec_num=random_exec, render_belief = not self.render_belief)

					if all_terminal or num_steps>=self.max_steps:
						break
					step += 1
					self.step = step
					for p in self.pomdp_tasks:
						p.num_step = self.step
					self.agent_pomdp_solver.num_step = self.step
					# sleep(1)

					# print ("Step: ", num_steps)
					# self.render(self.pomdp_tasks,str(step)+": "+self.agent_pomdp.actions_names[action],num=num_steps,final_rew=np.sum(total_belief_reward),exec_num=random_exec)

					# if print_status:
						# if "I'll be back" in self.agent_pomdp.actions[action,max_pomdp].name:
					# set_trace()
					# sleep(5)				

				print ("Done " + str(random_exec))
				self.final_total_reward[random_exec,:] = total_reward
				self.final_total_belief_reward[random_exec,:] = total_belief_reward
				self.final_total_max_Q[random_exec,:] = total_max_Q
				self.final_num_steps[random_exec] = num_steps
				self.final_satisfaction[random_exec,:] = satisfaction
				self.final_total_satisfaction[random_exec,:] = total_satisfaction
				self.final_unsatisfaction[random_exec,:] = unsatisfaction
				self.final_total_unsatisfaction[random_exec,:] = total_unsatisfaction
				self.planning_time[random_exec,:] = duration
				self.tree_sizes[random_exec,:] = tree_size

				if self.agentpomdp_vs_greedy or self.hybrid_vs_greedy:
					self.final_total_reward_other[random_exec,:] = total_reward_other
					self.final_total_belief_reward_other[random_exec,:] = total_belief_reward_other
					self.final_total_max_Q_other[random_exec,:] = total_max_Q_other
					self.final_satisfaction_other[random_exec,:] = satisfaction_other
					self.final_total_satisfaction_other[random_exec,:] = total_satisfaction
					self.final_unsatisfaction_other[random_exec,:] = unsatisfaction_other
					self.final_total_unsatisfaction_other[random_exec,:] = total_unsatisfaction
					self.tree_sizes_other[random_exec,:] = tree_size_other
					self.planning_time_other[random_exec,:] = duration_other

					print ("total reward other: ", np.sum(self.final_total_reward_other[random_exec,:]))
					print ("total belief reward other: ", np.sum(self.final_total_belief_reward_other[random_exec,:]))
					print ("total max Q other: ", np.sum(self.final_total_max_Q_other[random_exec,:]))
					print ("total satisfaction other: ", np.sum(self.final_satisfaction_other[random_exec,:])/np.sum(self.final_total_satisfaction_other[random_exec,:]))
					print ("total unsatisfaction other: ", np.sum(self.final_unsatisfaction_other[random_exec,:])/np.sum(self.final_total_unsatisfaction_other[random_exec,:]))
					print ("total tree size other: ", np.sum(self.tree_sizes_other[random_exec,:]))
					print ("planning time other: ", np.sum(self.planning_time_other[random_exec,:]))

				print ("num steps: ", np.sum(self.final_num_steps[random_exec]))
				print ("total reward: ", np.sum(self.final_total_reward[random_exec,:]))
				print ("total belief reward: ", np.sum(self.final_total_belief_reward[random_exec,:]))
				print ("total max Q: ", np.sum(self.final_total_max_Q[random_exec,:]))
				print ("total satisfaction: ", np.sum(self.final_satisfaction[random_exec,:])/np.sum(self.final_total_satisfaction[random_exec,:]))
				print ("total unsatisfaction: ", np.sum(self.final_unsatisfaction[random_exec,:])/np.sum(self.final_total_unsatisfaction[random_exec,:]))
				print ("planning time: ", np.sum(self.planning_time[random_exec,:]))
				print ("tree size: ", np.sum(self.tree_sizes[random_exec,:]))
				print ("_________________________________________________________________________")
				if num_steps>=self.max_steps:
					print ("EXCEEDED MAX STEPS")
					pass
					# set_trace()
				# set_trace()
				if not self.run_on_cobot:
					self.robot.reset()
				else:
					self.robot.reset(self.pomdp_tasks[0].kitchen_pos)

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
		pomdp_task, max_pomdp, action, max_Q, tree_size, time_step = self.optimal_agent_pomdp(self.horizon, self.pomdp_tasks, self.agent_pomdp, self.agent_pomdp_solver)	
		
		end_time = time.clock()	
		planning_time += end_time - start_time

		obss = []
		obss_tuple = []
		observations = []
		all_terminal = True

		total_reward = 0
		total_belief_reward = 0		
		satisfaction = 0
		unsatisfaction = 0
		total_satisfaction = 0
		total_unsatisfaction = 0
		position = None

		# rew_out, exp_time = self.agent_pomdp_solver.run_RTDP_bel_Q (beliefs=self.agent_pomdp_solver.beliefs, actions=self.agent_pomdp.actions[action,:], horizon=time_step, max_horizon=time_step, one_action=True, gamma=self.gamma, tree_s=1,all_poss_actions=True)
		# total_belief_reward += rew_out

		sorted_list_pomdps = list(range(len(self.pomdp_solvers)))
		sorted_list_pomdps.remove(max_pomdp)
		sorted_list_pomdps.insert(0, max_pomdp)
		# print ("list of pomdps: ", sorted_list_pomdps)
		# set_trace()
		new_beliefs = []
		unexpected_observations = False
		for i in sorted_list_pomdps:
			if print_status or print_status_short:
				solver_prob = self.pomdp_solvers[i].belief.prob
				# state_str = "-- initial belief POMDP " + str(i) + ": " + str(self.pomdp_tasks[i].get_state_tuple(self.pomdp_solvers[i].belief.prob[0][1]))
				state_str = "-- initial belief: "
				for p in solver_prob:
					state_str += "(" + str(p[0]) + "," + str(self.pomdp_tasks[i].get_state_tuple(p[1])) + ")"

				# for p in solver_prob:
				# 	st = self.pomdp_tasks[i].get_state_tuple(p[1])			
				# 	state_str += ", " + str(st[self.pomdp_tasks[i].feature_indices['customer_satisfaction']]) + "-" + \
				# 	str(st[self.pomdp_tasks[i].feature_indices['current_request']]) + ":" + str(round(p[0],2))
				print (state_str)

			new_state_index, obs, reward, terminal, debug_info, position = self.pomdp_tasks[i].step(self.agent_pomdp.actions[action,i],start_state=None,simulate=False, robot=self.run_on_cobot, selected_pomdp=(i == max_pomdp))

			all_terminal = all_terminal and terminal

			if not simulate_step:
				terminal_prev[i] = terminal

			obss.insert(i,obs) # indices
			obss_tuple.insert(i,self.pomdp_tasks[i].get_observation_tuple(obs))

			pomdp_solver = self.pomdp_solvers[i]			
			
			if not simulate_step:
				if pomdp_solver.compute_1_over_eta (pomdp_solver.belief.prob, self.agent_pomdp.actions[action,i], obs, all_poss_actions=True,horizon=None) == 0:
					unexpected_observation = True
					if self.step > 3:
						pomdp_solver.compute_1_over_eta (pomdp_solver.belief.prob, self.agent_pomdp.actions[action,i], obs, all_poss_actions=True,horizon=None)
						set_trace()
					# set_trace()
					print ("UNEXPECTED EVENT DETECTEDED")
				else:
					unexpected_observation = False

				if not unexpected_observation: ##  and not self.mismatch
					pomdp_solver.update_current_belief(self.agent_pomdp.actions[action,i],obs,all_poss_actions=True, horizon=None)

				elif unexpected_observation:
					new_belief = self.pomdp_tasks[i].adapt_pomdp(pomdp_solver, self.agent_pomdp, pomdp_solver.belief, self.agent_pomdp.actions[action,i], obs)
					pomdp_solver.reset(new_belief)
					unexpected_observations = True
					self.mismatch = True

				new_beliefs.append(deepcopy(pomdp_solver.get_belief()))


		for i in range(len(self.pomdp_solvers)):
			obs_tuple = obss_tuple[i]
			observations.extend(obs_tuple[0:len(obs_tuple)-2]) ## hack
		observations.extend(obs_tuple[len(obs_tuple)-2:len(obs_tuple)]) #hack
		# set_trace()
		obs_index = None
		
		if not unexpected_observations:
			state_str = "-- updated belief agent: "			
			self.agent_pomdp_solver.update_current_belief(self.agent_pomdp.actions[action,:],obss,all_poss_actions=True, horizon=None)
			for solver_prob in self.agent_pomdp_solver.beliefs:
				for p in solver_prob.prob:
					state_str += "(" + str(p[0]) + "," + str(self.pomdp_tasks[0].get_state_tuple(p[1])) + ")"
				print ("\n")
			print (state_str)

		else:
			self.agent_pomdp_solver.reset(new_beliefs) 

		if print_status:
			for i in range(len(self.pomdp_solvers)):			
				solver_prob = self.pomdp_solvers[i].belief.prob
				state_str = "-- updated belief: "
				for p in solver_prob:
					state_str += "(" + str(p[0]) + "," + str(self.pomdp_tasks[i].get_state_tuple(p[1])) + ")"
				print (state_str)
				if state_str == "-- updated belief: ":
					set_trace()

		return all_terminal, position, satisfaction, unsatisfaction, total_reward, total_belief_reward, total_satisfaction, total_unsatisfaction, action, obss, obs_index, max_Q, tree_size, planning_time, max_pomdp
		


	def write_to_file(self,file):
		if file:
			filename = self.test_folder + self.model_folder + '/' + 'tables-' + str(len(self.tasks)) + '_simple-' + str(self.simple) \
			 + '_greedy-' + str(self.greedy) +  '_horizon-' + str(self.horizon) + '_#execs-' + \
			 str(self.num_random_executions) + '_seed-' + str(self.seed);

			if self.greedy:
				filename += "_noop-" + str(self.no_op)
				if self.hybrid_3T:
					filename += "_hybrid_3T-" + str(True)
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
			data['final_total_reward'] = self.final_total_reward
			data['final_total_belief_reward'] = self.final_total_belief_reward
			data['final_total_max_Q'] = self.final_total_max_Q
			data['final_total_satisfaction'] = self.final_total_satisfaction
			data['final_satisfaction'] = self.final_satisfaction
			data['final_total_unsatisfaction'] = self.final_total_unsatisfaction
			data['final_unsatisfaction'] = self.final_unsatisfaction
			data['planning_time'] = self.planning_time
			data['tree_sizes'] = self.tree_sizes

			if self.agentpomdp_vs_greedy or self.hybrid_vs_greedy:
				data['final_total_reward_other'] = self.final_total_reward_other
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
		print ("greedy: ", self.greedy, file=f)
		print ("simple: ", self.simple, file=f)
		print ("seed: ", self.seed, file=f)
		print ("no_op: ", self.no_op, file=f)
		print ("hybrid: ", self.greedy_hybrid, file=f)
		print ("shani: ", self.shani_baseline, file=f)
		print ("H_POMDP: ", self.hierarchical_baseline, file=f)

		if self.agentpomdp_vs_greedy or self.hybrid_vs_greedy:
			print ("final total reward other: ",np.mean(np.sum(self.final_total_reward_other,axis=1)), file=f)
			print ("final total belief reward other: ",np.mean(np.sum(self.final_total_belief_reward_other,axis=1)), file=f)
			print ("final total max_Q other: ",np.mean(np.sum(self.final_total_max_Q_other,axis=1)), file=f)
			print ("final total satisfaction other: ", np.mean(np.sum(self.final_satisfaction_other,axis=1)/np.sum(self.final_total_satisfaction_other,axis=1)), file=f)
			print ("final total unsatisfaction other: ", np.mean(np.sum(self.final_unsatisfaction_other,axis=1)/np.sum(self.final_total_unsatisfaction_other,axis=1)), file=f)
			print ("tree sizes other: ", np.mean(np.sum(self.tree_sizes_other,axis=1)), file=f)

		print ("DONEEEEE!", file=f)

		if file:
			f.close()

	def optimal_agent_pomdp (self, horizon, pomdp_tasks, agent_pomdp, agent_pomdp_solver):
		global print_status
		action_values = np.zeros(horizon*2)
		# Q, a, max_time, Q_a, tree_size, exp_time_steps,_ = agent_pomdp_solver.compute_V(None,horizon,horizon,one_action=False,gamma=self.gamma,tree_s=1,all_poss_actions=True)
		Q, a, min_time, Q_a, tree_size, exp_time_steps, _ = self.agent_pomdp_solver.run_RTDP_bel(None,horizon,horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=True)

		if print_status:
			print ("robot: ", (self.robot.get_feature('x').value, self.robot.get_feature('y').value))
			print ("reward, selected action: ",(Q,a))
			# print ("action values: ", action_values)

			# if horizon == 2:
			# 	print (self.agent_pomdp.actions_names[int(action_values[0])], " , ", \
			# 		self.agent_pomdp.actions_names[int(action_values[1])])
			# if horizon == 3:
			# 	print (self.agent_pomdp.actions_names[int(action_values[0])], " , ", \
			# 		self.agent_pomdp.actions_names[int(action_values[1])], " , ", \
			# 		self.agent_pomdp.actions_names[int(action_values[2])])
			print ("-- Optimal Max Q --")
			print (Q_a)

		
		min_action = np.argmin(Q_a) ## was a
		min_Q = Q
		min_pomdp, action = agent_pomdp.get_pomdp(min_action, self.current_pomdp)
		pomdp_task = pomdp_tasks[min_pomdp]	
		min_time_step = exp_time_steps[min_action][0]
		
		if print_status:
			print ("time steps: ", min_time_step)
			print ("POMDP ", min_pomdp)
			print ("final selected action: ",action.name)
			if len(pomdp_tasks) == len(self.pomdp_tasks):
				if self.LB is not None and self.LB - min_Q > 0.001:
					print ("WRONG LB")
					set_trace()
				if (self.UB is not None and self.UB - min_Q < -0.001) and not self.shani_baseline:
					print ("WRONG UB")
					set_trace()

		return pomdp_task, min_pomdp, min_action, min_Q, tree_size, min_time_step



	def solve_ind_pomdps (self, selected_pomdp_solvers, horizon, no_op):
		global print_status

		Vs = np.zeros((len(self.pomdp_solvers),1))
		all_actions_time_steps = []
		pomdp_Q = []
		tree_size = 0
		upper_bound = 0
		
		for i in selected_pomdp_solvers:
			pomdp_solver = self.pomdp_solvers[i]

			if print_status:
				print ("goal: ", (pomdp_solver.env.task.table.goal_x, pomdp_solver.env.task.table.goal_y))
				print ("initial belief: ", self.pomdp_tasks[i].get_state_tuple(pomdp_solver.belief.prob[0][1]))
				print ("robot: ", (self.robot.get_feature('x').value, self.robot.get_feature('y').value))
			
			# max_Q, max_a, max_time, Q_a, tree_s, exp_time_steps,_ = pomdp_solver.compute_V(pomdp_solver.belief,horizon,horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
			min_Q, min_a, min_time, Q_a, tree_size, exp_time_steps, _ = pomdp_solver.run_RTDP_bel(pomdp_solver.belief,horizon,horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
			all_actions_time_steps.append(exp_time_steps)
			Vs[i] = min_Q
			tree_size += tree_s
			pomdp_Q.append(Q_a)
			upper_bound += min_Q
			if print_status:
				print ("reward, selected action: ",(min_Q,min_a), Q_a)
				# set_trace()

		return upper_bound, tree_size, all_actions_time_steps, Vs, pomdp_Q

	def compute_lower_bound (self, selected_pomdp_solvers, Vs, V_traj, horizon, no_op):
		global print_status
		lower_bound = np.Inf
		for i in selected_pomdp_solvers:
			min_Vs = 0
			min_V_str = ""
			for j in selected_pomdp_solvers:
				if i != j:
					# pomdp_solver = self.pomdp_solvers[j]
					# min_V,_,_,_,_ = pomdp_solver.compute_Q_one_action(pomdp_solver.belief,self.pomdp_tasks[j].no_action,horizon,horizon,one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
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

	def compute_min_Qs_greedily (self, selected_pomdp_solvers, pomdp_Q, V_traj, all_actions_time_steps, horizon, no_op):
		# selected pomdps are the ones that we want to sum
		global print_status
		min_Qs = np.zeros((self.agent_pomdp.nA,1))
		min_time_steps = np.zeros((self.agent_pomdp.nA,1))	
		count = 0

		if print_status:
			print ("--- POMDPs: ", selected_pomdp_solvers) 


		for action in range(self.agent_pomdp.actions.shape[0]):
			# if print_status:
			# 	print_str = "action: " + str(action) + " "
			for i in range(self.agent_pomdp.actions.shape[1]):
				pomdp_solver = self.pomdp_solvers[i]
				if self.agent_pomdp.pomdp_actions[action] in selected_pomdp_solvers:
					if i in selected_pomdp_solvers:
						ind_act = self.agent_pomdp.actions[action,i]					
						if not no_op or (ind_act in self.pomdp_tasks[i].valid_actions):
							min_Qs[count] += pomdp_Q[i][ind_act.id][0]
							min_time_steps[count] =  all_actions_time_steps[i][ind_act.id]
							# if print_status:
							# 	print_str += str(pomdp_Q[i][ind_act.id][0]) + " "
						else:
							try:
								rew, exp_time = pomdp_solver.run_RTDP_bel_Q (self.pomdp_solvers[i].belief, action=ind_act, horizon=horizon, max_horizon=horizon, \
									one_action=False, gamma=self.gamma, tree_s=0,all_poss_actions=not no_op)
							except:
								set_trace()	
							min_Qs[count] += rew
							# if print_status:
							# 	print_str += str(rew) + " "
					else:
						# rew, tree_s, exp_time,_,_ = pomdp_solver.compute_Q_one_action(pomdp_solver.belief,self.pomdp_tasks[i].no_action,horizon,horizon, \
						# 	one_action=False,gamma=self.gamma, tree_s=1, all_poss_actions=not no_op)
						# tree_size += tree_s
						rew = V_traj[i]
						min_Qs[count] += rew
						# if print_status:
						# 	print_str += str(rew) + " "
				else:
					min_Qs[count] = np.Inf
					# if print_status:
					# 	print_str += "-inf "

			# if print_status:
			# 	print (print_str)

			count += 1

		return min_Qs, tree_size, min_time_steps

	def select_min_Q_pomdp (self, min_Qs, min_time_steps):
		min_action = np.argmin(min_Qs)
		min_time_step = min_time_steps[min_action][0]
		min_Q = np.min(min_Qs)
		min_pomdp, pomdp_action = self.agent_pomdp.get_pomdp(min_action, self.current_pomdp)
		pomdp_task = self.pomdp_tasks[min_pomdp]	

		return pomdp_task, min_pomdp, min_action, min_Q, min_time_step, pomdp_action

	def run_optimal_planner_on_k(self, selected_pomdps, horizon):
		sub_pomdp_tasks = [self.pomdp_tasks[i] for i in selected_pomdps]
		sub_pomdp_solvers = [self.pomdp_solvers[i] for i in selected_pomdps]
		sub_tasks = [self.tasks[i] for i in selected_pomdps]

		beliefs = []

		for solver in sub_pomdp_solvers:
			beliefs.append(solver.belief.prob)

		if print_status:
			print ("** selected POMDPS: ", selected_pomdps)
		self.current_pomdp = None
		sub_agent_pomdp = AgentPOMDP(sub_pomdp_tasks, sub_pomdp_solvers, sub_tasks, self.robot, self.random)
		sub_agent_pomdp_solver = MultiIndGoalPOMDPSolver(sub_agent_pomdp,beliefs,self.random)
		# sub_agent_pomdp_solver = POMDPSolver(sub_agent_pomdp,sub_agent_pomdp.get_beliefs(beliefs),self.random)

		if print_status:
			print (beliefs)
			print ("======= running optimal planner on " + str(len(selected_pomdps)) + " POMDP =======")
		sub_pomdp_task, sub_min_pomdp, sub_action, sub_min_Q, sub_tree_size, sub_time_step = \
			self.optimal_agent_pomdp (horizon, sub_pomdp_tasks, sub_agent_pomdp, sub_agent_pomdp_solver)
		min_time_step = sub_time_step
		if print_status:
			print ("==================================================")
			print ("time steps: ", min_time_step)

		return sub_agent_pomdp, sub_pomdp_task, sub_min_pomdp, sub_action, sub_min_Q, sub_tree_size, sub_time_step


	def render(self, pomdp_tasks,actions_names, num=None, final_rew=None, exec_num=0, render_belief=False):
		if render_belief:
			self.render_beliefs(pomdp_tasks,actions_names, num, final_rew, exec_num)
		else:
			render_trace = True
			# plt.close()
			# plt.clf()
			margin = 0.5
			x_low = self.robot.get_feature("x").low - margin
			x_high = self.robot.get_feature("x").high + margin
			y_low = self.robot.get_feature("y").low - margin
			y_high = self.robot.get_feature("y").high + margin
			coords = [x_low,x_high,y_low,y_high]
			drawTables(coords, self.restaurant.tables,pomdp_tasks,self.pomdp_solvers,actions_names,final_rew)
			drawRobot(self.restaurant.tables, pomdp_tasks, self.robot, 0.5)
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
				dirct = '../tests/' + self.model_folder + '/simple-' + str(self.simple) + '-horizon-' + str(self.horizon) + '-tables-' + str(len(self.tasks)) + '_example_' + str(self.example_num) + '/greedy-' + str(self.greedy) + '/'
				if not os.path.exists(dirct):
					os.makedirs(dirct)

				# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

				plt.savefig(dirct + str(exec_num) + "_" + t + '-greedy-' + str(self.greedy) + ".png",bbox_inches="tight")
				plt.close()
				# img = plt.imread(dirct + str(exec_num) + "_" + t + '-greedy-' + str(self.greedy) + ".png")
				# # print(my_img.shape)
				# # my_clipped_img = my_img[100:900,100:900]
				# plt.figure()
				# plt.imshow(img)
				# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
				# plt.savefig(dirct + str(exec_num) + "_" + t + '-greedy-' + str(self.greedy) + ".png")
				# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

				# set_trace()
				# plt.close()
				# set_trace()

			if render_trace:
				# set_trace() 
				render_trace = False

			sleep(PAUSE_TIME)
			plt.close()

			# if self.finish_render:
			#     plt.show(block=False)
			#     plt.pause(0.0000000001)

			# if close:
			#     plt.close()

	def render_beliefs(self, pomdp_tasks,actions_names, num=None, final_rew=None, exec_num=0):
		render_trace = False
		# plt.clf()
		# plt.close()
		plt.close('all')
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
			# plt.close()
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





