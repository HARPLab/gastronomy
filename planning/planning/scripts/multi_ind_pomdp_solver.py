from pdb import set_trace
import numpy as np
from copy import deepcopy, copy
from time import sleep
import math
import sys


from draw_env import *
from pomdp_solver import *
import time as system_time

print_status = False

cashing = None
NORMALIZE = False
NORMALIZATION_PRECISION = 0.01
precision = None

class BoundPair:
	def __init__(self, lb=None, ub=None):
		self.UB = ub
		self.LB = lb

class MultiIndPOMDPSolver:
	def __init__(self, env, initial_beliefs, random):
		# print("cashing: ", cashing)
		self.random = random
		self.beliefs = []
		for i in range(len(env.pomdp_tasks)):
			self.beliefs.append(Belief(initial_beliefs[i]))

		if env.extra_pomdp_solvers is not None:
			self.extra_beliefs = []
			for i in range(len(env.extra_pomdp_solvers)):
				blf = env.extra_pomdp_solvers[i].belief
				self.extra_beliefs.append(blf)

		self.env = env
		self.envs = env.pomdp_tasks
		# self.planning_time = 0
		


	############################33## use belief_prob
	def update_belief (self, env, belief_prob, action, observation, all_poss_actions,horizon):
		global NORMALIZE
		new_belief_prob = []
		sum_normalized = 0.0
		count_normalized = 0.0
		# action.print()
		for s_p in env.get_possible_next_states(observation, belief_prob, action):
			for o_p in env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					pr_obs = o_p[1]
					if pr_obs != 0:						
						over_eta = self.compute_1_over_eta (env,belief_prob,action,observation, all_poss_actions,horizon)
						if over_eta != 0:
							eta = 1.0/over_eta
							up_b_tr = self.update_belief_tr(env,belief_prob,s_p,action,all_poss_actions,horizon)
							if up_b_tr != 0:
								pr = eta * pr_obs * up_b_tr
								new_belief_prob.append((pr,s_p))
								if pr < NORMALIZATION_PRECISION:
									count_normalized += 1
									sum_normalized += pr
						# else:

					# 	else:
					# 		new_belief_prob[s_p] = 0
					# else:
					# 	new_belief_prob[s_p] = 0
		if NORMALIZE:
			# print (new_belief_prob, sum_normalized, count_normalized)
			if not count_normalized == 0:
				normalized_new_belief_prob = []
				for (pr,st) in new_belief_prob:
					if pr >= NORMALIZATION_PRECISION: 
						new_pr = pr + sum_normalized/(len(new_belief_prob)-count_normalized)
						normalized_new_belief_prob.append((new_pr,st))
				new_belief_prob = normalized_new_belief_prob
				# print (new_belief_prob)
				# set_trace()

		return new_belief_prob


	def update_belief_tr (self, env, belief_prob, state_p, action, all_poss_actions,horizon):
		sum_s = 0
		for (prob,state) in belief_prob:
			outcomes, steps = env.simulate_action(state,action,all_poss_actions,horizon)
			for outcome in outcomes:
				if outcome[1] == state_p:
					sum_s += prob * outcome[0]
		return sum_s

	def compute_1_over_eta (self, env, belief_prob, action, observation, all_poss_actions,horizon):
		sum_s_p = 0
		# for s_p in range(belief_prob.shape[0]): 
		for s_p in env.get_possible_next_states(observation, belief_prob, action):
			for o_p in env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					sum_s_p += o_p[1] * self.update_belief_tr(env,belief_prob,s_p,action,all_poss_actions,horizon)
		return sum_s_p
	###################################
	def update_current_belief(self, actions, obs_indices, all_poss_actions,horizon):
		# if action not in self.env.navigation_actions:
		for i in range(len(self.envs)):
			self.beliefs[i].prob = self.update_belief(self.envs[i], self.beliefs[i].prob, actions[i], obs_indices[i], all_poss_actions,horizon)
		# else:
		# 	self.belief.prob = self.update_belief (self.belief.prob, self.env.no_action, obs_index)
		# 	self.belief.pos, reward = self.env.simulate_navigation_action(action,self.belief.pos)

	def are_feasible_obs(self, possible_obss,obss,actions):
		prev_obs = None
		num_eq = 1.0
		# set_trace()
		all_envs = copy(self.envs)
		if self.env.extra_pomdp_solvers is not None and len(self.env.extra_pomdp_solvers) != 0:
			for extra_solver in self.env.extra_pomdp_solvers:
				all_envs.append(extra_solver.env)

		for e in range(len(all_envs)): 
			env = all_envs[e]
			new_obs = all_envs[e].get_observation_tuple(possible_obss[e][obss[e]])
			
			if not all_envs[e].task.table.package:
				if prev_obs is not None and not (prev_obs[env.obs_feature_indices["x"]] == new_obs[env.obs_feature_indices["x"]] and \
					prev_obs[env.obs_feature_indices["y"]] == new_obs[env.obs_feature_indices["y"]]):
					return False, num_eq
			else:
				if prev_obs is not None and actions[e-1].id == actions[e].id and actions[e].id != 1:
					if not prev_obs[all_envs[e-1].obs_feature_indices["ping_loc_success"]] == new_obs[all_envs[e].obs_feature_indices["ping_loc_success"]]:
						return False, num_eq
					else:
						num_eq += 1.0
			prev_obs = new_obs


		return True, num_eq

	def compute_immediate_reward (self, env, beliefs, action, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		immediate_r = 0
		immediate_time = 0
		for (prob,state) in beliefs.prob:
			outcomes, steps = env.simulate_action(state,action,all_poss_actions,horizon)
			for outcome in outcomes:				
				if prob != 0:
					immediate_r += prob * outcome[0] * outcome[2]
					immediate_time += prob * outcome[0] * steps

		return immediate_time, immediate_r

	def compute_Q (self, beliefs, actions, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False, LB_UB=False, extra_beliefs=None):
		tree_size = tree_s + 1
		immediate_r = 0 
		immediate_time = 0
		leaf_beliefs = []
		leaf_node = False
		time_steps = horizon
		extra_actions = []
		if HPOMDP:
			# set_trace()
			time,rew = self.compute_immediate_reward (self.envs[int(actions[0].id)], beliefs[actions[0].id], actions[0], horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
			if self.envs[int(actions[0].id)].task.table.package:
				time = horizon
			immediate_r += rew
			immediate_time += time
			time_steps = int(np.round(immediate_time))

			env_count = 0
			for env in self.envs:
				if env_count != int(actions[0].id):
					time,rew = self.compute_immediate_reward (env, beliefs[env_count], actions[env_count], time_steps, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
					immediate_r += rew

				actions[env_count].time_steps = time_steps
				env_count += 1

			reward = BoundPair(immediate_r,immediate_r)

		else:
			env_count = 0
			for env in self.envs:
				time,rew = self.compute_immediate_reward (env, beliefs[env_count], actions[env_count], horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
				# if "drive" in actions[env_count].name:
				# 	print (env_count, actions[env_count].name)
				# 	print ("rew: ", rew)
				# 	# print (self.compute_immediate_reward (env, beliefs[env_count], env.actions[1], horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP))
				# 	set_trace()
				immediate_r += rew
				if env_count == 0:
					immediate_time += time

				env_count += 1

			######### extra pomdps
			if extra_beliefs is not None:
				env_count = 0
				for env_solver in self.env.extra_pomdp_solvers: 
					extra_all_poss_actions = True
					if not env_solver.env.task.table.package:
						if not actions[0].type:
							action = actions[0]
						else:
							action = env_solver.env.noop_actions[str(int(np.round(immediate_time)))]
					else:
						if actions[0].id >= env_solver.env.non_navigation_actions_len:
							action = actions[0]
						else:
							action = env_solver.env.noop_actions[str(int(np.round(immediate_time)))]
					extra_actions.append(action)
					time,rew = env_solver.compute_immediate_reward (extra_beliefs[env_count], action, horizon, max_horizon, one_action, gamma, tree_s, extra_all_poss_actions, HPOMDP)
					# print ("time:", time, "rew:",rew)
					immediate_r += rew
					env_count += 1
			######### extra pomdps

			reward = BoundPair(immediate_r,immediate_r)

		immediate_time = int(np.round(immediate_time))
		exp_time = BoundPair(immediate_time,immediate_time)

		if print_status:
			print ("1_ action: ", [a.id for a in actions], " horizon: ", horizon, ", ", [a.name for a in actions])
			# for a in actions:
			# 	if "load" in a.name and horizon == 3:
			# 		set_trace()
			print ("action length: ", int(immediate_time))
			print ("immediate reward: ", immediate_r)
			print ("tree size: ", tree_size)
			for a in actions:
				if max_horizon == 4:
					print ("-----------------------------------------------")
					print ("-----------------------------------------------")
					if "load" in a.name:
						set_trace()
				# if max_horizon == 2:
				# 	print ("-----------------------------------------------")
				# 	print ("-----------------------------------------------")
				# 	if "drive" in a.name:
				# 		set_trace()
				# 		break

		if immediate_r == -np.Inf:
			return BoundPair(-np.Inf,-np.Inf),0,BoundPair(0,0),0,leaf_beliefs

		if not one_action:
			if horizon <= immediate_time:
				if not LB_UB:
					if horizon < immediate_time:
						if print_status:
							print ("2_ horizon: ", horizon, " action steps: ", immediate_time)
							print ("---- action bigger than horizon ----")
						return BoundPair(-np.Inf,-np.Inf),0,BoundPair(0,0),0,leaf_beliefs
					elif horizon == immediate_time:	
						if print_status:
							print ("3_ reward: ", reward.UB - immediate_r, reward.UB)
						return reward, tree_size, exp_time, int(immediate_time), leaf_beliefs
				else:
					if max_horizon < immediate_time:
						if print_status:
							print ("4_ horizon: ", horizon, " action steps: ", immediate_time)
							print ("---- action bigger than horizon ----")
						return BoundPair(-np.Inf,-np.Inf),0,BoundPair(0,0),0,leaf_beliefs
					elif max_horizon == int(immediate_time):
						if print_status:
							print ("9_ horizon: ", horizon, " action steps: ", immediate_time)
							print ("---- action equal max horizon ----")
						return reward, tree_size, exp_time, int(immediate_time), leaf_beliefs
					else:
						start_time = system_time.clock() 
						if print_status:
							print ("8_ horizon: ", horizon, " action steps: ", immediate_time, " max horizon: ", max_horizon)
						possible_obss = []
						possible_obss_len = 1
						possible_obss_dim = ()
						for e in range(len(self.envs)):
							env = self.envs[e]
							obss = list(env.get_possible_obss(beliefs[e].prob,all_poss_actions,time_steps))
							possible_obss.append(obss)
							possible_obss_len *= len(obss)
							possible_obss_dim += (len(obss),)

						######### extra pomdps
						if extra_beliefs is not None:
							for e in range(len(self.env.extra_pomdp_solvers)):
								env = self.env.extra_pomdp_solvers[e].env
								obss = list(env.get_possible_obss(extra_beliefs[e].prob,all_poss_actions,time_steps))
								possible_obss.append(obss)
								possible_obss_len *= len(obss)
								possible_obss_dim += (len(obss),)
						######### extra pomdps

						count = 0
						while count < possible_obss_len:					
							next_beliefs_value = BoundPair(0.0,0.0)

							new_beliefs = []
							eta = 1
							obss = self.get_tuple(count,possible_obss_dim)
							action_set = list(actions) + extra_actions
							feasible, num_eq = self.are_feasible_obs(possible_obss,obss,action_set)
							if feasible:
								for e in range(len(self.envs)):
									env = self.envs[e]					
									obs = possible_obss[e][obss[e]]
									new_belief_prob = self.update_belief(env, beliefs[e].prob, actions[e], obs, all_poss_actions,time_steps)

									if len(new_belief_prob) > 0:
										new_beliefs.append(Belief(new_belief_prob))
										if num_eq == 1.0 or e == 0:
											eta *= self.compute_1_over_eta(env,beliefs[e].prob,actions[e],obs,all_poss_actions,time_steps)

								######### extra pomdps
								if extra_beliefs is not None:
									extra_new_beliefs = []
									for e in range(len(self.env.extra_pomdp_solvers)): 
										env = self.env.extra_pomdp_solvers[e].env
										extra_all_poss_actions = True
										action = extra_actions[e]
										obs = possible_obss[e+len(self.envs)][obss[e+len(self.envs)]]
										extra_new_belief_prob = self.env.extra_pomdp_solvers[e].update_belief(extra_beliefs[e].prob, action, obs, extra_all_poss_actions,time_steps)

										if len(extra_new_belief_prob) > 0:
											extra_new_beliefs.append(Belief(extra_new_belief_prob))
											if num_eq == 1.0:
												eta *= self.env.extra_pomdp_solvers[e].compute_1_over_eta(extra_beliefs[e].prob,action,obs,extra_all_poss_actions,time_steps)
								######### extra pomdps

								if len(new_beliefs) == len(self.envs) and (self.env.extra_pomdp_solvers is None or (len(extra_new_beliefs) == len(self.env.extra_pomdp_solvers))):	
									# if max_horizon-(immediate_time) == 0:
									# if num_eq != 1.0:
									# 	print (eta, num_eq)
									# 	set_trace()

									if self.env.extra_pomdp_solvers is not None and len(self.env.extra_pomdp_solvers) != 0:
										LB, UB, tree_s = self.compute_LB_UB(new_beliefs, max_horizon-(immediate_time), gamma, tree_s, all_poss_actions, extra_new_beliefs)
									else:
										LB, UB, tree_s = self.compute_LB_UB(new_beliefs, max_horizon-(immediate_time), gamma, tree_s, all_poss_actions)

									if print_status:
										print ("10_ reward: ", eta, LB, UB)
									next_beliefs_value.UB =  eta * UB
									next_beliefs_value.LB =  eta * LB

									reward.UB += (gamma ** int(immediate_time)) * next_beliefs_value.UB
									reward.LB += (gamma ** int(immediate_time)) * next_beliefs_value.LB


							count += 1		
						# elif horizon < immediate_time:
						# 	# set_trace()
						# 	LB, UB, tree_s = self.compute_LB_UB(beliefs, max_horizon, gamma, tree_s, all_poss_actions)

						# 	reward.UB = UB
						# 	reward.LB = LB

						# 	exp_time.UB = 0
						# 	exp_time.LB = 0		

						if print_status:
							print ("5_ reward: ", reward.LB, reward.UB, " exp_time: ", exp_time.LB, exp_time.UB, " immediate_time: ", int(immediate_time))
							# set_trace()	
						end_time = system_time.clock()	
						# print ("duration: ", end_time - start_time)
						# self.planning_time += end_time - start_time
						return reward, tree_size, exp_time, int(immediate_time), leaf_beliefs

			elif horizon > immediate_time:
				env_count = 0
				possible_obss = []
				possible_obss_len = 1
				possible_obss_dim = ()
				for e in range(len(self.envs)):
					env = self.envs[e]
					obss = list(env.get_possible_obss(beliefs[e].prob,all_poss_actions,time_steps))
					possible_obss.append(obss)
					possible_obss_len *= len(obss)
					possible_obss_dim += (len(obss),)

				######### extra pomdps
				if extra_beliefs is not None:
					for e in range(len(self.env.extra_pomdp_solvers)):
						env = self.env.extra_pomdp_solvers[e].env
						obss = list(env.get_possible_obss(extra_beliefs[e].prob,all_poss_actions,time_steps))
						possible_obss.append(obss)
						possible_obss_len *= len(obss)
						possible_obss_dim += (len(obss),)
				######### extra pomdps

				count = 0
				while count < possible_obss_len:
					env_count = 0
					
					next_beliefs_value = BoundPair(0.0,0.0)
					next_beliefs_time = BoundPair(0.0,0.0)

					new_beliefs = []
					eta = 1
					obss = self.get_tuple(count,possible_obss_dim)
					action_set = list(actions) + extra_actions
					feasible, num_eq = self.are_feasible_obs(possible_obss,obss,action_set)
					if feasible:
						for e in range(len(self.envs)):
							env = self.envs[e]					
							obs = possible_obss[e][obss[e]]
							new_belief_prob = self.update_belief(env, beliefs[e].prob, actions[e], obs, all_poss_actions,time_steps)

							if len(new_belief_prob) > 0:
								new_beliefs.append(Belief(new_belief_prob))
								if num_eq == 1.0 or e == 0:
									eta *= self.compute_1_over_eta(env,beliefs[e].prob,actions[e],obs,all_poss_actions,time_steps)

						######### extra pomdps
						if extra_beliefs is not None:
							extra_new_beliefs = []
							for e in range(len(self.env.extra_pomdp_solvers)): 
								env = self.env.extra_pomdp_solvers[e].env
								extra_all_poss_actions = True
								action = extra_actions[e]
								obs = possible_obss[e+len(self.envs)][obss[e+len(self.envs)]]
								extra_new_belief_prob = self.env.extra_pomdp_solvers[e].update_belief(extra_beliefs[e].prob, action, obs, extra_all_poss_actions,time_steps)

								if len(extra_new_belief_prob) > 0:
									extra_new_beliefs.append(Belief(extra_new_belief_prob))
									if num_eq == 1.0:
										eta *= self.env.extra_pomdp_solvers[e].compute_1_over_eta(extra_beliefs[e].prob,action,obs,extra_all_poss_actions,time_steps)
						######### extra pomdps

						if len(new_beliefs) == len(self.envs) and (self.env.extra_pomdp_solvers is None or (len(extra_new_beliefs) == len(self.env.extra_pomdp_solvers))):
							# if num_eq != 1.0:
							# 	print (eta, num_eq)
							# 	set_trace()

							if self.env.extra_pomdp_solvers is not None and len(self.env.extra_pomdp_solvers) != 0:
								V,_,max_time,_,tree_size,_,leaf_beliefs = self.compute_V(new_beliefs, horizon-(immediate_time), max_horizon-(immediate_time), one_action, gamma, tree_size, all_poss_actions, HPOMDP, LB_UB, extra_new_beliefs)
							else:
								V,_,max_time,_,tree_size,_,leaf_beliefs = self.compute_V(new_beliefs, horizon-(immediate_time), max_horizon-(immediate_time), one_action, gamma, tree_size, all_poss_actions, HPOMDP, LB_UB)
							# set_trace()
							next_beliefs_value.UB +=  eta * V.UB
							next_beliefs_time.UB += eta * max_time.UB

							next_beliefs_value.LB +=  eta * V.LB
							next_beliefs_time.LB += eta * max_time.LB

							reward.UB += (gamma ** int(immediate_time)) * next_beliefs_value.UB
							reward.LB += (gamma ** int(immediate_time)) * next_beliefs_value.LB

							exp_time.UB += next_beliefs_time.UB
							exp_time.LB += next_beliefs_time.LB


					count += 1

				exp_time.LB = int(round(exp_time.LB))
				exp_time.UB = int(round(exp_time.UB))
		if print_status:
			if max_horizon == 4:
				print ("**************************************************")
				set_trace()
			print ("6_ reward: ", reward.LB, reward.UB, " exp_time: ", exp_time.LB, exp_time.UB, " immediate_time: ", int(immediate_time))
			# set_trace()

		return reward, tree_size, exp_time, int(immediate_time), leaf_beliefs

	# def compute_UB (self, beliefs, actions, max_horizon, gamma, tree_s=0, all_poss_actions=False):
	# 	env_count = 0
	# 	# LB = -np.Inf
	# 	UB = 0

	# 	for j in range(0,len(self.env.pomdp_solvers)):
	# 		rew, tree_s, exp_time,_,_ = self.env.pomdp_solvers[j].compute_Q (beliefs[j], action=actions[j], horizon=max_horizon, max_horizon=max_horizon, \
	# 						one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)
	# 		UB += rew
	# 		env_count += 1

	# 	return UB, tree_s

	def compute_LB_UB (self, beliefs, max_horizon, gamma, tree_s=0, all_poss_actions=False, extra_beliefs=None):
		global cashing
		# horizons = np.full((len(self.env.pomdp_solvers),),None)

		# if print_status:
		# 	for j in range(len(beliefs)):
		# 		pr_str = "["
		# 		for i in range(len(beliefs[j].prob)):
		# 			pr_str += "(" + str(beliefs[j].prob[i][0]) + "," + str(self.env.pomdp_tasks[j].get_state_tuple(beliefs[j].prob[i][1])) + ")"
		# 		pr_str += "]"
		# 		pr_str += "\n"
		# 		print (pr_str)

		# 	if extra_beliefs is not None and len(extra_beliefs) == 2:
		# 		for j in range(len(extra_beliefs)):
		# 			pr_str = "["
		# 			for i in range(len(extra_beliefs[j].prob)):
		# 				pr_str += "(" + str(extra_beliefs[j].prob[i][0]) + "," + str(self.env.extra_pomdp_solvers[j].env.get_state_tuple(extra_beliefs[j].prob[i][1])) + ")"
		# 			pr_str += "]"
		# 			pr_str += "\n"
		# 			print (pr_str)
		# 			# set_trace()

		if max_horizon == 0:
			return 0, 0, 0

		env_count = 0
		LB = -np.Inf
		UB = 0
		# start_time = system_time.clock()
		V_traj = np.full((len(self.env.pomdp_solvers),),None)
		# print ("beliefs: ", beliefs[0].prob, beliefs[1].prob)
		for j in range(0,len(self.env.pomdp_solvers)):
			pomdp_solver = self.env.pomdp_solvers[j]
			min_V,_,_,_,_ = pomdp_solver.compute_Q_one_action(beliefs[j],self.env.pomdp_tasks[j].noop_actions['1'],max_horizon,max_horizon,\
				one_action=False,gamma=gamma, tree_s=tree_s, all_poss_actions=all_poss_actions)
			V_traj[j] = min_V
			# print ("belief" + str(j) + ": ", pomdp_solver.env.get_state_tuple(beliefs[0].prob[0][1]),"min_V:", min_V)

		if extra_beliefs is not None:
			extra_V_traj = np.full((len(self.env.extra_pomdp_solvers),),None)
			for j in range(0,len(self.env.extra_pomdp_solvers)):
				pomdp_solver = self.env.extra_pomdp_solvers[j]
				min_V,_,_,_,_ = pomdp_solver.compute_Q_one_action(extra_beliefs[j],pomdp_solver.env.noop_actions['1'],max_horizon,max_horizon,\
					one_action=False,gamma=gamma, tree_s=tree_s, all_poss_actions=all_poss_actions)
				extra_V_traj[j] = min_V
				# print ("extra belief: ", self.env.extra_pomdp_solvers[j].env.get_state_tuple(extra_beliefs[0].prob[0][1]),"min_V:", min_V)

		for j in range(0,len(self.env.pomdp_solvers)):
			if cashing:
				rew_UB, rew_LB = self.add_to_belief_library(self.env.pomdp_solvers[j], beliefs[j], max_horizon, gamma, tree_s, all_poss_actions)
			else:
				rew_t_t,_,exp_time,_,_,_,_ = self.env.pomdp_solvers[j].compute_V (beliefs[j], horizon=max_horizon, max_horizon=max_horizon, \
						one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)
				rew_UB = rew_t_t
				rew_LB = rew_t_t

			# if (rew_t_t - rew_LB) < -0.001 or (rew_UB - rew_t_t) < -0.001:
			# 	print ("UB: ", rew_UB)
			# 	print ("V: ", rew_t_t)
			# 	print ("LB: ", rew_LB)
			# 	set_trace()
			# if np.fabs(rew_t - rew_t_t) > 0.001:
			# 	print ("UB: ", rew_t)
			# 	print ("V: ", rew_t_t)
			# 	set_trace()
			UB += rew_UB
			mask = np.full(len(self.env.pomdp_solvers), True); mask[env_count] = False;
			vale_of_noops = np.sum(V_traj[mask])
			LB = max(LB,rew_LB + vale_of_noops)
			env_count += 1
			# print ("rew: ", rew_LB, rew_UB, vale_of_noops, LB,UB)
			# set_trace()

		######### extra pomdps
		if extra_beliefs is not None and len(extra_beliefs) != 0:
			# print ("rew: ", LB,UB)
			LB += np.sum(extra_V_traj)
			# print ("rew: ", LB,UB)
			env_count = 0
			for j in range(0,len(self.env.extra_pomdp_solvers)):
				if cashing:
					rew_UB, rew_LB = self.add_to_belief_library(self.env.extra_pomdp_solvers[j], extra_beliefs[j], max_horizon, gamma, tree_s, all_poss_actions)
				else:
					rew_t_t,_,exp_time,_,_,_,_ = self.env.extra_pomdp_solvers[j].compute_V (extra_beliefs[j], horizon=max_horizon, max_horizon=max_horizon, \
							one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)
					rew_UB = rew_t_t
					rew_LB = rew_t_t
				
				# if rew_UB + UB > UB:
				# 	print ("UB: ", UB, rew_UB + UB)
					# set_trace()
				UB += rew_UB
				mask = np.full(len(self.env.extra_pomdp_solvers), True); mask[env_count] = False;
				vale_of_noops = np.sum(extra_V_traj[mask])
				# if rew_LB + vale_of_noops + np.sum(V_traj) > LB:
				# 	print ("LB: ", vale_of_noops, np.sum(V_traj), LB-np.sum(extra_V_traj), rew_LB + vale_of_noops + np.sum(V_traj))
					# set_trace()
				LB = max(LB,rew_LB + vale_of_noops + np.sum(V_traj))
				# set_trace()

				env_count += 1

		######### extra pomdps
		# end_time = system_time.clock()	
		return LB, UB, tree_s

	def add_to_belief_library(self, pomdp_solver, belief, max_horizon, gamma, tree_s, all_poss_actions):
		# for (pr,state) in states:
			# (max_horizon,states) not in pomdp_solver.env.belief_library
		rew_t_t = None
		part_of_action_space = all_poss_actions
		if part_of_action_space in pomdp_solver.env.belief_library.keys() and \
			max_horizon in pomdp_solver.env.belief_library[part_of_action_space].keys():
			if belief in pomdp_solver.env.belief_library[part_of_action_space][max_horizon].keys():
				rew_t_t = pomdp_solver.env.belief_library[part_of_action_space][max_horizon][belief]

				# rew_t_t2,_,exp_time,_,_,_,_ = pomdp_solver.compute_V (belief, horizon=max_horizon, max_horizon=max_horizon, \
				# 		one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)

				# if rew_t_t2 != rew_t_t:
				# 	set_trace()

		if rew_t_t is None:
			rew_t_t,_,exp_time,_,_,_,_ = pomdp_solver.compute_V (belief, horizon=max_horizon, max_horizon=max_horizon, \
						one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)

			if len(pomdp_solver.env.belief_library.keys()) == 0:
				pomdp_solver.env.belief_library[True] = {}
				pomdp_solver.env.belief_library[False] = {}

			if max_horizon not in pomdp_solver.env.belief_library[part_of_action_space].keys():
				pomdp_solver.env.belief_library[part_of_action_space][max_horizon] = {}
			pomdp_solver.env.belief_library[part_of_action_space][max_horizon][belief] = rew_t_t

		return rew_t_t, rew_t_t

	# def add_to_belief_library(self, pomdp_solver, states, max_horizon, gamma, tree_s, all_poss_actions):
	# 	max_a = None
	# 	max_Q = None
	# 	max_min_Q = None

	# 	states = states.prob

	# 	for act in pomdp_solver.env.feasible_actions:
	# 		a = act.id
	# 		new_Q = 0.0
	# 		min_Q = None

	# 		for (pr,state) in states:
	# 			if (max_horizon,state) not in pomdp_solver.env.belief_library.keys():
	# 				pomdp_solver.env.belief_library[(max_horizon,state)] = {}
	# 				Q, tree_size, exp_time, immediate_time, new_b = pomdp_solver.compute_Q (Belief([(1.0,state)]), act, horizon=max_horizon, max_horizon=max_horizon, \
	# 											one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)
	# 				pomdp_solver.env.belief_library[(max_horizon,state)][a] = Q
	# 			else:
	# 				if a in pomdp_solver.env.belief_library[(max_horizon,state)].keys():
	# 					Q = pomdp_solver.env.belief_library[(max_horizon,state)][a]
	# 				else:
	# 					Q, tree_size, exp_time, immediate_time, new_b = pomdp_solver.compute_Q (Belief([(1.0,state)]), act, horizon=max_horizon, max_horizon=max_horizon, \
	# 											one_action=False, gamma=gamma, tree_s=tree_s,all_poss_actions=all_poss_actions)
	# 					pomdp_solver.env.belief_library[(max_horizon,state)][a] = Q

	# 			new_Q += pr*Q
	# 			if min_Q is None or min_Q > Q:
	# 				min_Q = Q

	# 		if act in pomdp_solver.env.pomdps_actions and (max_Q is None or new_Q > max_Q):
	# 			max_Q = new_Q
	# 			max_a = act

	# 		if act in pomdp_solver.env.pomdps_actions and (max_min_Q is None or min_Q > max_min_Q):
	# 			max_min_Q = min_Q

	# 	return max_Q, max_min_Q


	def get_tuple (self, index, dim):
		state = np.unravel_index(index,dim)
		new_state = list(state)
		return new_state

	def compute_V (self, beliefs=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False, LB_UB=False, extra_beliefs=None):
		if beliefs is None:
			beliefs = self.beliefs
			if self.env.extra_pomdp_solvers is not None and len(self.env.extra_pomdp_solvers) != 0:
				extra_beliefs = self.extra_beliefs

		max_Q = BoundPair()
		max_a = BoundPair()
		tree_size = tree_s
		time_steps = np.full((self.env.nA,1),1)
		max_time = BoundPair()
		Q_a = BoundPair(np.full((self.env.nA,1),-np.Inf),np.full((self.env.nA,1),-np.Inf))
		leaf_beliefs = []
		count = 0
		
		for act in self.env.feasible_actions:
			a = self.env.feasible_actions_index[count]
			new_Q, tree_size, exp_time, immediate_time, new_b = self.compute_Q(beliefs, act, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP, LB_UB, extra_beliefs)
			
			Q_a.UB[a] = new_Q.UB; Q_a.LB[a] = new_Q.LB; 

			# if abs(new_Q.UB + 2.27280325) < 0.01:
			# 	set_trace()

			time_steps[a] = immediate_time 
			# self.Q[self.env.get_state_index(belief)].append((a,new_Q))
			if max_Q.UB is None or new_Q.UB > max_Q.UB:
				max_Q.UB = new_Q.UB
				max_a.UB = act
				max_time.UB = exp_time.UB
				leaf_beliefs = new_b

			if max_Q.LB is None or new_Q.LB > max_Q.LB:
				max_Q.LB = new_Q.LB
				max_a.LB = act
				max_time.LB = exp_time.LB
				leaf_beliefs = new_b
			
			if print_status:
				# set_trace()
				if horizon == 2:
					print ("**********************************")
				# print ("Q: ", new_Q,a, self.env.actions[a], " horizon: ", horizon)
				# print ("max Q: ", max_Q,max_a, self.env.actions[max_a], " horizon: ", horizon)
			# action_values[horizon-1] = max_a
			# action_values[max_horizon+(horizon-1)] = max_Q
			count += 1


		if max_Q.UB - max_Q.LB < precision:
			print ("UB is less than LB")
			print ("LB,UB ", max_Q.LB, max_Q.UB, max_Q.UB-max_Q.LB)
			# set_trace()
		return max_Q, max_a, max_time, Q_a, tree_size, time_steps, leaf_beliefs
		