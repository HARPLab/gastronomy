from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import math
import sys


from draw_env import *
from pomdp_solver import *

print_status = False

class MultiIndPOMDPSolver:
	def __init__(self, env, initial_beliefs, random):
		self.random = random
		self.beliefs = []
		self.env = env
		for i in range(len(env.pomdp_tasks)):
			self.beliefs.append(Belief(initial_beliefs[i]))
		self.envs = env.pomdp_tasks


	############################33## use belief_prob
	def update_belief (self, env, belief_prob, action, observation, all_poss_actions,horizon):
		new_belief_prob = []
		for s_p in env.get_possible_next_states(observation):
			for o_p in env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					pr_obs = o_p[1]
					if pr_obs != 0:						
						over_eta = self.compute_1_over_eta (env,belief_prob,action,observation, all_poss_actions,horizon)
						if over_eta != 0:
							eta = 1.0/over_eta
							up_b_tr = self.update_belief_tr(env,belief_prob,s_p,action,all_poss_actions,horizon)
							if up_b_tr != 0:
								new_belief_prob.append((eta * pr_obs * up_b_tr,s_p))

					# 	else:
					# 		new_belief_prob[s_p] = 0
					# else:
					# 	new_belief_prob[s_p] = 0

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
		for s_p in env.get_possible_next_states(observation):
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

	def are_feasible_obs(self, possible_obss,obss):
		prev_obs = None
		for e in range(len(self.envs)): 
			env = self.envs[e]
			new_obs = self.envs[e].get_observation_tuple(possible_obss[e][obss[e]])
			
			if prev_obs is not None and not (prev_obs[env.obs_feature_indices["x"]] == new_obs[env.obs_feature_indices["x"]] and \
				prev_obs[env.obs_feature_indices["y"]] == new_obs[env.obs_feature_indices["y"]]):
				return False
			prev_obs = new_obs

		return True

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

	def compute_Q (self, beliefs, actions, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		tree_size = tree_s + 1
		immediate_r = 0 
		immediate_time = 0
		leaf_beliefs = []
		time_steps = horizon
		if HPOMDP:
			time,rew = self.compute_immediate_reward (self.envs[int(actions[0].id)], beliefs[actions[0].id], actions[0], horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
			immediate_r += rew
			immediate_time += time
			time_steps = int(np.round(immediate_time))

			env_count = 0
			for env in self.envs:
				if env_count != int(actions[0].id):
					time,rew = self.compute_immediate_reward (env, beliefs[env_count], actions[env_count], time_steps, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
					immediate_r += rew

				env_count += 1

		else:
			env_count = 0
			for env in self.envs:
				time,rew = self.compute_immediate_reward (env, beliefs[env_count], actions[env_count], horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
				immediate_r += rew
				if env_count == 0:
					immediate_time += time

				env_count += 1

		immediate_time = np.round(immediate_time)
		exp_time = int(immediate_time)
		reward = immediate_r##/immediate_time*((1.0-gamma**immediate_time)/(1.0-gamma))

		if print_status:
			if horizon == 3:
				print ("***********************************************")
			print ("action: ", [a.id for a in actions], " horizon: ", horizon, ", ", [a.name for a in actions])
			print ("action length: ", int(immediate_time))
			print ("immediate reward: ", immediate_r)
			print ("tree size: ", tree_size)

		if immediate_r == -np.Inf:
			return -np.Inf,0,0,0,leaf_beliefs

		if not one_action:
			if horizon < exp_time:
				if print_status:
					print ("horizon: ", horizon, " action steps: ", exp_time)
					print ("---- action bigger than horizon ----")
					# set_trace()
				return -np.Inf,0,0,0,leaf_beliefs

			elif horizon > exp_time:
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

				count = 0

				while count < possible_obss_len:
					env_count = 0
					next_beliefs_value = 0
					next_beliefs_time = 0
					new_beliefs = []
					eta = 1
					obss = self.get_tuple(count,possible_obss_dim)
					if self.are_feasible_obs(possible_obss,obss):
						for e in range(len(self.envs)):
							env = self.envs[e]					
							obs = possible_obss[e][obss[e]]
							new_belief_prob = self.update_belief(env, beliefs[e].prob, actions[e], obs, all_poss_actions,time_steps)

							if len(new_belief_prob) > 0:
								new_beliefs.append(Belief(new_belief_prob))
								eta *= self.compute_1_over_eta(env,beliefs[e].prob,actions[e],obs,all_poss_actions,time_steps)

						if len(new_beliefs) == len(self.envs):
							V,_,max_time,_,tree_size,_,leaf_beliefs = self.compute_V(new_beliefs, horizon-int(immediate_time), max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
							next_beliefs_value +=  eta * V
							next_beliefs_time += eta * max_time

							reward += (gamma ** int(immediate_time)) * next_beliefs_value
							exp_time += next_beliefs_time


					count += 1
		if print_status:
			print ("reward: ", reward - immediate_r, reward)
			# set_trace()

		return reward, tree_size, int(np.round(exp_time)), int(immediate_time), leaf_beliefs

	def get_tuple (self, index, dim):
		state = np.unravel_index(index,dim)
		new_state = list(state)
		return new_state

	def compute_V (self, beliefs=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		if beliefs is None:
			beliefs = self.beliefs
		max_Q = None
		max_a = None
		tree_size = tree_s 
		time_steps = np.full((self.env.nA,1),1)
		max_time = None
		Q_a = np.full((self.env.nA,1),-np.Inf)
		leaf_beliefs = []
		count = 0
		
		for act in self.env.feasible_actions:
			a = self.env.feasible_actions_index[count]
			new_Q, tree_size, exp_time, immediate_time, new_b = self.compute_Q(beliefs, act, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			Q_a[a] = new_Q
			time_steps[a] = immediate_time
			# self.Q[self.env.get_state_index(belief)].append((a,new_Q))
			if max_Q is None or new_Q > max_Q:
				max_Q = new_Q
				max_a = act
				max_time = exp_time
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

		return max_Q, max_a, max_time, Q_a, tree_size, time_steps, leaf_beliefs
		