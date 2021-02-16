from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import math
import sys


from draw_env import *
from goal_pomdp_solver import *

print_status = False

class Beliefs:
	def __init__(self, initial_beliefs):
		self.probs = []
		for b in initial_beliefs:
			self.probs.append(b)

	def get_string(self):
		beliefs_str = ""
		for belief in self.probs:
			beliefs_str += "[" + belief.get_string() + "]"

		return beliefs_str

	def __hash__(self):
		states_str = self.get_string()
		return hash(states_str)

	def __eq__(self, other):
		if len(other.probs) != len(self.probs):
			return False

		for i in range(len(self.probs)):
			if len(other.probs[i].prob) != len(self.probs[i].prob):
				return False
			elif other.probs[i].get_string() != self.probs[i].get_string():
				return False
		return True

class MultiIndGoalPOMDPSolver:
	def __init__(self, env, initial_beliefs, random):
		self.num_of_iterations = 1000
		self.random = random
		self.beliefs = []
		self.env = env
		self.threshold = 1
		for i in range(len(env.pomdp_tasks)):
			self.beliefs.append(Belief(initial_beliefs[i]))
		self.envs = env.pomdp_tasks
		self.num_step = 0

	def reset (self, new_beliefs_state):
		self.beliefs = new_beliefs_state

	############################33## use belief_prob
	def update_belief (self, env, belief_prob, action, observation, all_poss_actions,horizon):
		new_belief_prob = []
		for s_p in env.get_possible_next_states(belief_prob, action, all_poss_actions, horizon):
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
		for s_p in env.get_possible_next_states(belief_prob, action, all_poss_actions, horizon):
			for o_p in env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					sum_s_p += o_p[1] * self.update_belief_tr(env,belief_prob,s_p,action,all_poss_actions,horizon)
		return sum_s_p
	###################################
	def get_tuple (self, index, dim):
		state = np.unravel_index(index,dim)
		new_state = list(state)
		return new_state

	def update_current_belief(self, actions, obs_indices, all_poss_actions,horizon):
		# if action not in self.env.navigation_actions:
		for i in range(len(self.envs)):
			self.beliefs[i].prob = self.update_belief(self.envs[i], self.beliefs[i].prob, actions[i], obs_indices[i], all_poss_actions,horizon)
		# else:
		# 	self.belief.prob = self.update_belief (self.belief.prob, self.env.no_action, obs_index)
		# 	self.belief.pos, reward = self.env.simulate_navigation_action(action,self.belief.pos)

	def are_feasible_obs(self, possible_obss, obss, actions):
		prev_obs = None
		for e in range(len(self.envs)): 
			env = self.envs[e]
			# if actions[e].type != 2:
			new_obs_pair = self.envs[e].get_observation_tuple(possible_obss[e][obss[e]])
			obs_type = new_obs_pair[0]
			new_obs = new_obs_pair[1]
			new_x = new_obs[env.obs_feature_indices[obs_type]["x"]]
			new_y = new_obs[env.obs_feature_indices[obs_type]["y"]]
			# else:
			# 	new_obs = self.envs[e].get_state_tuple(possible_obss[e][obss[e]][1])
			# 	new_x = new_obs[env.feature_indices["x"]]
			# 	new_y = new_obs[env.feature_indices["y"]]
			
			if prev_obs is not None and not (prev_obs["x"] == new_x and prev_obs["y"] == new_y):
				return False
			else:
				prev_obs = {}
				prev_obs["x"] = new_x
				prev_obs["y"] = new_y

		return True

######################################################################################################
	def compute_cost (self, env, belief, action, horizon, max_horizon, gamma, tree_s=0, all_poss_actions=False):
		immediate_r = 0 
		immediate_time = 0
		terminal = True
		sampled_next_state = None
		rand_num = self.random.choice(100)
		sum_prob = 0

		for (prob,state) in belief.prob:
			outcomes, steps = env.simulate_action(state,action,all_poss_actions,horizon)

			# print (state, action, outcomes)
			for outcome in outcomes:				
				if prob != 0:
					immediate_r += prob * outcome[0] * outcome[2]
					immediate_time += prob * outcome[0] * steps
					terminal = terminal and outcome[3]

					######################################################################RTDP-bel
					if len(self.sampled_states) != 0 and sampled_next_state is None:
						sum_prob += outcome[0]*100	
						if rand_num < sum_prob:
							sampled_next_state = outcome[1]
					######################################################################RTDP-bel
		immediate_time = np.round(immediate_time)
		exp_time = int(immediate_time)
		reward = immediate_r 

		if sampled_next_state is None:
			print (sampled_next_state)
			set_trace()
			outcomes, steps = env.simulate_action(state,action,all_poss_actions,horizon)
		return reward, exp_time, sampled_next_state, terminal

	def compute_Q (self, beliefs, actions, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		tree_size = tree_s + 1
		sampled_observations = None
		env_count = 0
		immediate_time = 0
		immediate_reward = 0
		total_reward = 0
		immediate_terminal = True
		sampled_next_states = []
		time_steps = horizon
		for env_count in range(len(self.envs)):
			reward, time, sampled_next_state, terminal = self.compute_cost (self.envs[env_count], beliefs[env_count], actions[env_count], horizon, max_horizon, gamma, tree_s, all_poss_actions)
			if env_count == 0:
				immediate_time = time

			immediate_reward += reward
			immediate_terminal = immediate_terminal and terminal
			sampled_next_states.append(sampled_next_state)

		total_reward = immediate_reward
		env_count = 0
		possible_obss = []
		possible_obss_len = 1
		possible_obss_dim = ()
		for e in range(len(self.envs)):
			env = self.envs[e]
			# if actions[e].type != 2:
			obss = list(env.get_possible_obss(beliefs[e].prob,all_poss_actions,time_steps))
			# else:
				# obss= leaf_beliefs

			possible_obss.append(obss)
			possible_obss_len *= len(obss)
			possible_obss_dim += (len(obss),)

		count = 0
		Q_obs = []

		while count < possible_obss_len:
			env_count = 0
			next_beliefs_value = 0
			new_beliefs = []
			eta = 1
			obss = self.get_tuple(count,possible_obss_dim)
			observations = []
			if self.are_feasible_obs(possible_obss, obss, actions):
				for e in range(len(self.envs)):
					env = self.envs[e]					
					obs = possible_obss[e][obss[e]]

					new_belief_prob = self.update_belief(env, beliefs[e].prob, actions[e], obs, all_poss_actions,time_steps)

					if len(new_belief_prob) > 0:
						new_beliefs.append(Belief(new_belief_prob))
						eta *= self.compute_1_over_eta(env,beliefs[e].prob,actions[e],obs,all_poss_actions,time_steps)
						observations.append(obs)


				if len(new_beliefs) == len(self.envs):
					# if print_status:
					# 	actions[0].print()
					# 	print (beliefs, reward)
					# 	for e in range(len(beliefs)):
					# 		belief = beliefs[e]
					# 		for b in belief.prob:
					# 			print ("belief: ", (b[0],self.envs[e].get_state_tuple(b[1])))
					# 	for e in range(len(observations)):
					# 		ooo = self.envs[e].get_observation_tuple(observations[e])
					# 		print ("obs: ", ooo)
					# 		if actions[0].type == Action_Type.CLARIFICATION:
					# 			an = self.envs[0].obs_feature_indices[Observation_Type.HUMAN_INPUT]["answer"]
					# 			print("answer: ", ooo[1][an])
					# 	print (new_beliefs, reward)
					# 	for e in range(len(new_beliefs)):
					# 		belief = new_beliefs[e]
					# 		for b in belief.prob:
					# 			print ("belief: ", (b[0],self.envs[e].get_state_tuple(b[1])))

					V = self.get_hashed_value (new_beliefs, actions, all_poss_actions)
					next_beliefs_value =  eta * V
					Q_obs.append((observations,eta,V))
					# if print_status:
					# 	print (observations, eta, V)

					total_reward += (gamma ** int(immediate_time)) * next_beliefs_value

			count += 1
		# print (Q_obs)
		return total_reward, immediate_reward, tree_size, int(immediate_time), sampled_next_states, immediate_terminal, Q_obs

	def get_hashed_value (self, new_beliefs, actions, all_poss_actions):
		# if actions[0].type == Action_Type.CLARIFICATION and self.num_step == 3:
		# 	print ("hashed val: ", self.env.get_from_belief_library(self.envs, Beliefs(new_beliefs), all_poss_actions))
			# set_trace()
		return self.env.get_from_belief_library(self.envs, Beliefs(new_beliefs), all_poss_actions)

	def compute_V (self, beliefs=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		global print_status
		if beliefs is None:
			beliefs = self.beliefs

		min_Q = None
		min_R = None
		min_a = None
		Q_a = np.full((self.env.nA,1),np.Inf)
		tree_size = tree_s
		time_steps = np.full((self.env.nA,1),1)
		min_time = None
		min_sampled_next_states = None
		min_terminal = None
		count = 0
		min_Q_obs = None
		error = 0
		for act in self.env.feasible_actions:
			# if act[0].type == Action_Type.CLARIFICATION and self.num_step == 3:
			# 	set_trace()
			# if self.num_step >= 3:
			# 	if "no op" in act[0].name:
			# 		print_status = True
			# 	elif act[0].type == Action_Type.CLARIFICATION:
			# 		print_status = True
			# 	else:
			# 		print_status = False
			a = self.env.feasible_actions_index[count]
			new_Q, new_R, tree_size, immediate_time, sampled_next_states, terminal, Q_obs = self.compute_Q(beliefs, act, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			# if self.num_step >= 3:
			# 	print ("Q: ", new_Q, " action: ", act[0].name)
			Q_a[a] = new_Q
			time_steps[a] = immediate_time
			
			if (min_Q is None or new_Q < min_Q):
				min_Q = new_Q
				min_R = new_R
				min_a = act
				min_time = immediate_time
				min_sampled_next_states = sampled_next_states
				min_terminal = terminal
				min_Q_obs = Q_obs

			count += 1

		# if print_status or self.num_step >= 3:
		# 	# for belief in beliefs:
		# 	# 	for b in belief.prob:
		# 	# 		print ("belief: ", (b[0],self.env.get_state_tuple(b[1])))
		# 	for e in range(len(self.envs)):
		# 		print ("--- compute V: ", " minQ: ", min_Q, " min_act: ", min_a[e].name, " min_time: ", min_time, " min_sampled_next_state: ", min_sampled_next_states[e])
		
		self.env.add_to_belief_library(Beliefs(beliefs), min_Q, all_poss_actions)
		if not min_terminal:
			###################################################################### RTDP-bel
			sampled_observations = []
			if len(self.sampled_states) != 0 and min_sampled_next_states is not None:
				for e in range(len(self.envs)):
					env = self.envs[e]
					obss = env.simulate_observation(min_sampled_next_states[e],min_a[e])
					if len(obss) == 0:
						set_trace()
						obss = env.simulate_observation(min_sampled_next_states[e],min_a[e])
					rand_num = self.random.choice(100)
					sum_prob = 0
					for (temp_obs,prob) in obss:
						sum_prob += prob*100	
						if rand_num < sum_prob:
							sampled_observations.append(temp_obs)
							break
			###################################################################### RTDP-bel
			new_beliefs = []
			selected_obs = []
			for e in range(len(self.envs)):
				env = self.envs[e]			
				obs = sampled_observations[e]
				selected_obs.append(obs)
				# if actions[e].type != 2:
				new_belief_prob = self.update_belief(env, beliefs[e].prob, min_a[e], obs, all_poss_actions,time_steps)

				if len(new_belief_prob) > 0:
					new_beliefs.append(Belief(new_belief_prob))

			if len(new_beliefs) == len(self.envs) and horizon > 0:
				self.sampled_states = min_sampled_next_states
				next_V, _, _, _, _, _, _, next_error = self.compute_V(new_beliefs, horizon-min_time, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
				min_VV = min_R
				for o in min_Q_obs:
					if o[0] == selected_obs:
						min_VV += (gamma ** int(immediate_time)) * o[1] * next_V
					else:
						min_VV += (gamma ** int(immediate_time)) * o[1] * o[2]
				# if min_VV != np.Inf:
				# 	set_trace()
				error += np.abs(min_VV-min_Q) + next_error
				# print (error, next_error, min_VV, min_Q)
				# set_trace()
				# if print_status or self.num_step >= 3:
				# 	for e in range(len(self.envs)):
				# 		print ("--- compute V: ", " minQ: ", min_Q, " min_act: ", min_a[e].name, " min_time: ", min_time, " min_sampled_next_state: ", min_sampled_next_states[e])

				# if print_status or self.num_step >= 3:
				# 	for e in range(len(self.envs)):
				# 		print ("*** compute V: ", " minQ: ", min_VV, " min_act: ", min_a[e].name, " min_time: ", min_time, " min_sampled_next_state: ", min_sampled_next_states[e])

				min_Q = min_VV
			elif horizon <= 0:
				return np.Inf, self.env.all_no_ops['1'], 1, Q_a, tree_size, time_steps, None, np.Inf

		else:
			print ("Terminal: ", max_horizon - horizon)
			# set_trace()
			# for e in range(len(beliefs)):
			# 	belief = beliefs[e]
			# 	for b in belief.prob:
			# 		print ("belief: ", (b[0],self.envs[e].get_state_tuple(b[1])))
		
		self.env.add_to_belief_library(Beliefs(beliefs), min_Q, all_poss_actions)
		return min_Q, min_a, min_time, Q_a, tree_size, time_steps, None, error
		
	def run_RTDP_bel (self, beliefs=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		global print_status
		# if self.num_step >= 3:
		# 	print_status = True
		if beliefs is None:
			beliefs = self.beliefs
		action_len = 20
		action_history = np.full((action_len,1),-1)
		error_history = np.full((action_len,1),-1)
		count = 0
		while count < self.num_of_iterations:
			self.sampled_states = []
			for belief in beliefs:
				rand_num = self.random.choice(100)
				sum_prob = 0
				for (prob,state) in belief.prob:
					sum_prob += prob*100
					if rand_num < sum_prob:	
						self.sampled_states.append(state)
						break

			min_Q, min_a, min_time, Q_a, tree_size, time_steps, leaf_beliefs, error = self.compute_V (beliefs, horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
			for e in range(len(self.envs)):
				print ("action: ", min_a[e].name, " Q: ", min_Q, " time: ", min_time)
				print ("error: ", error)
			
			# min_action_id = np.argmin(Q_a)
			# if np.all(action_history == min_action_id):
			# 	print ("action selected at iteration ", count)
			# 	break

			# action_history[count%action_len] = min_action_id

			# min_action_id = np.argmin(Q_a)
			error_history[count%action_len] = error
			if np.mean(error_history) < self.threshold:
				print ("action selected at iteration ", count)
				break

			count += 1

		return min_Q, min_a, min_time, Q_a, tree_size, time_steps, None


	def run_RTDP_bel_Q (self, beliefs=None, actions=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		num_of_iterations = 1
		tree_size = 0
		if beliefs is None:
			beliefs = self.beliefs
		
		count = 0
		min_Q = 0
		while count < num_of_iterations:
			self.sampled_states = []
			for belief in beliefs:
				rand_num = self.random.choice(100)
				sum_prob = 0
				for (prob,state) in belief.prob:
					sum_prob += prob*100
					if rand_num < sum_prob:	
						self.sampled_states.append(state)
						break


			######################################################################################
			cost, tree_size, immediate_time, sampled_next_states, terminal = self.compute_Q(beliefs, actions, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			min_time = immediate_time
			if not terminal:
				sampled_observations = []
				if len(self.sampled_states) != 0 and sampled_next_states is not None:
					for e in range(len(self.envs)):
						env = self.envs[e]
						obss = env.simulate_observation(sampled_next_states[e],actions[e])
						rand_num = self.random.choice(100)
						sum_prob = 0
						for (temp_obs,prob) in obss:
							sum_prob += prob*100	
							if rand_num < sum_prob:
								sampled_observations.append(temp_obs)
								break
				###################################################################### RTDP-bel
				new_beliefs = []
				for e in range(len(self.envs)):
					env = self.envs[e]					
					obs = sampled_observations[e]

					# if actions[e].type != 2:
					new_belief_prob = self.update_belief(env, beliefs[e].prob, actions[e], obs, all_poss_actions,horizon)

					if len(new_belief_prob) > 0:
						new_beliefs.append(Belief(new_belief_prob))

				if len(new_beliefs) == len(self.envs):
					min_Q, min_a, min_time, Q_a, tree_size, time_steps, leaf_beliefs = self.compute_V (new_beliefs, horizon-min_time, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
					# if print_status:
					for e in range(len(self.envs)):
						print ("action: ", min_a[e].name, " Q: ", min_Q, " time: ", min_time)
					print (Q_a)
				# set_trace()

			count += 1

		return min_Q+cost, immediate_time