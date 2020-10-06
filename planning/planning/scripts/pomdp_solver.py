from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import math


from draw_env import *
print_status = False
NORMALIZE = False
NORMALIZATION_PRECISION = 0.01

class Belief:
	def __init__(self, initial_belief):
		self.prob = initial_belief

	def get_string(self):
		states = self.prob
		def myFunc(e):
			return e[1]
		states.sort(key=myFunc)
		states_str = ""
		for (p,s) in states:
			states_str += "("+str(round(p,2))+","+str(s)+")"
		return states_str

	def __hash__(self):
		states_str = self.get_string()
		# print (states_str)
		return hash(states_str)

	def __eq__(self, other):
		if len(other.prob) == len(self.prob):
			# print (self.get_string(), "==",other.get_string())
			if other.get_string() == self.get_string():
				return True
			# if len(other.prob) > 2:
			# 	set_trace()
			return False
		else:
			return False

class POMDPSolver:
	def __init__(self, env, initial_belief, random):
		self.random = random
		self.belief = Belief(initial_belief)
		self.env = env


	############################33## use belief_prob
	def get_readable_string(self):
		states = self.belief.prob
		def myFunc(e):
			return e[1]
		states.sort(key=myFunc)
		states_str = ""
		for (p,s) in states:
			states_str += "("+str(round(p,2))+","+str(self.env.get_state_tuple(s))+")"
		return states_str

	def update_belief (self, belief_prob, action, observation, all_poss_actions,horizon):
		global NORMALIZE
		new_belief_prob = []
		sum_normalized = 0.0
		count_normalized = 0.0
		for s_p in self.env.get_possible_next_states(observation, belief_prob, action):
			for o_p in self.env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					pr_obs = o_p[1]
					if pr_obs != 0:						
						over_eta = self.compute_1_over_eta (belief_prob,action,observation,all_poss_actions,horizon)
						if over_eta != 0:
							eta = 1.0/over_eta
							up_b_tr = self.update_belief_tr(belief_prob,s_p,action,all_poss_actions,horizon)
							if up_b_tr != 0:
								pr = eta * pr_obs * up_b_tr
								new_belief_prob.append((pr,s_p))
								if pr < NORMALIZATION_PRECISION:
									count_normalized += 1
									sum_normalized += pr
						# else:
						# 	new_belief_prob[s_p] = 0
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


	def update_belief_tr (self, belief_prob, state_p, action, all_poss_actions,horizon):
		sum_s = 0
		for (prob,state) in belief_prob:
			outcomes, steps = self.env.simulate_action(state,action,all_poss_actions,horizon)
			for outcome in outcomes:
				if outcome[1] == state_p:
					sum_s += prob * outcome[0]
		return sum_s

	def compute_1_over_eta (self, belief_prob, action, observation, all_poss_actions,horizon):
		sum_s_p = 0
		# for s_p in range(belief_prob.shape[0]): 
		for s_p in self.env.get_possible_next_states(observation, belief_prob, action):
			for o_p in self.env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					sum_s_p += o_p[1] * self.update_belief_tr(belief_prob,s_p,action,all_poss_actions,horizon)
		return sum_s_p
	###################################
	def update_current_belief(self, action, obs_index, all_poss_actions,horizon):
		# if action not in self.env.navigation_actions:
		# for b in self.belief.prob:
		# 	print ('{0:.16f}'.format(b[0]),b[1])
		self.belief.prob = self.update_belief (self.belief.prob,action,obs_index,all_poss_actions,horizon)
		# for b in self.belief.prob:
		# 	print ('{0:.16f}'.format(b[0]),b[1])
		# else:
		# 	self.belief.prob = self.update_belief (self.belief.prob, self.env.no_action, obs_index)
		# 	self.belief.pos, reward = self.env.simulate_navigation_action(action,self.belief.pos)

	def compute_immediate_reward (self, belief, action, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		immediate_r = 0
		immediate_time = 0
		for (prob,state) in belief.prob:
			outcomes, steps = self.env.simulate_action(state,action,all_poss_actions,horizon)
			for outcome in outcomes:				
				if prob != 0:
					immediate_r += prob * outcome[0] * outcome[2]
					immediate_time += prob * outcome[0] * steps

		return immediate_time, immediate_r

	def compute_Q (self, belief, action, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		tree_size = tree_s + 1

		immediate_r = 0 
		immediate_time = 0
		terminal = True
		leaf_beliefs = []

		for (prob,state) in belief.prob:
			outcomes, steps = self.env.simulate_action(state,action,all_poss_actions,horizon)
			# print (state, action, outcomes)
			for outcome in outcomes:				
				if prob != 0:
					immediate_r += prob * outcome[0] * outcome[2]
					immediate_time += prob * outcome[0] * steps
					terminal = terminal and outcome[3]
					if HPOMDP:
						leaf_beliefs.append((prob * outcome[0],outcome[1],outcome[2],outcome[3],1,False))

		immediate_time = np.round(immediate_time)
		exp_time = int(immediate_time)
		reward = immediate_r ## this is wrong, I should include gamma in teh actual computatyion of reward

		# print (action, immediate_time, exp_time, horizon, horizon > exp_time, horizon-exp_time)
		if print_status:
			print ("***********************************************")
			print ("action: ", action.id, " horizon: ", horizon, ", ", action.name)
			print ("action length: ", int(immediate_time))
			print ("immediate reward: ", immediate_r, " reward: ", reward)
			print ("tree size: ", tree_size)
			# set_trace()
		
		if immediate_r == -np.Inf:
			return -np.Inf,0,0,0,leaf_beliefs

		if not one_action:
			if horizon < exp_time:
				if print_status:
					print ("horizon: ", horizon, " action steps: ", exp_time)
					print ("---- action bigger than horizon ----")
				return -np.Inf,0,0,0, leaf_beliefs

			elif horizon == exp_time or terminal:
				if print_status:
					print ("reward: ", reward - immediate_r, reward)
					# set_trace()
				return reward, tree_size, int(np.round(exp_time)), int(immediate_time), leaf_beliefs

			elif horizon > exp_time and not terminal:
				leaf_beliefs = []
				next_beliefs_value = 0
				next_beliefs_time = 0
				possible_obss = self.env.get_possible_obss(belief.prob,all_poss_actions,horizon)
				# print ("get possible observations", len(possible_obss))

				for obs in possible_obss: ## hack
				# for obs in range(self.env.nO):
					new_belief_prob = self.update_belief(belief.prob, action, obs, all_poss_actions,horizon)

					if len(new_belief_prob) > 0: ## and not set(belief.prob) == set(new_belief_prob):
						# print ("next horizon")
						V,_,max_time,_,tree_size,_,new_bs = self.compute_V(Belief(new_belief_prob), horizon-exp_time, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
						one_over_eta = self.compute_1_over_eta(belief.prob,action,obs,all_poss_actions,horizon)
						next_beliefs_value += one_over_eta * V		
						next_beliefs_time += one_over_eta * max_time
						if HPOMDP:						
							for new_b in new_bs:			
								new_b2 = (new_b[0]*one_over_eta, new_b[1], reward + \
									(gamma**int(immediate_time))*(new_b[2]), new_b[3], new_b[4], new_b[5])
								leaf_beliefs.append(new_b2)

				reward += (gamma ** int(immediate_time)) * next_beliefs_value
				exp_time += next_beliefs_time
				# reward += next_beliefs_value

		if print_status:
			print ("reward: ", reward - immediate_r, reward)
			print (leaf_beliefs)
			# if horizon == 3:
			# 	set_trace()

		return reward, tree_size, int(np.round(exp_time)), int(immediate_time), leaf_beliefs

	def compute_V (self, belief=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		if belief is None:
			belief = self.belief
		max_Q = None
		max_a = None
		Q_a = np.full((self.env.nA,1),-np.Inf)
		tree_size = tree_s
		time_steps = np.full((self.env.nA,1),1)
		max_time = None
		leaf_beliefs = []
		for act in self.env.feasible_actions:
			a = act.id
			new_Q, tree_size, exp_time, immediate_time, new_b = self.compute_Q(belief, act, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			Q_a[a] = new_Q
			time_steps[a] = immediate_time
			# self.Q[self.env.get_state_index(belief)].append((act,new_Q))
			if act in self.env.pomdps_actions and (max_Q is None or new_Q > max_Q):
				max_Q = new_Q
				max_a = act
				max_time = exp_time
				if HPOMDP:						
					leaf_beliefs = new_b

			# print (new_Q,act)
			# if horizon == 2:
			# 	print ("Q: ", new_Q, act, " horizon: ", horizon)

		return max_Q, max_a, max_time, Q_a, tree_size, time_steps, leaf_beliefs
		
	def compute_Q_one_action (self, belief, action, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		# print_status = True
		tree_size = tree_s + 1

		immediate_r = 0 
		immediate_time = 0
		terminal = False
		leaf_beliefs = []

		for (prob,state) in belief.prob:
			outcomes, steps = self.env.simulate_action(state,action,all_poss_actions,horizon)
			for outcome in outcomes:				
				if prob != 0:
					immediate_r += prob * outcome[0] * outcome[2]
					immediate_time += prob * outcome[0] * steps
					terminal = terminal and outcome[3]
					if HPOMDP:
						leaf_beliefs.append((prob * outcome[0],outcome[1],outcome[2],outcome[3],1,False))

		
		immediate_time = np.round(immediate_time)
		exp_time = int(immediate_time)
		reward = immediate_r #/immediate_time*((1.0-gamma**immediate_time)/(1.0-gamma))

		# if one_action and immediate_time != horizon:
		# 	set_trace()


		if print_status:
			print ("***********************************************")
			# print (belief.prob)
			print_str = ""
			for i in range(len(belief.prob)):
				print_str += "(" + str(belief.prob[i][0]) + "," + str(self.env.get_state_tuple(belief.prob[i][1])) + ")"
			print (print_str)
			print ("action: ", action.id, " horizon: ", horizon, ", ", action.name)
			print ("action length: ", int(immediate_time))
			print ("immediate reward: ", immediate_r, " reward: ", reward)
			print ("tree size: ", tree_size)

		if immediate_r == -np.Inf:
			return -np.Inf,0,0,0,leaf_beliefs

		if not one_action:
			if horizon < exp_time:
				if print_status:
					print ("horizon: ", horizon, " action steps: ", exp_time)
					print ("---- action bigger than horizon ----")
				return -np.Inf,0,0,0,leaf_beliefs

			elif horizon == exp_time or terminal:
				if print_status:
					print ("reward: ", reward - immediate_r, reward)
					# set_trace()
				return reward, tree_size, int(np.round(exp_time)), int(immediate_time), leaf_beliefs

			elif horizon > exp_time and not terminal:
				leaf_beliefs = []
				next_beliefs_value = 0
				next_beliefs_time = 0
				possible_obss = self.env.get_possible_obss(belief.prob,all_poss_actions,horizon)
				# print ("get possible observations", len(possible_obss))

				for obs in possible_obss: ## hack
				# for obs in range(self.env.nO):
					new_belief_prob = self.update_belief(belief.prob, action, obs, all_poss_actions,horizon)

					if len(new_belief_prob) > 0: ## and not set(belief.prob) == set(new_belief_prob):
						# print ("next horizon")
						# print (belief.prob, self.env.get_state_tuple(belief.prob[0][1]))
						# print (self.env.get_observation_tuple(obs), new_belief_prob)
						V, tree_size, max_time,_,new_bs = self.compute_Q_one_action(Belief(new_belief_prob), action, horizon-exp_time, max_horizon, one_action, gamma, tree_size, \
							all_poss_actions, HPOMDP)
						one_over_eta = self.compute_1_over_eta(belief.prob,action,obs,all_poss_actions,horizon)
						next_beliefs_value += one_over_eta * V		
						next_beliefs_time += one_over_eta * max_time
						if HPOMDP:						
							for new_b in new_bs:			
								new_b2 = (new_b[0]*one_over_eta, new_b[1], reward + \
									(gamma**int(immediate_time))*(new_b[2]), new_b[3], new_b[4], new_b[5])
								leaf_beliefs.append(new_b2)

				reward += (gamma ** int(immediate_time)) * next_beliefs_value
				exp_time += next_beliefs_time
				# reward += next_beliefs_value

		if print_status:
			print ("reward: ", reward - immediate_r, reward)
			print ("**********************************")
			# set_trace()
		return reward, tree_size, int(np.round(exp_time)), int(immediate_time), leaf_beliefs
