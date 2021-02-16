from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import math
from pomdp_client import *
from goal_pomdp_solver import Belief

from draw_env import *
print_status = False


class POMDPSolver:
	def __init__(self, env, initial_belief, random = None):
		self.random = random
		self.belief = Belief(initial_belief)
		self.env = env

	def reset (self, new_belief_state):
		self.belief = Belief(new_belief_state)

	def get_belief (self):
		return self.belief
	############################33## use belief_prob
	def update_belief (self, belief_prob, action, observation, all_poss_actions, horizon):
		new_belief_prob = []
		for s_p in self.env.get_possible_next_states(belief_prob, action, all_poss_actions, horizon):
			# print ("s:",self.env.get_state_tuple(s_p))
			for o_p in self.env.simulate_observation(s_p,action):
				# print ("o:",self.env.get_observation_tuple(o_p[0]))
				if o_p[0] == observation:
					pr_obs = o_p[1]
					if pr_obs != 0:						
						over_eta = self.compute_1_over_eta (belief_prob,action,observation,all_poss_actions,horizon)
						if over_eta != 0:
							eta = 1.0/over_eta
							up_b_tr = self.update_belief_tr(belief_prob,s_p,action,all_poss_actions,horizon)
							if up_b_tr != 0:
								# print (self.env.task.table.id,eta,pr_obs,up_b_tr,self.env.get_state_tuple(s_p))
								new_belief_prob.append((eta * pr_obs * up_b_tr,s_p))
						# else:
						# 	new_belief_prob[s_p] = 0
					# else:
					# 	new_belief_prob[s_p] = 0
		# else:
		# 	if print_status:
		# 		print ("Errorrrrrrr - update_belief in pomdp_solver.py")
		# 		set_trace()

		return new_belief_prob

	def update_belief_tr (self, belief_prob, state_p, action, all_poss_actions,horizon):
		sum_s = 0
		for (prob,state) in belief_prob:
			outcomes, steps = self.env.simulate_action(state,action,all_poss_actions,horizon)
			for outcome in outcomes:
				if outcome[1] == state_p:
					sum_s += prob * outcome[0]
		return sum_s

	def compute_1_over_eta (self, belief_prob, action, observation, all_poss_actions, horizon):
		sum_s_p = 0
		# for s_p in range(belief_prob.shape[0]): 
		for s_p in self.env.get_possible_next_states(belief_prob, action, all_poss_actions, horizon):
			for o_p in self.env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					sum_s_p += o_p[1] * self.update_belief_tr(belief_prob,s_p,action,all_poss_actions,horizon)
		return sum_s_p
	###################################
	def update_current_belief(self, action, obs_index, all_poss_actions,horizon):
		# if action not in self.env.navigation_actions:
		# for b in self.belief.prob:
		# 	print ('{0:.16f}'.format(b[0]),b[1])
		# if action.type != 2:
		self.belief.prob = self.update_belief (self.belief.prob,action,obs_index,all_poss_actions,horizon)
		# else:
			# state = self.env.get_state_tuple(obs_index) # assuming this function will get the state index instead of the observation
			# self.belief.prob = Belief((1.0,state))
		# for b in self.belief.prob:
		# 	print ('{0:.16f}'.format(b[0]),b[1])
		# else:
		# 	self.belief.prob = self.update_belief (self.belief.prob, self.env.no_action, obs_index)
		# 	self.belief.pos, reward = self.env.simulate_navigation_action(action,self.belief.pos)


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
					if HPOMDP: #  or action.type == 2
						leaf_beliefs.append((prob * outcome[0],outcome[1],outcome[2],outcome[3],1,False))

		immediate_time = np.round(immediate_time)
		exp_time = int(immediate_time)
		reward = immediate_r ## this is wrong, I should include gamma in teh actual computatyion of reward

		# print (action, immediate_time, exp_time, horizon, horizon > exp_time, horizon-exp_time)
		if print_status:
			# if (horizon == 3):
			# 	set_trace()
			print ("***********************************************")
			print ("action: ", action.id, " horizon: ", horizon, ", ", action.name)
			print ("action length: ", int(immediate_time))
			print ("immediate reward: ", immediate_r, " reward: ", reward)
			print ("tree size: ", tree_size)
			for b in belief.prob:
				print ("belief: ", (b[0],self.env.get_state_tuple(b[1])))
			# if action.id == 5:
			# 	set_trace()

			# if horizon == 3 and self.env.task.table.id == 2:
			# 	set_trace()
			# if action.type == 2 and len(belief.prob) > 1 and horizon == 3:
			# 	set_trace()

			# if len(belief.prob) > 1 and horizon == 3 and self.env.task.table.id == 2:
			# 	set_trace()
		
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
				next_beliefs_value = 0
				next_beliefs_time = 0

				# if action.type != 2:
				leaf_beliefs = []
				possible_obss = self.env.get_possible_obss(belief.prob,all_poss_actions,horizon)
				# print ("get possible observations", len(possible_obss))

				for obs in possible_obss: ## hack
					# print (len(self.env.get_state_tuple(belief.prob[0][1])), self.env.get_state_tuple(belief.prob[0][1]))
					# print (len(self.env.get_observation_tuple(obs)[1]), self.env.get_observation_tuple(obs))
					new_belief_prob = self.update_belief(belief.prob, action, obs, all_poss_actions,horizon)
					# print (new_belief_prob)

					if len(new_belief_prob) > 0: ## and not set(belief.prob) == set(new_belief_prob):
						# print ("next horizon")
						# print (new_belief_prob)
						# print (self.env.get_state_tuple(new_belief_prob[0][1]))
						V,_,max_time,_,tree_size,_,new_bs = self.compute_V(Belief(new_belief_prob), horizon-exp_time, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
						one_over_eta = self.compute_1_over_eta(belief.prob,action,obs,all_poss_actions,horizon)
						next_beliefs_value += one_over_eta * V		
						next_beliefs_time += one_over_eta * max_time
						if HPOMDP:						
							for new_b in new_bs:			
								new_b2 = (new_b[0]*one_over_eta, new_b[1], reward + \
									(gamma**int(immediate_time))*(new_b[2]), new_b[3], new_b[4], new_b[5])
								leaf_beliefs.append(new_b2)

				# else: # oracle
				# 	for b in leaf_beliefs:
				# 		new_belief_prob = [(1.0,b[1])]
				# 		V,_,max_time,_,tree_size,_,new_bs = self.compute_V(Belief(new_belief_prob), horizon-exp_time, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
				# 		one_over_eta = b[0]
				# 		next_beliefs_value += one_over_eta * V		
				# 		next_beliefs_time += one_over_eta * max_time


				reward += (gamma ** int(immediate_time)) * next_beliefs_value
				exp_time += next_beliefs_time
				# reward += next_beliefs_value

		if print_status:
			print ("reward: ", reward - immediate_r, reward)
			print (leaf_beliefs)

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
					# print (self.env.get_state_tuple(outcome[1]),prob,outcome[0],self.env.get_state_tuple(outcome[1]))
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
			print (belief.prob)
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

