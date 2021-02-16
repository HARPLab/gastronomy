from pdb import set_trace
import numpy as np
from copy import deepcopy
from time import sleep
import math
from pomdp_client import *

from draw_env import *
print_status = False

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
			states_str += "("+str(round(p,1))+","+str(s)+")"
		return states_str

	def __hash__(self):
		states_str = self.get_string()
		return hash(states_str)

	def __eq__(self, other):
		if len(other.prob) == len(self.prob):
			if other.get_string() == self.get_string():
				return True
			return False
		else:
			return False

class GoalPOMDPSolver:
	def __init__(self, env, initial_belief, random = None):
		self.random = random
		self.belief = Belief(initial_belief)
		self.env = env
		self.num_of_iterations = 1000

	def reset (self, new_belief_state):
		self.belief = Belief(new_belief_state)

	def get_belief (self):
		return self.belief
	############################33## use belief_prob
	def update_belief (self, belief_prob, action, observation, all_poss_actions, horizon):
		new_belief_prob = []
		for s_p in self.env.get_possible_next_states(belief_prob, action, all_poss_actions, horizon):
			for o_p in self.env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					pr_obs = o_p[1]
					if pr_obs != 0:						
						over_eta = self.compute_1_over_eta (belief_prob,action,observation,all_poss_actions,horizon)
						if over_eta != 0:
							eta = 1.0/over_eta
							up_b_tr = self.update_belief_tr(belief_prob,s_p,action,all_poss_actions,horizon)
							if up_b_tr != 0:
								new_belief_prob.append((eta * pr_obs * up_b_tr,s_p))

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
		for s_p in self.env.get_possible_next_states(belief_prob, action, all_poss_actions, horizon):
			for o_p in self.env.simulate_observation(s_p,action):
				if o_p[0] == observation:
					sum_s_p += o_p[1] * self.update_belief_tr(belief_prob,s_p,action,all_poss_actions,horizon)
		return sum_s_p
	###################################
	def update_current_belief(self, action, obs_index, all_poss_actions,horizon):
		self.belief.prob = self.update_belief (self.belief.prob,action,obs_index,all_poss_actions,horizon)

###############################################################################################################################
	def compute_cost (self, belief, action, horizon, max_horizon, gamma, tree_s=0, all_poss_actions=False):
		immediate_r = 0 
		immediate_time = 0
		terminal = True
		sampled_next_state = None
		rand_num = self.random.choice(100)
		sum_prob = 0

		for (prob,state) in belief.prob:
			outcomes, steps = self.env.simulate_action(state,action,all_poss_actions,horizon)

			# print (state, action, outcomes)
			for outcome in outcomes:				
				if prob != 0:
					immediate_r += prob * outcome[0] * outcome[2]
					immediate_time += prob * outcome[0] * steps
					terminal = terminal and outcome[3]

					######################################################################RTDP-bel
					if self.sampled_state is not None and sampled_next_state is None:
						sum_prob += outcome[0]*100	
						if rand_num < sum_prob:
							sampled_next_state = outcome[1]
					######################################################################RTDP-bel

		immediate_time = np.round(immediate_time)
		exp_time = int(immediate_time)
		reward = immediate_r 
		return reward, exp_time, sampled_next_state, terminal

	def compute_Q (self, belief, action, horizon, max_horizon, one_action, gamma, tree_s=0, all_poss_actions=False, HPOMDP=False):
		
		tree_size = tree_s + 1
		sampled_observation = None
		reward, immediate_time, sampled_next_state, terminal = self.compute_cost (belief, action, horizon, max_horizon, gamma, tree_s, all_poss_actions)
		
		possible_obss = self.env.get_possible_obss(belief.prob,all_poss_actions,horizon)
		next_beliefs_value = 0

		for obs in possible_obss: ## hack
			new_belief_prob = self.update_belief(belief.prob, action, obs, all_poss_actions,horizon)

			if len(new_belief_prob) > 0: 
				V = self.get_hashed_value (Belief(new_belief_prob), action, all_poss_actions)
				one_over_eta = self.compute_1_over_eta(belief.prob,action,obs,all_poss_actions,horizon)
				next_beliefs_value += one_over_eta * V		
			
				if print_status:
					print ("new_belief_prob: ", new_belief_prob, " obss: ", len(possible_obss), "-", obs)
					for b in new_belief_prob:
						print ("new belief: ", (b[0],self.env.get_state_tuple(b[1])))

		if print_status:
			print ("***********************************************")
			print ("action: ", action.id, " horizon: ", horizon, ", ", action.name)
			print ("action length: ", int(immediate_time))
			print ("immediate cost: ", reward, " future cost: ", (gamma ** int(immediate_time)) * next_beliefs_value)
			
			for b in belief.prob:
				print ("belief: ", (b[0],self.env.get_state_tuple(b[1])))
			

		reward += (gamma ** int(immediate_time)) * next_beliefs_value

		return reward, tree_size, int(immediate_time), sampled_next_state, terminal

	def get_hashed_value (self, new_belief_prob, action, all_poss_actions):
		return self.env.get_from_belief_library(new_belief_prob, action, all_poss_actions)

	def compute_V (self, belief=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		if belief is None:
			belief = self.belief
		min_Q = None
		min_a = None
		Q_a = np.full((self.env.nA,1),np.Inf)
		tree_size = tree_s
		time_steps = np.full((self.env.nA,1),1)
		min_time = None
		min_sampled_next_state = None
		min_termianl = None
		if print_status:
			print ("compute V: ", belief.prob, " h: ", horizon, " sampled_state: ", self.sampled_state)
		for act in self.env.feasible_actions:
			a = act.id
			new_Q, tree_size, immediate_time, sampled_next_state, terminal = self.compute_Q(belief, act, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			Q_a[a] = new_Q
			time_steps[a] = immediate_time
			if print_status:
				print ("compute V: ", belief.prob, " Q: ", new_Q, " act: ", act.name, " a: ", a, " time: ", immediate_time)
			if act in self.env.pomdps_actions and (min_Q is None or new_Q < min_Q):
				min_Q = new_Q
				min_a = act
				min_time = immediate_time
				min_sampled_next_state = sampled_next_state
				min_termianl = terminal


		self.env.add_to_belief_library(belief, min_Q, all_poss_actions)
		if print_status:
			for b in belief.prob:
				print ("belief: ", (b[0],self.env.get_state_tuple(b[1])))
			print ("compute V: ", belief.prob, " minQ: ", min_Q, " min_act: ", min_a.name, " min_time: ", min_time, " min_sampled_next_state: ", min_sampled_next_state)
		if not min_termianl:
			possible_obss = self.env.simulate_observation(min_sampled_next_state,min_a)
			###################################################################### RTDP-bel
			rand_num = self.random.choice(100)
			sum_prob = 0
			sampled_observation = None
			if self.sampled_state is not None and sampled_next_state is not None:
				for (temp_obs,prob) in possible_obss:
					sum_prob += prob*100	
					if sampled_observation is None and rand_num < sum_prob:
						sampled_observation = temp_obs
						break
				obs = sampled_observation
			###################################################################### RTDP-bel
			new_belief_prob = self.update_belief(belief.prob, min_a, obs, all_poss_actions,horizon)
			# print (new_belief_prob)

			if len(new_belief_prob) > 0 and horizon > 0:
				self.sampled_state = min_sampled_next_state
				self.compute_V(Belief(new_belief_prob), horizon-min_time, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			elif horizon <= 0:
				return np.Inf, self.env.noop_actions['1'], 1, Q_a, tree_size, time_steps, None
		else:
			print ("Terminal: ", horizon)
			for b in belief.prob:
				print ("belief: ", (b[0],self.env.get_state_tuple(b[1])))
			# set_trace()
		return min_Q, min_a, min_time, Q_a, tree_size, time_steps, None

	
	def run_RTDP_bel (self, belief=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		if belief is None:
			belief = self.belief
		action_len = 20
		action_history = np.full((action_len,1),-1)
		count = 0
		while count < self.num_of_iterations:
			rand_num = self.random.choice(100)
			sum_prob = 0
			self.sampled_state = None
			for (prob,state) in belief.prob:
				sum_prob += prob*100
				if self.sampled_state is None and rand_num < sum_prob:	
					self.sampled_state = state
					break
			# set_trace()
			min_Q, min_a, min_time, Q_a, tree_size, time_steps, leaf_beliefs = self.compute_V (belief, horizon, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)
			print ("action: ", min_a.name, " Q: ", min_Q, " time: ", min_time)
			if np.all(action_history == min_a.id):
				print ("action selected at iteration ", count)
				break

			action_history[count%action_len] = min_a.id
			count += 1

		return min_Q, min_a, min_time, Q_a, tree_size, time_steps, None

	def run_RTDP_bel_Q (self, belief=None, action=None, horizon=100, max_horizon=100, one_action=None, gamma=1.0, tree_s=None, all_poss_actions=False, HPOMDP=False):
		num_of_iterations = 1
		if belief is None:
			belief = self.belief
		
		count = 0
		min_Q = 0
		while count < num_of_iterations:
			rand_num = self.random.choice(100)
			sum_prob = 0
			self.sampled_state = None
			for (prob,state) in belief.prob:
				sum_prob += prob*100
				if self.sampled_state is None and rand_num < sum_prob:	
					self.sampled_state = state
					break
			###################################################################### RTDP-bel
			cost, tree_size, immediate_time, sampled_next_state, terminal = self.compute_Q(belief, action, horizon, max_horizon, one_action, gamma, tree_size, all_poss_actions, HPOMDP)
			min_time = immediate_time
			if not terminal:
				possible_obss = self.env.simulate_observation(sampled_next_state,action)				
				rand_num = self.random.choice(100)
				sum_prob = 0
				sampled_observation = None
				if self.sampled_state is not None and sampled_next_state is not None:
					for (temp_obs,prob) in possible_obss:
						sum_prob += prob*100	
						if sampled_observation is None and rand_num < sum_prob:
							sampled_observation = temp_obs
							break
					obs = sampled_observation
				
				new_belief_prob = self.update_belief(belief, action, obs, all_poss_actions,horizon)
				self.sampled_state = sampled_next_state
				###################################################################### RTDP-bel
				min_Q, min_a, min_time, Q_a, tree_size, time_steps, leaf_beliefs = self.compute_V (new_belief_prob, horizon-min_time, max_horizon, one_action, gamma, tree_s, all_poss_actions, HPOMDP)

			
			print (min_Q, min_a.name, min_time, Q_a)
			count += 1

		return min_Q+cost, immediate_time