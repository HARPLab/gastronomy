import numpy as np
import gym
import time
from pdb import set_trace
import math
import pylab as plt
from math import exp

EPS_GREEDY = "epsilon_greedy"
EPS_GREEDY_W_DECAY = "epsilon_greedy_w_decay"

class Q_learning:
	def __init__(self,env,num_actions,num_states,epsilon,gamma,alpha,exploration_policy):
		self.print_status = True
		self.q = np.zeros((num_states,num_actions))
		self.num_actions = num_actions
		self.num_states = num_states
		self.gamma = gamma
		self.alpha = alpha
		self.initial_Q_value = 0
		self.env = env

		self.exp_policy = exploration_policy
		if exploration_policy == EPS_GREEDY:
			## epsilon greedy
			self.epsilon = epsilon

		if exploration_policy == EPS_GREEDY_W_DECAY:
			## epsilon greedy with decay
			self.epsilon_start_val = 1.0
			self.epsilon_end_val = 0.1
			self.epsilon = self.epsilon_start_val

	def update(self,state,action,nextstate,reward,k,action_highlevel):
		max_next_q = np.max([self.get_q(nextstate,a) for a in range(0,self.num_actions)])
		if action_highlevel:
			new_value = reward + math.pow(self.gamma,k) * max_next_q
		else:
			new_value = reward + self.gamma * max_next_q
		self.q[state,action] = self.q[state,action] + self.alpha * (new_value - self.q[state,action])

	def get_q(self,state,action):
		return self.q[state,action]

	def get_Q(self):
		return self.q

	# epsilon greedy
	def select_epsilon_greedy_action(self,state):
		if np.random.random_sample() < self.epsilon:
			return np.random.randint(low=0,high=self.num_actions)
		else:
			q = [self.get_q(state,a) for a in range(0,self.num_actions)]
			q_max = np.max(q)
			q_max_indices = [index for index in range(0,self.num_actions) if q[index] == q_max]
			max_q_action = np.random.choice(q_max_indices)
			return max_q_action

	def select_epsilon_greedy_w_decay_action(self,state,num_steps):
		# epsilon_decay = 1.0/num_steps
		if np.random.random_sample() < self.epsilon:
			return np.random.randint(low=0,high=self.num_actions)
		else:
			q = [self.get_q(state,a) for a in range(0,self.num_actions)]
			q_max = np.max(q)
			q_max_indices = [index for index in range(0,self.num_actions) if q[index] == q_max]
			max_q_action = np.random.choice(q_max_indices)
			return max_q_action

		# self.epsilon = max(self.epsilon - epsilon_decay,self.epsilon_end_val)

	def select_greedy_action(self,state):
		q = [self.get_q(state,a) for a in range(0,self.num_actions)]
		q_max = np.max(q)
		q_max_indices = [index for index in range(0,self.num_actions) if q[index] == q_max]
		max_q_action = np.random.choice(q_max_indices)
		return max_q_action

	def get_V(self):
		return np.amax(self.q, axis=1);

	def get_policy(self):
		return np.argmax(self.q, axis=1);

	def train(self,num_episodes,num_test_episodes,max_steps = 10000000):
		if self.print_status:
			print ("******* q learning")
		test_every = 1
		self.train_curve = np.zeros(int(num_episodes/test_every))
		self.success_curve = np.zeros(int(num_episodes/test_every))
		
		for e in range(0,num_episodes):
			curr_state = self.env.reset()
			# print ("new episode .........")
			is_terminal = False
			num_steps = 0
			if self.exp_policy == EPS_GREEDY_W_DECAY:
				self.epsilon = max(self.epsilon_start_val-((self.epsilon_start_val-self.epsilon_end_val)*(1-exp(-e/num_episodes))),self.epsilon_end_val)
			while not is_terminal:
				if self.exp_policy == EPS_GREEDY:
					# print ("state: ", curr_state)
					action = self.select_epsilon_greedy_action(curr_state)
				elif self.exp_policy == EPS_GREEDY_W_DECAY:
					action = self.select_epsilon_greedy_w_decay_action(curr_state,num_episodes*5)

				nextstate, reward, is_terminal, debug_info = self.env.step(action)
				k = debug_info['steps']
				action_highlevel = debug_info['action_highlevel']
				# print (curr_state, nextstate, reward, is_terminal, debug_info)
				self.update(curr_state,action,nextstate,reward, k, action_highlevel)
				curr_state = nextstate

				num_steps += 1
				if num_steps == max_steps:
					break

			# if e % test_every == 0:
			# 	test_reward,num_success = self.test(num_test_episodes,max_steps=200)
			# 	self.train_curve[int(e/test_every)] = test_reward
			# 	self.success_curve[int(e/test_every)] = num_success
			# 	print ("episode%d, reward, success%.4f: ", test_reward, (e,num_success))

	def test(self,num_episodes,show = False,max_steps=10000000):
		# observations = []
		total_reward = 0.0
		num_success = 0
		for e in range(0,num_episodes):
			curr_state = self.env.reset()
			is_terminal = False
			num_steps = 0
			while not is_terminal:
				action = self.select_greedy_action(curr_state)
				nextstate, reward, is_terminal, debug_info = self.env.step(action)
				# observations.append(debug_info["state"])
				total_reward += reward
				curr_state = nextstate
				if show:
					self.env.render()
					time.sleep(1)

				num_steps += 1
				if num_steps == max_steps:
					break

			if 'goal' in debug_info:
				# for minigrid envs
				num_success += debug_info['goal']
			else:
				# for discrete envs
				num_success += (is_terminal)

		# data = np.asarray(observations)
		# mean = np.mean(data, axis=0)
		# cov = np.cov(data, rowvar=0)

		return total_reward/num_episodes, num_success/(num_episodes+0.0)

	def show_value_function(self,name):
	    val_f = self.env.get_value_function(self.get_V())
	    sizeVal = int(math.sqrt(len(val_f)))                 
	    im = plt.imshow(np.reshape(val_f,(sizeVal,sizeVal),order='F'))
	    plt.colorbar(im,orientation='vertical')
	    # plt.show()
	    plt.savefig(name+'_q_learning.png')


if __name__ == "__main__":
	## 
	import projectrwm.envs
	num_train_episodes = 10000
	num_test_episodes = 10
	# envs: 'Deterministic-8x8-FrozenLake-v0', 'Stochastic-4x4-FrozenLake-v0', 'Stochastic-8x8-FrozenLake-v0', 'Deterministic-4x4-FrozenLake-v0'
	env_name = 'Empty8gym-v1' # Taxi-v2'
	env = gym.make(env_name)
	print ("**************************************")
	print ("env name: ",env_name)
	print("# of actions: ",env.action_space)
	print("# of states: ", env.observation_space)
	print ("**************************************")
	print ("Q-learning started.")
	# set_trace()
	agent = Q_learning(env,num_actions=env.action_space.n,num_states=env.observation_space.n,epsilon=0.1,gamma=0.99,alpha=0.05,exploration_policy=EPS_GREEDY_W_DECAY)

	agent.train(num_train_episodes,num_test_episodes,EPS_GREEDY)

	total_reward = agent.test(num_test_episodes,show=True)

	print ("final test reward: ", total_reward)

	print ("Q-learning ended.")