# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import gym

import numpy as np
import math
import pylab as plt
import time

def evaluate_policy(env, gamma, policy, max_iterations=int(1e6), tol=1e-5):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros((env.nS))
    i=0
    delta = np.inf
    while i < max_iterations and delta > tol:
      delta = 0
      for s in range(env.nS):
        v = V[s]
        a = policy[s]
        r = 0
        for pstate in env.P[s][a]:
          k = pstate[4]
          high_level = pstate[5]

          if high_level:
            if pstate[3] is False:
              r += pstate[0]*(pstate[2] + math.pow(gamma,k) * V[pstate[1]])
            else:
              r += pstate[0]*pstate[2]

          else:
            if pstate[3] is False:
              r += pstate[0]*(pstate[2] + gamma * V[pstate[1]])
            else:
              r += pstate[0]*pstate[2]

        V[s] = r
        delta = max(delta, abs(V[s] - v))
      i+=1
    return V, i


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    policy = np.zeros((env.nS))
    for s in range(env.nS):
      m = -np.inf
      opt_a = -1
      for a in range(env.nA):
        r = 0
        for pstate in env.P[s][a]:
          k = pstate[4]
          high_level = pstate[5]

          if high_level:
            if pstate[3] is False:
              r += pstate[0]*(pstate[2] + math.pow(gamma,k) * value_function[pstate[1]])
            else:
              r += pstate[0]*pstate[2]

          else:
            if pstate[3] is False:
              r += pstate[0]*(pstate[2] + gamma * value_function[pstate[1]])
            else:
              r += pstate[0]*pstate[2]

        if r > m:
          m = r
          opt_a = a
      policy[s] = int(opt_a)
    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    for s in range(env.nS):
      old_pi = policy[s]
      m = -np.inf
      opt_a = -1
      for a in range(env.nA):
        r = 0
        for pstate in env.P[s][a]:
          k = pstate[4]
          high_level = pstate[5]

          if high_level:
            if pstate[3] is False:
              r += pstate[0]*(pstate[2] + math.pow(gamma,k) * value_func[pstate[1]])
            else:
              r += pstate[0]*pstate[2]

          else:
            if pstate[3] is False:
              r += pstate[0]*(pstate[2] + gamma * value_func[pstate[1]])
            else:
              r += pstate[0]*pstate[2]

          #print (r)
        #print (m)
        #print (r)
        if r > m:
          m = r
          opt_a = a
      if opt_a != old_pi:
        policy[s] = int(opt_a)
        policy_stable = False
    return policy_stable, policy.astype(int)


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    policy_stable = False
    i = 0
    total_ite = 0
    while policy_stable is False and i < max_iterations:
      val_func, iterations = evaluate_policy(env, gamma, policy)
      policy_stable, policy = improve_policy(env, gamma, val_func, policy)
      i+=1
      total_ite += iterations
    return policy.astype(int), val_func, i, total_ite


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    print ("**** value iteration started ...")
    start_time = time.time()
    V = np.zeros((env.nS))
    A = np.zeros((env.nS)) 
    Q = np.zeros((env.nS,env.nA))
    ite = 0
    delta = np.inf
    print ("state size: ",env.nS, " action size: ",env.nA)
    while ite < max_iterations and delta > tol:
      delta = 0
      for s in range(env.nS):
        v = V[s]
        m = -np.inf
        opt_a = -1
        for a in range(env.nA):
          r = 0
          for pstate in env.P[s][a]:
            # print (pstate)
            k = pstate[4]
            high_level = pstate[5]

            if high_level:
              if pstate[3] is False:
                r += pstate[0]*(pstate[2] + math.pow(gamma,k) * V[pstate[1]])
              else:
                r += pstate[0]*pstate[2]

            else:
              if pstate[3] is False:
                r += pstate[0]*(pstate[2] + gamma * V[pstate[1]])
              else:
                r += pstate[0]*pstate[2]
          Q[s][a] = r
          if r > m:
            m = r
            opt_a = a
        V[s] = m
        A[s] = int(opt_a)
        delta = max(delta, abs(V[s] - v))
        #print (V)
        #print (V[s])
        #print (v)
      ite+=1
    end_time = time.time()
    elapsed = end_time - start_time
    print ("**** value iteration ended: ",elapsed)
    return V, ite, A.astype(int), Q


def print_policy(env, policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """

    # for i in range(0,env.grid_size):
    #     for j in range(0,env.grid_size):
    #         new_state_index = env.get_state_index((j,i))
    #         print (action_names[policy[new_state_index]],end=' ')
    #     print ("\n")

    # print(str_policy)
    pass

def show_value_function(env,val_func):
    # val_f = env.get_value_function(val_func)
    im = plt.imshow(np.reshape(val_func,(val_func.shape[0],val_func.shape[1]),order='C'))
    plt.colorbar(im,orientation='vertical')
    plt.show()
    # plt.savefig(str(len(val_func))+'_value_iteration.png')

def execute_policy(env,policy,Q,V):
    total_reward = 0    
    count = 1
    for i in range(1,count+1): ## STOCHASTIC CASE
        partial_reward = 0;
        num_steps = 0;
        nextstate = env.get_state()
        while True:
            print (nextstate)
            action = policy[nextstate]
            print ("action: ", env.get_action_name(action)) ## ,Q[nextstate]
            nextstate, reward, is_terminal, debug_info = env.step(action,Q,V)
            
            env.render()

            partial_reward += reward ## total reward, not discounted
            num_steps += 1

            if is_terminal:
                total_reward += partial_reward
                break
        
        print (i,num_steps,total_reward)

    total_reward = total_reward/count
    return total_reward,num_steps