import table_path_code as resto
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import cv2
import matplotlib.pylab as plt
import math
import copy
import time
import decimal
import random
import os
from pandas.plotting import table
import matplotlib.gridspec as gridspec
import klampt_smoothing as chunkify
from collections import defaultdict
from shapely.geometry import LineString
import scipy.interpolate as interpolate
import pprint
import matplotlib.patheffects as path_effects


import sys
# sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/ModelPredictiveTrajectoryGenerator/')
# sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/StateLatticePlanner/')

# start 		= r.get_start()
# goals 		= r.get_goals_all()
# goal 		= r.get_current_goal()
# observers 	= r.get_observers()
# tables 		= r.get_tables()
# waypoints 	= r.get_waypoints()
# SCENARIO_IDENTIFIER = r.get_scenario_identifier()

FLAG_SAVE 				= True
FLAG_VIS_GRID 			= False
FLAG_EXPORT_HARDCODED 	= False
FLAG_REDO_PATH_CREATION = False #True #False #True #False
FLAG_REDO_ENVIR_CACHE 	= False #True
FLAG_MIN_MODE			= False
FLAG_EXPORT_LATEX_MAXES = False

VISIBILITY_TYPES 		= resto.VIS_CHECKLIST
NUM_CONTROL_PTS 		= 3

NUMBER_STEPS 		= 30
PATH_TIMESTEPS 		= 15

resto_pickle = 'pickle_vis'
vis_pickle = 'pickle_resto'
FILENAME_PATH_ASSESS = 'path_assessment/'

FLAG_PROB_HEADING = False
FLAG_PROB_PATH = True
FLAG_EXPORT_SPLINE_DEBUG = False

# PATH_COLORS = [(138,43,226), (0,255,255), (255,64,64), (0,201,87)]

SAMPLE_TYPE_CENTRAL 			= 'central-new'
SAMPLE_TYPE_CENTRAL_SPARSE 		= 'central-sprs-new'
SAMPLE_TYPE_DEMO 				= 'demo2'
SAMPLE_TYPE_CURVE_TEST			= 'ctest'
SAMPLE_TYPE_NEXUS_POINTS 		= 'nn_fin7'
SAMPLE_TYPE_NEXUS_POINTS_ONLY 	= 'nn_only'
SAMPLE_TYPE_SPARSE				= 'sparse'
SAMPLE_TYPE_SYSTEMATIC 			= 'systematic'
SAMPLE_TYPE_HARDCODED 			= 'hardcoded'
SAMPLE_TYPE_VISIBLE 			= 'visible'
SAMPLE_TYPE_INZONE 				= 'in_zone'
SAMPLE_TYPE_SHORTEST			= 'minpaths'
SAMPLE_TYPE_FUSION				= 'fusion'


ENV_START_TO_HERE 		= 'start_to_here'
ENV_HERE_TO_GOALS 		= 'here_to_goals'
ENV_VISIBILITY_PER_OBS 	= 'vis_per_obs'
ENV_PROB_G_HERE			= 'prob_to_here'

premade_path_sampling_types = [SAMPLE_TYPE_DEMO, SAMPLE_TYPE_SHORTEST, SAMPLE_TYPE_CURVE_TEST]
non_metric_columns = ["path", "goal", 'path_length', 'path_cost', 'sample_points']

bug_counter = defaultdict(int)
curvatures = []
max_curvatures = []


def f_cost(t1, t2):
	a = resto.dist(t1, t2)
	# return a
	return np.abs(a * a)

def f_path_length(t1, t2):
	a = resto.dist(t1, t2)
	return a
	# return np.abs(a * a)

def f_path_cost(path):
	cost = 0
	for i in range(len(path) - 1):
		cost = cost + f_cost(path[i], path[i + 1])

	return cost

def f_convolved(val_list, f_function):
	tstamps = range(len(val_list))
	ret = []
	for t in tstamps:
		ret.append(f_function(t) * val_list[t])
	return ret

def f_vis_exp1(t, pt, aud):
	return (f_og(t) * f_vis3(pt, aud))


def f_og(t, path):
	# len(path)
	return NUMBER_STEPS - t

def f_novis(t, obs):
	return 1

# # Given the observers of a given location, in terms of distance and relative heading
# # Ada final equation TODO verify all correct
# def f_vis_single(p, observers):
# 	# dist_units = 100
# 	angle_cone = 120.0 / 2
# 	distance_cutoff = 2000

# 	# Given a list of entries in the format 
# 	# ((obsx, obsy), angle, distance)
# 	if len(observers) == 0:
# 		return 1
	
# 	vis = 0
# 	for obs in observers:
# 		if obs == None:
# 			return 0
# 		else:
# 			angle, dist = obs.get_obs_to_pt_relationship(p)
# 			# print((angle, dist))

# 		if angle < angle_cone and dist < distance_cutoff:
# 			vis += np.abs(angle_cone - angle)

# 	# print(vis)
# 	return vis

def f_naked(t, pt, aud, path):
	return decimal.Decimal(1.0)

# Ada final equation
# f_VIS TODO VERIFY
def f_exp_single(t, pt, aud, path):
	# if this is the omniscient case, return the original equation
	if len(aud) == 0 and path is not None:
		return float(60 - t)
		# return float(len(path) - t)
	elif len(aud) == 0:
		# print('ping')
		return 1.0

	# if in the (x, y) OR (x, y, t) case we can totally 
	# still run this equation
	val = get_visibility_of_pt_w_observers(pt, aud, normalized=False)
	return val

def f_exp_single_normalized(t, pt, aud, path):
	# if this is the omniscient case, return the original equation
	if len(aud) == 0 and path is not None:
		return float(len(path) - t + 1)
		# return float(len(path) - t)
	elif len(aud) == 0:
		# print('ping')
		return 1.0

	# if in the (x, y) OR (x, y, t) case we can totally 
	# still run this equation
	val = get_visibility_of_pt_w_observers(pt, aud, normalized=True)
	return val


# ADA TODO MASTER VISIBILITY EQUATION
def get_visibility_of_pt_w_observers(pt, aud, normalized=True):
	observers = []
	score = []

	reasonable_set_sizes = [0, 1, 5]
	if len(aud) not in reasonable_set_sizes:
		print(len(aud))
		exit()

	# section for alterating calculculation for a few 
	# out of the whole set; mainly for different combination techniques
	# if len(aud) == 5:
	# 	aud = [aud[2], aud[4]]

	MAX_DISTANCE = 500
	for observer in aud:
		obs_orient 	= observer.get_orientation() + 90
		# if obs_orient != 300:
		# 	print(obs_orient)
		# 	exit()
		obs_FOV 	= observer.get_FOV()

		angle 		= angle_between_points(observer.get_center(), pt)
		distance 	= resto.dist(pt, observer.get_center())
		# print("~~~")
		# print(observer.get_center())
		# print(distance)
		# print(pt)
		
		# print(ang)
		a = angle - obs_orient
		signed_angle_diff = (a + 180) % 360 - 180
		angle_diff = abs(signed_angle_diff)

		# if (pt[0] % 100 == 0) and (pt[1] % 100 == 0):
		# 	print(str(pt) + " -> " + str(observer.get_center()) + " = angle " + str(angle))
		# 	print("observer looking at... " + str(obs_orient))
		# 	print("angle diff = " + str(angle_diff))

		# print(angle, distance)
		# observation = (pt, angle, distance)
		# observers.append(observation)

		half_fov = (obs_FOV / 2.0)
		# print(half )
		if angle_diff < half_fov:
			from_center = half_fov - angle_diff
			if normalized:
				from_center = from_center / (half_fov)

			# from_center = from_center * from_center
			score.append(from_center)
		else:
			if normalized:
				score.append(0)
			else:
				score.append(1)

		# 	# full credit at the center of view
		# 	offset_multiplier = np.abs(angle_diff) / obs_FOV

		# 	# # 1 if very close
		# 	# distance_bonus = (MAX_DISTANCE - distance) / MAX_DISTANCE
		# 	# score += (distance_bonus*offset_multiplier)
		# 	score = offset_multiplier
		# 	score = distance

	# combination method for multiple viewers: minimum value
	if len(score) > 0:
		# score = min(score)
		score = sum(score)
	else:
		score = 0
	return score

# Ada: Final equation
# TODO Cache this result for a given path so far and set of goals
def prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):
	start = r.get_start()
	g_array = []
	g_target = 0
	for g in goals:
		p_raw = unnormalized_prob_goal_given_path(r, p_n1, pt, g, goals, cost_path_to_here, exp_settings)
		g_array.append(p_raw)
		if g is goal:
			g_target = p_raw

	if(sum(g_array) == 0):
		print("weird g_array")
		return 0

	return g_target / (sum(g_array))

# Ada: final equation
def unnormalized_prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):
	decimal.getcontext().prec = 60
	is_og = exp_settings['prob_og']

	start = r.get_start()

	if is_og:
		c1 = decimal.Decimal(cost_path_to_here)
	else:
		c1 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(r.get_start()), resto.to_xy(pt), exp_settings))	

	
	c2 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(pt), resto.to_xy(goal), exp_settings))
	c3 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(start), resto.to_xy(goal), exp_settings))

	# print(c2)
	# print(c3)
	a = np.exp((-c1 + -c2))
	b = np.exp(-c3)
	# print(a)
	# print(b)

	ratio 		= a / b

	if math.isnan(ratio):
		ratio = 0

	return ratio

def prob_goal_given_heading(start, pn, pt, goal, goals, cost_path_to_here):

	g_probs = prob_goals_given_heading(pn, pt, goals)
	g_index = goals.index(goal)

	return g_probs[g_index]


def f_angle_prob(heading, goal_theta):
	diff = (np.abs(np.abs(heading - goal_theta) - 180))
	return diff * diff


def prob_goals_given_heading(p0, p1, goals):
	# works with
	# diff = (np.abs(np.abs(heading - goal_theta) - 180))

	# find the heading from start to pt
	# start to pt
	# TODO theta
	heading = resto.angle_between(p0, p1)
	# print("heading: " + str(heading))
	
	# return an array of the normalized prob of each goal from this current heading
	# for each goal, find the probability this angle is pointing to it



	probs = []
	for goal in goals:
		# 180 		= 0
		# 0 or 360 	= 1
		# divide by other options, 
		# so if 2 in same dir, 50/50 odds
		goal_theta = resto.angle_between(p0, goal)
		prob = f_angle_prob(heading, goal_theta)
		probs.append(prob)


	divisor = sum(probs)
	# divisor = 1.0

	return np.true_divide(probs, divisor)
	# return ratio


def tests_prob_heading():
	p1 = (0,0)
	p2 = (0,1)

	goals 	= [(1,1), (-1,1), (0, -1)]
	correct = [ 0.5,  0.5, -0.]
	result 	= prob_goals_given_heading(p1, p2, goals)

	if np.array_equal(correct, result):
		pass
	else:
		print("Error in heading probabilities")

	goals 	= [(4,0), (5,0)]
	correct = [ 0.5,  0.5]
	result 	= prob_goals_given_heading(p1, p2, goals)

	print(goals)
	print(result)

	goals 	= [(4,1), (4,1)]
	correct = [ 0.5,  0.5]
	result 	= prob_goals_given_heading(p1, p2, goals)	

	print(goals)
	print(result)

	print("ERR")
	goals 	= [(1,1), (-1,1), (-1,-1), (1,-1)]
	correct = [ 0.5,  0.5, 0, 0]
	result 	= prob_goals_given_heading(p1, p2, goals)	

	print(goals)
	print(result)

	goals 	= [(2,0), (0,2), (-2,0), (0,-2)]
	correct = [ 0.25,  0.5, 0.25, 0]
	result 	= prob_goals_given_heading(p1, p2, goals)	

	print(goals)
	print(result)


def get_costs_along_path(path):
	output = []
	ci = 0
	csf = 0
	for pi in range(len(path)):
		
		cst = f_cost(path[ci], path[pi])
		csf = csf + cst
		log = (path[pi], csf)
		ci = pi
		output.append(log)
		
	return output

# returns a list of the path length so far at each point
def get_path_length(path):
	total = 0
	output = [0]

	for i in range(len(path) - 1):
		link_length = f_path_length(path[i], path[i + 1])
		total = total + link_length
		output.append(total)

	return output, total

	# output = []
	# ci = 0
	# csf = 0
	# total = 0
	# for pi in range(len(path)):
	# 	cst = f_path_length(path[ci], path[pi])
	# 	total += cst
	# 	ci = pi
	# 	output.append(total) #log
		
	# return output, total

def get_min_viable_path(r, goal, exp_settings):
	path_option = construct_single_path_with_angles_spline(exp_settings, r.get_start(), goal, [], fn_export_from_exp_settings(exp_settings))
	path_option = chunkify.chunkify_path(r, exp_settings, path_option)
	return path_option

def get_min_viable_path_length(r, goal, exp_settings):
	path_option = get_min_viable_path(r, goal, exp_settings)
	return get_path_length(path_option)[1]

def get_min_direct_path(r, p0, p1, exp_settings):
	path_option = [p0, p1]
	path_option = chunkify.chunkify_path(r, exp_settings, path_option)
	return path_option

def get_dist(p0, p1):
	p0_x, p0_y = p0
	p1_x, p1_y = p1

	min_distance = np.sqrt((p0_x-p1_x)**2 + (p0_y-p1_y)**2)
	return min_distance

def get_min_direct_path_cost_between(r, p0, p1, exp_settings):
	dist = get_dist(p0, p1)
	dt = chunkify.get_dt(exp_settings)
	cost_chunk = dt * dt
	num_chunks = int()

	leftover = dist - (dt*num_chunks)
	cost = (num_chunks * cost_chunk) + (leftover*leftover)

	return cost
	# f_path_cost(path_option)

def get_min_direct_path_length(r, p0, p1, exp_settings):
	return get_dist(p0, p1)

# Given a 
def f_legibility(r, goal, goals, path, aud, f_function, exp_settings):
	FLAG_is_denominator = exp_settings['is_denominator']
	if f_function is None and FLAG_is_denominator:
		f_function = f_exp_single
	elif f_function is None:
		f_function = f_exp_single_normalized

	if path is None or len(path) == 0:
		return 0
	# min_realistic_path_length = exp_settings['min_path_length'][goal]
	# print("min_realistic_path_length -> " + str(min_realistic_path_length))
	
	legibility = decimal.Decimal(0)
	divisor = decimal.Decimal(0)
	total_dist = decimal.Decimal(0)

	if 'lambda' in exp_settings and exp_settings['lambda'] != '':
		LAMBDA = decimal.Decimal(exp_settings['lambda'])
		epsilon = decimal.Decimal(exp_settings['epsilon'])
	else:
		# TODO verify this
		LAMBDA = 1.0
		epsilon = 1.0

	start = path[0]
	total_cost = decimal.Decimal(0)
	aug_path = get_costs_along_path(path)

	path_length_list, length_of_total_path = get_path_length(path)
	length_of_total_path = decimal.Decimal(length_of_total_path)

	# Previously this was a variable, 
	# now it's constant due to our constant-speed chunking
	delta_x = decimal.Decimal(1.0) #length_of_total_path / len(aug_path)

	t = 1
	p_n = path[0]
	divisor = epsilon
	numerator = decimal.Decimal(0.0)

	f_log = []
	p_log = []
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))
		prob_goal_given = prob_goal_given_path(r, p_n, pt, goal, goals, cost_to_here, exp_settings)
		f_log.append(float(f))
		p_log.append(prob_goal_given)

		if prob_goal_given > 1 or prob_goal_given < 0:
			print(prob_goal_given)
			print("!!!")

		if FLAG_is_denominator or len(aud) == 0:
			numerator += (prob_goal_given * f) # * delta_x)
			divisor += f #* delta_x
		else:
			numerator += (prob_goal_given * f) # * delta_x)
			divisor += decimal.Decimal(1.0) #* delta_x

		t = t + 1
		total_cost += decimal.Decimal(f_cost(p_n, pt))
		p_n = pt

	if divisor == 0:
		legibility = 0
	else:
		legibility = (numerator / divisor)

	total_cost =  - LAMBDA*total_cost
	overall = legibility + total_cost

	# if len(aud) == 0:
	# 	print(numerator)
	# 	print(divisor)
	# 	print(f_log)
	# 	print(p_log)
	# 	print(legibility)
	# 	print(overall)
	# 	print()

	if legibility > 1.0 or legibility < 0:
		# print("BAD L ==> " + str(legibility))
		# r.get_obs_label(aud)
		goal_index = r.get_goal_index(goal)
		category = r.get_obs_label(aud)
		bug_counter[goal_index, category] += 1

	elif (legibility == 1):
		goal_index = r.get_goal_index(goal)
		category = r.get_obs_label(aud)
		bug_counter[goal_index, category] += 1

		# print(len(aud))
		if exp_settings['kill_1'] == True:
			overall = 0.0

	return overall

# Given a 
def f_env(r, goal, goals, path, aud, f_function, exp_settings):
	fov = exp_settings['fov']
	FLAG_is_denominator = exp_settings['is_denominator']
	if path is None or len(path) == 0:
		return 0

	if f_function is None and FLAG_is_denominator:
		f_function = f_exp_single
	elif f_function is None:
		f_function = f_exp_single_normalized

	if FLAG_is_denominator:
		vis_cutoff = 1
	else:
		half_fov = fov / 2.0
		vis_cutoff = 0

	count = 0
	aug_path = get_costs_along_path(path)

	path_length_list, length_of_total_path = get_path_length(path)
	length_of_total_path = decimal.Decimal(length_of_total_path)

	epsilon = exp_settings['epsilon']

	env_readiness = -1
	t = 1
	p_n = path[0]
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))
	
		# if f is greater than 0, this indicates being in-view
		if f > vis_cutoff:
			count += 1
			if env_readiness == -1:
				env_readiness = (len(aug_path) - t + 1)

		# if it's not at least 0, then out of sight, not part of calc
		else:
			count = 0.0

		t += 1

	return count, env_readiness, len(aug_path)


def get_costs(path, target, obs_sets):
	vals = []

	for aud in obs_sets:
		new_val = f_cost()

	return vals

def get_legibilities(resto, path, target, goals, obs_sets, f_vis, exp_settings):
	vals = {}
	f_vis = exp_settings['f_vis']

	# print("manually: naked")
	naked_prob = f_legibility(resto, target, goals, path, [], f_naked, exp_settings)
	vals['naked'] = naked_prob

	for key in obs_sets.keys():
		aud = obs_sets[key]
		new_leg = f_legibility(resto, target, goals, path, aud, None, exp_settings)
		new_env, x = f_env(resto, target, goals, path, aud, f_vis, exp_settings)

		vals[key] = new_leg
		vals[key + "-env"] = new_env

	return vals


# https://medium.com/@jaems33/understanding-robot-motion-path-smoothing-5970c8363bc4
def smooth_slow(path, weight_data=0.5, weight_smooth=0.1, tolerance=1):
	"""
	Creates a smooth path for a n-dimensional series of coordinates.
	Arguments:
		path: List containing coordinates of a path
		weight_data: Float, how much weight to update the data (alpha)
		weight_smooth: Float, how much weight to smooth the coordinates (beta).
		tolerance: Float, how much change per iteration is necessary to keep iterating.
	Output:
		new: List containing smoothed coordinates.
	"""

	dims = len(path[0])
	new = [[0, 0]] * len(path)
	# print(new)
	change = tolerance

	while change >= tolerance:
		change = 0.0
		prev_change = change
		
		for i in range(1, len(new) - 1):
			for j in range(dims):

				x_i = path[i][j]
				y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j]

				y_i_saved = y_i
				y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
				new[i][j] = y_i

				change += abs(y_i - y_i_saved)

		# print(change)
		if prev_change == change:
			return new
	return new


def smoothed(blocky_path, r):
	points = []
	
	xys = blocky_path

	ts = [t/NUMBER_STEPS for t in range(NUMBER_STEPS + 1)]
	bezier = resto.make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points


	return smooth(blocky_path)
	return blocky_path

def generate_single_path_grid(restaurant, target, vis_type, n_control):
	sample_pts 	= restaurant.sample_points(n_control, target, vis_type)
	path = construct_single_path(restaurant.get_start(), target, sample_pts)
	path 		= smoothed(path, restaurant)
	return path


def generate_single_path(restaurant, target, vis_type, n_control):
	valid_path = False

	while (not valid_path):
		sample_pts 	= restaurant.sample_points(n_control, target, vis_type)
		path 		= construct_single_path_bezier(restaurant.get_start(), target, sample_pts)
		valid_path  = is_valid_path(restaurant, path)

		if (not valid_path):
			# print("regenerating")
			pass

	return path

def at_pt(a, b, tol):
	return (abs(a - b) < tol)

def construct_single_path(start, end, sample_pts):
	points = [start]
	GRID_SIZE = 10
	# NUMBER_STEPS should be the final length

	# randomly walk there
	cx, cy, ctheta = start
	targets = sample_pts + [end]

	# print(sample_pts)

	for target in targets:
		tx, ty = resto.to_xy(target)
		x_sign, y_sign = 1, 1
		if tx < cx:
			x_sign = -1
		if ty < cy:
			y_sign = -1

		counter = 0

		# print("cx:" + str(cx) + " tx:" + str(tx))
		# print("cy:" + str(cy) + " ty:" + str(ty))

		# print(not at_pt(cx, tx, GRID_SIZE))
		# print(not at_pt(cy, ty, GRID_SIZE))

		# Abs status 
		while not at_pt(cx, tx, GRID_SIZE) or not at_pt(cy, ty, GRID_SIZE):
			# print("in loop")
			counter = counter + 1
			axis = random.randint(0, 1)
			if axis == 0 and not at_pt(cx, tx, GRID_SIZE):
				cx = cx + (x_sign * GRID_SIZE)
			elif not at_pt(cy, ty, GRID_SIZE):
				cy = cy + (y_sign * GRID_SIZE)

			new_pt = (cx, cy)
			points.append(new_pt)

		points.append(target)

	return points

def get_max_turn_along_path(path):
	angle_list = []
	is_counting = False
	for i in range(len(path) - 4):
		p1, p2, p3 = path[i], path[i + 2], path[i + 4]
		angle = angle_of_turn([p1, p2], [p2, p3])
		print(str((p1, p2, p3)) + "->" + str(angle))

		if resto.dist(p1,p2) > 2 or resto.dist(p2, p3) > 2:
			angle_list.append(abs(angle))
			curvatures.append(angle)
		# else:
		# 	print("too short, rejected")
	
	# print(angle_list)
	max_curvature = max(angle_list)
	# min_curvature = min(angle_list)
	# print(max_curvature)
	max_curvatures.append(max_curvature)
	# print(angle_list.index(max_curvature))

	return max_curvature

# def check_curvature(path):
# 	lx = [x for x,y in path]
# 	ly = [y for x,y in path]

# 	#first derivatives 
# 	dx= np.gradient(lx)
# 	dy = np.gradient(ly)

# 	#second derivatives 
# 	d2x = np.gradient(dx)
# 	d2y = np.gradient(dy)

# 	#calculation of curvature from the typical formula
# 	curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
# 	# curvature = curvature[~np.isnan(curvature)]
# 	curvature = curvature[2:-2]
# 	print(curvature)
# 	max_curvature = max(curvature)
# 	print(max_curvature)

# 	# curvatures.append(max_curvature)
# 	return max_curvature

def get_hi_low_of_pts(r):
	pt_list = copy.copy(r.get_goals_all())
	pt_list.append(r.get_start())

	first = pt_list[0]
	px, py, ptheta = first
	low_x, hi_x = px, px
	low_y, hi_y = py, py

	for pt in pt_list:
		px, py = pt[0], pt[1]

		if low_x > px:
			low_x = px

		if low_y > py:
			low_y = py

		if hi_x < px:
			hi_x = px

		if hi_y < py:
			hi_y = py

	return low_x, hi_x, low_y, hi_y


def is_valid_path(r, path, exp_settings):
	return True
	tables = r.get_tables()
	# print(len(tables))

	start = r.get_start()
	sx, sy, stheta = start
	gx0, gy0, gt0 = r.get_goals_all()[0]
	gx1, gy1, gt1 = r.get_goals_all()[1]
	# print("sampling central")
	
	low_x, hi_x, low_y, hi_y = get_hi_low_of_pts(r)

	for p in path:
		if p[0] < start[0] - 2:
			# print(p)
			return False

	line = LineString(path)
	if not line.is_simple:
		return False

	# max_turn = get_max_turn_along_path(path)
	# if max_turn >= exp_settings['angle_cutoff']:
	# 	return False

	# Checks for table intersection
	for t in tables:
		if t.intersects_path(path):
			return False

	BOUND_CHECK_RIGHT = True
	right_buffer = exp_settings['right-bound']
	# Checks for remaining in bounds
	
	for i in range(len(path) - 1):
		pt1 = path[i]
		pt2 = path[i + 1]
		
		# print((pt1, pt2))

		px, py = pt1[0], pt1[1]

		if BOUND_CHECK_RIGHT:
			if px > hi_x + right_buffer:
				return False

		if px < 0:
			return False
		if py < 0:
			return False
		if px > 1350:
			return False
		if py > 1000:
			return False




	return True

def as_tangent(start_angle):
	# start_angle is assumed to be in degrees
	a = np.deg2rad(start_angle)

	dx = np.cos(a)
	dy = np.sin(a)

	return [dx, dy]

def as_tangent_test(sa):
	sa = 90
	print(sa)
	print(as_tangent(sa))
	sa = 0
	print(sa)
	print(as_tangent(sa))

def path_formatted(xs, ys):
	# print(ys)
	xs = [int(x) for x in xs]
	ys = [int(y) for y in ys]
	return list(zip(xs, ys))

def get_pre_goal_pt(goal, exp_settings):
	x, y, theta = goal
	k = exp_settings['angle_strength']
	# print(k)

	if theta == resto.DIR_NORTH:
		y = y - k
	if theta == resto.DIR_SOUTH:
		y = y + k
	if theta == resto.DIR_EAST:
		x = x + k
	if theta == resto.DIR_WEST:
		x = x - k

	return (x, y, theta)

# https://hal.archives-ouvertes.fr/hal-03017566/document
def construct_single_path_with_angles_bspline(exp_settings, start, goal, sample_pts, fn, is_weak=False):
	if len(sample_pts) == 0:
		return [start, goal]
		# return construct_single_path_with_angles_spline(exp_settings, start, goal, sample_pts, fn, is_weak=False)

	x, y = [], []
	xy_0 = start
	xy_n = goal
	xy_mid = sample_pts

	xy_pre_n = get_pre_goal_pt(goal, exp_settings)

	x.append(xy_0[0])
	y.append(xy_0[1])

	for i in range(len(sample_pts)):
		spt = sample_pts[i]
		sx = spt[0]
		sy = spt[1]
		x.append(sx)
		y.append(sy)

	x.append(xy_pre_n[0])
	y.append(xy_pre_n[1])

	x.append(xy_n[0])
	y.append(xy_n[1])

	# Subtract 90 to turn path angle into tangent
	start_angle = xy_0[2] - 90
	# Do the reverse for the ending point
	end_angle 	= xy_n[2] + 90

	# Strength of how much we're enforcing the exit angle
	k = exp_settings['angle_strength']

	x = np.array(x)
	y = np.array(y)

	# print(path_formatted(x, y))

	tck,u = interpolate.splprep([x,y],s=0)
	unew = np.arange(0,1.01,0.01)
	out = interpolate.splev(unew,tck)

	path = path_formatted(out[0], out[1])
	return path

def get_pre_goal_pt_xy(pt, exp_settings):
	new_x = pt[0]
	new_y = pt[1]

	offset = exp_settings['waypoint_offset']
	if pt[2] == resto.DIR_NORTH:
		new_y -= offset
	elif pt[2] == resto.DIR_SOUTH:
		new_y += offset

	return (new_x, new_y)

# https://hal.archives-ouvertes.fr/hal-03017566/document
def construct_single_path_with_angles_spline(exp_settings, start, goal, sample_pts, fn, is_weak=False):
	# print("WITH ANGLE")
	xy_0 = start
	xy_n = goal
	xy_mid = sample_pts	

	x = []
	y = []

	x.append(xy_0[0])
	y.append(xy_0[1])

	for i in range(len(sample_pts)):
		spt = sample_pts[i]
		sx = spt[0]
		sy = spt[1]
		x.append(sx)
		y.append(sy)

	# xy_pre_goal = get_pre_goal_pt_xy(goal, exp_settings)
	# x.append(xy_pre_goal[0])
	# y.append(xy_pre_goal[1])

	x.append(xy_n[0])
	y.append(xy_n[1])

	# print(x)
	# print(y)
	# exit()

	# Subtract 90 to turn path angle into tangent
	start_angle = xy_0[2] - 90
	# Do the reverse for the ending point
	end_angle 	= xy_n[2] + 90

	# Strength of how much we're enforcing the exit angle
	k = exp_settings['angle_strength']
	
	if is_weak:
		t1 = np.array(as_tangent(start_angle)) * k * .001
	else:
		t1 = np.array(as_tangent(start_angle)) * k

	tn = np.array(as_tangent(end_angle)) * k

	# print(type(t1))
	# tangent vectors

	# print("Tangents")
	# print(t1)
	# print(tn)


	Px=np.concatenate(([t1[0]],x,[tn[0]]))
	Py=np.concatenate(([t1[1]],y,[tn[1]]))

	# interpolation equations
	n = len(x)
	phi = np.zeros((n+2,n+2))
	for i in range(n):
		phi[i+1,i]=1
		phi[i+1,i+1]=4
		phi[i+1,i+2]=1

	# end condition constraints
	phi=np.zeros((n+2,n+2))
	for i in range(n):
		phi[i+1,i] = 1
		phi[i+1,i+1] = 4
		phi[i+1,i+2] = 1 
	phi[0,0] = -3
	phi[0,2] = 3
	phi[n+1,n-1] = -3
	phi[n+1,n+1] = 3
	# passage matrix
	phi_inv = np.linalg.inv(phi)
	# control points
	Qx=6*phi_inv.dot(Px)
	Qy=6*phi_inv.dot(Py)
	# figure plot
	# plt.figure(figsize=(12, 5))
	t=np.linspace(0,1,num=101)

	length = 1000
	width = 1375

	plt.xlim([0, width])
	plt.ylim([0, length])

	x_all = []
	y_all = []

	for k in range(0,n-1):
		x_t = 1.0/6.0*(((1-t)**3)*Qx[k]+(3*t**3-6*t**2+4)*Qx[k+1]+(-3*t**3+3*t**2+3*t+1)*Qx[k+2]+(t**3)*Qx[k+3])
		y_t = 1.0/6.0*(((1-t)**3)*Qy[k]+(3*t**3-6*t**2+4)*Qy[k+1]+(-3*t**3+3*t**2+3*t+1)*Qy[k+2]+(t**3)*Qy[k+3]) 
		
		x_all.extend(x_t)
		y_all.extend(y_t)

	if FLAG_EXPORT_SPLINE_DEBUG:
		plt.plot(x_t,y_t,'k',linewidth=2.0,color='orange')

		print("Saving the path I made lalalala")
		plt.plot(x, y, 'ko', label='fit knots',markersize=15.0)
		plt.plot(Qx, Qy, 'o--', label='control points',markersize=15.0)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend(loc='upper left', ncol=2)
		fn_spline = fn_pathpickle_from_exp_settings(exp_settings) + 'sample-cubic_spline_imposed_tangent_direction.png'
		plt.savefig(fn_spline)
		# plt.show()
		plt.clf()

	return path_formatted(x_all, y_all)

def construct_single_path_bezier(start, end, sample_pts):
	points = []
	
	xys = [start] + sample_pts + [end]

	ts = [t/NUMBER_STEPS for t in range(NUMBER_STEPS + 1)]
	bezier = resto.make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points

def create_path_options(num_paths, target, restaurant, vis_type):
	path_list = []
	for i in range(num_paths):
		path_option = generate_single_path(restaurant, target, vis_type)
		path_list.append(path_option)

	return path_list

def generate_paths(num_paths, restaurant, vis_types):
	path_options = {}
	for target in restaurant.get_goals_all():
		for vis_type in vis_types:
			path_options[target][vis_type] = create_path_options(num_paths, target, restaurant, vis_type)
	return path_options


def get_vis_labels():
	vis_labels, dummy = get_visibilities([], [], [], [])
	return vis_labels

def minMax(x):
	return pd.Series(index=['min','max'],data=[x.min(),x.max()])

def determine_lambda(r):
	start = r.get_start()
	goals = r.get_goals_all()
	lambda_val = 0
	costs = []

	for g in goals:
		p = generate_single_path(r, g, None, 0)

		p_cost = f_path_cost(p)
		costs.append(p_cost)


	final_cost = max(costs)


	pass

def inspect_heatmap(df):
	# print(df)

	length 		= df['x'].max()
	width 		= df['y'].max()
	max_multi 	= df['VIS_MULTI'].max()
	max_a 		= df['VIS_A'].max()
	max_b 		= df['VIS_B'].max()
	max_omni 	= df['VIS_OMNI'].max()

	# print((length, width))
	print((max_omni, max_multi))

	img = np.zeros((length,width), np.uint8)

	# df = df.transpose()
	for x in range(width):
		for y in range(length):
			val = df[(df['x'] == x) & (df['y'] == y) ]
			v = val['VIS_MULTI']
			fill = int(255.0 * (v / max_multi) )
			img[x, y] = fill

		print(x)


	cv2.imwrite('multi_heatmap'+ '.png', img) 

# df.at[i,COL_PATHING] = get_pm_label(row)

def inspect_visibility(options, restaurant, ti, fn):
	options = options[0]

	for pkey in options.keys():
		print(pkey)
		path = options[pkey][0]
		# print('saving fig')


		t = range(len(path))
		v = get_vis_graph_info(path, restaurant)
		# vo, va, vb, vm = v

		fig = plt.figure()
		ax1 = fig.add_subplot(111)

		for key in v.keys():
			ax1.scatter(t, v[key], s=10, c='r', marker="o", label=key)

		# ax1.scatter(t, va, s=10, c='b', marker="o", label="Vis A")
		# ax1.scatter(t, vb, s=10, c='y', marker="o", label="Vis B")
		# ax1.scatter(t, vm, s=10, c='g', marker="o", label="Vis Multi")

		ax1.set_title('visibility of ' + pkey)
		plt.legend(loc='upper left');
		
		plt.savefig(fn + "-" + str(ti) + "-" + pkey + '-vis' + '.png')
		plt.clf()

		# f1 = f_convolved(v1, f_og)
		# f2 = f_convolved(v2, f_og)
		# f3 = f_convolved(v3, f_og)
		# f4 = f_convolved(v4, f_og)
		# f5 = f_convolved(v5, f_og)

		# fig = plt.figure()
		# ax1 = fig.add_subplot(111)

		# ax1.scatter(x, f1, s=10, c='b', marker="o", label=vl1)
		# ax1.scatter(x, f2, s=10, c='r', marker="o", label=vl2)
		# ax1.scatter(x, f3, s=10, c='g', marker="o", label=vl3)
		# ax1.scatter(x, f4, s=10, c='y', marker="o", label=vl4)
		# ax1.scatter(x, f5, s=10, c='grey', marker="o", label=vl5)
		# ax1.set_title('f_remix for best path to goal ' + goal)
		# plt.legend(loc='upper left');
			
def get_vis_graph_info(path, restaurant):
	vals_dict = {}

	obs_sets = restaurant.get_obs_sets()

	for aud_i in obs_sets.keys():
		vals = []
		for t in range(len(path)):

			# goal, goals, path, df_obs
			new_val = f_remix3(t, path[t], obs_sets[aud_i], path)
			# print(new_val)
			# exit()

			vals.append(new_val)

		vals_dict[aud_i] = vals

	return vals_dict
	# return vo, va, vb, vm



def inspect_details(detail_dict, fn):
	if FLAG_PROB_HEADING:
		return
	return

	vis_labels = get_vis_labels()
	vl1 = vis_labels[0]
	vl2 = vis_labels[1]
	vl3 = vis_labels[2]
	vl4 = vis_labels[3]
	vl5 = vis_labels[4]

	for pkey in detail_dict.keys():
		# print('saving fig')
		paths_details = detail_dict[pkey]

		for detail in paths_details:
			v1 = detail[vl1]
			v2 = detail[vl2]
			v3 = detail[vl3]
			v4 = detail[vl4]
			v5 = detail[vl5]

			goal_index = detail['target_index']
			goal = resto.UNITY_GOAL_NAMES[goal_index]

			x = range(len(v1))
	
			fig = plt.figure()
			ax1 = fig.add_subplot(111)

			ax1.scatter(x, v1, s=10, c='b', marker="o", label=vl1)
			ax1.scatter(x, v2, s=10, c='r', marker="o", label=vl2)
			ax1.scatter(x, v3, s=10, c='g', marker="o", label=vl3)
			ax1.scatter(x, v4, s=10, c='y', marker="o", label=vl4)
			ax1.scatter(x, v5, s=10, c='grey', marker="o", label=vl5)
			ax1.set_title('visibility of best path to goal ' + goal)
			plt.legend(loc='upper left');
			
			plt.savefig(fn + 'vis' + '.png')
			plt.clf()

			f1 = f_convolved(v1, f_og)
			f2 = f_convolved(v2, f_og)
			f3 = f_convolved(v3, f_og)
			f4 = f_convolved(v4, f_og)
			f5 = f_convolved(v5, f_og)

			fig = plt.figure()
			ax1 = fig.add_subplot(111)

			ax1.scatter(x, f1, s=10, c='b', marker="o", label=vl1)
			ax1.scatter(x, f2, s=10, c='r', marker="o", label=vl2)
			ax1.scatter(x, f3, s=10, c='g', marker="o", label=vl3)
			ax1.scatter(x, f4, s=10, c='y', marker="o", label=vl4)
			ax1.scatter(x, f5, s=10, c='grey', marker="o", label=vl5)
			ax1.set_title('f_remix for best path to goal ' + goal)
			plt.legend(loc='upper left');
			
			plt.savefig(fn + goal + '-' + pkey + '-convolved' + '.png')
			plt.clf()



def combine_list_of_dicts(all_options):
	new_dict = {}
	keys = {}

	for option in all_options:
		keys = option.keys() | keys

	for key in keys:
		new_dict[key] = []

	for option in all_options:
		for key in keys:
			new_dict[key].append(option[key])
	
	return new_dict

def get_hardcoded():
	start = (104, 477)
	end = (1035, 567)
	l1 = construct_single_path_bezier(start, end, [(894, 265)])

	labels = ['max-lo-fcombo', 'max-la-fcombo', 'max-lb-fcombo', 'max-lm-fcombo']
	p1 = [(104, 477), (141, 459), (178, 444), (215, 430), (251, 417), (287, 405), (322, 395), (357, 386), (391, 379), (425, 373), (459, 368), (492, 365), (525, 363), (557, 363), (588, 364), (620, 366), (651, 370), (681, 375), (711, 381), (740, 389), (769, 398), (798, 409), (826, 421), (854, 434), (881, 449), (908, 465), (934, 483), (960, 502), (985, 522), (1010, 543), (1035, 567)]
	p2 = [(104, 477), (147, 447), (190, 419), (231, 394), (272, 371), (312, 350), (351, 331), (390, 315), (427, 301), (464, 289), (499, 280), (534, 273), (568, 268), (601, 265), (634, 265), (665, 267), (696, 271), (726, 277), (755, 286), (783, 297), (810, 310), (836, 325), (862, 343), (886, 363), (910, 385), (933, 410), (955, 437), (976, 466), (996, 497), (1016, 531), (1035, 567)]
	p3 = [(104, 477), (124, 447), (145, 419), (167, 394), (190, 371), (213, 350), (237, 332), (262, 315), (288, 301), (314, 290), (341, 280), (369, 273), (397, 268), (427, 266), (457, 265), (487, 267), (519, 271), (551, 278), (584, 286), (617, 297), (652, 310), (687, 326), (722, 343), (759, 363), (796, 386), (834, 410), (873, 437), (912, 466), (952, 497), (993, 531), (1035, 567)]
	p4 = [(104, 477), (146, 446), (187, 418), (228, 392), (268, 369), (307, 348), (345, 329), (383, 313), (420, 298), (456, 286), (491, 277), (525, 269), (559, 264), (592, 262), (624, 261), (656, 263), (686, 267), (716, 274), (745, 282), (774, 293), (801, 307), (828, 322), (854, 340), (879, 361), (904, 383), (928, 408), (950, 435), (973, 464), (994, 496), (1015, 530), (1035, 567)]
	p5 = [(104, 477), (98, 509), (95, 540), (95, 569), (97, 596), (101, 620), (108, 643), (118, 663), (130, 682), (145, 698), (162, 712), (182, 725), (204, 735), (229, 743), (256, 749), (286, 753), (318, 755), (353, 755), (390, 753), (430, 749), (472, 742), (517, 734), (565, 724), (615, 711), (667, 697), (722, 680), (779, 662), (839, 641), (902, 618), (967, 593), (1035, 567)]

	# options = {}
	# options[labels[0]] = [p5]
	# options[labels[1]] = [p1]
	# options[labels[2]] = [p2]
	# options[labels[3]] = [p3]

	# RSS Workshop paper points 
	options = {}
	options[labels[0]] = [p5] # RED
	options[labels[1]] = [p3] # YELLOW
	options[labels[2]] = [p2] # BLUE
	options[labels[3]] = [l1] # GREEN

	return options


# remove invalid paths
def trim_paths(r, all_paths_dict, goal, exp_settings, reverse=False):
	trimmed_paths = []
	trimmed_sp = []
	removed_paths = []

	all_paths = all_paths_dict['path']
	sp = all_paths_dict['sp']

	print(len(all_paths))
	print(len(sp))

	for pi in range(len(all_paths)):
		p = all_paths[pi]
		is_valid = is_valid_path(r, p, exp_settings)
		if is_valid and reverse == False:
			trimmed_paths.append(p)
			trimmed_sp.append(sp[pi])
		elif not is_valid and reverse == True:
			trimmed_paths.append(p)
			trimmed_sp.append(sp[pi])

		if not is_valid and reverse == False:
			removed_paths.append(p)
		elif is_valid and reverse == True:
			removed_paths.append(p)

	if reverse == False:
		print("Paths trimmed: " + str(len(all_paths)) + " -> " + str(len(trimmed_paths)))
	return trimmed_paths, removed_paths, trimmed_sp

def get_mirrored(r, sample_sets):
	start = r.get_start()
	sx, sy, st = start
	mirror_sets = []
	# print(path)
	for ss in sample_sets:
		new_path = []
		for p in ss:
			new_y_offset = (sy - p[1])
			new_y = sy + new_y_offset
			new_pt = (p[0], new_y)
			new_path.append(new_pt)

		mirror_sets.append(new_path)
	return mirror_sets

def get_mirrored_path(r, path):
	start = r.get_start()
	sx, sy, st = start
	new_path = []
	for p in path:
		new_y_offset = (sy - p[1])
		new_y = sy + new_y_offset
		new_pt = (p[0], new_y)
		new_path.append(new_pt)
	return new_path

def get_sample_points_sets(r, start, goal, exp_settings):
	# sampling_type = 'systematic'
	# sampling_type = 'visible'
	# sampling_type = 'in_zone'

	sample_sets = []
	# resolution = 10
	SAMPLE_BUFFER = 150

	sampling_type = exp_settings['sampling_type']

	if sampling_type == SAMPLE_TYPE_SYSTEMATIC or sampling_type == SAMPLE_TYPE_FUSION:
		width = r.get_width()
		length = r.get_length()

		xi_range = range(int(width / resolution))
		yi_range = range(int(length / resolution))

		for xi in xi_range:
			for yi in yi_range:
				x = int(resolution * xi)
				y = int(resolution * yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

	if sampling_type == SAMPLE_TYPE_DEMO:
		start = (104, 477)
		end = (1035, 567)
		l1 = construct_single_path_bezier(start, end, [(894, 265)])

		p1 = [(104, 477), (141, 459), (178, 444), (215, 430), (251, 417), (287, 405), (322, 395), (357, 386), (391, 379), (425, 373), (459, 368), (492, 365), (525, 363), (557, 363), (588, 364), (620, 366), (651, 370), (681, 375), (711, 381), (740, 389), (769, 398), (798, 409), (826, 421), (854, 434), (881, 449), (908, 465), (934, 483), (960, 502), (985, 522), (1010, 543), (1035, 567)]
		p2 = [(104, 477), (147, 447), (190, 419), (231, 394), (272, 371), (312, 350), (351, 331), (390, 315), (427, 301), (464, 289), (499, 280), (534, 273), (568, 268), (601, 265), (634, 265), (665, 267), (696, 271), (726, 277), (755, 286), (783, 297), (810, 310), (836, 325), (862, 343), (886, 363), (910, 385), (933, 410), (955, 437), (976, 466), (996, 497), (1016, 531), (1035, 567)]
		p3 = [(104, 477), (124, 447), (145, 419), (167, 394), (190, 371), (213, 350), (237, 332), (262, 315), (288, 301), (314, 290), (341, 280), (369, 273), (397, 268), (427, 266), (457, 265), (487, 267), (519, 271), (551, 278), (584, 286), (617, 297), (652, 310), (687, 326), (722, 343), (759, 363), (796, 386), (834, 410), (873, 437), (912, 466), (952, 497), (993, 531), (1035, 567)]
		p4 = [(104, 477), (146, 446), (187, 418), (228, 392), (268, 369), (307, 348), (345, 329), (383, 313), (420, 298), (456, 286), (491, 277), (525, 269), (559, 264), (592, 262), (624, 261), (656, 263), (686, 267), (716, 274), (745, 282), (774, 293), (801, 307), (828, 322), (854, 340), (879, 361), (904, 383), (928, 408), (950, 435), (973, 464), (994, 496), (1015, 530), (1035, 567)]
		p5 = [(104, 477), (98, 509), (95, 540), (95, 569), (97, 596), (101, 620), (108, 643), (118, 663), (130, 682), (145, 698), (162, 712), (182, 725), (204, 735), (229, 743), (256, 749), (286, 753), (318, 755), (353, 755), (390, 753), (430, 749), (472, 742), (517, 734), (565, 724), (615, 711), (667, 697), (722, 680), (779, 662), (839, 641), (902, 618), (967, 593), (1035, 567)]
		p6 = l1

		p1 = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 426), (203, 420), (203, 412), (204, 403), (204, 393), (205, 381), (206, 367), (207, 351), (210, 333), (213, 314), (216, 294), (221, 271), (229, 248), (238, 224), (249, 200), (264, 176), (283, 153), (306, 134), (334, 118), (368, 109), (406, 107), (448, 112), (492, 122), (536, 136), (578, 151), (619, 167), (658, 184), (695, 200), (730, 216), (763, 231), (794, 245), (823, 258), (850, 268), (875, 278), (897, 285), (918, 291), (936, 295), (952, 297), (966, 298), (978, 296), (987, 292), (994, 287), (999, 282), (1001, 275), (1003, 270), (1004, 265), (1004, 261), (1004, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 426), (203, 420), (203, 412), (204, 403), (204, 393), (205, 381), (206, 367), (207, 351), (210, 333), (213, 314), (216, 294), (221, 271), (229, 248), (238, 224), (249, 200), (264, 176), (283, 153), (306, 134), (334, 118), (368, 109), (406, 107), (448, 112), (492, 122), (536, 136), (578, 151), (619, 167), (658, 184), (695, 200), (730, 216), (763, 231), (794, 245), (823, 258), (850, 268), (875, 278), (897, 285), (918, 291), (936, 295), (952, 297), (966, 298), (978, 296), (987, 292), (994, 287), (999, 282), (1001, 275), (1003, 270), (1004, 265), (1004, 261), (1004, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 435), (217, 434), (224, 433), (232, 432), (243, 430), (256, 427), (270, 426), (287, 423), (307, 420), (328, 416), (352, 412), (379, 408), (407, 403), (438, 398), (472, 393), (508, 388), (547, 382), (587, 377), (629, 370), (673, 364), (719, 359), (765, 353), (808, 349), (849, 345), (885, 343), (918, 341), (947, 341), (972, 342), (992, 344), (1010, 347), (1022, 349), (1031, 352), (1028, 348), (1022, 341), (1018, 333), (1015, 323), (1012, 314), (1009, 304), (1007, 295), (1006, 287), (1006, 279), (1005, 272), (1005, 267), (1005, 263), (1005, 259), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 435), (233, 435), (244, 434), (256, 434), (271, 433), (288, 432), (308, 431), (330, 430), (354, 428), (381, 427), (410, 425), (442, 424), (475, 422), (512, 420), (551, 418), (591, 417), (634, 414), (678, 412), (724, 410), (770, 408), (816, 406), (858, 404), (896, 402), (931, 401), (960, 400), (985, 399), (1006, 398), (1021, 395), (1032, 390), (1034, 382), (1031, 373), (1027, 362), (1022, 350), (1018, 339), (1014, 328), (1011, 317), (1009, 307), (1007, 297), (1006, 288), (1006, 280), (1005, 273), (1005, 267), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 437), (217, 437), (224, 437), (233, 437), (244, 437), (256, 437), (271, 437), (288, 437), (308, 437), (330, 437), (354, 437), (380, 437), (410, 437), (441, 437), (475, 437), (511, 437), (549, 437), (590, 436), (632, 436), (676, 436), (722, 435), (768, 435), (813, 434), (855, 433), (894, 432), (928, 431), (957, 429), (982, 426), (1002, 423), (1017, 416), (1027, 408), (1029, 396), (1027, 384), (1024, 372), (1021, 359), (1017, 346), (1014, 334), (1011, 322), (1009, 311), (1008, 301), (1006, 291), (1005, 283), (1005, 276), (1005, 270), (1005, 264), (1005, 261), (1005, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (223, 436), (231, 436), (241, 436), (252, 436), (265, 436), (281, 436), (298, 436), (318, 435), (339, 435), (363, 435), (389, 435), (417, 434), (448, 434), (480, 433), (514, 433), (550, 432), (588, 431), (627, 430), (668, 429), (709, 428), (749, 427), (786, 426), (819, 424), (849, 422), (877, 419), (901, 415), (922, 410), (941, 405), (956, 397), (969, 388), (979, 377), (987, 365), (992, 353), (997, 341), (999, 329), (1001, 317), (1003, 306), (1003, 297), (1004, 288), (1004, 280), (1004, 273), (1004, 267), (1004, 262), (1004, 259), (1005, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 435), (214, 435), (218, 435), (223, 434), (229, 434), (237, 434), (246, 433), (257, 433), (270, 432), (285, 431), (301, 430), (321, 430), (342, 429), (366, 428), (393, 427), (423, 426), (455, 424), (491, 423), (529, 421), (569, 418), (611, 415), (652, 412), (691, 408), (728, 402), (764, 397), (797, 391), (827, 384), (855, 377), (880, 370), (903, 361), (923, 353), (941, 344), (956, 334), (969, 325), (979, 315), (988, 305), (994, 296), (998, 288), (1001, 280), (1003, 273), (1004, 267), (1004, 263), (1004, 260), (1005, 258), (1005, 257), (1005, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 454), (203, 461), (203, 469), (203, 479), (203, 489), (205, 502), (206, 516), (209, 532), (213, 549), (218, 568), (225, 587), (234, 607), (246, 627), (262, 647), (280, 665), (304, 682), (332, 695), (364, 704), (401, 709), (443, 709), (486, 706), (529, 700), (572, 693), (614, 684), (653, 676), (691, 667), (726, 658), (760, 649), (792, 641), (821, 633), (847, 626), (872, 620), (894, 615), (915, 611), (932, 606), (948, 604), (962, 603), (973, 602), (983, 602), (991, 603), (997, 605), (1000, 607), (1002, 610), (1004, 613), (1004, 615), (1004, 617), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(203, 437), (203, 438), (203, 438), (203, 440), (203, 442), (203, 445), (203, 449), (202, 455), (202, 462), (202, 471), (202, 481), (202, 493), (203, 507), (204, 522), (205, 539), (207, 558), (211, 578), (217, 599), (223, 622), (233, 645), (244, 669), (259, 692), (279, 713), (303, 732), (332, 747), (367, 757), (406, 760), (449, 758), (493, 752), (537, 743), (579, 732), (620, 721), (659, 708), (697, 697), (732, 684), (765, 673), (795, 662), (824, 652), (850, 643), (875, 634), (896, 627), (916, 621), (933, 615), (949, 611), (962, 608), (973, 606), (983, 606), (991, 606), (997, 607), (1000, 609), (1003, 612), (1004, 614), (1004, 616), (1004, 618), (1004, 618)], ((1005, 617, 0), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 435), (233, 435), (244, 434), (257, 434), (272, 433), (289, 432), (309, 431), (331, 430), (356, 429), (382, 427), (412, 426), (444, 425), (478, 424), (515, 423), (554, 422), (596, 421), (638, 420), (684, 420), (730, 420), (777, 421), (824, 423), (867, 425), (907, 428), (942, 433), (972, 439), (997, 448), (1017, 457), (1031, 470), (1040, 483), (1043, 497), (1041, 511), (1038, 524), (1033, 536), (1028, 548), (1023, 558), (1019, 568), (1015, 577), (1011, 584), (1009, 592), (1007, 598), (1006, 603), (1005, 607), (1005, 611), (1005, 614), (1005, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'b'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (225, 436), (233, 436), (244, 435), (257, 435), (272, 435), (290, 434), (309, 434), (331, 433), (356, 433), (383, 432), (413, 432), (444, 431), (479, 431), (515, 431), (555, 431), (596, 431), (639, 431), (684, 432), (730, 433), (777, 435), (824, 437), (867, 440), (906, 443), (940, 448), (971, 454), (996, 461), (1016, 470), (1031, 481), (1040, 493), (1043, 506), (1041, 519), (1038, 531), (1033, 542), (1028, 553), (1023, 563), (1018, 572), (1014, 580), (1011, 587), (1008, 593), (1007, 599), (1006, 604), (1005, 608), (1005, 612), (1005, 614), (1005, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 438), (218, 438), (225, 439), (233, 440), (244, 441), (257, 442), (272, 444), (289, 446), (309, 448), (332, 450), (356, 453), (383, 456), (412, 459), (445, 463), (479, 466), (515, 471), (555, 475), (596, 480), (639, 485), (684, 491), (731, 495), (778, 502), (822, 507), (863, 512), (900, 517), (934, 522), (963, 526), (989, 530), (1009, 534), (1027, 538), (1039, 543), (1045, 548), (1043, 556), (1038, 562), (1031, 568), (1025, 574), (1019, 580), (1015, 585), (1011, 592), (1009, 597), (1007, 602), (1006, 606), (1005, 610), (1005, 613), (1005, 615), (1005, 617), (1005, 617), (1005, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (204, 439), (205, 441), (205, 444), (207, 448), (208, 454), (210, 461), (212, 469), (216, 478), (220, 490), (225, 503), (231, 517), (237, 533), (245, 550), (255, 569), (266, 588), (279, 609), (294, 630), (312, 651), (331, 671), (355, 691), (383, 709), (413, 723), (448, 734), (486, 739), (527, 738), (568, 734), (608, 726), (647, 717), (683, 706), (718, 695), (751, 684), (781, 672), (810, 662), (836, 652), (861, 642), (883, 634), (904, 627), (922, 619), (938, 614), (953, 610), (965, 607), (976, 604), (985, 604), (992, 604), (997, 606), (1001, 607), (1003, 610), (1004, 613), (1004, 615), (1004, 617), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 453), (203, 459), (203, 467), (203, 475), (205, 485), (206, 496), (209, 509), (214, 522), (220, 537), (227, 552), (237, 567), (251, 583), (267, 598), (287, 612), (311, 624), (339, 633), (371, 641), (407, 644), (447, 646), (489, 645), (532, 642), (575, 639), (617, 634), (656, 629), (694, 625), (730, 620), (764, 615), (795, 610), (824, 606), (851, 604), (876, 601), (898, 599), (918, 597), (935, 596), (951, 596), (964, 596), (975, 597), (985, 599), (992, 602), (997, 604), (1001, 608), (1003, 611), (1004, 613), (1004, 615), (1004, 617), (1004, 617)]}

		paths = list(p1.values())

		# p1 = chunkify.chunkify_path(exp_settings, p1)
		# p2 = chunkify.chunkify_path(exp_settings, p2)
		# p3 = chunkify.chunkify_path(exp_settings, p3)
		# p4 = chunkify.chunkify_path(exp_settings, p4)
		# p5 = chunkify.chunkify_path(exp_settings, p5)
		# p6 = chunkify.chunkify_path(exp_settings, p6)
		# p7 = get_min_viable_path(r, goal, exp_settings)

		sample_sets = paths

	if sampling_type == SAMPLE_TYPE_NEXUS_POINTS or sampling_type == SAMPLE_TYPE_NEXUS_POINTS_ONLY:
		sample_sets = []
		imported_0 = {((1005, 257, 180), 'naked'): [(504, 107)], ((1005, 257, 180), 'omni'): [(504, 107)], ((1005, 257, 180), 'a'): [(504, 407)], ((1005, 257, 180), 'b'): [(804, 407)], ((1005, 257, 180), 'c'): [(804, 407)], ((1005, 257, 180), 'd'): [(804, 407)], ((1005, 257, 180), 'e'): [(804, 407)], ((1005, 617, 0), 'naked'): [(504, 407)], ((1005, 617, 0), 'omni'): [(504, 407)], ((1005, 617, 0), 'a'): [(504, 407)], ((1005, 617, 0), 'b'): [(504, 407)], ((1005, 617, 0), 'c'): [(504, 407)], ((1005, 617, 0), 'd'): [(504, 407)], ((1005, 617, 0), 'e'): [(504, 407)]}
		imported_1 = {((1005, 257, 180), 'naked'): [(508, 111)], ((1005, 257, 180), 'omni'): [(508, 111)], ((1005, 257, 180), 'a'): [(503, 411)], ((1005, 257, 180), 'b'): [(800, 411)], ((1005, 257, 180), 'c'): [(807, 411)], ((1005, 257, 180), 'd'): [(807, 411)], ((1005, 257, 180), 'e'): [(807, 402)], ((1005, 617, 0), 'naked'): [(500, 411)], ((1005, 617, 0), 'omni'): [(500, 411)], ((1005, 617, 0), 'a'): [(500, 411)], ((1005, 617, 0), 'b'): [(499, 411)], ((1005, 617, 0), 'c'): [(499, 411)], ((1005, 617, 0), 'd'): [(499, 411)], ((1005, 617, 0), 'e'): [(500, 411)]}
		imported_2 = {((1005, 257, 180), 'naked'): [(624, 107)], ((1005, 257, 180), 'omni'): [(624, 107)], ((1005, 257, 180), 'a'): [(474, 407)], ((1005, 257, 180), 'b'): [(774, 407)], ((1005, 257, 180), 'c'): [(924, 407)], ((1005, 257, 180), 'd'): [(984, 407)], ((1005, 257, 180), 'e'): [(984, 407)], ((1005, 617, 0), 'naked'): [(414, 737)], ((1005, 617, 0), 'omni'): [(414, 737)], ((1005, 617, 0), 'a'): [(414, 737)], ((1005, 617, 0), 'b'): [(414, 737)], ((1005, 617, 0), 'c'): [(804, 647)], ((1005, 617, 0), 'd'): [(804, 647)], ((1005, 617, 0), 'e'): [(804, 587)]}
		imported_3 = {((1005, 257, 180), 'naked'): [(534, 122)], ((1005, 257, 180), 'omni'): [(534, 122)], ((1005, 257, 180), 'a'): [(624, 422)], ((1005, 257, 180), 'b'): [(744, 422)], ((1005, 257, 180), 'c'): [(909, 422)], ((1005, 257, 180), 'd'): [(984, 407)], ((1005, 257, 180), 'e'): [(984, 407)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(399, 752)], ((1005, 617, 0), 'b'): [(429, 752)], ((1005, 617, 0), 'c'): [(819, 647)], ((1005, 617, 0), 'd'): [(819, 647)], ((1005, 617, 0), 'e'): [(819, 572)]}
		imported_4 = {((1005, 257, 180), 'naked'): [(533, 125)], ((1005, 257, 180), 'omni'): [(533, 125)], ((1005, 257, 180), 'a'): [(619, 423)], ((1005, 257, 180), 'b'): [(743, 425)], ((1005, 257, 180), 'c'): [(908, 425)], ((1005, 257, 180), 'd'): [(987, 402)], ((1005, 257, 180), 'e'): [(987, 402)], ((1005, 617, 0), 'naked'): [(426, 723)], ((1005, 617, 0), 'omni'): [(426, 723)], ((1005, 617, 0), 'a'): [(396, 755)], ((1005, 617, 0), 'b'): [(426, 755)], ((1005, 617, 0), 'c'): [(822, 650)], ((1005, 617, 0), 'd'): [(822, 650)], ((1005, 617, 0), 'e'): [(822, 573)]}
		imported_5 = {((1005, 257, 180), 'naked'): [(540, 124)], ((1005, 257, 180), 'omni'): [(540, 124)], ((1005, 257, 180), 'a'): [(617, 429)], ((1005, 257, 180), 'b'): [(737, 433)], ((1005, 257, 180), 'c'): [(992, 413)], ((1005, 257, 180), 'd'): [(992, 413)], ((1005, 257, 180), 'e'): [(990, 409)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(390, 757)], ((1005, 617, 0), 'b'): [(426, 763)], ((1005, 617, 0), 'c'): [(830, 648)], ((1005, 617, 0), 'd'): [(830, 648)], ((1005, 617, 0), 'e'): [(824, 571)]}

		imported_res_2 = {((1005, 257, 180), 'naked'): [(1152, 431)], ((1005, 257, 180), 'omni'): [(1152, 431)], ((1005, 257, 180), 'a'): [(486, 431)], ((1005, 257, 180), 'b'): [(738, 431)], ((1005, 257, 180), 'c'): [(1044, 431)], ((1005, 257, 180), 'd'): [(1152, 431)], ((1005, 257, 180), 'e'): [(1152, 431)], ((1005, 617, 0), 'naked'): [(426, 725)], ((1005, 617, 0), 'omni'): [(426, 725)], ((1005, 617, 0), 'a'): [(372, 761)], ((1005, 617, 0), 'b'): [(426, 761)], ((1005, 617, 0), 'c'): [(426, 731)], ((1005, 617, 0), 'd'): [(426, 725)], ((1005, 617, 0), 'e'): [(1116, 449)]}
		imported_res_3 = {((1005, 257, 180), 'naked'): [(545, 123)], ((1005, 257, 180), 'omni'): [(545, 123)], ((1005, 257, 180), 'a'): [(616, 434)], ((1005, 257, 180), 'b'): [(734, 434)], ((1005, 257, 180), 'c'): [(983, 426)], ((1005, 257, 180), 'd'): [(992, 413)], ((1005, 257, 180), 'e'): [(990, 409)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(375, 768)], ((1005, 617, 0), 'b'): [(429, 776)], ((1005, 617, 0), 'c'): [(832, 648)], ((1005, 617, 0), 'd'): [(832, 648)], ((1005, 617, 0), 'e'): [(824, 571)]}

		imported_res_4 = {((1005, 257, 180), 'naked'): [(564, 107)], ((1005, 257, 180), 'omni'): [(564, 107)], ((1005, 257, 180), 'a'): [(384, 407)], ((1005, 257, 180), 'b'): [(744, 407)], ((1005, 257, 180), 'c'): [(924, 407)], ((1005, 257, 180), 'd'): [(1014, 407)], ((1005, 257, 180), 'e'): [(1014, 407)], ((1005, 617, 0), 'naked'): [(834, 557)], ((1005, 617, 0), 'omni'): [(834, 557)], ((1005, 617, 0), 'a'): [(834, 497)], ((1005, 617, 0), 'b'): [(834, 587)], ((1005, 617, 0), 'c'): [(1014, 467)], ((1005, 617, 0), 'd'): [(1014, 467)], ((1005, 617, 0), 'e'): [(1014, 467)]}
		imported_res_5 = {((1005, 257, 180), 'naked'): [(1114, 422)], ((1005, 257, 180), 'omni'): [(1114, 422)], ((1005, 257, 180), 'a'): [(384, 432)], ((1005, 257, 180), 'b'): [(734, 432)], ((1005, 257, 180), 'c'): [(969, 432)], ((1005, 257, 180), 'd'): [(1154, 432)], ((1005, 257, 180), 'e'): [(1154, 432)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(374, 642)], ((1005, 617, 0), 'b'): [(649, 762)], ((1005, 617, 0), 'c'): [(1009, 622)], ((1005, 617, 0), 'd'): [(1154, 442)], ((1005, 617, 0), 'e'): [(1154, 442)]}
		imported_res_6 = {((1005, 257, 180), 'naked'): [(1149, 431)], ((1005, 257, 180), 'omni'): [(1149, 431)], ((1005, 257, 180), 'a'): [(387, 433)], ((1005, 257, 180), 'b'): [(731, 434)], ((1005, 257, 180), 'c'): [(958, 433)], ((1005, 257, 180), 'd'): [(1151, 434)], ((1005, 257, 180), 'e'): [(1167, 429)], ((1005, 617, 0), 'naked'): [(426, 789)], ((1005, 617, 0), 'omni'): [(426, 789)], ((1005, 617, 0), 'a'): [(371, 643)], ((1005, 617, 0), 'b'): [(640, 775)], ((1005, 617, 0), 'c'): [(1004, 623)], ((1005, 617, 0), 'd'): [(1163, 444)], ((1005, 617, 0), 'e'): [(1163, 443)]}

		nexus_icon_3 = {((1005, 257, 180), 'naked'): [(644, 397)], ((1005, 257, 180), 'omni'): [(644, 397)], ((1005, 257, 180), 'a'): [(1134, 427)], ((1005, 257, 180), 'b'): [(1114, 427)], ((1005, 257, 180), 'c'): [(914, 427)], ((1005, 257, 180), 'd'): [(734, 427)], ((1005, 257, 180), 'e'): [(384, 427)], ((1005, 617, 0), 'naked'): [(664, 547)], ((1005, 617, 0), 'omni'): [(664, 547)], ((1005, 617, 0), 'a'): [(1124, 447)], ((1005, 617, 0), 'b'): [(1124, 447)], ((1005, 617, 0), 'c'): [(824, 647)], ((1005, 617, 0), 'd'): [(664, 757)], ((1005, 617, 0), 'e'): [(374, 637)]}
		nexus_icon = {((1005, 257, 180), 'naked'): [(1150, 431)], ((1005, 257, 180), 'omni'): [(1150, 431)], ((1005, 257, 180), 'a'): [(1180, 430)], ((1005, 257, 180), 'b'): [(1152, 434)], ((1005, 257, 180), 'c'): [(971, 434)], ((1005, 257, 180), 'd'): [(734, 433)], ((1005, 257, 180), 'e'): [(386, 434)], ((1005, 617, 0), 'naked'): [(429, 790)], ((1005, 617, 0), 'omni'): [(429, 790)], ((1005, 617, 0), 'a'): [(1178, 442)], ((1005, 617, 0), 'b'): [(1172, 445)], ((1005, 617, 0), 'c'): [(971, 444)], ((1005, 617, 0), 'd'): [(643, 788)], ((1005, 617, 0), 'e'): [(372, 644)]}
		nexus_icon_2 = {((1005, 257, 180), 'naked'): [(594, 197)], ((1005, 257, 180), 'omni'): [(594, 197)], ((1005, 257, 180), 'a'): [(1134, 427)], ((1005, 257, 180), 'b'): [(1114, 427)], ((1005, 257, 180), 'c'): [(914, 427)], ((1005, 257, 180), 'd'): [(734, 427)], ((1005, 257, 180), 'e'): [(384, 427)], ((1005, 617, 0), 'naked'): [(414, 727)], ((1005, 617, 0), 'omni'): [(414, 727)], ((1005, 617, 0), 'a'): [(1124, 447)], ((1005, 617, 0), 'b'): [(1124, 447)], ((1005, 617, 0), 'c'): [(824, 647)], ((1005, 617, 0), 'd'): [(664, 757)], ((1005, 617, 0), 'e'): [(374, 637)]}

		central_20_points = {((1005, 257, 180), 'naked'): [(384, 107)], ((1005, 257, 180), 'omni'): [(384, 107)], ((1005, 257, 180), 'a'): [(1124, 427)], ((1005, 257, 180), 'b'): [(1124, 427)], ((1005, 257, 180), 'c'): [(904, 427)], ((1005, 257, 180), 'd'): [(764, 427)], ((1005, 257, 180), 'e'): [(384, 427)], ((1005, 617, 0), 'naked'): [(664, 707)], ((1005, 617, 0), 'omni'): [(664, 707)], ((1005, 617, 0), 'a'): [(1124, 447)], ((1005, 617, 0), 'b'): [(1124, 447)], ((1005, 617, 0), 'c'): [(824, 647)], ((1005, 617, 0), 'd'): [(664, 747)], ((1005, 617, 0), 'e'): [(664, 507)]}
		central_15_points = {((1005, 257, 180), 'naked'): [(384, 107)], ((1005, 257, 180), 'omni'): [(384, 107)], ((1005, 257, 180), 'a'): [(1119, 422)], ((1005, 257, 180), 'b'): [(1119, 422)], ((1005, 257, 180), 'c'): [(909, 422)], ((1005, 257, 180), 'd'): [(744, 422)], ((1005, 257, 180), 'e'): [(384, 422)], ((1005, 617, 0), 'naked'): [(399, 722)], ((1005, 617, 0), 'omni'): [(399, 722)], ((1005, 617, 0), 'a'): [(1059, 452)], ((1005, 617, 0), 'b'): [(1059, 452)], ((1005, 617, 0), 'c'): [(819, 647)], ((1005, 617, 0), 'd'): [(429, 752)], ((1005, 617, 0), 'e'): [(399, 587)]}

		central_15_points = {((1005, 257, 180), 'naked'): [(384, 107)], ((1005, 257, 180), 'omni'): [(384, 107)], ((1005, 257, 180), 'a'): [(1104, 422)], ((1005, 257, 180), 'b'): [(1104, 422)], ((1005, 257, 180), 'c'): [(924, 422)], ((1005, 257, 180), 'd'): [(744, 422)], ((1005, 257, 180), 'e'): [(384, 422)], ((1005, 617, 0), 'naked'): [(429, 737)], ((1005, 617, 0), 'omni'): [(429, 737)], ((1005, 617, 0), 'a'): [(1059, 452)], ((1005, 617, 0), 'b'): [(1059, 452)], ((1005, 617, 0), 'c'): [(834, 632)], ((1005, 617, 0), 'd'): [(429, 767)], ((1005, 617, 0), 'e'): [(429, 587)]}

		# central points with new method
		central_15_points = {((1005, 257, 180), 'naked'): [(389, 107)], ((1005, 257, 180), 'omni'): [(389, 107)], ((1005, 257, 180), 'a'): [(954, 357)], ((1005, 257, 180), 'b'): [(949, 392)], ((1005, 257, 180), 'c'): [(904, 432)], ((1005, 257, 180), 'd'): [(764, 427)], ((1005, 257, 180), 'e'): [(399, 427)], ((1005, 617, 0), 'naked'): [(374, 732)], ((1005, 617, 0), 'omni'): [(374, 732)], ((1005, 617, 0), 'a'): [(949, 437)], ((1005, 617, 0), 'b'): [(949, 442)], ((1005, 617, 0), 'c'): [(844, 627)], ((1005, 617, 0), 'd'): [(649, 762)], ((1005, 617, 0), 'e'): [(374, 607)]}
		central_15_points = {((1005, 257, 180), 'naked'): [(399, 107)], ((1005, 257, 180), 'omni'): [(399, 107)], ((1005, 257, 180), 'a'): [(924, 362)], ((1005, 257, 180), 'b'): [(924, 407)], ((1005, 257, 180), 'c'): [(894, 422)], ((1005, 257, 180), 'd'): [(759, 422)], ((1005, 257, 180), 'e'): [(399, 422)], ((1005, 617, 0), 'naked'): [(384, 737)], ((1005, 617, 0), 'omni'): [(384, 737)], ((1005, 617, 0), 'a'): [(924, 437)], ((1005, 617, 0), 'b'): [(924, 437)], ((1005, 617, 0), 'c'): [(879, 437)], ((1005, 617, 0), 'd'): [(384, 752)], ((1005, 617, 0), 'e'): [(384, 602)]}

		# best with cutoff 20
		central_15_points = {((1005, 257, 180), 'naked'): [(399, 107)], ((1005, 257, 180), 'omni'): [(399, 107)], ((1005, 257, 180), 'a'): [(894, 332)], ((1005, 257, 180), 'b'): [(894, 422)], ((1005, 257, 180), 'c'): [(894, 422)], ((1005, 257, 180), 'd'): [(759, 422)], ((1005, 257, 180), 'e'): [(399, 422)], ((1005, 617, 0), 'naked'): [(384, 722)], ((1005, 617, 0), 'omni'): [(384, 722)], ((1005, 617, 0), 'a'): [(894, 422)], ((1005, 617, 0), 'b'): [(804, 467)], ((1005, 617, 0), 'c'): [(789, 662)], ((1005, 617, 0), 'd'): [(384, 752)], ((1005, 617, 0), 'e'): [(384, 647)]}

		best_with_cutoff_20 	= {((1005, 257, 180), 'naked'): [(387, 107)], ((1005, 257, 180), 'omni'): [(387, 107)], ((1005, 257, 180), 'a'): [(855, 248)], ((1005, 257, 180), 'b'): [(894, 425)], ((1005, 257, 180), 'c'): [(894, 434)], ((1005, 257, 180), 'd'): [(738, 434)], ((1005, 257, 180), 'e'): [(387, 434)], ((1005, 617, 0), 'naked'): [(372, 716)], ((1005, 617, 0), 'omni'): [(372, 716)], ((1005, 617, 0), 'a'): [(885, 434)], ((1005, 617, 0), 'b'): [(861, 434)], ((1005, 617, 0), 'c'): [(384, 719)], ((1005, 617, 0), 'd'): [(384, 764)], ((1005, 617, 0), 'e'): [(372, 650)]}
		nexus_icon4 			= {((1005, 257, 180), 'naked'): [(399, 107)], ((1005, 257, 180), 'omni'): [(399, 107)], ((1005, 257, 180), 'a'): [(1014, 347)], ((1005, 257, 180), 'b'): [(1014, 407)], ((1005, 257, 180), 'c'): [(894, 422)], ((1005, 257, 180), 'd'): [(759, 407)], ((1005, 257, 180), 'e'): [(534, 407)], ((1005, 617, 0), 'naked'): [(384, 752)], ((1005, 617, 0), 'omni'): [(384, 752)], ((1005, 617, 0), 'a'): [(1014, 452)], ((1005, 617, 0), 'b'): [(1014, 452)], ((1005, 617, 0), 'c'): [(984, 497)], ((1005, 617, 0), 'd'): [(384, 767)], ((1005, 617, 0), 'e'): [(384, 617)]}

		demo_sept19_1 = {((1005, 257, 180), 'naked'): [(414, 107)], ((1005, 257, 180), 'omni'): [(414, 107)], ((1005, 257, 180), 'a'): [(924, 407)], ((1005, 257, 180), 'b'): [(924, 407)], ((1005, 257, 180), 'c'): [(924, 407)], ((1005, 257, 180), 'd'): [(774, 407)], ((1005, 257, 180), 'e'): [(414, 407)], ((1005, 617, 0), 'naked'): [(384, 737)], ((1005, 617, 0), 'omni'): [(384, 737)], ((1005, 617, 0), 'a'): [(924, 437)], ((1005, 617, 0), 'b'): [(924, 467)], ((1005, 617, 0), 'c'): [(864, 587)], ((1005, 617, 0), 'd'): [(384, 767)], ((1005, 617, 0), 'e'): [(384, 647)]}
		# demo_sept19_2 = {((1005, 257, 180), 'naked'): [(389, 107)], ((1005, 257, 180), 'omni'): [(389, 107)], ((1005, 257, 180), 'a'): [(959, 387)], ((1005, 257, 180), 'b'): [(959, 387)], ((1005, 257, 180), 'c'): [(914, 432)], ((1005, 257, 180), 'd'): [(749, 432)], ((1005, 257, 180), 'e'): [(389, 432)], ((1005, 617, 0), 'naked'): [(374, 722)], ((1005, 617, 0), 'omni'): [(374, 722)], ((1005, 617, 0), 'a'): [(954, 442)], ((1005, 617, 0), 'b'): [(954, 442)], ((1005, 617, 0), 'c'): [(859, 612)], ((1005, 617, 0), 'd'): [(694, 727)], ((1005, 617, 0), 'e'): [(374, 652)]}
		# demo_sept19_3 = {((1005, 257, 180), 'naked'): [(389, 107)], ((1005, 257, 180), 'omni'): [(389, 107)], ((1005, 257, 180), 'a'): [(959, 392)], ((1005, 257, 180), 'b'): [(959, 392)], ((1005, 257, 180), 'c'): [(914, 432)], ((1005, 257, 180), 'd'): [(749, 432)], ((1005, 257, 180), 'e'): [(389, 432)], ((1005, 617, 0), 'naked'): [(374, 722)], ((1005, 617, 0), 'omni'): [(374, 722)], ((1005, 617, 0), 'a'): [(954, 442)], ((1005, 617, 0), 'b'): [(954, 442)], ((1005, 617, 0), 'c'): [(854, 622)], ((1005, 617, 0), 'd'): [(649, 762)], ((1005, 617, 0), 'e'): [(374, 657)]}

		demo_sept19_4 = {((1005, 257, 180), 'naked'): [(414, 107)], ((1005, 257, 180), 'omni'): [(414, 107)], ((1005, 257, 180), 'a'): [(954, 377)], ((1005, 257, 180), 'b'): [(954, 407)], ((1005, 257, 180), 'c'): [(924, 407)], ((1005, 257, 180), 'd'): [(834, 407)], ((1005, 257, 180), 'e'): [(414, 407)], ((1005, 617, 0), 'naked'): [(384, 737)], ((1005, 617, 0), 'omni'): [(384, 737)], ((1005, 617, 0), 'a'): [(954, 437)], ((1005, 617, 0), 'b'): [(954, 467)], ((1005, 617, 0), 'c'): [(894, 527)], ((1005, 617, 0), 'd'): [(384, 767)], ((1005, 617, 0), 'e'): [(384, 647)]}
		demo_sept19_5 = {((1005, 257, 180), 'naked'): [(409, 102)], ((1005, 257, 180), 'omni'): [(409, 102)], ((1005, 257, 180), 'a'): [(959, 388)], ((1005, 257, 180), 'b'): [(956, 437)], ((1005, 257, 180), 'c'): [(912, 435)], ((1005, 257, 180), 'd'): [(747, 435)], ((1005, 257, 180), 'e'): [(387, 435)], ((1005, 617, 0), 'naked'): [(379, 732)], ((1005, 617, 0), 'omni'): [(379, 732)], ((1005, 617, 0), 'a'): [(958, 438)], ((1005, 617, 0), 'b'): [(955, 440)], ((1005, 617, 0), 'c'): [(856, 621)], ((1005, 617, 0), 'd'): [(646, 764)], ((1005, 617, 0), 'e'): [(371, 656)]}
		demo_sept19_6 = {((1005, 257, 180), 'naked'): [(399, 107)], ((1005, 257, 180), 'omni'): [(399, 107)], ((1005, 257, 180), 'a'): [(849, 257)], ((1005, 257, 180), 'b'): [(864, 422)], ((1005, 257, 180), 'c'): [(864, 422)], ((1005, 257, 180), 'd'): [(774, 422)], ((1005, 257, 180), 'e'): [(444, 422)], ((1005, 617, 0), 'naked'): [(384, 752)], ((1005, 617, 0), 'omni'): [(384, 752)], ((1005, 617, 0), 'a'): [(864, 422)], ((1005, 617, 0), 'b'): [(864, 452)], ((1005, 617, 0), 'c'): [(849, 617)], ((1005, 617, 0), 'd'): [(384, 767)], ((1005, 617, 0), 'e'): [(384, 617)]}

		demo_sept19_6 = {((1005, 257, 180), 'naked'): [(374, 107)], ((1005, 257, 180), 'omni'): [(374, 107)], ((1005, 257, 180), 'a'): [(934, 357)], ((1005, 257, 180), 'b'): [(934, 412)], ((1005, 257, 180), 'c'): [(904, 432)], ((1005, 257, 180), 'd'): [(759, 432)], ((1005, 257, 180), 'e'): [(374, 432)], ((1005, 617, 0), 'naked'): [(379, 727)], ((1005, 617, 0), 'omni'): [(379, 727)], ((1005, 617, 0), 'a'): [(934, 437)], ((1005, 617, 0), 'b'): [(934, 442)], ((1005, 617, 0), 'c'): [(854, 627)], ((1005, 617, 0), 'd'): [(649, 762)], ((1005, 617, 0), 'e'): [(379, 602)]}

		imported_0 = list(imported_0.values())
		imported_1 = list(imported_1.values())
		imported_2 = list(imported_2.values())
		imported_3 = list(imported_3.values())
		imported_4 = list(imported_4.values())
		imported_5 = list(imported_5.values())
		# imported_5 = list(imported_6.values())

		imported_res_2 = list(imported_res_2.values())
		imported_res_3 = list(imported_res_3.values())
		imported_res_4 = list(imported_res_4.values())
		imported_res_5 = list(imported_res_5.values())
		imported_res_6 = list(imported_res_6.values())

		demo_sept19_1 = list(demo_sept19_1.values())
		demo_sept19_4 = list(demo_sept19_4.values())
		demo_sept19_5 = list(demo_sept19_5.values())
		demo_sept19_6 = list(demo_sept19_6.values())

		# central_20_points = list(central_20_points.values())
		# central_15_points = list(central_15_points.values())
		# best_with_cutoff_20 = list(best_with_cutoff_20.values())


		# nexus_icon = list(nexus_icon.values())[7:9]
		nexus_icon = [[(429, 790)], [(1150, 431)]]

		imported = []
		# imported.extend(imported_1)
		# imported.extend(imported_2)
		# imported.extend(imported_3)
		# imported.extend(imported_4)
		# imported.extend(imported_5)
		# imported.extend(imported_res_2)
		# imported.extend(imported_res_3)
		# imported.extend(imported_res_4)
		# imported.extend(imported_res_5)
		# imported.extend(imported_res_6)
		# imported.extend(nexus_icon)
		# imported.extend(central_20_points)
		# imported.extend(demo_sept19_4)
		# imported.extend(demo_sept19_5)
		imported.extend(demo_sept19_6)



		new_imported = []
		for imp in imported:
			if imp not in new_imported:
				new_imported.append(imp)
		imported = new_imported

		# mirror_sets = get_mirrored(r, imported)
		# for p in mirror_sets:
		# 	if p not in imported:
		# 		imported.append(p)


		if sampling_type == SAMPLE_TYPE_NEXUS_POINTS:
			resolution = 3

			augmented = [] #imported
			search_hi = 9
			search_lo = -1 * search_hi

			for xi in range(search_lo, search_hi, resolution):
				for yi in range(search_lo, search_hi, resolution):
					for imp in imported:
						new_set = []
						for p in imp:
							print(p)
							# print(xi, yi)
							new_pt = (p[0] + xi, p[1] + yi)
							new_set.append(new_pt)
						augmented.append(new_set)

			# # print(augmented)
			sample_sets.extend(augmented)
		else:
			sample_sets.extend(imported)
	
		print("Done sampling")

	if sampling_type == SAMPLE_TYPE_CURVE_TEST:
		# test_dict = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 425), (204, 420), (204, 413), (204, 404), (205, 394), (206, 382), (208, 369), (209, 354), (213, 338), (216, 320), (221, 301), (228, 281), (237, 259), (248, 238), (262, 217), (280, 197), (303, 180), (329, 166), (361, 159), (397, 157), (436, 162), (479, 170), (523, 184), (566, 199), (608, 216), (647, 231), (685, 247), (721, 262), (756, 276), (788, 289), (818, 300), (846, 309), (872, 317), (896, 321), (917, 325), (936, 326), (953, 325), (968, 321), (979, 316), (988, 308), (995, 300), (999, 291), (1001, 283), (1003, 275), (1004, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 425), (204, 420), (204, 413), (204, 404), (205, 394), (206, 382), (208, 369), (209, 354), (213, 338), (216, 320), (221, 301), (228, 281), (237, 259), (248, 238), (262, 217), (280, 197), (303, 180), (329, 166), (361, 159), (397, 157), (436, 162), (479, 170), (523, 184), (566, 199), (608, 216), (647, 231), (685, 247), (721, 262), (756, 276), (788, 289), (818, 300), (846, 309), (872, 317), (896, 321), (917, 325), (936, 326), (953, 325), (968, 321), (979, 316), (988, 308), (995, 300), (999, 291), (1001, 283), (1003, 275), (1004, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (215, 436), (220, 435), (227, 435), (235, 435), (245, 434), (256, 433), (269, 433), (285, 432), (301, 431), (320, 430), (341, 430), (363, 429), (387, 428), (414, 427), (443, 426), (473, 426), (505, 426), (539, 425), (574, 426), (611, 426), (650, 427), (688, 429), (724, 430), (758, 430), (790, 430), (820, 429), (848, 426), (874, 422), (897, 416), (918, 409), (936, 400), (951, 390), (965, 378), (975, 365), (983, 352), (990, 339), (995, 327), (998, 315), (1000, 303), (1003, 293), (1003, 284), (1004, 276), (1004, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (216, 436), (222, 436), (230, 435), (239, 435), (250, 435), (263, 434), (278, 433), (295, 433), (314, 432), (335, 432), (358, 431), (384, 430), (411, 429), (440, 428), (472, 428), (505, 427), (540, 426), (577, 426), (615, 426), (655, 426), (695, 426), (734, 427), (771, 428), (805, 429), (836, 430), (863, 430), (889, 429), (911, 427), (931, 423), (949, 416), (963, 408), (974, 396), (983, 384), (989, 370), (994, 357), (997, 343), (1000, 330), (1001, 317), (1003, 306), (1003, 295), (1004, 286), (1004, 277), (1004, 270), (1004, 265), (1004, 260), (1005, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'c'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 436), (233, 436), (244, 435), (257, 435), (272, 435), (289, 434), (309, 434), (330, 433), (355, 432), (381, 432), (411, 431), (443, 431), (476, 430), (513, 429), (552, 428), (592, 428), (635, 427), (679, 426), (725, 426), (772, 426), (818, 425), (861, 426), (899, 426), (934, 427), (964, 428), (989, 429), (1009, 430), (1025, 429), (1034, 423), (1033, 413), (1029, 400), (1024, 387), (1020, 374), (1016, 360), (1013, 346), (1010, 333), (1008, 320), (1007, 308), (1006, 297), (1006, 288), (1005, 279), (1005, 272), (1005, 266), (1005, 261), (1005, 259), (1005, 257), (1005, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (218, 435), (225, 435), (235, 434), (246, 433), (259, 432), (275, 431), (293, 430), (313, 429), (336, 427), (361, 425), (389, 423), (420, 421), (453, 419), (488, 417), (527, 414), (567, 411), (610, 409), (655, 407), (701, 404), (750, 402), (799, 400), (848, 398), (894, 397), (936, 396), (973, 396), (1006, 397), (1034, 399), (1056, 402), (1072, 405), (1075, 406), (1065, 401), (1053, 392), (1043, 382), (1034, 371), (1027, 359), (1021, 346), (1016, 334), (1012, 321), (1009, 310), (1008, 299), (1006, 289), (1005, 280), (1005, 273), (1005, 267), (1005, 262), (1005, 259), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (218, 435), (225, 435), (235, 434), (246, 433), (259, 432), (275, 431), (293, 430), (313, 429), (336, 427), (361, 425), (389, 423), (420, 421), (453, 419), (488, 417), (527, 414), (567, 411), (610, 409), (655, 407), (701, 404), (750, 402), (799, 400), (848, 398), (894, 397), (936, 396), (973, 396), (1006, 397), (1034, 399), (1056, 402), (1072, 405), (1075, 406), (1065, 401), (1053, 392), (1043, 382), (1034, 371), (1027, 359), (1021, 346), (1016, 334), (1012, 321), (1009, 310), (1008, 299), (1006, 289), (1005, 280), (1005, 273), (1005, 267), (1005, 262), (1005, 259), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (202, 438), (202, 440), (200, 443), (198, 447), (196, 452), (193, 459), (189, 467), (185, 476), (180, 487), (175, 499), (170, 514), (165, 529), (160, 547), (155, 566), (152, 586), (150, 608), (150, 630), (155, 652), (166, 673), (183, 691), (208, 701), (240, 706), (277, 705), (319, 698), (365, 688), (414, 674), (463, 660), (511, 645), (558, 630), (603, 615), (646, 602), (686, 589), (725, 578), (761, 569), (795, 561), (827, 554), (856, 549), (882, 547), (906, 545), (927, 546), (946, 549), (962, 553), (975, 559), (985, 566), (992, 575), (998, 583), (1000, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(204, 437), (203, 437), (203, 437), (202, 438), (202, 440), (200, 443), (198, 447), (196, 452), (193, 459), (189, 467), (185, 476), (180, 487), (175, 499), (170, 514), (165, 529), (160, 547), (155, 566), (152, 586), (150, 608), (150, 630), (155, 652), (166, 673), (183, 691), (208, 701), (240, 706), (277, 705), (319, 698), (365, 688), (414, 674), (463, 660), (511, 645), (558, 630), (603, 615), (646, 602), (686, 589), (725, 578), (761, 569), (795, 561), (827, 554), (856, 549), (882, 547), (906, 545), (927, 546), (946, 549), (962, 553), (975, 559), (985, 566), (992, 575), (998, 583), (1000, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'a'): [(204, 437), (203, 437), (203, 437), (202, 438), (201, 440), (199, 443), (197, 446), (193, 451), (189, 458), (184, 465), (178, 474), (173, 485), (166, 497), (158, 510), (150, 525), (141, 542), (133, 560), (126, 580), (120, 601), (115, 623), (115, 647), (120, 669), (133, 688), (157, 700), (187, 706), (224, 705), (267, 699), (313, 689), (364, 676), (415, 661), (466, 647), (515, 632), (562, 618), (608, 605), (651, 592), (693, 581), (732, 571), (769, 562), (802, 556), (834, 551), (862, 548), (889, 546), (912, 546), (933, 548), (951, 551), (965, 556), (978, 562), (987, 570), (994, 578), (999, 587), (1002, 595), (1003, 602), (1004, 607), (1004, 612), (1004, 615), (1004, 617), (1004, 617)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (205, 438), (207, 440), (210, 443), (214, 447), (218, 451), (224, 457), (231, 464), (239, 472), (249, 481), (260, 491), (274, 504), (288, 517), (305, 530), (323, 546), (344, 562), (366, 579), (390, 596), (416, 615), (444, 633), (475, 649), (507, 666), (542, 682), (579, 694), (618, 704), (658, 707), (696, 703), (732, 694), (764, 681), (794, 666), (821, 650), (846, 633), (869, 616), (889, 600), (908, 586), (925, 572), (940, 561), (955, 553), (967, 548), (978, 545), (988, 549), (994, 555), (999, 563), (1001, 573), (1002, 582), (1003, 590), (1004, 598), (1004, 604), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 442), (215, 445), (220, 449), (227, 454), (235, 460), (245, 468), (256, 476), (270, 486), (285, 496), (302, 508), (321, 520), (342, 535), (365, 550), (390, 565), (418, 582), (447, 599), (479, 617), (513, 634), (548, 651), (585, 667), (625, 682), (666, 695), (707, 704), (746, 707), (783, 704), (815, 695), (844, 682), (870, 667), (892, 650), (911, 633), (928, 615), (942, 599), (955, 584), (966, 570), (975, 558), (983, 550), (991, 546), (997, 548), (1000, 556), (1002, 564), (1003, 574), (1004, 583), (1004, 591), (1004, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 440), (216, 443), (222, 445), (229, 450), (239, 454), (250, 459), (264, 465), (279, 472), (296, 480), (315, 489), (337, 499), (361, 509), (387, 520), (415, 532), (446, 544), (479, 557), (514, 571), (551, 584), (590, 597), (631, 610), (673, 622), (717, 633), (759, 641), (799, 646), (835, 647), (868, 643), (897, 636), (922, 626), (943, 614), (959, 600), (973, 586), (984, 572), (992, 559), (998, 548), (1002, 538), (1006, 535), (1006, 543), (1006, 553), (1005, 563), (1005, 572), (1005, 582), (1005, 591), (1005, 598), (1005, 605), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 440), (216, 443), (222, 445), (229, 450), (239, 454), (250, 459), (264, 465), (279, 472), (296, 480), (315, 489), (337, 499), (361, 509), (387, 520), (415, 532), (446, 544), (479, 557), (514, 571), (551, 584), (590, 597), (631, 610), (673, 622), (717, 633), (759, 641), (799, 646), (835, 647), (868, 643), (897, 636), (922, 626), (943, 614), (959, 600), (973, 586), (984, 572), (992, 559), (998, 548), (1002, 538), (1006, 535), (1006, 543), (1006, 553), (1005, 563), (1005, 572), (1005, 582), (1005, 591), (1005, 598), (1005, 605), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)]}
		test_dict = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (237, 436), (250, 436), (265, 436), (282, 435), (301, 435), (324, 435), (348, 435), (377, 434), (407, 434), (441, 433), (477, 433), (516, 433), (558, 432), (603, 432), (650, 431), (700, 431), (752, 430), (806, 430), (861, 430), (918, 429), (974, 429), (1029, 429), (1079, 430), (1124, 430), (1162, 431), (1191, 432), (1210, 434), (1198, 432), (1178, 428), (1157, 422), (1136, 415), (1115, 406), (1096, 397), (1078, 386), (1062, 375), (1049, 363), (1038, 351), (1028, 338), (1021, 326), (1016, 314), (1012, 303), (1008, 293), (1007, 283), (1005, 276), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (237, 436), (250, 436), (265, 436), (282, 435), (301, 435), (324, 435), (348, 435), (377, 434), (407, 434), (441, 433), (477, 433), (516, 433), (558, 432), (603, 432), (650, 431), (700, 431), (752, 430), (806, 430), (861, 430), (918, 429), (974, 429), (1029, 429), (1079, 430), (1124, 430), (1162, 431), (1191, 432), (1210, 434), (1198, 432), (1178, 428), (1157, 422), (1136, 415), (1115, 406), (1096, 397), (1078, 386), (1062, 375), (1049, 363), (1038, 351), (1028, 338), (1021, 326), (1016, 314), (1012, 303), (1008, 293), (1007, 283), (1005, 276), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (215, 436), (221, 436), (228, 436), (236, 435), (245, 435), (257, 435), (270, 434), (285, 434), (302, 434), (320, 433), (341, 433), (363, 433), (387, 432), (413, 432), (442, 432), (472, 432), (504, 432), (537, 432), (573, 433), (610, 433), (648, 434), (687, 435), (723, 436), (757, 436), (789, 435), (819, 433), (847, 430), (873, 425), (896, 418), (917, 410), (936, 401), (951, 390), (964, 378), (975, 365), (983, 352), (990, 339), (994, 326), (998, 314), (1001, 302), (1002, 292), (1003, 283), (1004, 275), (1004, 269), (1004, 263), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (216, 436), (222, 436), (230, 436), (239, 436), (250, 435), (263, 435), (278, 435), (295, 435), (314, 434), (335, 434), (358, 434), (383, 433), (411, 433), (440, 433), (472, 432), (505, 432), (540, 432), (577, 432), (615, 432), (655, 432), (695, 433), (734, 434), (771, 435), (804, 436), (835, 436), (864, 436), (888, 434), (911, 432), (931, 426), (948, 419), (962, 410), (974, 399), (982, 386), (989, 372), (993, 358), (997, 344), (999, 331), (1001, 318), (1003, 306), (1003, 296), (1004, 286), (1004, 278), (1004, 271), (1004, 265), (1004, 261), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (212, 436), (218, 436), (225, 436), (235, 436), (246, 436), (259, 436), (274, 436), (292, 435), (312, 435), (335, 435), (360, 435), (388, 434), (418, 434), (451, 434), (486, 433), (524, 433), (564, 433), (607, 432), (651, 432), (697, 432), (744, 432), (794, 432), (842, 432), (888, 432), (929, 433), (966, 434), (999, 434), (1025, 436), (1047, 436), (1062, 435), (1064, 428), (1057, 418), (1048, 407), (1040, 394), (1032, 381), (1026, 367), (1020, 353), (1016, 340), (1013, 327), (1010, 314), (1007, 303), (1006, 292), (1006, 283), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (238, 436), (250, 436), (265, 436), (282, 436), (301, 436), (324, 435), (349, 435), (377, 435), (407, 435), (441, 434), (478, 434), (517, 434), (559, 433), (604, 433), (651, 433), (701, 433), (752, 432), (806, 432), (862, 432), (919, 432), (976, 432), (1031, 432), (1081, 432), (1126, 433), (1163, 434), (1193, 435), (1213, 436), (1199, 434), (1178, 430), (1157, 423), (1135, 416), (1115, 407), (1095, 397), (1078, 387), (1063, 375), (1049, 363), (1038, 350), (1029, 338), (1021, 326), (1015, 314), (1011, 302), (1008, 292), (1007, 283), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (238, 436), (250, 436), (265, 435), (282, 435), (301, 435), (324, 434), (349, 434), (377, 433), (407, 433), (441, 432), (478, 432), (516, 431), (559, 431), (604, 430), (651, 429), (700, 429), (752, 428), (806, 428), (862, 427), (919, 427), (976, 427), (1031, 426), (1081, 427), (1125, 427), (1163, 428), (1193, 429), (1213, 431), (1199, 429), (1179, 426), (1157, 420), (1135, 413), (1115, 405), (1096, 396), (1078, 385), (1062, 374), (1049, 362), (1037, 350), (1029, 337), (1021, 325), (1015, 314), (1011, 302), (1008, 292), (1007, 283), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(203, 437), (204, 437), (204, 437), (203, 438), (203, 440), (203, 443), (203, 448), (204, 453), (204, 460), (204, 469), (205, 479), (206, 490), (207, 503), (210, 518), (212, 534), (216, 552), (222, 571), (229, 591), (238, 612), (249, 633), (263, 653), (282, 672), (305, 688), (332, 701), (364, 708), (400, 709), (440, 704), (483, 694), (527, 681), (570, 666), (611, 650), (651, 635), (688, 619), (725, 605), (759, 592), (791, 580), (821, 569), (849, 560), (875, 554), (898, 549), (920, 546), (939, 546), (956, 548), (970, 552), (981, 558), (989, 566), (995, 574), (999, 583), (1002, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(203, 437), (204, 437), (204, 437), (204, 439), (204, 441), (204, 444), (204, 448), (205, 454), (206, 461), (207, 469), (208, 480), (210, 491), (212, 505), (216, 520), (220, 536), (226, 554), (232, 573), (240, 594), (251, 614), (262, 636), (278, 657), (296, 677), (319, 695), (346, 709), (377, 718), (414, 722), (452, 718), (494, 709), (537, 696), (579, 680), (620, 664), (658, 647), (695, 631), (731, 616), (764, 601), (796, 588), (825, 576), (852, 567), (877, 559), (900, 553), (921, 549), (940, 548), (956, 549), (970, 552), (981, 559), (990, 566), (995, 574), (999, 583), (1002, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1005, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'a'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 449), (202, 455), (202, 462), (202, 471), (202, 482), (202, 494), (202, 509), (203, 525), (204, 542), (206, 561), (209, 582), (214, 604), (219, 628), (228, 652), (238, 677), (252, 701), (270, 722), (293, 741), (323, 754), (357, 760), (395, 758), (437, 750), (482, 736), (526, 719), (569, 702), (610, 683), (650, 664), (688, 646), (724, 629), (758, 614), (791, 600), (820, 587), (848, 576), (874, 567), (897, 560), (918, 556), (938, 554), (954, 554), (968, 557), (980, 562), (989, 568), (995, 576), (999, 584), (1001, 592), (1003, 600), (1004, 606), (1004, 611), (1004, 614), (1004, 616), (1004, 617)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (204, 439), (204, 441), (204, 444), (205, 449), (205, 455), (206, 462), (208, 471), (209, 482), (211, 494), (214, 508), (218, 524), (222, 542), (227, 561), (234, 581), (241, 603), (251, 626), (262, 650), (276, 674), (293, 697), (314, 719), (338, 738), (367, 753), (401, 762), (439, 763), (480, 756), (523, 743), (565, 727), (606, 709), (644, 690), (682, 671), (717, 652), (750, 635), (782, 618), (811, 603), (839, 589), (865, 578), (888, 569), (910, 562), (929, 557), (947, 554), (962, 555), (975, 557), (985, 563), (993, 570), (998, 578), (1001, 586), (1003, 594), (1004, 601), (1004, 607), (1004, 612), (1004, 615), (1005, 617), (1005, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 440), (216, 443), (222, 445), (229, 450), (239, 454), (250, 459), (263, 465), (279, 472), (296, 480), (315, 489), (337, 499), (361, 509), (387, 520), (415, 533), (446, 545), (479, 558), (513, 571), (550, 585), (589, 598), (630, 611), (672, 622), (716, 633), (758, 641), (798, 646), (834, 646), (867, 642), (896, 635), (920, 625), (941, 613), (958, 599), (971, 585), (982, 571), (990, 558), (997, 547), (1001, 538), (1005, 535), (1006, 542), (1005, 552), (1005, 562), (1005, 572), (1005, 582), (1005, 590), (1005, 598), (1005, 604), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (213, 439), (219, 441), (226, 443), (235, 445), (247, 449), (261, 453), (276, 457), (295, 462), (316, 467), (339, 474), (365, 481), (394, 489), (426, 497), (460, 506), (497, 515), (536, 526), (578, 536), (623, 546), (669, 557), (718, 569), (769, 580), (821, 590), (874, 600), (927, 609), (977, 617), (1023, 621), (1063, 623), (1096, 620), (1122, 613), (1139, 599), (1142, 582), (1135, 566), (1123, 552), (1109, 541), (1093, 533), (1077, 529), (1062, 528), (1048, 530), (1036, 535), (1027, 543), (1020, 552), (1014, 562), (1010, 572), (1008, 581), (1006, 590), (1005, 598), (1005, 604), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (213, 439), (219, 441), (226, 443), (235, 445), (247, 449), (261, 453), (276, 457), (295, 462), (316, 467), (339, 474), (365, 481), (394, 489), (426, 497), (460, 506), (497, 515), (536, 526), (578, 536), (623, 546), (669, 557), (718, 569), (769, 580), (821, 590), (874, 600), (927, 609), (977, 617), (1023, 621), (1063, 623), (1096, 620), (1122, 613), (1139, 599), (1142, 582), (1135, 566), (1123, 552), (1109, 541), (1093, 533), (1077, 529), (1062, 528), (1048, 530), (1036, 535), (1027, 543), (1020, 552), (1014, 562), (1010, 572), (1008, 581), (1006, 590), (1005, 598), (1005, 604), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)]}

		sample_sets = list(test_dict.values())
		# print(len(sample_sets))
		sample_sets = [sample_sets[3], sample_sets[6], sample_sets[8]]
		sample_sets = [sample_sets[2]]

		mirror_sets = [] #get_mirrored(r, sample_sets)
		# print(len(mirror_sets))
		for p in mirror_sets:
			sample_sets.append(p)

		# print(len(sample_sets))
		# exit()
		# print(test_dict.keys())
		# sample_sets = [sample_sets[0]]


	if sampling_type == SAMPLE_TYPE_SHORTEST or sampling_type == SAMPLE_TYPE_FUSION:
		goals = r.get_goals_all()
		# sample_sets = []

		for g in goals:
			min_path = get_min_viable_path(r, goal, exp_settings)
			sample_sets.append(min_path)

	if sampling_type == SAMPLE_TYPE_CENTRAL or sampling_type == SAMPLE_TYPE_FUSION:
		resolution = exp_settings['resolution']
		low_x, hi_x, low_y, hi_y = get_hi_low_of_pts(r)
		# print(get_hi_low_of_pts(r))

		SAMPLE_BUFFER = 150

		hi_y 	+= SAMPLE_BUFFER
		low_y 	-= SAMPLE_BUFFER
		hi_x 	+= SAMPLE_BUFFER

		xi_range = range(low_x, hi_x, resolution)
		yi_range = range(low_y, hi_y, resolution)

		for xi in range(low_x, hi_x, resolution):
			for yi in range(low_y, hi_y, resolution):
				# print(xi, yi)
				x = int(xi)
				y = int(yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

		# print(point_set)

	if sampling_type == SAMPLE_TYPE_CENTRAL_SPARSE:
		resolution_sparse = exp_settings['resolution'] * 3

		low_x, hi_x, low_y, hi_y = get_hi_low_of_pts(r)
		
		SAMPLE_BUFFER = 150

		hi_y 	+= SAMPLE_BUFFER
		low_y 	-= SAMPLE_BUFFER
		hi_x 	+= SAMPLE_BUFFER

		xi_range = range(low_x, hi_x, resolution_sparse)
		yi_range = range(low_y, hi_y, resolution_sparse)

		# print(list(xi_range))
		# print(list(yi_range))


		count = 0
		for xi in xi_range:
			for yi in yi_range:
				# print(xi, yi)
				x = int(xi)
				y = int(yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

		mirror_sets = get_mirrored(r, sample_sets)
		for p in mirror_sets:
			sample_sets.append(p)

		# print(sample_sets)
		# print(start)
		# exit()

		# k = len(sample_sets)
		# k = int(.3*k)
		# k = 44
		# sample_sets = sample_sets[k:k+5]

		# print(sample_sets)

	if sampling_type == SAMPLE_TYPE_HARDCODED:
		sx, sy, stheta = start
		gx, gy, gt = goal

		mx = int((sx + gx)/2.0)
		my = int((sy + gy)/2.0)

		resolution = 200
		point_set = []
		sample_sets.append(point_set)

		# point_set = [(sx + resolution,		sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx + 2*resolution,	sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx + 3*resolution,	sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx + 4*resolution,	sy)]
		# sample_sets.append(point_set)


		point_set = [(mx + resolution,		my)]
		sample_sets.append(point_set)
		point_set = [(mx + 2*resolution,	my)]
		sample_sets.append(point_set)
		point_set = [(mx + 3*resolution,	my)]
		sample_sets.append(point_set)
		point_set = [(mx + 4*resolution,	my)]
		sample_sets.append(point_set)


		# point_set = [(sx - resolution,sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx,sy + resolution)]
		# sample_sets.append(point_set)
		# point_set = [(sx,sy - resolution)]
		# sample_sets.append(point_set)
		# point_set = [(sx - resolution,sy + resolution)]
		# sample_sets.append(point_set)
		# point_set = [(sx - resolution,sy - resolution)]
		# sample_sets.append(point_set)

		# point_set = [(gx - resolution,gy)]
		# sample_sets.append(point_set)
		# # point_set = [(gx - resolution,gy + resolution)]
		# # sample_sets.append(point_set)
		# # point_set = [(gx - resolution,gy - resolution)]
		# # sample_sets.append(point_set)
		# point_set = [(gx - 2*resolution,gy)]
		# sample_sets.append(point_set)

		# # point_set = [(gx - 2*resolution,gy + resolution)]
		# # sample_sets.append(point_set)
		# # point_set = [(gx - 2*resolution,gy - resolution)]
		# # sample_sets.append(point_set)

		# point_set = [(gx - 3*resolution,gy)]
		# sample_sets.append(point_set)

		# point_set = [(gx - 4*resolution,gy)]
		# sample_sets.append(point_set)


	return sample_sets

def lam_to_str(lam):
	return str(lam)#.replace('.', ',')

def eps_to_str(eps):
	return str(eps)#.replace('.', ',')

def fn_export_from_exp_settings(exp_settings):
	fn = ""
	title 				= exp_settings['title']
	sampling_type 		= exp_settings['sampling_type']
	eps 				= exp_settings['epsilon']
	lam 				= exp_settings['lambda']
	n_chunks 			= exp_settings['num_chunks']
	chunking_type 		= exp_settings['chunk_type']
	astr 				= exp_settings['angle_strength']
	FLAG_is_denominator = exp_settings['is_denominator']
	rez 				= exp_settings['resolution']
	f_label 			= exp_settings['f_vis_label']
	fov 				= exp_settings['fov']
	prob_og 			= exp_settings['prob_og']
	right_bound 		= exp_settings['right-bound']
	# exp_settings['f_vis']			= f_exp_single_normalized
	# exp_settings['angle_cutoff']	= 70

	is_denom = 0
	if FLAG_is_denominator:
		is_denom = 1

	is_denom 	= str(is_denom)
	eps 		= eps_to_str(eps)
	lam 		= lam_to_str(lam)
	prob_og 	= str(int(prob_og))

	unique_title = str(title) + "_fnew=" + str(is_denom) + "_"
	unique_title += str(sampling_type) + "-rez" + str(rez) + "-la" + lam + "_" + str(chunking_type) + "-" + str(n_chunks) 
	unique_title += "-as-" + str(astr) + 'fov=' + str(fov)
	unique_title += "-rb" + str(right_bound)
	unique_title += 'pog=' + str(prob_og)

	fn = FILENAME_PATH_ASSESS + unique_title + "/"

	if not os.path.exists(fn):
		os.mkdir(fn)

	fn += unique_title
	return fn


def fn_pathpickle_from_exp_settings(exp_settings, goal_index):
	sampling_type = exp_settings['sampling_type']
	n_chunks = exp_settings['num_chunks']
	angle_str = exp_settings['angle_strength']
	res = exp_settings['resolution']

	fn_pickle = FILENAME_PATH_ASSESS + "export-" + sampling_type + "-g" + str(goal_index)
	fn_pickle += "ch" + str(n_chunks) +"as" + str(angle_str) + "res" + str(res) +  ".pickle"
	print("{" + fn_pickle + "}")
	return fn_pickle


def fn_pathpickle_envir_cache(exp_settings):
	sampling_type = exp_settings['sampling_type']
	n_chunks = exp_settings['num_chunks']
	angle_str = exp_settings['angle_strength']
	res = exp_settings['resolution']
	f_vis_label 	= exp_settings['f_vis_label']
	FLAG_is_denominator = exp_settings['is_denominator']
	is_denom = 0
	if FLAG_is_denominator:
		is_denom = 1
	is_denom = str(is_denom)

	fn_pickle = FILENAME_PATH_ASSESS + "export-envir-cache-" + str(sampling_type) + "-" + str(f_vis_label)
	fn_pickle += "ch" + str(n_chunks) +"as" + str(angle_str) + "res" + str(res) +  ".pickle"

	# print("{" + fn_pickle + "}")
	return fn_pickle

def numpy_to_image(data):
	pretty_data = copy.copy(data.T)
	pretty_data = cv2.flip(pretty_data, 0)
	return pretty_data


def export_envir_cache_pic(r, data, label, g_index, obs_label, exp_settings):
	pretty_data = numpy_to_image(data)	

	# We use this to flip vertically
	r_height = 1000

	fig, ax = plt.subplots()
	ax.imshow(pretty_data, interpolation='nearest')
	obs_sets = r.get_obs_sets()
	xs, ys = [], []


	okey = ""
	for ok in obs_sets.keys():
		obs_xy = obs_sets[ok]
		if obs_xy is not None and len(obs_xy) > 0:
			obs_hexes = r.get_obs_sets_hex()
			obs_xy = obs_xy[0].get_center()
			color = obs_hexes[ok]
			x = obs_xy[0]
			y = r_height - obs_xy[1]
			xs.append(x)
			ys.append(y)
			ax.plot([x], [y], 'o', markersize=15, color='white')
			ax.plot([x], [y], 'o', markersize=12, color=color)
	
	
	xs, ys = [], []

	for goal in r.get_goals_all():
		gx, gy = goal[0], goal[1]
		xs.append(gx)
		ys.append(r_height - gy)

	ax.plot(xs, xs, 'o', markersize=12, color='black')
	ax.plot(xs, ys, 'o', markersize=10, color='white')
	xs, ys = [], []

	for table in r.get_tables():
		# table = table.get_center()
		# tx, ty = table[0], table[1]
		# xs.append(tx)
		# ys.append(r_height - ty)
		tx, ty = [], []

		table_shape = list(table.get_pretty_shape().exterior.coords)[1:]
		tx = [int(x[0]) for x in table_shape]
		ty = [1000 - int(x[1]) for x in table_shape]

		plt.fill(tx, ty, color='#FF8200', edgecolor='black', linewidth=1)

	# ax.plot(xs, ys, 'yo')
	xs, ys = [], []

	start = r.get_start()
	sx = start[0]
	sy = r_height - start[1]
	xs.append(sx)
	ys.append(sy)

	ax.plot(xs, ys, 's', color='white', markersize=16)
	ax.plot(xs, ys, 's', color='gray', markersize=15)
	xs, ys = [], []

	print("*** {" + obs_label + "}")
	title = ""
	if obs_label in ['a', 'b', 'c', 'd', 'e']:
		title = "$V_{" + obs_label + "}$"
	elif obs_label == 'omni':
		title = "$V_{o}$"

	plt.text(150, 380, title, fontsize=46, color='#FFFFFF')
	plt.axis('off')

	plt.tight_layout()
	plt.xlim([100, 1150])
	plt.ylim([250, 900])
	plt.gca().invert_yaxis()
	plt.savefig(fn_export_from_exp_settings(exp_settings) + "g="+ str(g_index) +  '-' + obs_label + '-plot'  + '.png')
	plt.clf()

def get_envir_cache(r, exp_settings):
	f_vis 		= exp_settings['f_vis']
	f_vis_label	= exp_settings['f_vis_label']
	
	fn_pickle = fn_pathpickle_envir_cache(exp_settings)

	if os.path.isfile(fn_pickle):
		with open(fn_pickle, "rb") as f:
			try:
				envir_cache = pickle.load(f)		
				print("\tImported pickle of envir cache @ " + f_pickle)

			except Exception: # so many things could go wrong, can't be more specific.
				pass

	if FLAG_REDO_ENVIR_CACHE or not os.path.isfile(fn_pickle):
		envir_cache = {}
		tic = time.perf_counter()
		print("Getting start to here dict")
		envir_cache[ENV_START_TO_HERE] = get_dict_cost_start_to_here(r, exp_settings)
		toc = time.perf_counter()
		print(f"\tCalculated start to here in {toc - tic:0.4f} seconds")
		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()


		print("Getting here to goals dict")
		tic = time.perf_counter()
		envir_cache[ENV_HERE_TO_GOALS] = get_dict_cost_here_to_goals_all(r, exp_settings)
		toc = time.perf_counter()
		print(f"\tCalculated here to goals in {toc - tic:0.4f} seconds")
		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()

		print("Getting visibility per obs dict")
		tic = time.perf_counter()
		envir_cache[ENV_VISIBILITY_PER_OBS] = get_dict_vis_per_obs_set(r, exp_settings, f_vis)
		toc = time.perf_counter()
		print(f"\tCalculated vis per obs in {toc - tic:0.4f} seconds")
		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()

		print("Done with pickle")



		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()

	return envir_cache

def get_dict_cost_here_to_goals_all(r, exp_settings):
	goals = r.get_goals_all()
	all_goals = {}

	for g_index in range(len(goals)):
		g = goals[g_index]
		all_goals[g] = get_dict_cost_here_to_goal(r, g, exp_settings)
		export_envir_cache_pic(r, all_goals[g], 'here-to-goal', g_index, "", exp_settings)

	return all_goals

# Get pre-calculated cost from here to goal
# returns (x,y) -> cost_here_to_goal
# for the entire environment
def get_dict_cost_here_to_goal(r, goal, exp_settings):
	dict_start_to_goal = np.zeros((r.get_width(), r.get_length()))
	pt_goal = resto.to_xy(goal)

	for i in range(r.get_width()): #r.get_sampling_width():
		# print(str(i) + "... ", end='')
		# if i % 15 ==0:
		# 	print()
		for j in range(r.get_length()): #r.get_sampling_length():
			pt = (i, j)
			val = get_min_direct_path_cost_between(r, pt, pt_goal, exp_settings)
			dict_start_to_goal[i, j] = val

	return dict_start_to_goal

# Get pre-calculated dict of cost from here to goal
# returns (x,y) -> cost_start_to_here
# for the entire environment
def get_dict_cost_start_to_here(r, exp_settings):
	dict_start_to_goal = np.zeros((r.get_width(), r.get_length()))
	start = resto.to_xy(r.get_start())
	# print(dict_start_to_goal.shape)

	for i in range(r.get_width()):
		# print(str(i) + "... ", end='')
		# if i % 15 ==0:
		# 	print()
		for j in range(r.get_length()): #r.get_sampling_length():
			pt = (i, j)
			# print(pt)
			val = get_min_direct_path_cost_between(r, start, resto.to_xy(pt), exp_settings)
			dict_start_to_goal[i, j] = val
	
	export_envir_cache_pic(r, dict_start_to_goal, 'start-to-here', "-", "", exp_settings)
	return dict_start_to_goal

def get_dict_vis_per_obs_set(r, exp_settings, f_vis):
	# f_vis = f_exp_single(t, pt, aud, path)
	
	if f_vis == "" or True:
		f_vis = f_exp_single_normalized

	obs_sets = r.get_obs_sets()
	all_vis_dict = {}
	for ok in obs_sets.keys():
		print("Getting obs vis for " + ok)
		os = obs_sets[ok]
		os_vis = np.zeros((r.get_width(), r.get_length()))
		# print(os_vis.shape)

		for i in range(r.get_width()): #r.get_sampling_length():
			# print(str(i) + "... ", end='')
			# if i % 15 ==0:
			# 	print()
			for j in range(r.get_length()): #r.get_sampling_width():
				# print(str(j) + "... ", end='')
				pt = (i, j)
				val = f_vis(None, pt, os, None)
				# if len(os) > 0:
				# if len(os) > 0 and (i % 50 == 0) and (j % 50 == 0):
				# 	print(str(pt) + " -> " + str(os[0].get_center()) + " = " + str(val))
				os_vis[i, j] = val

		all_vis_dict[ok] = os_vis
		print("\texporting " + ok)
		export_envir_cache_pic(r, os_vis, 'obs_angle', "", ok, exp_settings)

	return all_vis_dict

def title_from_exp_settings(exp_settings):
	title = exp_settings['title']
	sampling_type = exp_settings['sampling_type']
	eps = exp_settings['epsilon']
	lam = exp_settings['lambda']
	n_chunks 		= exp_settings['num_chunks']
	chunking_type 	= exp_settings['chunk_type']
	angle_strength = exp_settings['angle_strength']

	FLAG_is_denominator = exp_settings['is_denominator']
	rez 				= exp_settings['resolution']
	f_label 			= exp_settings['f_vis_label']
	fov 				= exp_settings['fov']
	prob_og 			= exp_settings['prob_og']
	right_bound 		= exp_settings['right-bound']

	eps = eps_to_str(eps)
	lam = lam_to_str(lam)

	cool_title = str(title) + ": " + str(sampling_type) + " ang_str=" + str(angle_strength)
	cool_title += "\nright_bound=" + str(right_bound) + " fov=" + str(fov)
	cool_title += "\nlam=" + lam + "     prob_og=" + str(prob_og)
	cool_title += "\nn=" + str(n_chunks) + " distr=" + str(chunking_type)

	return cool_title

# Convert sample points into actual useful paths
def get_paths_from_sample_set(r, exp_settings, goal_index):
	sampling_type = exp_settings['sampling_type']

	sample_pts = get_sample_points_sets(r, r.get_start(), r.get_goals_all()[goal_index], exp_settings)
	print("\tSampled " + str(len(sample_pts)) + " points using the sampling type {" + sampling_type + "}")

	target = r.get_goals_all()[goal_index]
	all_paths = []
	fn_pickle = fn_pathpickle_from_exp_settings(exp_settings, goal_index)

	print("\t Looking for import @ " + fn_pickle)

	if not FLAG_REDO_PATH_CREATION and os.path.isfile(fn_pickle):
		print("\tImporting preassembled paths")
		with open(fn_pickle, "rb") as f:
			try:
				path_dict = pickle.load(f)		
				print("\tImported pickle of combo (goal=" + str(goal_index) + ", sampling=" + str(sampling_type) + ")")
				print("imported " + str(len(all_paths['path'])) + " paths")
				return path_dict

			except Exception: # so many things could go wrong, can't be more specific.
				pass
	else:
		if sampling_type not in premade_path_sampling_types:
			print("\tDidn't import; assembling set of paths")
			all_paths = []
			# If I don't yet have a path
			for point_set in sample_pts:
				
				# path_option = construct_single_path_with_angles(exp_settings, r.get_start(), target, point_set, fn_export_from_exp_settings(exp_settings))
				# path_option = chunkify.chunkify_path(exp_settings, path_option)
				# all_paths.append(path_option)

				# path_option_2 = construct_single_path_with_angles_spline(exp_settings, r.get_start(), target, point_set, fn_export_from_exp_settings(exp_settings), is_weak=True)
				# path_option_2 = chunkify.chunkify_path(r, exp_settings, path_option_2)
				# all_paths.append(path_option_2)

				try:
					path_option_2 = construct_single_path_with_angles_spline(exp_settings, r.get_start(), target, point_set, fn_export_from_exp_settings(exp_settings), is_weak=True)
					path_option_2 = chunkify.chunkify_path(r, exp_settings, path_option_2)
					all_paths.append(path_option_2)
				except Exception:
					print("RIP")
					all_paths.append([])
		else:
			all_paths = sample_pts

	path_dict = {'path': all_paths, 'sp': sample_pts}

	dbfile = open(fn_pickle, 'wb')
	pickle.dump(path_dict, dbfile)
	dbfile.close()
	print("\tSaved paths for faster future run on combo (goal=" + str(goal_index) + ", sampling=" + str(sampling_type) + ")")

	return path_dict

# TODO Ada
def create_systematic_path_options_for_goal(r, exp_settings, start, goal, img, num_paths=500):
	all_paths = []
	target = goal

	label = exp_settings['title']
	sampling_type = exp_settings['sampling_type']


	fn = FILENAME_PATH_ASSESS + label + "_best_path" + ".png"
	goal_index = r.get_goal_index(goal)

	title = title_from_exp_settings(exp_settings)

	min_paths = [get_min_viable_path(r, goal, exp_settings)]
	resto.export_raw_paths(r, img, min_paths, title, fn_export_from_exp_settings(exp_settings)+ "_g" + str(goal_index) + "-min")

	all_paths_dict = get_paths_from_sample_set(r, exp_settings, goal_index)
	resto.export_raw_paths(r, img, all_paths_dict['path'], title, fn_export_from_exp_settings(exp_settings)+ "_g" + str(goal_index) + "-all")

	trimmed_paths, removed_paths, trimm_sp = trim_paths(r, all_paths_dict, goal, exp_settings)
	resto.export_raw_paths(r, img, trimmed_paths, title, fn_export_from_exp_settings(exp_settings) + "_g" + str(goal_index) + "-trimmed")
	resto.export_raw_paths(r, img, removed_paths, title, fn_export_from_exp_settings(exp_settings) + "_g" + str(goal_index) + "-rmvd")

	return trimmed_paths, trimm_sp


def experimental_scenario_single():
	generate_type = resto.TYPE_EXP_SINGLE

	# SETUP FROM SCRATCH, AND SAVE OPTIONS
	if True:
		# Create the restaurant scene from our saved description of it
		print("PLANNER: Creating layout of type " + str(generate_type))
		r 	= resto.Restaurant(generate_type)
		# print("PLANNER: get visibility info")

		if FLAG_VIS_GRID:
			# If we'd like to make a graph of what the visibility score is at different points
			# df_vis = r.get_visibility_of_pts_pandas(f_visibility)

			# dbfile = open(vis_pickle, 'ab') 
			# pickle.dump(df_vis, dbfile)					  
			# dbfile.close()
			# print("Saved visibility map")

			# df_vis.to_csv('visibility.csv')
			# print("Visibility point grid created")
			pass
		
		# pickle the map for future use
		dbfile = open(resto_pickle, 'ab') 
		pickle.dump(r, dbfile)					  
		dbfile.close()
		print("Saved restaurant pickle")

	# OR LOAD FROM FILE
	else:
		dbfile = open(resto_pickle, 'rb')
		r = pickle.load(dbfile)
		print("Imported pickle of restaurant")


		if FLAG_VIS_GRID:
			dbfile = open(vis_pickle, 'rb')
			df_vis = pickle.load(dbfile)
			print("Imported pickle of vis")


	return r


def image_to_planner(resto, pt):
	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	# planner always starts at (0,0)
	gx, gy, gtheta = resto.get_start()


	x = float(x) - gx
	y = float(y) - gy
	nx = x #(x / UNITY_SCALE_X) + UNITY_OFFSET_X
	ny = y #(y / UNITY_SCALE_Y) + UNITY_OFFSET_Y
	
	if len(pt) == 3:
		return (int(ny), int(nx), theta)
	return (int(ny), int(nx))

def planner_to_image(resto, pt):
	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	# planner always starts at (0,0)
	gx, gy, gtheta = resto.get_start()


	x = float(x) + gx
	y = float(y) + gy
	nx = y #(x / UNITY_SCALE_X) + UNITY_OFFSET_X
	ny = x #(y / UNITY_SCALE_Y) + UNITY_OFFSET_Y
	
	if len(pt) == 3:
		return (int(ny), int(nx), np.deg2rad(theta))
	return (int(ny), int(nx))

def angle_between_points(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	angle = np.arctan2(y2 - y1, x2 - x1)

	# ang1 = np.arctan2(*p1[::-1])
	# ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg(angle)

def angle_between_lines(l1, l2):
	# l1_x1, l1_y1 = l1[0]
	# l1_x2, l1_y2 = l1[1]
	# l2_x1, l2_y1 = l2[0]
	# l2_x2, l2_y2 = l2[1]

	p1a, p1b = l1
	p2a, p2b = l2

	a1 = angle_between_points(p1a, p1b)
	a2 = angle_between_points(p2a, p2b)
	angle = (a1 - a2)

	# cosTh = np.dot(l1,l2)
	# sinTh = np.cross(l1,l2)
	# angle = np.rad2deg(np.arctan2(sinTh,cosTh))
	return angle

def angle_of_turn(l1, l2):
	return (angle_between_lines(l1, l2))

def print_states(resto, states, label):
	img = resto.get_img()

	cv2.circle(img, planner_to_image(resto, (0,0)), 5, (138,43,226), 5)
	for s in states:
		x, y, t = planner_to_image(resto, s)
		print((x,y, t))

		COLOR_RED = (138,43,226)
		cv2.circle(img, (x, y), 5, COLOR_RED, 5)

		angle = s[2];
		length = 20;
		
		ax =  int(x + length * np.cos(angle * np.pi / 180.0))
		ay =  int(y + length * np.sin(angle * np.pi / 180.0))
		cv2.arrowedLine(img, (x,y), (ax, ay), COLOR_RED, 2)
		# print((ax, ay))

	cv2.imwrite(FILENAME_PATH_ASSESS + 'samples-' + label + '.png', img)

def print_path(resto, xc, yc, yawc, label):
	img = resto.get_img()

	cv2.circle(img, planner_to_image(resto, (0,0)), 5, (138,43,226), 5)
	for i in range(len(xc)):
		x = int(xc[i])
		y = int(yc[i])
		yaw = yawc[i]

		COLOR_RED = (138,43,226)
		cv2.circle(img, (x, y), 5, COLOR_RED, 5)

		angle = yaw;
		length = 20;
		
		ax =  int(x + length * np.cos(angle * np.pi / 180.0))
		ay =  int(y + length * np.sin(angle * np.pi / 180.0))
		cv2.arrowedLine(img, (x,y), (ax, ay), COLOR_RED, 2)

	cv2.imwrite(FILENAME_PATH_ASSESS + 'path-' + label + '.png', img)

def make_path_libs(resto, goal):
	start = resto.get_start()
	sx, sy, stheta = image_to_planner(resto, start)
	gx, gy, gtheta = image_to_planner(resto, goal)
	print("FINDING ROUTE TO GOAL " + str(goal))
	show_animation = False

	min_distance = np.sqrt((sx-gx)**2 + (sy-gy)**2)
	target_states = [image_to_planner(resto, goal)]
	# :param goal_angle: goal orientation for biased sampling
	# :param ns: number of biased sampling
	# :param nxy: number of position sampling
	# :param nxy: number of position sampling
	# :param nh: number of heading sampleing
	# :param d: distance of terminal state
	# :param a_min: position sampling min angle
	# :param a_max: position sampling max angle
	# :param p_min: heading sampling min angle
	# :param p_max: heading sampling max angle
	# :return: states list

	k0 = 0.0
	nxy = 7
	nh = 9
	# verify d is reasonable
	d = min_distance
	print("D=" + str(d))
	a_min = - np.deg2rad(15.0)
	a_max = np.deg2rad(15.0)
	p_min = - np.deg2rad(15.0)
	p_max = np.deg2rad(15.0)
	print("calculating states")
	states = slp.calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
	print(states)

	print_states(resto, states, 'calc-unif')

	print("calculating results")
	result = slp.generate_path(states, k0)
	print(result)

	for table in result:
		xc, yc, yawc = slp.motion_model.generate_trajectory(
			table[3], table[4], table[5], k0)

		print("gen trajectory")
		print((xc, yc, yawc))

		if show_animation:
			plt.plot(xc, yc, "-r")
			print(xc, yc)

	if show_animation:
		plt.grid(True)
		plt.axis("equal")
		plt.show()

	print("Done")


def export_path_options_for_each_goal(restaurant, best_paths, exp_settings):
	# print(best_paths)
	img = restaurant.get_img()
	 #cv2.flip(img, 0)
	# cv2.imwrite(FILENAME_PATH_ASSESS + unique_key + 'empty.png', empty_img)

	fn = FILENAME_PATH_ASSESS
	title = "" #title_from_exp_settings(exp_settings)

	# flip required for orientation
	font_size = 1
	y0, dy = 50, 50
	for i, line in enumerate(title.split('\n')):
	    y = y0 + i*dy
	    cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (209, 80, 0, 255), 3)
	
	empty_img = img
	all_img = img

	color_dict = restaurant.get_obs_sets_colors()

	goal_imgs = {}
	for pkey in best_paths.keys():
		goal 		= pkey[0]
		# audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		goal_imgs[goal_index] = copy.copy(empty_img)

	for pkey in best_paths.keys():
		path = best_paths[pkey]
		path = restaurant.path_to_printable_path(path)
		path_img = img.copy()
		
		goal 		= pkey[0]
		audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		goal_img = goal_imgs[goal_index]
		obs_key = pkey[1]

		if obs_key == 'shortest':
			break

		solo_img = restaurant.get_obs_img(obs_key)
		
		color = color_dict[audience]

		# Draw the path  
		for i in range(len(path) - 1):
			a = path[i]
			b = path[i + 1]
			
			cv2.line(solo_img, a, b, color, thickness=3, lineType=8)
			cv2.circle(solo_img, a, 4, color, 4)

			if audience is not 'naked':
				cv2.line(goal_img, a, b, color, thickness=3, lineType=8)
				cv2.line(all_img, a, b, color, thickness=3, lineType=8)
				cv2.circle(goal_img, a, 4, color, 4)
				cv2.circle(all_img, a, 4, color, 4)
		
		title = exp_settings['title']

		sampling_type = exp_settings['sampling_type']
		cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_solo_path-g=' + str(goal_index)+ "-aud=" + str(audience) + '.png', solo_img) 
		print("exported image of " + str(pkey) + " for goal " + str(goal_index))


	for goal_index in goal_imgs.keys():
		goal_img = goal_imgs[goal_index]
		cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_goal_' + str(goal_index) + '.png', goal_img) 

	reply = cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_overview_yay'+ '.png', all_img)
	# TODO: actually export pics for them


# video_lengths = {}
# video_lengths[('G_ME', 'LO')]       = 10.13668 56
# video_lengths[('G_ME', 'LA')]       = 9.636692 53
# video_lengths[('G_ME', 'LB')]       = 9.803356 54 = 0.18154363 OR 
# video_lengths[('G_ME', 'LC')]       = 9.803356 54
# video_lengths[('G_ME', 'LD')]       = 9.636692 53
# video_lengths[('G_ME', 'LE')]       = 9.303364 51

# video_lengths[('G_AWAY', 'LO')]     = 10.13668 = 56
# video_lengths[('G_AWAY', 'LA')]     = 9.803356 = 54
# video_lengths[('G_AWAY', 'LB')]     = 9.803356 = 54
# video_lengths[('G_AWAY', 'LC')]     = 9.636692 = 53 = 0.181824377 OR 0.185321
# video_lengths[('G_AWAY', 'LD')]     = 10.13668 = 56
# video_lengths[('G_AWAY', 'LE')]     = 9.803356 = 54 = 0.181012143 OR 0.184968981
# timestep = 0.333324

def stamps_to_steps(stamp_list):
	stepsize = 185 # .185 * 1000ms/s
	stamp_timesteps = {}

	for i in range(60):
		stamp_timesteps[i] = 0
	
	for stamp in stamp_list:
		timestep = int(float(stamp) / stepsize) + 1
		stamp_timesteps[timestep] += 1

	all_steps = list(stamp_timesteps.values())
	return stamp_timesteps, sum(all_steps)/len(stamp_timesteps.values())

def export_path_moments_confusion_for_each_goal(restaurant, best_paths, stamps, exp_settings):
	img = restaurant.get_img()
	 #cv2.flip(img, 0)
	# cv2.imwrite(FILENAME_PATH_ASSESS + unique_key + 'empty.png', empty_img)

	goals_dict 	= {(1005, 257, 180):'G_ME', (1005, 617, 0):'G_AWAY'}
	viewers	= ['VA', 'VB', 'VC', 'VD', 'VE', None]
	paths 	= ['LA', 'LB', 'LC', 'LD', 'LE']
	path_dict 	= {'a':'LA', 'b':'LB', 'c':'LC', 'd':'LD', 'e':'LE', 'omni':'LO'}
	viewer_dict	= {'a':'VA', 'b':'VB', 'c':'VC', 'd':'VD', 'e':'VE', None:None}

	path_set = [((1005, 257, 180), 'omni'), ((1005, 257, 180), 'a'), ((1005, 257, 180), 'b'), ((1005, 257, 180), 'c'), ((1005, 257, 180), 'd'), ((1005, 257, 180), 'e'), ((1005, 617, 0), 'omni'), ((1005, 617, 0), 'a'), ((1005, 617, 0), 'b'), ((1005, 617, 0), 'c'), ((1005, 617, 0), 'd'), ((1005, 617, 0), 'e')]
	fn = FILENAME_PATH_ASSESS
	title = "" #title_from_exp_settings(exp_settings)
	
	empty_img = img
	all_img = img

	color_dict = restaurant.get_obs_sets_colors()

	# goal_imgs = {}
	# for pkey in path_set:
	# 	goal 		= pkey[0]
	# 	# audience 	= pkey[1]
	# 	goal_index 	= restaurant.get_goal_index(goal)

	# 	goal_imgs[goal_index] = copy.copy(empty_img)

	for pkey in path_set:
		path = best_paths[pkey]
		path = restaurant.path_to_printable_path(path)
		path_img = img.copy()
		
		goal 		= pkey[0]
		audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		# goal_img = goal_imgs[goal_index]
		obs_key = pkey[1]

		t_goal = goals_dict[goal]
		t_path = path_dict[audience]
		# then iterate through all the possible viewers to mark them
		t_obs = None
		
		if obs_key == 'shortest':
			break

		solo_img = restaurant.get_obs_img(obs_key)
		all_obs_img = copy.copy(solo_img)

		for obs in ['a', 'b', 'c', 'd', 'e', None]:
			solo_img = restaurant.get_obs_img(obs_key)
			t_obs = obs
		
			stamp_key = (t_goal, t_path, viewer_dict[t_obs])
			stamp_list = stamps[stamp_key]
			stamp_timesteps, stamp_mean = stamps_to_steps(stamp_list)

			color = color_dict[audience]
			if t_obs != None:
				obs_color = color_dict[t_obs]
			else:
				obs_color = (255, 255, 255)
				t_obs = "all"

			# Draw the path  
			for i in range(len(path) - 1):
				a = path[i]
				b = path[i + 1]
				
				cv2.line(solo_img, a, b, color, thickness=3, lineType=8)
				
				k = 3.0 # scale factor for viewing
				num_issues = stamp_timesteps[i]
				size_dot = int(k * (num_issues / stamp_mean))
				if size_dot < 0:
					size_dot = 0

				if num_issues > stamp_mean:
					cv2.circle(solo_img, a, size_dot + 1, (0,0,0), size_dot + 1)
					cv2.circle(solo_img, a, size_dot, obs_color, size_dot)

					cv2.circle(all_obs_img, a, size_dot + 1, (0,0,0), size_dot + 1)
					cv2.circle(all_obs_img, a, size_dot, obs_color, size_dot)

				if audience is not 'naked':
					# cv2.line(goal_img, a, b, color, thickness=3, lineType=8)
					cv2.line(all_img, a, b, color, thickness=3, lineType=8)
					cv2.line(all_obs_img, a, b, (0, 0, 0), thickness=3, lineType=8)
					# cv2.circle(goal_img, a, 4, color, 4)
					# cv2.circle(all_img, a, 4, color, 4)
			
			title = exp_settings['title']

			sampling_type = exp_settings['sampling_type']
			# fn = fn_export_from_exp_settings(exp_settings)
			fn = "path_assessment/spring2022_v1/" + 'mocon_solo_path-g=' + str(t_goal)+ "-xi=" + str(t_path) + "-obs-" + t_obs + '.png'
			cv2.imwrite(fn, solo_img) 
			print(fn)
			print("exported mocon image of " + str(pkey) + " for goal " + str(goal_index) + "-xi=" + str(t_path) + "-obs-" + t_obs)


		fn = "path_assessment/spring2022_v1/" + 'mocon_all_obs_path-g=' + str(t_goal)+ "-xi=" + str(t_path) + '.png'
		cv2.imwrite(fn, all_obs_img) 
		print("exported mocon image of " + str(pkey) + " for goal " + str(goal_index))






def get_columns_metric(r, df):
	columns = df.columns.tolist()
	for col in non_metric_columns:
		if col in columns:
			columns.remove(col)
	return columns

def get_columns_env(r, df):
	columns = df.columns.tolist()
	new_cols = []
	for col in columns:
		if 'env' in col:
			new_cols.append(col)
	return new_cols

def get_columns_legibility(r, df):
	columns = df.columns.tolist()
	new_cols = ['naked']
	for col in r.get_obs_sets().keys():
		new_cols.append(col)
	return new_cols

def dict_to_leg_df(r, data, exp_settings):
	df = pd.DataFrame.from_dict(data)
	columns = get_columns_metric(r, df)

	for col in columns:
		df = df.astype({col: float})

	export_legibility_df(r, df, exp_settings)

	return df

def rgb_to_hex(red, green, blue):
	"""Return color as #rrggbb for the given color values."""
	return '#%02x%02x%02x' % (red, green, blue)

def export_legibility_df(r, df, exp_settings):
	title = exp_settings['title']
	sampling_type = exp_settings['sampling_type']

	df.to_csv(fn_export_from_exp_settings(exp_settings) + "_legibilities.csv")

	df.describe().to_csv(fn_export_from_exp_settings(exp_settings) + "_description.csv")

	print(get_columns_metric(r, df))
	print(get_columns_env(r, df))
	print(get_columns_legibility(r, df))

	# columns_env = get_columns_pure_vis(r, df)
	# make_overview_plot(r, df, exp_settings, columns_env, 'env')

	columns_env = get_columns_env(r, df)
	make_overview_plot(r, df, exp_settings, columns_env, 'env')

	columns_legi = get_columns_legibility(r, df)
	make_overview_plot(r, df, exp_settings, columns_legi, 'legi')

def make_overview_plot(r, df, exp_settings, columns, label):
	all_goals = df["goal"].unique().tolist()
	df_array = []

	g_index = 0
	if False:
		for g in all_goals:
			df_new = df[df['goal'] == g]
			df_array.append(df_new)


			df_new.plot.box(vert=False) # , by=["goal"]
			# bp = df.boxplot(by="goal") #, column=columns)
			# bp = df.groupby('goal').boxplot()

			plt.tight_layout()
			#save the plot as a png file
			plt.savefig(fn_export_from_exp_settings(exp_settings) + "g="+ str(g_index) +  '-desc_plot_' + label  + '.png')
			plt.clf()

			g_index += 1


	obs_palette = r.get_obs_sets_hex()
	goal_labels = r.get_goal_labels()

	goals_list = r.get_goals_all()

	if FLAG_MIN_MODE:
		if 'omni' in columns:
			columns = ['omni', 'c']
		if 'omni-env' in columns:
			columns = ['omni-env', 'c-env']


	df_a = df[df['goal'] == goals_list[0]]
	df_b = df[df['goal'] == goals_list[1]]

	df_a.loc[:,"goal"] = df_a.loc[:, "goal"].map(goal_labels)
	df_b.loc[:,"goal"] = df_b.loc[:, "goal"].map(goal_labels)

	df_a = df_a[columns]
	df_b = df_b[columns]

	# make the total overview plot
	contents_a = np.round(df_a.describe(), 2)
	contents_b = np.round(df_b.describe(), 2)

	contents_a.loc['count'] = contents_a.loc['count'].astype(int).astype(str)
	contents_b.loc['count'] = contents_b.loc['count'].astype(int).astype(str)

	fig_grid = plt.figure(figsize=(10, 6), constrained_layout=True)
	gs = gridspec.GridSpec(ncols=2, nrows=2,
						 width_ratios=[1, 1], wspace=None,
						 hspace=None, height_ratios=[1, 2], figure=fig_grid)

	cool_title = title_from_exp_settings(exp_settings)
	plt.suptitle(cool_title)

	# gs.update(wspace=1)
	ax1 = plt.subplot(gs[0, :1], )
	ax2 = plt.subplot(gs[0, 1:])
	ax3 = plt.subplot(gs[1, 0:2])

	ax1.axis('off')
	ax2.axis('off')

	table_a = table(ax1, contents_a, loc="center")
	table_b = table(ax2, contents_b, loc="center")

	table_a.auto_set_font_size(False)
	table_b.auto_set_font_size(False)

	table_a.set_fontsize(6)
	table_b.set_fontsize(6)

	# plt.savefig(FILENAME_PATH_ASSESS + title + "_" + sampling_type+  '-table'  + '.png')

	key_cols = columns
	key_cols.append('goal')
	mdf = df[key_cols].melt(id_vars=['goal'])
	ax3 = sns.stripplot(x="goal", y="value", hue="variable", data=mdf, palette=obs_palette, split=True, linewidth=1, edgecolor='gray')	
	if label == 'env':
		ax3.set_ylabel('Size of maximum envelope of visibility')
	else:
		ax3.set_ylabel('Legibility with regard to goal')
	ax3.set_xlabel('Goal')

	ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
	
	# df_new.plot.box(vert=False) # , by=["goal"]
	plt.tight_layout()
	# fig.tight_layout()
	#save the plot as a png file
	plt.savefig(fn_export_from_exp_settings(exp_settings) +  '-desc_plot_' + label + '.png')
	plt.clf()
	

def export_table_all_viewers(r, best_paths, exp_settings):
	# if stat_type == 'env':
	# 	f_function = f_env
	# elif stat_type == 'leg':
	# 	f_function = f_legibility
	# else:
	# 	print("Problem")
	# 	exit()

	obs_sets = r.get_obs_sets()
	obs_keys = list(obs_sets.keys())

	# (target, observer) = value
	data = []

	for key in best_paths:
		path = best_paths[key]
		gkey, target_aud_key = key

		for aud_key in obs_keys:
			actual_audience = obs_sets[aud_key]
		
			f_leg_value = f_legibility(r, gkey, r.get_goals_all(), path, actual_audience, None, exp_settings)
			f_env_value, f_env_first, f_env_max = f_env(r, gkey, r.get_goals_all(), path, actual_audience, None, exp_settings)
			f_env_percent = (f_env_value / f_env_max) * 100.0
			f_env_earliest_percent = (f_env_first / f_env_max) * 100.0

			datum = {'goal':gkey, 'target_aud':target_aud_key, 'actual_aud':aud_key, 'legibility':f_leg_value, 'env':f_env_value, 'env_pct':f_env_percent, 'env_first':f_env_earliest_percent}
			data.append(datum)

			if target_aud_key == 'omni' and aud_key == 'omni':
				print(gkey)
				print(f_leg_value)
				print(f_env_value)

	df = pd.DataFrame.from_dict(data)
	# print(data)


	fig_grid = plt.figure(figsize=(10, 6), constrained_layout=True)
	gs = gridspec.GridSpec(ncols=2, nrows=2,
						 width_ratios=[1, 1], wspace=None,
						 hspace=None, height_ratios=[1, 1], figure=fig_grid)

	cool_title = title_from_exp_settings(exp_settings)
	plt.suptitle(cool_title)

	goals_list = r.get_goals_all()

	df['legibility'] = df['legibility'].astype(float)
	df['env'] = df['env'].astype(int)

	df_a = df[df['goal'] == goals_list[0]]
	df_b = df[df['goal'] == goals_list[1]]

	df_a_rounded = np.round(df_a, 3)
	df_b_rounded = np.round(df_b, 3)

	# df_c = df[df['goal'] == goals_list[0]]
	# df_d = df[df['goal'] == goals_list[1]]

	# df_a.loc[:,"goal"] = df_a.loc[:, "goal"].map(goal_labels)
	# df_b.loc[:,"goal"] = df_b.loc[:, "goal"].map(goal_labels)

	cols_env = get_columns_legibility(r, df)
	cols_leg = get_columns_env(r, df)


	# df_a = df_a[cols_leg]
	# df_b = df_b[cols_leg]

	# df_c = df_c[cols_env]
	# df_d = df_d[cols_env]

	# print(df.pivot(index='target_aud', columns='actual_aud'))

	# gs.update(wspace=1)
	ax1 = plt.subplot(gs[0, :1], )
	ax2 = plt.subplot(gs[0, 1:])
	ax3 = plt.subplot(gs[1, :1], )
	ax4 = plt.subplot(gs[1, 1:])

	ax1.set_title("Legibility for paths to goal 0")
	ax2.set_title("Legibility for paths to goal 1")
	ax3.set_title("Theoretical Max\n Envelope of Readiness for paths to goal 0")
	ax4.set_title("Theoretical Max\n Envelope of Readiness for paths to goal 1")

	ax1.set_xlabel("Target Audience")
	ax2.set_xlabel("Target Audience")
	ax3.set_xlabel("Target Audience")
	ax4.set_xlabel("Target Audience")

	ax1.set_ylabel("Optimized for Audience")
	ax2.set_ylabel("Optimized for Audience")
	ax3.set_ylabel("Optimized for Audience")
	ax4.set_ylabel("Optimized for Audience")

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')

	# .pivot_table(values='value', index='label', columns='type')
	pt_a = df_a_rounded.pivot_table(values='legibility', index='target_aud', columns='actual_aud', aggfunc='first')
	pt_b = df_b_rounded.pivot_table(values='legibility', index='target_aud', columns='actual_aud', aggfunc='first')
	pt_c = df_a.pivot_table(values='env', index='target_aud', columns='actual_aud', aggfunc='first')
	pt_d = df_b.pivot_table(values='env', index='target_aud', columns='actual_aud', aggfunc='first')

	table_a = table(ax1, pt_a, loc="center")
	table_b = table(ax2, pt_b, loc="center")
	table_c = table(ax3, pt_c, loc="center")
	table_d = table(ax4, pt_d, loc="center")

	# table_a.auto_set_font_size(False)
	# table_b.auto_set_font_size(True)

	# table_a.set_fontsize(6)
	# table_b.set_fontsize(6)

	plt.tight_layout()
	plt.savefig(fn_export_from_exp_settings(exp_settings) +  '-desc_table_voila' + '.png')
	plt.clf()

	if FLAG_EXPORT_LATEX_MAXES:
		print("LATEX EXPORT")
		latex_c = df_a.pivot_table(values='env_pct', index='target_aud', columns='actual_aud', aggfunc='first')
		latex_d = df_b.pivot_table(values='env_pct', index='target_aud', columns='actual_aud', aggfunc='first')

		print(latex_c.to_latex(index=True, index_names=True))
		print(latex_d.to_latex(index=True, index_names=True))

		print("RAW")
		latex_c = df_a.pivot_table(values='env', index='target_aud', columns='actual_aud', aggfunc='first')
		latex_d = df_b.pivot_table(values='env', index='target_aud', columns='actual_aud', aggfunc='first')

		print(latex_c.to_latex(index=True, index_names=True))
		print(latex_d.to_latex(index=True, index_names=True))

		print("EARLIEST SEEN F_ENV")
		latex_c = df_a.pivot_table(values='env_first', index='target_aud', columns='actual_aud', aggfunc='first')
		latex_d = df_b.pivot_table(values='env_first', index='target_aud', columns='actual_aud', aggfunc='first')

		print(latex_c.to_latex(index=True, index_names=True))
		print(latex_d.to_latex(index=True, index_names=True))

		# df.to_csv(FILENAME_PATH_ASSESS + target_type + "-" + goal + "-" + analysis_label + ".csv")


	return None

# TODO: verify is indexing correctly and grabbing best overall, 
# not best in short zone
def get_best_paths_from_df(r, df, exp_settings):
	best_index = {}
	best_paths = {}
	best_lookup = {}
	best_sample_points = {}

	# symmetry check
	cached_omni_bottom = [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 425), (203, 420), (203, 412), (203, 403), (203, 393), (204, 380), (205, 365), (206, 350), (207, 332), (210, 313), (214, 292), (219, 270), (225, 246), (233, 221), (244, 197), (258, 172), (276, 150), (299, 130), (328, 115), (362, 107), (400, 107), (441, 115), (485, 129), (529, 145), (572, 163), (613, 182), (652, 201), (690, 219), (725, 237), (759, 254), (791, 268), (820, 282), (848, 293), (873, 303), (897, 310), (918, 316), (937, 318), (954, 319), (968, 317), (980, 312), (988, 305), (995, 298), (999, 290), (1002, 282), (1003, 274), (1004, 268), (1004, 263), (1004, 260), (1004, 258), (1004, 257)]
	cached_omni_top = get_mirrored_path(r, cached_omni_bottom)

	goals = df['goal'].unique()
	columns = get_columns_legibility(r, df)
	if FLAG_MIN_MODE:
		columns = ['omni', 'c']

	# print("GOALS")
	# print(goals)
	# print(columns)
	FLAG_USE_CACHED = False

	for goal in goals:
		is_goal =  df['goal']==goal
		for col in columns:
			df_goal 	= df[is_goal]
			column 		= df_goal[col]
			# print(column)
			max_index 	= pd.to_numeric(column).idxmax()

			if column is 'omni' and FLAG_USE_CACHED:
				if goal is goals[0]:
					df.index[df['path'] == cached_omni_top].tolist()[0]
				elif goal is goals[0]:
					df.index[df['path'] == cached_omni_bottom].tolist()[0]
				else:
					print("sad")
					exit()

			best_index[(goal, col)] = max_index
			best_paths[(goal, col)] = df.iloc[max_index]['path']
			best_sample_points[(goal, col)] = df.iloc[max_index]['sample_points']
			best_lookup[(goal, col)] = df.iloc[max_index]

	omni_case_goal_0 = best_lookup[(goals[0], 'omni')]
	omni_case_goal_1 = best_lookup[(goals[1], 'omni')]

	if omni_case_goal_0['omni'] > omni_case_goal_1['omni']:
		best_paths[(goals[1], 'omni')] = get_mirrored_path(r, best_paths[(goals[0], 'omni')])
	elif omni_case_goal_1['omni'] > omni_case_goal_0['omni']:
		best_paths[(goals[0], 'omni')] = get_mirrored_path(r, best_paths[(goals[1], 'omni')])


	export_table_all_viewers(r, best_paths, exp_settings)
	
	print("BEST SAMPLE POINTS")
	print(best_sample_points)
	file1 = open(fn_export_from_exp_settings(exp_settings) + "_SAMPLE_POINTS.txt","w+")
	file1.write(str(best_sample_points))
	file1.close()

	return best_paths, best_index

def analyze_all_paths(r, paths_for_analysis_dict, exp_settings):
	paths 		= None
	goals 		= r.get_goals_all()
	if FLAG_MIN_MODE:
		obs_sets_old 	= r.get_obs_sets()
		obs_sets 	= {}
		obs_sets['omni'] 	= obs_sets_old['omni']
		obs_sets['c'] 		= obs_sets_old['c']
	else:
		obs_sets 	= r.get_obs_sets()

	all_entries = []
	key_index 	= 0

	df_all = []
	data = []

	for key in paths_for_analysis_dict:
		goal 	= key
		paths 	= paths_for_analysis_dict[key]['paths']
		sp 	= paths_for_analysis_dict[key]['sp']

		for pi in range(len(paths)):
			path = paths[pi]
			f_vis = exp_settings['f_vis']
			datum = get_legibilities(r, path, goal, goals, obs_sets, f_vis, exp_settings)
			datum['path'] = path
			datum['goal'] = goal
			datum['sample_points'] = sp[pi]
			datum['path_length'] = get_path_length(path)[1]
			datum['path_cost'] = f_path_cost(path)
			data.append(datum)
			# datum = [goal_index] + []


	if len(data) == 0:
		return None

	# data_frame of all paths overall
	df = dict_to_leg_df(r, data, exp_settings)

	best_paths, best_index = get_best_paths_from_df(r, df, exp_settings)

	export_path_options_for_each_goal(r, best_paths, exp_settings)
	return best_paths

# Function for when I've found the best path using the main code
# but want to iterate a bunch of times for denser analytics, more sampling, etc.
def export_best_options():
	FLAG_EXPORT_JUST_TEASER 	= True
	FLAG_EXPORT_MOCON 			= False


	r = experimental_scenario_single()
	exp_settings = defaultdict(float)
	exp_settings['prob_og'] = False
	exp_settings['sampling_type'] = 'custom'
	rb = 40
	km = False
	exp_settings['title'] 			= 'fall2021'
	exp_settings['resolution']		= 15
	exp_settings['f_vis_label']		= 'fall2021'
	exp_settings['epsilon'] 		= 0 #1e-12 #eps #decimal.Decimal(1e-12) # eps #.000000000001
	exp_settings['lambda'] 			= 0 #lam #decimal.Decimal(1e-12) #lam #.000000000001
	exp_settings['num_chunks']		= 50
	exp_settings['chunk-by-what']	= chunkify.CHUNK_BY_DURATION
	exp_settings['chunk_type']		= chunkify.CHUNKIFY_LINEAR	# CHUNKIFY_LINEAR, CHUNKIFY_TRIANGULAR, CHUNKIFY_MINJERK
	exp_settings['angle_strength']	= 500 # is what was used  astr #40
	exp_settings['min_path_length'] = {}
	exp_settings['is_denominator']	= False
	exp_settings['f_vis']			= f_exp_single_normalized
	exp_settings['kill_1']			= km
	exp_settings['angle_cutoff']	= 70
	exp_settings['fov']	= 120
	exp_settings['prob_og']			= False
	exp_settings['right-bound']		= rb
	exp_settings['waypoint_offset']	= 20


	img = r.get_img()
	cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_empty.png', img)

	# good set of bottom paths
	paths = {((1005, 257, 180), 'naked'): [(203, 437), (203, 436), (203, 436), (203, 434), (203, 432), (203, 429), (203, 424), (203, 418), (203, 411), (203, 402), (203, 392), (203, 379), (204, 365), (205, 350), (206, 332), (209, 313), (213, 292), (217, 270), (224, 247), (233, 224), (245, 199), (259, 175), (278, 153), (301, 133), (330, 118), (364, 109), (403, 106), (445, 109), (489, 118), (533, 130), (575, 143), (617, 157), (655, 172), (693, 187), (728, 201), (762, 214), (792, 227), (821, 238), (848, 249), (872, 257), (895, 265), (915, 272), (933, 277), (949, 280), (963, 282), (975, 282), (984, 281), (992, 279), (998, 275), (1001, 270), (1003, 266), (1004, 263), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'omni'): [(203, 437), (203, 436), (203, 436), (203, 434), (203, 432), (203, 429), (203, 424), (203, 418), (203, 411), (203, 402), (203, 392), (203, 379), (204, 365), (205, 350), (206, 332), (209, 313), (213, 292), (217, 270), (224, 247), (233, 224), (245, 199), (259, 175), (278, 153), (301, 133), (330, 118), (364, 109), (403, 106), (445, 109), (489, 118), (533, 130), (575, 143), (617, 157), (655, 172), (693, 187), (728, 201), (762, 214), (792, 227), (821, 238), (848, 249), (872, 257), (895, 265), (915, 272), (933, 277), (949, 280), (963, 282), (975, 282), (984, 281), (992, 279), (998, 275), (1001, 270), (1003, 266), (1004, 263), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 435), (225, 434), (234, 433), (246, 432), (259, 430), (275, 429), (293, 427), (313, 425), (336, 423), (361, 420), (389, 417), (420, 413), (454, 410), (489, 406), (527, 403), (568, 399), (611, 394), (656, 389), (703, 385), (752, 379), (801, 374), (850, 370), (895, 365), (936, 362), (973, 358), (1006, 356), (1034, 353), (1057, 351), (1074, 350), (1085, 347), (1077, 342), (1066, 337), (1055, 332), (1045, 326), (1035, 320), (1027, 313), (1021, 305), (1015, 297), (1012, 289), (1009, 283), (1006, 276), (1005, 270), (1005, 266), (1005, 262), (1005, 259), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(203, 437), (204, 437), (204, 437), (206, 437), (209, 437), (213, 437), (218, 437), (226, 437), (235, 437), (246, 437), (260, 437), (276, 437), (294, 437), (314, 437), (337, 437), (363, 437), (391, 436), (422, 436), (455, 436), (491, 436), (530, 435), (571, 435), (613, 434), (659, 434), (706, 433), (754, 432), (804, 431), (854, 429), (901, 427), (944, 425), (982, 423), (1016, 419), (1043, 416), (1066, 411), (1081, 403), (1085, 392), (1080, 381), (1070, 370), (1060, 359), (1050, 349), (1041, 338), (1033, 329), (1026, 318), (1020, 309), (1015, 300), (1012, 292), (1009, 284), (1006, 277), (1006, 271), (1005, 266), (1005, 262), (1005, 259), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 437), (217, 437), (224, 437), (233, 437), (244, 437), (256, 437), (271, 437), (288, 437), (307, 438), (329, 438), (353, 438), (380, 438), (408, 438), (440, 438), (473, 438), (509, 438), (547, 438), (587, 437), (629, 437), (673, 436), (718, 435), (764, 434), (808, 432), (850, 430), (887, 427), (920, 424), (949, 420), (973, 415), (994, 409), (1009, 401), (1019, 391), (1024, 379), (1024, 366), (1022, 354), (1020, 343), (1017, 331), (1013, 321), (1011, 310), (1009, 301), (1008, 292), (1006, 284), (1005, 277), (1005, 271), (1005, 266), (1005, 262), (1005, 259), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (223, 436), (231, 436), (240, 436), (252, 436), (266, 436), (281, 436), (298, 435), (318, 435), (340, 435), (364, 434), (390, 434), (418, 433), (449, 432), (481, 431), (516, 430), (552, 429), (590, 427), (629, 426), (670, 423), (711, 421), (750, 418), (787, 415), (820, 411), (850, 407), (876, 402), (900, 397), (921, 391), (939, 384), (954, 375), (967, 367), (978, 357), (986, 347), (991, 336), (996, 325), (999, 315), (1001, 305), (1002, 297), (1003, 288), (1004, 280), (1004, 274), (1004, 268), (1004, 264), (1004, 261), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (215, 436), (220, 436), (226, 436), (233, 436), (242, 435), (252, 435), (264, 435), (277, 434), (292, 433), (310, 433), (328, 432), (349, 430), (372, 429), (396, 428), (423, 426), (452, 424), (483, 421), (516, 419), (551, 416), (588, 413), (627, 409), (665, 404), (701, 400), (736, 394), (768, 389), (798, 383), (827, 377), (853, 370), (877, 363), (899, 355), (918, 348), (936, 340), (951, 332), (964, 323), (975, 315), (983, 306), (990, 298), (995, 290), (999, 282), (1002, 276), (1003, 270), (1004, 265), (1004, 262), (1004, 259), (1004, 258), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (202, 454), (202, 461), (202, 470), (202, 480), (202, 492), (203, 506), (204, 521), (205, 538), (208, 557), (211, 577), (217, 598), (223, 620), (233, 643), (245, 666), (260, 688), (280, 709), (305, 726), (335, 739), (371, 746), (410, 747), (452, 742), (497, 734), (541, 723), (583, 710), (624, 697), (663, 684), (700, 670), (736, 658), (769, 645), (799, 634), (828, 624), (854, 615), (879, 607), (901, 601), (920, 595), (938, 592), (954, 590), (967, 589), (978, 590), (987, 592), (994, 595), (999, 600), (1002, 604), (1003, 608), (1004, 612), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(203, 437), (203, 438), (203, 438), (203, 440), (203, 442), (203, 445), (203, 450), (203, 456), (203, 463), (203, 472), (203, 482), (203, 495), (204, 509), (205, 524), (206, 542), (209, 561), (213, 582), (217, 604), (224, 627), (233, 650), (245, 675), (259, 699), (278, 721), (301, 741), (330, 756), (364, 765), (403, 768), (445, 765), (489, 756), (533, 744), (575, 731), (617, 717), (655, 702), (693, 687), (728, 673), (762, 660), (792, 647), (821, 636), (848, 625), (872, 617), (895, 609), (915, 602), (933, 597), (949, 594), (963, 592), (975, 592), (984, 593), (992, 595), (998, 599), (1001, 604), (1003, 608), (1004, 611), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 436), (235, 436), (246, 436), (259, 435), (275, 435), (293, 435), (313, 435), (336, 434), (362, 434), (390, 434), (421, 433), (455, 433), (491, 433), (529, 432), (570, 432), (613, 432), (658, 432), (705, 432), (754, 433), (803, 434), (853, 435), (901, 436), (944, 438), (983, 440), (1016, 444), (1044, 448), (1066, 454), (1081, 462), (1085, 474), (1080, 486), (1071, 498), (1061, 510), (1051, 520), (1042, 531), (1033, 542), (1026, 552), (1020, 562), (1015, 571), (1012, 580), (1009, 588), (1007, 595), (1006, 601), (1005, 607), (1005, 611), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'b'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 436), (235, 436), (246, 436), (259, 436), (275, 436), (293, 436), (313, 435), (336, 435), (362, 435), (390, 435), (421, 435), (455, 435), (491, 435), (529, 435), (570, 435), (613, 435), (658, 436), (705, 436), (754, 437), (803, 438), (853, 439), (901, 441), (944, 443), (982, 445), (1016, 449), (1044, 453), (1066, 459), (1081, 466), (1085, 477), (1080, 489), (1071, 501), (1061, 511), (1051, 522), (1041, 533), (1033, 544), (1026, 553), (1020, 563), (1015, 572), (1012, 581), (1009, 589), (1006, 595), (1006, 602), (1005, 607), (1005, 611), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (209, 437), (213, 437), (218, 438), (225, 439), (235, 440), (246, 441), (259, 443), (275, 444), (293, 446), (313, 448), (336, 450), (362, 453), (390, 456), (421, 460), (454, 463), (490, 467), (528, 470), (569, 474), (612, 479), (657, 484), (704, 488), (752, 494), (802, 499), (850, 503), (896, 508), (937, 511), (974, 515), (1007, 517), (1034, 520), (1057, 522), (1075, 523), (1085, 527), (1077, 531), (1066, 536), (1055, 541), (1044, 547), (1035, 553), (1027, 561), (1020, 568), (1015, 576), (1012, 584), (1009, 591), (1006, 597), (1005, 603), (1005, 608), (1005, 612), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (204, 439), (204, 441), (204, 444), (205, 449), (205, 454), (206, 462), (208, 471), (209, 480), (211, 493), (214, 506), (217, 522), (221, 539), (226, 558), (233, 578), (241, 599), (250, 622), (262, 645), (277, 668), (294, 691), (315, 713), (340, 732), (369, 748), (402, 758), (441, 762), (483, 760), (525, 752), (567, 740), (608, 727), (647, 713), (684, 699), (719, 684), (752, 670), (783, 656), (812, 644), (839, 632), (864, 622), (887, 614), (907, 605), (926, 599), (943, 595), (957, 592), (970, 590), (981, 591), (989, 592), (996, 596), (1000, 600), (1003, 604), (1003, 609), (1004, 612), (1004, 615), (1004, 616), (1004, 617)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (202, 447), (202, 452), (202, 459), (202, 466), (202, 474), (203, 484), (205, 495), (208, 507), (211, 521), (218, 535), (226, 549), (236, 564), (250, 579), (267, 593), (287, 605), (312, 615), (341, 622), (374, 627), (410, 628), (450, 627), (493, 624), (537, 619), (580, 614), (622, 608), (661, 602), (700, 596), (736, 590), (769, 586), (801, 582), (830, 578), (857, 576), (882, 574), (904, 573), (924, 574), (941, 575), (957, 577), (970, 580), (980, 584), (989, 588), (995, 593), (999, 598), (1002, 603), (1003, 608), (1004, 612), (1004, 614), (1004, 616), (1004, 617)]}
	bottom_paths = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 426), (203, 420), (203, 412), (204, 403), (204, 393), (205, 381), (206, 367), (207, 351), (210, 333), (213, 314), (216, 294), (221, 271), (229, 248), (238, 224), (249, 200), (264, 176), (283, 153), (306, 134), (334, 118), (368, 109), (406, 107), (448, 112), (492, 122), (536, 136), (578, 151), (619, 167), (658, 184), (695, 200), (730, 216), (763, 231), (794, 245), (823, 258), (850, 268), (875, 278), (897, 285), (918, 291), (936, 295), (952, 297), (966, 298), (978, 296), (987, 292), (994, 287), (999, 282), (1001, 275), (1003, 270), (1004, 265), (1004, 261), (1004, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 426), (203, 420), (203, 412), (204, 403), (204, 393), (205, 381), (206, 367), (207, 351), (210, 333), (213, 314), (216, 294), (221, 271), (229, 248), (238, 224), (249, 200), (264, 176), (283, 153), (306, 134), (334, 118), (368, 109), (406, 107), (448, 112), (492, 122), (536, 136), (578, 151), (619, 167), (658, 184), (695, 200), (730, 216), (763, 231), (794, 245), (823, 258), (850, 268), (875, 278), (897, 285), (918, 291), (936, 295), (952, 297), (966, 298), (978, 296), (987, 292), (994, 287), (999, 282), (1001, 275), (1003, 270), (1004, 265), (1004, 261), (1004, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 435), (217, 434), (224, 433), (232, 432), (243, 430), (256, 427), (270, 426), (287, 423), (307, 420), (328, 416), (352, 412), (379, 408), (407, 403), (438, 398), (472, 393), (508, 388), (547, 382), (587, 377), (629, 370), (673, 364), (719, 359), (765, 353), (808, 349), (849, 345), (885, 343), (918, 341), (947, 341), (972, 342), (992, 344), (1010, 347), (1022, 349), (1031, 352), (1028, 348), (1022, 341), (1018, 333), (1015, 323), (1012, 314), (1009, 304), (1007, 295), (1006, 287), (1006, 279), (1005, 272), (1005, 267), (1005, 263), (1005, 259), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 435), (233, 435), (244, 434), (256, 434), (271, 433), (288, 432), (308, 431), (330, 430), (354, 428), (381, 427), (410, 425), (442, 424), (475, 422), (512, 420), (551, 418), (591, 417), (634, 414), (678, 412), (724, 410), (770, 408), (816, 406), (858, 404), (896, 402), (931, 401), (960, 400), (985, 399), (1006, 398), (1021, 395), (1032, 390), (1034, 382), (1031, 373), (1027, 362), (1022, 350), (1018, 339), (1014, 328), (1011, 317), (1009, 307), (1007, 297), (1006, 288), (1006, 280), (1005, 273), (1005, 267), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 437), (217, 437), (224, 437), (233, 437), (244, 437), (256, 437), (271, 437), (288, 437), (308, 437), (330, 437), (354, 437), (380, 437), (410, 437), (441, 437), (475, 437), (511, 437), (549, 437), (590, 436), (632, 436), (676, 436), (722, 435), (768, 435), (813, 434), (855, 433), (894, 432), (928, 431), (957, 429), (982, 426), (1002, 423), (1017, 416), (1027, 408), (1029, 396), (1027, 384), (1024, 372), (1021, 359), (1017, 346), (1014, 334), (1011, 322), (1009, 311), (1008, 301), (1006, 291), (1005, 283), (1005, 276), (1005, 270), (1005, 264), (1005, 261), (1005, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (223, 436), (231, 436), (241, 436), (252, 436), (265, 436), (281, 436), (298, 436), (318, 435), (339, 435), (363, 435), (389, 435), (417, 434), (448, 434), (480, 433), (514, 433), (550, 432), (588, 431), (627, 430), (668, 429), (709, 428), (749, 427), (786, 426), (819, 424), (849, 422), (877, 419), (901, 415), (922, 410), (941, 405), (956, 397), (969, 388), (979, 377), (987, 365), (992, 353), (997, 341), (999, 329), (1001, 317), (1003, 306), (1003, 297), (1004, 288), (1004, 280), (1004, 273), (1004, 267), (1004, 262), (1004, 259), (1005, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 435), (214, 435), (218, 435), (223, 434), (229, 434), (237, 434), (246, 433), (257, 433), (270, 432), (285, 431), (301, 430), (321, 430), (342, 429), (366, 428), (393, 427), (423, 426), (455, 424), (491, 423), (529, 421), (569, 418), (611, 415), (652, 412), (691, 408), (728, 402), (764, 397), (797, 391), (827, 384), (855, 377), (880, 370), (903, 361), (923, 353), (941, 344), (956, 334), (969, 325), (979, 315), (988, 305), (994, 296), (998, 288), (1001, 280), (1003, 273), (1004, 267), (1004, 263), (1004, 260), (1005, 258), (1005, 257), (1005, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 442), (203, 445), (203, 449), (202, 455), (202, 462), (202, 471), (202, 481), (202, 493), (203, 506), (204, 522), (206, 538), (208, 556), (212, 576), (218, 597), (225, 618), (235, 641), (248, 663), (264, 684), (285, 703), (311, 718), (342, 727), (378, 732), (418, 729), (461, 722), (506, 711), (549, 698), (592, 684), (632, 670), (671, 655), (708, 642), (743, 628), (776, 616), (807, 606), (835, 596), (862, 588), (886, 581), (908, 577), (927, 573), (945, 572), (960, 572), (973, 574), (983, 577), (991, 582), (997, 588), (1000, 595), (1002, 601), (1004, 606), (1004, 610), (1004, 614), (1004, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'omni'): [(204, 437), (204, 438), (204, 438), (203, 439), (203, 441), (203, 444), (203, 448), (203, 454), (203, 462), (204, 471), (204, 481), (205, 493), (206, 507), (207, 523), (210, 541), (213, 560), (216, 580), (221, 603), (229, 626), (238, 650), (249, 674), (264, 698), (283, 721), (306, 740), (334, 756), (368, 765), (406, 767), (448, 762), (492, 752), (536, 738), (578, 723), (619, 707), (658, 690), (695, 674), (730, 658), (763, 643), (794, 629), (823, 616), (850, 606), (875, 596), (897, 589), (918, 583), (936, 579), (952, 577), (966, 576), (978, 578), (987, 582), (994, 587), (999, 592), (1001, 599), (1003, 604), (1004, 609), (1004, 613), (1004, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 436), (233, 436), (244, 436), (257, 436), (272, 436), (289, 435), (308, 435), (330, 435), (355, 435), (382, 434), (411, 434), (443, 434), (476, 434), (513, 434), (552, 433), (593, 433), (635, 433), (679, 433), (725, 433), (772, 434), (818, 434), (861, 435), (900, 436), (934, 437), (964, 439), (989, 442), (1009, 446), (1024, 452), (1033, 462), (1034, 474), (1031, 487), (1027, 499), (1023, 512), (1018, 525), (1015, 537), (1013, 549), (1010, 561), (1008, 571), (1007, 580), (1006, 589), (1005, 596), (1005, 603), (1005, 608), (1005, 612), (1005, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 436), (233, 436), (244, 436), (257, 436), (272, 436), (289, 436), (308, 436), (330, 436), (355, 436), (382, 436), (411, 436), (443, 436), (477, 436), (513, 436), (552, 436), (593, 437), (635, 437), (680, 437), (725, 438), (772, 438), (818, 439), (861, 440), (900, 441), (934, 442), (964, 444), (989, 446), (1009, 450), (1024, 456), (1033, 464), (1034, 476), (1031, 489), (1027, 501), (1023, 514), (1018, 526), (1015, 538), (1012, 550), (1010, 561), (1008, 572), (1006, 581), (1006, 590), (1005, 597), (1005, 603), (1005, 609), (1005, 612), (1005, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (211, 440), (216, 442), (222, 444), (230, 448), (240, 451), (251, 456), (265, 461), (281, 467), (298, 474), (318, 482), (341, 490), (365, 499), (392, 509), (421, 519), (452, 530), (485, 542), (521, 553), (559, 565), (598, 577), (639, 588), (682, 600), (726, 609), (767, 618), (805, 623), (841, 627), (872, 626), (900, 623), (924, 618), (945, 611), (962, 602), (977, 592), (988, 582), (996, 572), (1002, 564), (1005, 557), (1007, 555), (1007, 562), (1006, 570), (1006, 579), (1005, 586), (1005, 594), (1005, 600), (1005, 606), (1005, 610), (1005, 613), (1004, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (205, 439), (206, 440), (209, 443), (212, 447), (216, 452), (221, 458), (227, 466), (235, 475), (244, 485), (254, 497), (267, 510), (280, 524), (295, 541), (312, 558), (331, 577), (351, 596), (373, 616), (398, 637), (425, 659), (453, 681), (484, 701), (517, 721), (553, 738), (591, 752), (631, 760), (670, 761), (706, 755), (741, 744), (772, 731), (800, 714), (826, 696), (850, 679), (871, 661), (891, 645), (908, 629), (925, 615), (939, 603), (952, 593), (964, 584), (975, 578), (984, 575), (992, 575), (997, 579), (1001, 585), (1002, 592), (1003, 598), (1004, 604), (1004, 609), (1004, 612), (1004, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (202, 447), (202, 452), (202, 458), (202, 465), (203, 473), (204, 483), (206, 493), (208, 505), (213, 518), (219, 531), (228, 545), (239, 559), (254, 571), (272, 582), (294, 592), (320, 600), (350, 605), (383, 607), (420, 606), (460, 603), (504, 598), (548, 591), (591, 585), (633, 578), (672, 571), (710, 566), (746, 560), (780, 556), (811, 553), (840, 550), (867, 549), (891, 549), (913, 550), (932, 553), (950, 556), (964, 561), (976, 566), (985, 573), (992, 580), (997, 587), (1001, 595), (1002, 601), (1004, 606), (1004, 611), (1004, 614), (1004, 615), (1004, 616)]}
	# good top paths
	top_paths = {((1005, 257, 180), 'naked'): [(203, 437), (203, 436), (203, 436), (203, 434), (203, 432), (203, 429), (203, 425), (202, 419), (202, 412), (202, 403), (202, 393), (202, 381), (203, 367), (204, 352), (205, 335), (207, 316), (211, 296), (217, 275), (223, 252), (233, 229), (244, 205), (259, 182), (279, 161), (303, 142), (332, 127), (367, 117), (406, 114), (449, 116), (493, 122), (537, 131), (579, 142), (620, 153), (659, 166), (697, 177), (732, 190), (765, 201), (795, 212), (824, 222), (850, 231), (875, 240), (896, 247), (916, 253), (933, 259), (949, 263), (962, 266), (973, 268), (983, 268), (991, 268), (997, 267), (1000, 265), (1003, 262), (1004, 260), (1004, 258), (1004, 256), (1004, 256)], ((1005, 257, 180), 'omni'): [(203, 437), (203, 436), (203, 436), (203, 434), (203, 432), (203, 429), (203, 425), (202, 419), (202, 412), (202, 403), (202, 393), (202, 381), (203, 367), (204, 352), (205, 335), (207, 316), (211, 296), (217, 275), (223, 252), (233, 229), (244, 205), (259, 182), (279, 161), (303, 142), (332, 127), (367, 117), (406, 114), (449, 116), (493, 122), (537, 131), (579, 142), (620, 153), (659, 166), (697, 177), (732, 190), (765, 201), (795, 212), (824, 222), (850, 231), (875, 240), (896, 247), (916, 253), (933, 259), (949, 263), (962, 266), (973, 268), (983, 268), (991, 268), (997, 267), (1000, 265), (1003, 262), (1004, 260), (1004, 258), (1004, 256), (1004, 256)], ((1005, 257, 180), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 435), (217, 434), (224, 432), (233, 431), (244, 429), (256, 427), (271, 424), (288, 420), (308, 416), (330, 412), (354, 408), (381, 403), (410, 397), (442, 392), (477, 386), (513, 379), (552, 372), (593, 364), (636, 356), (681, 349), (727, 340), (772, 332), (815, 325), (855, 319), (891, 312), (923, 308), (953, 304), (977, 300), (999, 298), (1017, 297), (1031, 296), (1042, 296), (1042, 296), (1035, 294), (1028, 292), (1022, 288), (1016, 285), (1013, 281), (1009, 276), (1007, 271), (1006, 267), (1005, 264), (1005, 261), (1005, 258), (1005, 257), (1005, 257), (1005, 257)], ((1005, 257, 180), 'b'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 437), (218, 437), (225, 437), (234, 437), (245, 437), (258, 437), (273, 437), (290, 437), (310, 437), (332, 438), (356, 438), (384, 438), (413, 438), (445, 437), (480, 437), (516, 437), (556, 436), (597, 436), (640, 435), (685, 433), (731, 432), (778, 429), (824, 426), (867, 423), (906, 420), (940, 415), (970, 409), (995, 403), (1015, 394), (1030, 385), (1040, 374), (1043, 361), (1041, 349), (1037, 338), (1032, 327), (1027, 317), (1022, 308), (1018, 299), (1014, 292), (1011, 284), (1008, 278), (1007, 273), (1006, 268), (1005, 264), (1005, 261), (1005, 258), (1005, 257), (1005, 257), (1005, 257)], ((1005, 257, 180), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 437), (217, 437), (224, 437), (233, 437), (244, 437), (257, 437), (272, 437), (289, 437), (309, 437), (331, 438), (355, 438), (382, 438), (412, 438), (444, 437), (478, 437), (514, 437), (553, 436), (594, 436), (637, 434), (681, 433), (727, 431), (774, 429), (819, 426), (861, 423), (900, 419), (934, 414), (963, 409), (988, 402), (1008, 394), (1023, 384), (1032, 373), (1037, 361), (1036, 349), (1033, 337), (1029, 326), (1025, 317), (1020, 308), (1017, 299), (1012, 291), (1010, 284), (1008, 278), (1006, 272), (1006, 267), (1005, 264), (1005, 261), (1005, 258), (1005, 257), (1005, 257), (1005, 257)], ((1005, 257, 180), 'd'): [(203, 437), (203, 437), (204, 436), (205, 436), (208, 436), (212, 436), (217, 436), (224, 436), (232, 436), (242, 436), (253, 436), (267, 436), (283, 436), (301, 435), (322, 435), (344, 434), (369, 434), (396, 433), (425, 432), (457, 431), (490, 430), (526, 429), (563, 427), (603, 425), (643, 422), (685, 419), (728, 416), (768, 411), (805, 407), (838, 402), (867, 396), (894, 390), (917, 383), (937, 375), (954, 367), (968, 358), (979, 348), (988, 339), (994, 328), (998, 319), (1001, 310), (1003, 301), (1004, 293), (1004, 286), (1004, 279), (1004, 273), (1005, 268), (1005, 264), (1005, 261), (1005, 258), (1005, 257), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (215, 436), (220, 436), (227, 436), (235, 436), (245, 436), (256, 435), (269, 435), (284, 434), (301, 434), (320, 433), (340, 432), (362, 431), (387, 429), (413, 428), (442, 425), (472, 423), (503, 420), (537, 416), (572, 413), (609, 407), (647, 402), (684, 396), (718, 390), (751, 383), (781, 376), (810, 369), (836, 362), (860, 355), (882, 347), (903, 340), (921, 333), (937, 325), (951, 317), (963, 310), (974, 303), (983, 296), (989, 289), (994, 283), (998, 278), (1001, 272), (1003, 268), (1004, 264), (1004, 261), (1004, 258), (1004, 257), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 454), (203, 461), (203, 469), (203, 479), (203, 489), (205, 502), (206, 516), (209, 532), (213, 549), (218, 568), (225, 587), (234, 607), (246, 627), (262, 647), (280, 665), (304, 682), (332, 695), (364, 704), (401, 709), (443, 709), (486, 706), (529, 700), (572, 693), (614, 684), (653, 676), (691, 667), (726, 658), (760, 649), (792, 641), (821, 633), (847, 626), (872, 620), (894, 615), (915, 611), (932, 606), (948, 604), (962, 603), (973, 602), (983, 602), (991, 603), (997, 605), (1000, 607), (1002, 610), (1004, 613), (1004, 615), (1004, 617), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(203, 437), (203, 438), (203, 438), (203, 440), (203, 442), (203, 445), (203, 449), (202, 455), (202, 462), (202, 471), (202, 481), (202, 493), (203, 507), (204, 522), (205, 539), (207, 558), (211, 578), (217, 599), (223, 622), (233, 645), (244, 669), (259, 692), (279, 713), (303, 732), (332, 747), (367, 757), (406, 760), (449, 758), (493, 752), (537, 743), (579, 732), (620, 721), (659, 708), (697, 697), (732, 684), (765, 673), (795, 662), (824, 652), (850, 643), (875, 634), (896, 627), (916, 621), (933, 615), (949, 611), (962, 608), (973, 606), (983, 606), (991, 606), (997, 607), (1000, 609), (1003, 612), (1004, 614), (1004, 616), (1004, 618), (1004, 618)], ((1005, 617, 0), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 435), (233, 435), (244, 434), (257, 434), (272, 433), (289, 432), (309, 431), (331, 430), (356, 429), (382, 427), (412, 426), (444, 425), (478, 424), (515, 423), (554, 422), (596, 421), (638, 420), (684, 420), (730, 420), (777, 421), (824, 423), (867, 425), (907, 428), (942, 433), (972, 439), (997, 448), (1017, 457), (1031, 470), (1040, 483), (1043, 497), (1041, 511), (1038, 524), (1033, 536), (1028, 548), (1023, 558), (1019, 568), (1015, 577), (1011, 584), (1009, 592), (1007, 598), (1006, 603), (1005, 607), (1005, 611), (1005, 614), (1005, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'b'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (225, 436), (233, 436), (244, 435), (257, 435), (272, 435), (290, 434), (309, 434), (331, 433), (356, 433), (383, 432), (413, 432), (444, 431), (479, 431), (515, 431), (555, 431), (596, 431), (639, 431), (684, 432), (730, 433), (777, 435), (824, 437), (867, 440), (906, 443), (940, 448), (971, 454), (996, 461), (1016, 470), (1031, 481), (1040, 493), (1043, 506), (1041, 519), (1038, 531), (1033, 542), (1028, 553), (1023, 563), (1018, 572), (1014, 580), (1011, 587), (1008, 593), (1007, 599), (1006, 604), (1005, 608), (1005, 612), (1005, 614), (1005, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 438), (218, 438), (225, 439), (233, 440), (244, 441), (257, 442), (272, 444), (289, 446), (309, 448), (332, 450), (356, 453), (383, 456), (412, 459), (445, 463), (479, 466), (515, 471), (555, 475), (596, 480), (639, 485), (684, 491), (731, 495), (778, 502), (822, 507), (863, 512), (900, 517), (934, 522), (963, 526), (989, 530), (1009, 534), (1027, 538), (1039, 543), (1045, 548), (1043, 556), (1038, 562), (1031, 568), (1025, 574), (1019, 580), (1015, 585), (1011, 592), (1009, 597), (1007, 602), (1006, 606), (1005, 610), (1005, 613), (1005, 615), (1005, 617), (1005, 617), (1005, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (204, 439), (205, 441), (205, 444), (207, 448), (208, 454), (210, 461), (212, 469), (216, 478), (220, 490), (225, 503), (231, 517), (237, 533), (245, 550), (255, 569), (266, 588), (279, 609), (294, 630), (312, 651), (331, 671), (355, 691), (383, 709), (413, 723), (448, 734), (486, 739), (527, 738), (568, 734), (608, 726), (647, 717), (683, 706), (718, 695), (751, 684), (781, 672), (810, 662), (836, 652), (861, 642), (883, 634), (904, 627), (922, 619), (938, 614), (953, 610), (965, 607), (976, 604), (985, 604), (992, 604), (997, 606), (1001, 607), (1003, 610), (1004, 613), (1004, 615), (1004, 617), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 453), (203, 459), (203, 467), (203, 475), (205, 485), (206, 496), (209, 509), (214, 522), (220, 537), (227, 552), (237, 567), (251, 583), (267, 598), (287, 612), (311, 624), (339, 633), (371, 641), (407, 644), (447, 646), (489, 645), (532, 642), (575, 639), (617, 634), (656, 629), (694, 625), (730, 620), (764, 615), (795, 610), (824, 606), (851, 604), (876, 601), (898, 599), (918, 597), (935, 596), (951, 596), (964, 596), (975, 597), (985, 599), (992, 602), (997, 604), (1001, 608), (1003, 611), (1004, 613), (1004, 615), (1004, 617), (1004, 617)]}

	best_paths = {}
	best_paths[((1005, 257, 180), 'naked')]		= bottom_paths[((1005, 257, 180), 'naked')]
	best_paths[((1005, 257, 180), 'omni')]	= bottom_paths[((1005, 257, 180), 'omni')]
	best_paths[((1005, 257, 180), 'a')]		= bottom_paths[((1005, 257, 180), 'a')]
	best_paths[((1005, 257, 180), 'b')]		= bottom_paths[((1005, 257, 180), 'b')]
	best_paths[((1005, 257, 180), 'c')]		= bottom_paths[((1005, 257, 180), 'c')]
	best_paths[((1005, 257, 180), 'd')]		= bottom_paths[((1005, 257, 180), 'd')]
	best_paths[((1005, 257, 180), 'e')]		= bottom_paths[((1005, 257, 180), 'e')]

	best_paths[((1005, 617, 0), 'naked')]		= top_paths[((1005, 617, 0), 'naked')]
	best_paths[((1005, 617, 0), 'omni')]		= top_paths[((1005, 617, 0), 'omni')]
	best_paths[((1005, 617, 0), 'a')]			= top_paths[((1005, 617, 0), 'a')]
	best_paths[((1005, 617, 0), 'b')]			= top_paths[((1005, 617, 0), 'b')]
	best_paths[((1005, 617, 0), 'c')]			= top_paths[((1005, 617, 0), 'c')]
	best_paths[((1005, 617, 0), 'd')]			= top_paths[((1005, 617, 0), 'd')]
	best_paths[((1005, 617, 0), 'e')]			= top_paths[((1005, 617, 0), 'e')]

	best_paths = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (204, 434), (204, 432), (204, 429), (204, 424), (205, 418), (205, 411), (206, 402), (207, 391), (209, 379), (211, 365), (214, 349), (218, 331), (222, 312), (227, 292), (234, 270), (242, 247), (253, 223), (267, 199), (283, 175), (302, 153), (326, 132), (355, 117), (389, 108), (427, 107), (469, 112), (512, 124), (555, 139), (596, 156), (635, 174), (673, 191), (709, 209), (743, 226), (775, 241), (805, 256), (833, 269), (859, 280), (883, 289), (904, 297), (924, 303), (942, 306), (958, 307), (971, 306), (982, 303), (990, 298), (996, 291), (1000, 283), (1003, 277), (1003, 271), (1004, 265), (1004, 261), (1004, 259), (1004, 257), (1004, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (204, 434), (204, 432), (204, 429), (204, 424), (205, 418), (205, 411), (206, 402), (207, 391), (209, 379), (211, 365), (214, 349), (218, 331), (222, 312), (227, 292), (234, 270), (242, 247), (253, 223), (267, 199), (283, 175), (302, 153), (326, 132), (355, 117), (389, 108), (427, 107), (469, 112), (512, 124), (555, 139), (596, 156), (635, 174), (673, 191), (709, 209), (743, 226), (775, 241), (805, 256), (833, 269), (859, 280), (883, 289), (904, 297), (924, 303), (942, 306), (958, 307), (971, 306), (982, 303), (990, 298), (996, 291), (1000, 283), (1003, 277), (1003, 271), (1004, 265), (1004, 261), (1004, 259), (1004, 257), (1004, 257)], ((1005, 257, 180), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 435), (225, 434), (233, 433), (245, 432), (258, 431), (273, 429), (291, 427), (310, 425), (333, 423), (358, 421), (385, 418), (415, 415), (447, 412), (482, 409), (520, 405), (560, 402), (601, 398), (645, 395), (691, 391), (738, 387), (786, 384), (833, 381), (876, 379), (916, 377), (953, 377), (982, 377), (1009, 378), (1031, 379), (1047, 382), (1058, 384), (1052, 379), (1043, 372), (1034, 363), (1027, 353), (1021, 342), (1016, 331), (1013, 319), (1010, 309), (1008, 299), (1007, 289), (1006, 281), (1005, 273), (1005, 268), (1005, 263), (1005, 260), (1004, 258), (1004, 257)], ((1005, 257, 180), 'b'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 435), (234, 435), (245, 434), (258, 434), (273, 433), (291, 432), (311, 431), (333, 430), (358, 429), (386, 428), (416, 426), (448, 425), (483, 423), (521, 421), (560, 420), (602, 418), (646, 416), (692, 414), (739, 412), (787, 411), (835, 409), (879, 408), (920, 407), (955, 407), (987, 406), (1013, 407), (1035, 407), (1050, 406), (1058, 402), (1052, 394), (1045, 385), (1037, 373), (1030, 362), (1023, 350), (1018, 339), (1014, 326), (1011, 315), (1008, 304), (1007, 294), (1006, 285), (1005, 277), (1005, 271), (1005, 265), (1005, 261), (1005, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 435), (233, 435), (244, 434), (257, 434), (272, 433), (289, 432), (308, 431), (330, 430), (355, 428), (382, 427), (411, 426), (443, 424), (477, 423), (514, 421), (553, 419), (594, 417), (637, 415), (682, 414), (728, 412), (775, 410), (821, 409), (863, 408), (902, 407), (937, 406), (967, 406), (992, 407), (1013, 407), (1029, 406), (1039, 402), (1038, 393), (1033, 383), (1028, 371), (1023, 359), (1018, 347), (1014, 335), (1011, 323), (1009, 312), (1008, 301), (1006, 291), (1005, 283), (1005, 275), (1005, 269), (1005, 264), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (217, 436), (223, 435), (231, 435), (241, 434), (254, 433), (268, 432), (284, 431), (302, 430), (323, 429), (346, 428), (371, 426), (399, 425), (429, 423), (460, 421), (495, 420), (531, 417), (569, 416), (609, 414), (651, 412), (694, 410), (738, 409), (780, 407), (819, 407), (854, 406), (885, 407), (913, 407), (937, 407), (958, 406), (975, 404), (988, 400), (997, 393), (1002, 382), (1005, 370), (1006, 358), (1006, 345), (1006, 333), (1006, 321), (1005, 310), (1005, 300), (1005, 290), (1005, 281), (1005, 274), (1005, 268), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (205, 435), (206, 434), (208, 432), (210, 430), (213, 429), (218, 427), (223, 425), (229, 423), (237, 421), (247, 419), (259, 417), (272, 415), (287, 413), (305, 411), (325, 409), (348, 408), (373, 407), (401, 407), (432, 406), (466, 406), (503, 407), (542, 407), (584, 407), (626, 406), (666, 405), (705, 403), (741, 401), (776, 397), (808, 393), (838, 388), (865, 381), (889, 374), (912, 366), (931, 358), (948, 348), (962, 339), (974, 328), (983, 318), (991, 308), (995, 298), (999, 289), (1002, 281), (1003, 273), (1004, 268), (1004, 263), (1004, 260), (1004, 258), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 454), (203, 462), (203, 470), (203, 481), (204, 493), (205, 506), (206, 522), (209, 538), (211, 557), (216, 577), (222, 598), (229, 620), (240, 642), (252, 665), (269, 686), (290, 706), (315, 722), (346, 732), (382, 737), (421, 734), (463, 726), (507, 713), (551, 699), (593, 684), (633, 668), (672, 652), (709, 637), (743, 623), (776, 609), (807, 597), (835, 587), (862, 579), (886, 571), (908, 566), (928, 563), (946, 562), (961, 563), (974, 566), (984, 570), (992, 577), (997, 584), (1000, 591), (1003, 598), (1003, 604), (1004, 609), (1004, 613), (1004, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'omni'): [(204, 437), (204, 438), (204, 438), (204, 440), (204, 442), (204, 445), (204, 450), (205, 456), (205, 463), (206, 472), (207, 483), (209, 495), (211, 509), (214, 525), (218, 543), (222, 562), (227, 582), (234, 604), (242, 627), (253, 651), (267, 675), (283, 699), (302, 721), (326, 742), (355, 757), (389, 766), (427, 767), (469, 762), (512, 750), (555, 735), (596, 718), (635, 700), (673, 683), (709, 665), (743, 648), (775, 633), (805, 618), (833, 605), (859, 594), (883, 585), (904, 577), (924, 571), (942, 568), (958, 567), (971, 568), (982, 571), (990, 576), (996, 583), (1000, 591), (1003, 597), (1003, 603), (1004, 609), (1004, 613), (1004, 615), (1004, 617), (1004, 617)], ((1005, 617, 0), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 436), (234, 436), (245, 436), (258, 436), (273, 436), (291, 436), (311, 436), (333, 436), (359, 436), (386, 436), (416, 436), (449, 436), (484, 436), (521, 436), (561, 436), (603, 436), (646, 436), (692, 436), (740, 436), (788, 436), (836, 436), (881, 436), (922, 436), (959, 437), (990, 437), (1017, 438), (1038, 440), (1053, 443), (1058, 453), (1053, 464), (1046, 476), (1038, 489), (1031, 501), (1025, 515), (1020, 528), (1015, 540), (1012, 553), (1009, 564), (1008, 575), (1007, 584), (1006, 593), (1005, 600), (1005, 606), (1005, 611), (1005, 614), (1005, 616), (1005, 617)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (206, 437), (209, 437), (212, 437), (218, 437), (225, 438), (234, 438), (245, 439), (258, 439), (274, 440), (292, 441), (311, 442), (334, 443), (359, 444), (386, 445), (416, 447), (449, 448), (484, 450), (521, 452), (561, 453), (603, 455), (646, 457), (692, 459), (739, 461), (787, 462), (835, 464), (880, 465), (920, 466), (955, 467), (987, 467), (1013, 466), (1035, 466), (1051, 467), (1058, 471), (1052, 479), (1044, 489), (1037, 500), (1030, 511), (1023, 523), (1018, 535), (1014, 547), (1011, 558), (1008, 569), (1007, 579), (1006, 588), (1005, 596), (1005, 603), (1005, 608), (1005, 612), (1005, 615), (1005, 617), (1005, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 437), (212, 438), (217, 439), (224, 440), (232, 441), (243, 443), (255, 446), (270, 448), (287, 451), (306, 454), (327, 458), (351, 461), (377, 465), (406, 470), (437, 476), (470, 481), (506, 486), (544, 492), (584, 497), (625, 503), (669, 508), (714, 513), (759, 518), (802, 522), (842, 525), (878, 526), (910, 526), (939, 525), (963, 523), (984, 519), (1001, 516), (1014, 511), (1022, 508), (1022, 514), (1018, 522), (1015, 533), (1012, 543), (1009, 554), (1008, 565), (1006, 575), (1006, 584), (1005, 592), (1005, 599), (1005, 605), (1005, 610), (1005, 613), (1005, 615), (1005, 616), (1005, 616)], ((1005, 617, 0), 'd'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 449), (203, 455), (203, 463), (203, 472), (203, 482), (204, 494), (205, 509), (206, 525), (208, 542), (211, 562), (214, 582), (220, 605), (226, 628), (235, 652), (246, 677), (261, 701), (279, 723), (302, 743), (331, 758), (365, 765), (403, 766), (445, 759), (490, 747), (533, 732), (576, 715), (616, 697), (656, 680), (693, 663), (728, 645), (762, 631), (793, 616), (822, 604), (850, 592), (875, 583), (897, 576), (918, 570), (937, 567), (953, 566), (967, 567), (979, 569), (988, 575), (995, 581), (999, 588), (1002, 595), (1003, 601), (1004, 607), (1004, 611), (1004, 614), (1004, 616), (1004, 616)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 453), (203, 460), (203, 468), (204, 476), (205, 487), (207, 499), (210, 512), (213, 526), (218, 542), (225, 558), (235, 574), (247, 590), (263, 606), (282, 620), (305, 632), (332, 640), (363, 645), (399, 646), (437, 643), (479, 637), (523, 627), (567, 618), (609, 607), (649, 596), (688, 587), (725, 578), (760, 569), (792, 562), (823, 556), (851, 552), (876, 548), (900, 547), (921, 547), (940, 549), (956, 552), (969, 557), (981, 563), (989, 571), (995, 579), (999, 587), (1002, 594), (1003, 601), (1004, 607), (1004, 611), (1004, 614), (1004, 616), (1004, 616)]}
	best_paths = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (204, 434), (204, 432), (204, 429), (204, 424), (205, 418), (205, 411), (206, 402), (207, 391), (209, 379), (211, 365), (214, 349), (218, 332), (222, 312), (227, 291), (234, 270), (242, 247), (253, 223), (267, 199), (283, 175), (302, 153), (327, 133), (356, 118), (389, 108), (427, 107), (469, 112), (512, 123), (555, 138), (596, 154), (635, 171), (673, 189), (709, 207), (743, 223), (775, 238), (805, 253), (833, 266), (859, 276), (883, 286), (904, 294), (924, 300), (942, 303), (957, 305), (971, 304), (982, 300), (990, 296), (996, 289), (1000, 282), (1003, 276), (1003, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 256), (1004, 256)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (204, 434), (204, 432), (204, 429), (204, 424), (205, 418), (205, 411), (206, 402), (207, 391), (209, 379), (211, 365), (214, 349), (218, 332), (222, 312), (227, 291), (234, 270), (242, 247), (253, 223), (267, 199), (283, 175), (302, 153), (327, 133), (356, 118), (389, 108), (427, 107), (469, 112), (512, 123), (555, 138), (596, 154), (635, 171), (673, 189), (709, 207), (743, 223), (775, 238), (805, 253), (833, 266), (859, 276), (883, 286), (904, 294), (924, 300), (942, 303), (957, 305), (971, 304), (982, 300), (990, 296), (996, 289), (1000, 282), (1003, 276), (1003, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 256), (1004, 256)], ((1005, 257, 180), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (212, 436), (218, 435), (225, 434), (234, 433), (245, 432), (258, 431), (273, 429), (291, 428), (311, 426), (333, 423), (358, 421), (386, 419), (416, 416), (448, 413), (483, 409), (520, 406), (560, 403), (602, 399), (645, 396), (691, 392), (739, 388), (786, 385), (833, 382), (877, 379), (916, 378), (953, 377), (982, 376), (1009, 377), (1031, 378), (1047, 380), (1058, 381), (1052, 376), (1043, 369), (1035, 360), (1027, 349), (1021, 339), (1016, 328), (1013, 317), (1010, 307), (1008, 297), (1007, 288), (1006, 280), (1005, 273), (1005, 267), (1005, 262), (1005, 259), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (212, 436), (218, 436), (225, 435), (234, 435), (245, 434), (258, 434), (274, 433), (292, 432), (311, 431), (334, 430), (359, 429), (386, 428), (416, 426), (449, 425), (484, 424), (521, 422), (561, 420), (603, 419), (646, 417), (692, 415), (739, 413), (787, 411), (835, 410), (880, 408), (920, 407), (955, 407), (987, 406), (1013, 406), (1035, 406), (1051, 405), (1058, 399), (1052, 391), (1045, 382), (1038, 371), (1030, 360), (1024, 348), (1018, 336), (1014, 324), (1011, 313), (1008, 303), (1007, 293), (1006, 285), (1005, 277), (1005, 270), (1005, 265), (1005, 261), (1005, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 435), (234, 435), (245, 434), (257, 434), (272, 433), (290, 432), (309, 431), (331, 430), (356, 429), (383, 428), (412, 426), (444, 425), (478, 423), (515, 421), (554, 420), (595, 418), (637, 416), (682, 414), (729, 413), (776, 411), (822, 409), (864, 408), (903, 407), (937, 406), (967, 406), (993, 406), (1013, 405), (1029, 404), (1039, 400), (1038, 391), (1034, 380), (1029, 369), (1023, 357), (1018, 345), (1014, 334), (1012, 322), (1010, 311), (1008, 300), (1006, 291), (1005, 282), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (223, 435), (231, 435), (241, 434), (252, 433), (265, 432), (281, 431), (298, 430), (318, 429), (340, 427), (364, 426), (390, 424), (419, 422), (449, 421), (482, 419), (516, 417), (553, 415), (590, 413), (630, 411), (671, 409), (713, 408), (753, 407), (789, 406), (823, 406), (854, 406), (881, 406), (905, 405), (926, 403), (945, 400), (961, 395), (973, 387), (983, 378), (990, 366), (995, 355), (999, 343), (1001, 331), (1002, 319), (1003, 308), (1004, 298), (1004, 289), (1004, 280), (1004, 273), (1004, 267), (1004, 262), (1004, 259), (1005, 257), (1005, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (205, 435), (206, 434), (208, 433), (211, 431), (214, 429), (219, 427), (224, 426), (230, 424), (239, 421), (249, 419), (260, 417), (274, 415), (289, 413), (307, 412), (327, 410), (350, 409), (375, 408), (404, 407), (435, 406), (469, 406), (505, 406), (545, 406), (587, 405), (628, 404), (669, 403), (707, 400), (744, 398), (778, 394), (809, 389), (839, 384), (866, 377), (891, 370), (913, 363), (931, 354), (949, 345), (963, 336), (974, 325), (983, 315), (991, 306), (996, 296), (999, 287), (1002, 279), (1003, 272), (1004, 267), (1004, 262), (1004, 259), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 454), (203, 462), (203, 470), (203, 481), (204, 492), (205, 506), (206, 521), (209, 538), (211, 557), (216, 576), (222, 598), (229, 620), (240, 642), (253, 665), (269, 686), (290, 705), (315, 721), (346, 732), (382, 737), (421, 734), (463, 726), (508, 714), (551, 700), (593, 686), (634, 670), (672, 654), (709, 640), (744, 626), (777, 612), (807, 601), (835, 590), (862, 582), (887, 575), (908, 570), (928, 567), (946, 565), (961, 565), (974, 568), (984, 573), (992, 578), (997, 586), (1000, 593), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(204, 437), (204, 438), (204, 438), (204, 440), (204, 442), (204, 445), (204, 450), (205, 456), (205, 463), (206, 472), (207, 483), (209, 495), (211, 509), (214, 525), (218, 542), (222, 562), (227, 583), (234, 604), (242, 627), (253, 651), (267, 675), (283, 699), (302, 721), (327, 741), (356, 756), (389, 766), (427, 767), (469, 762), (512, 751), (555, 736), (596, 720), (635, 703), (673, 685), (709, 667), (743, 651), (775, 636), (805, 621), (833, 608), (859, 598), (883, 588), (904, 580), (924, 574), (942, 571), (957, 569), (971, 570), (982, 574), (990, 578), (996, 585), (1000, 592), (1003, 598), (1003, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 618), (1004, 618)], ((1005, 617, 0), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 436), (234, 436), (245, 436), (258, 436), (273, 436), (291, 436), (311, 436), (333, 436), (359, 436), (386, 436), (416, 436), (449, 435), (484, 435), (521, 435), (561, 435), (603, 435), (646, 435), (692, 435), (740, 435), (788, 435), (836, 435), (881, 436), (922, 436), (959, 437), (991, 437), (1017, 439), (1038, 441), (1053, 445), (1058, 455), (1053, 466), (1047, 478), (1039, 490), (1032, 503), (1026, 516), (1021, 529), (1016, 541), (1013, 553), (1010, 564), (1008, 575), (1007, 584), (1006, 593), (1005, 600), (1005, 606), (1005, 611), (1005, 614), (1004, 615), (1004, 616)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (206, 437), (209, 437), (212, 437), (218, 437), (225, 438), (234, 438), (245, 439), (258, 439), (274, 440), (291, 441), (311, 442), (334, 443), (359, 444), (386, 445), (416, 447), (449, 448), (484, 449), (521, 451), (561, 453), (603, 455), (647, 456), (692, 458), (739, 460), (787, 462), (835, 463), (880, 465), (920, 466), (955, 467), (987, 467), (1013, 467), (1035, 467), (1051, 468), (1058, 474), (1052, 482), (1045, 491), (1038, 502), (1030, 513), (1024, 525), (1018, 537), (1014, 549), (1011, 560), (1008, 570), (1007, 580), (1006, 588), (1005, 596), (1005, 603), (1005, 608), (1005, 612), (1005, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (211, 440), (216, 441), (222, 444), (230, 447), (240, 451), (251, 456), (265, 460), (280, 466), (298, 474), (317, 481), (339, 489), (364, 497), (391, 507), (419, 517), (450, 528), (483, 539), (519, 550), (556, 561), (595, 573), (636, 584), (679, 594), (722, 603), (764, 611), (802, 615), (837, 617), (869, 615), (897, 611), (921, 604), (942, 596), (959, 586), (973, 577), (984, 566), (993, 557), (999, 549), (1004, 543), (1006, 548), (1006, 556), (1006, 565), (1005, 574), (1005, 583), (1005, 591), (1005, 598), (1005, 604), (1005, 610), (1005, 613), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (204, 439), (205, 441), (205, 444), (207, 448), (208, 454), (210, 462), (212, 471), (215, 481), (219, 493), (224, 507), (229, 522), (235, 539), (244, 557), (252, 578), (262, 599), (275, 621), (289, 644), (305, 668), (324, 691), (347, 714), (372, 733), (402, 750), (435, 762), (473, 767), (513, 764), (554, 754), (595, 741), (634, 724), (671, 707), (706, 689), (739, 672), (770, 654), (800, 638), (827, 623), (853, 610), (876, 598), (898, 588), (917, 579), (935, 574), (951, 570), (965, 568), (977, 570), (987, 573), (994, 579), (998, 585), (1001, 593), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 453), (203, 460), (203, 468), (204, 476), (205, 487), (207, 499), (210, 511), (213, 526), (219, 541), (226, 557), (236, 573), (247, 590), (263, 606), (282, 620), (305, 632), (332, 640), (364, 646), (399, 646), (437, 644), (480, 638), (523, 629), (567, 620), (609, 609), (650, 599), (688, 589), (725, 580), (760, 572), (792, 565), (823, 560), (851, 555), (876, 552), (900, 550), (921, 550), (940, 552), (956, 555), (969, 560), (981, 565), (989, 573), (995, 580), (999, 588), (1002, 595), (1003, 602), (1004, 607), (1004, 612), (1004, 615), (1004, 617), (1004, 617)]}

	yay = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (204, 434), (204, 432), (204, 429), (204, 424), (205, 418), (205, 411), (206, 402), (207, 391), (209, 379), (211, 365), (214, 349), (218, 332), (222, 312), (227, 291), (234, 270), (242, 247), (253, 223), (267, 199), (283, 175), (302, 153), (327, 133), (356, 118), (389, 108), (427, 107), (469, 112), (512, 123), (555, 138), (596, 154), (635, 171), (673, 189), (709, 207), (743, 223), (775, 238), (805, 253), (833, 266), (859, 276), (883, 286), (904, 294), (924, 300), (942, 303), (957, 305), (971, 304), (982, 300), (990, 296), (996, 289), (1000, 282), (1003, 276), (1003, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 256), (1004, 256)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (204, 434), (204, 432), (204, 429), (204, 424), (205, 418), (205, 411), (206, 402), (207, 391), (209, 379), (211, 365), (214, 349), (218, 332), (222, 312), (227, 291), (234, 270), (242, 247), (253, 223), (267, 199), (283, 175), (302, 153), (327, 133), (356, 118), (389, 108), (427, 107), (469, 112), (512, 123), (555, 138), (596, 154), (635, 171), (673, 189), (709, 207), (743, 223), (775, 238), (805, 253), (833, 266), (859, 276), (883, 286), (904, 294), (924, 300), (942, 303), (957, 305), (971, 304), (982, 300), (990, 296), (996, 289), (1000, 282), (1003, 276), (1003, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 256), (1004, 256)], ((1005, 257, 180), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (212, 436), (218, 435), (225, 434), (234, 433), (245, 432), (258, 431), (273, 429), (291, 428), (311, 426), (333, 423), (358, 421), (386, 419), (416, 416), (448, 413), (483, 409), (520, 406), (560, 403), (602, 399), (645, 396), (691, 392), (739, 388), (786, 385), (833, 382), (877, 379), (916, 378), (953, 377), (982, 376), (1009, 377), (1031, 378), (1047, 380), (1058, 381), (1052, 376), (1043, 369), (1035, 360), (1027, 349), (1021, 339), (1016, 328), (1013, 317), (1010, 307), (1008, 297), (1007, 288), (1006, 280), (1005, 273), (1005, 267), (1005, 262), (1005, 259), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (212, 436), (218, 436), (225, 435), (234, 435), (245, 434), (258, 434), (274, 433), (292, 432), (311, 431), (334, 430), (359, 429), (386, 428), (416, 426), (449, 425), (484, 424), (521, 422), (561, 420), (603, 419), (646, 417), (692, 415), (739, 413), (787, 411), (835, 410), (880, 408), (920, 407), (955, 407), (987, 406), (1013, 406), (1035, 406), (1051, 405), (1058, 399), (1052, 391), (1045, 382), (1038, 371), (1030, 360), (1024, 348), (1018, 336), (1014, 324), (1011, 313), (1008, 303), (1007, 293), (1006, 285), (1005, 277), (1005, 270), (1005, 265), (1005, 261), (1005, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 435), (234, 435), (245, 434), (257, 434), (272, 433), (290, 432), (309, 431), (331, 430), (356, 429), (383, 428), (412, 426), (444, 425), (478, 423), (515, 421), (554, 420), (595, 418), (637, 416), (682, 414), (729, 413), (776, 411), (822, 409), (864, 408), (903, 407), (937, 406), (967, 406), (993, 406), (1013, 405), (1029, 404), (1039, 400), (1038, 391), (1034, 380), (1029, 369), (1023, 357), (1018, 345), (1014, 334), (1012, 322), (1010, 311), (1008, 300), (1006, 291), (1005, 282), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (223, 435), (231, 435), (241, 434), (252, 433), (265, 432), (281, 431), (298, 430), (318, 429), (340, 427), (364, 426), (390, 424), (419, 422), (449, 421), (482, 419), (516, 417), (553, 415), (590, 413), (630, 411), (671, 409), (713, 408), (753, 407), (789, 406), (823, 406), (854, 406), (881, 406), (905, 405), (926, 403), (945, 400), (961, 395), (973, 387), (983, 378), (990, 366), (995, 355), (999, 343), (1001, 331), (1002, 319), (1003, 308), (1004, 298), (1004, 289), (1004, 280), (1004, 273), (1004, 267), (1004, 262), (1004, 259), (1005, 257), (1005, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (205, 435), (206, 434), (208, 433), (211, 431), (214, 429), (219, 427), (224, 426), (230, 424), (239, 421), (249, 419), (260, 417), (274, 415), (289, 413), (307, 412), (327, 410), (350, 409), (375, 408), (404, 407), (435, 406), (469, 406), (505, 406), (545, 406), (587, 405), (628, 404), (669, 403), (707, 400), (744, 398), (778, 394), (809, 389), (839, 384), (866, 377), (891, 370), (913, 363), (931, 354), (949, 345), (963, 336), (974, 325), (983, 315), (991, 306), (996, 296), (999, 287), (1002, 279), (1003, 272), (1004, 267), (1004, 262), (1004, 259), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 454), (203, 462), (203, 470), (203, 481), (204, 492), (205, 506), (206, 521), (209, 538), (211, 557), (216, 576), (222, 598), (229, 620), (240, 642), (253, 665), (269, 686), (290, 705), (315, 721), (346, 732), (382, 737), (421, 734), (463, 726), (508, 714), (551, 700), (593, 686), (634, 670), (672, 654), (709, 640), (744, 626), (777, 612), (807, 601), (835, 590), (862, 582), (887, 575), (908, 570), (928, 567), (946, 565), (961, 565), (974, 568), (984, 573), (992, 578), (997, 586), (1000, 593), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(204, 437), (204, 438), (204, 438), (204, 440), (204, 442), (204, 445), (204, 450), (205, 456), (205, 463), (206, 472), (207, 483), (209, 495), (211, 509), (214, 525), (218, 542), (222, 562), (227, 583), (234, 604), (242, 627), (253, 651), (267, 675), (283, 699), (302, 721), (327, 741), (356, 756), (389, 766), (427, 767), (469, 762), (512, 751), (555, 736), (596, 720), (635, 703), (673, 685), (709, 667), (743, 651), (775, 636), (805, 621), (833, 608), (859, 598), (883, 588), (904, 580), (924, 574), (942, 571), (957, 569), (971, 570), (982, 574), (990, 578), (996, 585), (1000, 592), (1003, 598), (1003, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 618), (1004, 618)], ((1005, 617, 0), 'a'): [(203, 437), (203, 437), (204, 436), (206, 436), (208, 436), (212, 436), (218, 436), (225, 436), (234, 436), (245, 436), (258, 436), (273, 436), (291, 436), (311, 436), (333, 436), (359, 436), (386, 436), (416, 436), (449, 435), (484, 435), (521, 435), (561, 435), (603, 435), (646, 435), (692, 435), (740, 435), (788, 435), (836, 435), (881, 436), (922, 436), (959, 437), (991, 437), (1017, 439), (1038, 441), (1053, 445), (1058, 455), (1053, 466), (1047, 478), (1039, 490), (1032, 503), (1026, 516), (1021, 529), (1016, 541), (1013, 553), (1010, 564), (1008, 575), (1007, 584), (1006, 593), (1005, 600), (1005, 606), (1005, 611), (1005, 614), (1004, 615), (1004, 616)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (206, 437), (209, 437), (212, 437), (218, 437), (225, 438), (234, 438), (245, 439), (258, 439), (274, 440), (291, 441), (311, 442), (334, 443), (359, 444), (386, 445), (416, 447), (449, 448), (484, 449), (521, 451), (561, 453), (603, 455), (647, 456), (692, 458), (739, 460), (787, 462), (835, 463), (880, 465), (920, 466), (955, 467), (987, 467), (1013, 467), (1035, 467), (1051, 468), (1058, 474), (1052, 482), (1045, 491), (1038, 502), (1030, 513), (1024, 525), (1018, 537), (1014, 549), (1011, 560), (1008, 570), (1007, 580), (1006, 588), (1005, 596), (1005, 603), (1005, 608), (1005, 612), (1005, 615), (1004, 616), (1004, 616)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (211, 440), (216, 441), (222, 444), (230, 447), (240, 451), (251, 456), (265, 460), (280, 466), (298, 474), (317, 481), (339, 489), (364, 497), (391, 507), (419, 517), (450, 528), (483, 539), (519, 550), (556, 561), (595, 573), (636, 584), (679, 594), (722, 603), (764, 611), (802, 615), (837, 617), (869, 615), (897, 611), (921, 604), (942, 596), (959, 586), (973, 577), (984, 566), (993, 557), (999, 549), (1004, 543), (1006, 548), (1006, 556), (1006, 565), (1005, 574), (1005, 583), (1005, 591), (1005, 598), (1005, 604), (1005, 610), (1005, 613), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (204, 439), (205, 441), (205, 444), (207, 448), (208, 454), (210, 462), (212, 471), (215, 481), (219, 493), (224, 507), (229, 522), (235, 539), (244, 557), (252, 578), (262, 599), (275, 621), (289, 644), (305, 668), (324, 691), (347, 714), (372, 733), (402, 750), (435, 762), (473, 767), (513, 764), (554, 754), (595, 741), (634, 724), (671, 707), (706, 689), (739, 672), (770, 654), (800, 638), (827, 623), (853, 610), (876, 598), (898, 588), (917, 579), (935, 574), (951, 570), (965, 568), (977, 570), (987, 573), (994, 579), (998, 585), (1001, 593), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 448), (203, 453), (203, 460), (203, 468), (204, 476), (205, 487), (207, 499), (210, 511), (213, 526), (219, 541), (226, 557), (236, 573), (247, 590), (263, 606), (282, 620), (305, 632), (332, 640), (364, 646), (399, 646), (437, 644), (480, 638), (523, 629), (567, 620), (609, 609), (650, 599), (688, 589), (725, 580), (760, 572), (792, 565), (823, 560), (851, 555), (876, 552), (900, 550), (921, 550), (940, 552), (956, 555), (969, 560), (981, 565), (989, 573), (995, 580), (999, 588), (1002, 595), (1003, 602), (1004, 607), (1004, 612), (1004, 615), (1004, 617), (1004, 617)]}
	best_paths = yay
	# best_paths[((1005, 257, 180), 'omni')] = yay[((1005, 257, 180), 'omni')]
	# best_paths[((1005, 257, 180), 'a')] = yay[((1005, 257, 180), 'a')]
	# best_paths[((1005, 257, 180), 'e')] = yay[((1005, 257, 180), 'e')]
	# best_paths[((1005, 617, 0), 'omni')] = yay[((1005, 617, 0), 'omni')]
	# best_paths[((1005, 617, 0), 'a')] = yay[((1005, 617, 0), 'a')]
	# best_paths[((1005, 617, 0), 'e')] = yay[((1005, 617, 0), 'e')]

	best_paths[((1005, 617, 0), 'shortest')]	= get_min_viable_path(r, (1005, 617, 0), exp_settings)
	best_paths[((1005, 257, 180), 'shortest')]	= get_min_viable_path(r, (1005, 257, 180), exp_settings)

	if FLAG_EXPORT_JUST_TEASER:
		new_best_paths = {}

		new_best_paths[((1005, 617, 0), 'a')] = best_paths[((1005, 617, 0), 'a')]
		new_best_paths[((1005, 617, 0), 'e')] = best_paths[((1005, 617, 0), 'e')]
		new_best_paths[((1005, 617, 0), 'omni')] = best_paths[((1005, 617, 0), 'omni')]

		new_best_paths[((1005, 257, 180), 'a')] = best_paths[((1005, 257, 180), 'a')]
		new_best_paths[((1005, 257, 180), 'e')] = best_paths[((1005, 257, 180), 'e')]
		new_best_paths[((1005, 257, 180), 'omni')] = best_paths[((1005, 257, 180), 'omni')]


	if FLAG_EXPORT_MOCON:
		stamps = {}
		stamps[('G_ME', 'LA', None)] = [7557, 8307, 7729, 7501, 5183, 6349, 7548, 7293, 7540, 6597, 7230, 5466, 2822, 7389, 7695, 7162, 9221, 3471, 9247, 7190, 9941, 7601, 6778, 6753, 6023, 7259, 6860, 7201, 7524, 8474, 9324, 986, 2227, 2561, 2948, 3122, 3375, 3549, 3776, 3936, 4189, 4336, 4603, 4763, 5030, 5204, 5471, 5724, 2077, 2304, 2971, 3105, 3278, 3452, 3598, 3799, 3999, 4947, 5293, 5440, 5667, 6041, 6401, 7323, 7656, 7260, 6639, 7454, 7456, 4899, 7512, 3731, 7266, 8021, 4256, 5161, 2691, 6278, 6797, 7400, 7818, 2188, 6375, 7626, 8576, 447, 7497, 502, 7484, 5556, 6580, 6695, 3604, 6962, 2854, 5511, 6631, 7536, 6932, 7463, 6982, 3985, 7392, 8144, 6334, 7424, 6022, 7222, 7687, 5931, 7344, 7944, 7744, 7486, 7101, 7851, 7837, 7499, 8009, 5544, 7427, 7266, 7294, 6293, 6590, 4969, 7195, 7331, 7685, 3917, 6712, 7279, 471, 7571, 6945, 6999, 9796, 7389, 4670, 5803, 6653, 7253, 3331, 4232, 4814, 5081, 5697, 6582, 7164, 7381, 4853, 6965, 5818, 5498, 6317, 7262, 7077, 7296, 950, 7552, 7589, 2109, 7610, 6510, 7179, 6935, 7518, 6534, 7587, 7482, 7329, 6756, 7806, 7571, 7663, 1370, 1870, 2937, 3674, 4037, 4424, 5687, 828, 1462, 2244, 6811, 1216, 2316, 1418, 6805, 7338, 7632, 6707, 7353, 508, 584, 730, 862, 946, 1071, 1154, 1293, 1390, 1536, 1626, 1723, 1862, 1959, 2078, 2203, 2314, 2453, 2585, 2724, 2821, 3036, 3161, 3286, 3383, 3502, 3626, 3731, 3828, 3946, 4043, 4140, 4251, 4342, 4432, 4543, 4640, 4766, 4870, 4946, 5078, 5154, 5293, 5383, 5494, 5592, 5696, 5779, 5925, 2552, 7660, 7244, 7411, 7305, 8256, 2233, 5014, 3605, 2418, 4617, 6551, 8551, 9918, 7145, 7779, 3410, 7265, 4434, 5651, 7434, 2599, 4516, 5401, 7760, 6416, 7665, 8172, 512, 748, 997, 1330, 2296, 3495, 3633, 4330, 528, 962, 1294, 2179, 2546, 2795, 3056, 3962, 4377, 4527, 4693, 4993, 5127, 5628, 6143, 827, 1324, 1725, 2108, 3026, 3324, 4874, 5191, 5524, 6710, 658, 1111, 1391, 2360, 3395, 4026, 5028, 7323, 4991, 6891, 7407, 6043, 6739, 7440, 6952, 7453, 7385, 7637, 7788, 7881, 9271, 10341, 3742, 8341, 7800, 6394, 8762, 8198, 7726, 7139, 3070, 7734, 7347, 5004, 5081, 7896, 6829, 6527, 7427, 7505, 6750, 7317, 7796, 8763, 4047, 5418, 6154, 7576, 6891, 7749, 1036, 2069, 2900, 470, 1102, 1369, 1784, 3001, 6779, 7264, 5478, 7471, 7503, 7784, 7479, 4025, 6564, 7896, 8842, 1363, 7945, 7296, 6581, 6998, 7264, 5858, 7457, 6705, 5978, 7211, 7595, 1931, 5257, 6435, 6471, 5549, 7302, 5662, 6096, 7128, 7526, 7411, 3185, 7638, 7586, 7392, 7399, 6882, 6571, 7243, 7109, 7745, 7782, 6571, 1331, 2808, 3727, 1520, 3529, 4354, 5108, 6159, 7593, 1100, 1626, 1988, 2651, 2895, 3454, 3749, 4293, 6968, 7913, 8780, 8547, 7315, 7414, 6573, 7626, 7546, 7572, 7621, 6267, 969, 6475, 7401, 1039, 1711, 3381, 6793, 918, 2173, 3069, 3504, 4769, 7300, 7724, 1607, 6934, 1598, 2471, 2993, 3794, 7592, 8826, 560, 6809, 7325, 6746, 7347, 4524, 7707, 6308, 7611, 6777, 6525, 7392, 6216, 7757, 6499, 1811, 8231, 7210, 7684, 6828, 7193, 7357, 5617, 7537, 7195, 7661, 5093, 7545, 5700, 6260, 7548, 9477, 699, 7534, 6996, 7813, 6709, 6805, 7807, 7537, 7557, 7287, 4074, 10442, 6560, 7444, 7668, 7676, 7669, 7874, 7523, 8877, 8133, 7663, 8663, 6257, 6591, 8308, 7627, 6959, 7577, 7444, 2174, 7700, 7317, 7249, 8900, 8191, 7220, 939, 1410, 2001, 6734, 7380, 853, 1318, 2567, 799, 1282, 4896, 8627, 4343, 4693, 4943, 1034, 2334, 7301, 8672, 6656, 6309, 9098, 6769, 1143, 1827, 3045, 4480, 4816, 2537, 9094, 7634, 6312, 7661, 7591, 5288, 6753, 7596, 7608, 4287, 8614, 6489, 5101, 5385, 5652, 7037, 7333, 7884, 3509, 6882, 7466, 8722, 1159, 5618, 907, 2422, 4157, 5569, 1547, 6725, 7175, 5887, 6420, 646, 7800, 6971, 7536, 6672, 5683, 7482, 8452, 7217, 9364, 6884, 6660, 6042, 6458, 7684, 7502, 7851, 6801, 5929, 5210, 4172, 6665, 7231, 7179, 7293, 10220, 1324, 3367, 6465, 7332, 6596, 6958, 7411, 7691, 6561, 1025, 6377, 7193, 6712, 1551, 3633, 2252, 3817, 6448, 707, 1856, 3121, 3538, 4337, 4837, 6053, 6868, 9216, 3159, 8557, 9772, 8750, 8056, 7961, 8087, 3947, 4517, 2541, 6311, 3902, 5438, 4842, 8191, 5029, 7143, 8083, 6613, 7595, 7251, 8002, 4969, 5998, 6567, 7750, 7026, 3189, 6029, 2327, 7257, 7587, 6304, 7535, 7470, 6125, 7475, 1714, 3680, 4547, 5614, 6480, 7131, 1587, 3603, 5504, 7270, 628, 1596, 5097, 6746, 7162, 835, 1236, 3859, 7330, 5986, 7652, 3213, 4460, 6345, 6916, 7777, 1323, 7419, 2332, 3573, 6340, 7226, 7180, 4666, 5736, 5871, 7420, 7537, 7674, 6304, 7119, 7685, 7270, 3929, 8092, 6113, 6996, 6076, 7226, 5759, 677, 7342, 7775, 5911, 7614, 6518, 7392, 9857, 6821, 4863, 7480, 7393, 3190, 2156, 7234, 7833, 5126, 6843, 7123, 6543, 7575, 6594, 7441, 7331, 5859, 7373, 5193, 6778, 1288, 5072, 5862, 8758, 8320, 6919, 7452, 5426, 5171, 8108, 8689, 9772, 4310, 5754, 6969, 6979, 1715, 6798, 663, 1396, 5982, 7495, 7774, 6807, 7617, 7979, 6763, 8374, 7111, 8195, 7392, 7545, 7282, 7285, 4517, 6891, 7224, 6050, 7102, 7436, 7557, 7384, 7478, 7540, 7610, 6828, 5871, 5202, 8890, 6989, 1521, 7340, 7581, 6107, 8013, 7839, 7215, 7882, 7976, 6767, 8003, 7127, 7284, 7169, 7481, 7493, 7314, 7282, 5676, 7041, 7424, 7974, 1083, 1717, 745, 2826, 3295, 4775, 5408, 3649, 5450, 7149, 7513, 7277, 7860, 7397, 7698]
		stamps[('G_ME', 'LA', 'VA')] = [7293, 7162, 2077, 2304, 2971, 3105, 3278, 3452, 3598, 3799, 3999, 4947, 5293, 5440, 5667, 6041, 6401, 7323, 7656, 7512, 7266, 8021, 447, 7497, 8144, 7944, 7101, 7851, 7499, 7266, 6590, 7331, 7685, 7389, 7077, 6510, 7179, 6935, 7518, 7482, 1418, 6707, 2418, 4617, 6551, 8551, 9918, 7265, 827, 1324, 1725, 2108, 3026, 3324, 4874, 5191, 5524, 6710, 4991, 6891, 6952, 7788, 7800, 8762, 8198, 7726, 7139, 3070, 7734, 7505, 7796, 8763, 1102, 1369, 7503, 7784, 1363, 7945, 7399, 7315, 7414, 7546, 1598, 2471, 2993, 3794, 7592, 8826, 6746, 7757, 7195, 7661, 7545, 7548, 7534, 7807, 10442, 7676, 7669, 8308, 7700, 7317, 8191, 8672, 7591, 7608, 8614, 6882, 7466, 5887, 7800, 8452, 7502, 7851, 6958, 7691, 707, 1856, 3121, 3538, 4337, 4837, 6053, 6868, 8087, 6311, 8191, 8083, 7251, 8002, 2327, 7587, 7475, 3859, 7330, 7419, 7119, 7685, 3929, 677, 7342, 7775, 6821, 7480, 7833, 5126, 7123, 8758, 8320, 8108, 7495, 7979, 7545, 7285, 7557, 7478, 6828, 8890, 7340, 7839, 7215, 7882, 7976, 7127, 7284, 7041, 7424, 745, 2826, 3295, 4775, 5408, 7513]
		stamps[('G_ME', 'LA', 'VB')] = [7501, 7540, 2822, 7389, 7695, 7190, 9941, 6860, 7524, 8474, 9324, 6639, 7454, 7456, 7818, 8576, 7484, 6932, 7463, 7392, 7687, 7744, 7486, 7837, 8009, 6712, 7279, 7571, 6945, 4670, 5803, 6653, 7253, 6965, 7296, 7589, 2109, 7610, 7329, 6756, 7806, 7571, 7663, 6805, 7338, 508, 584, 730, 862, 946, 1071, 1154, 1293, 1390, 1536, 1626, 1723, 1862, 1959, 2078, 2203, 2314, 2453, 2585, 2724, 2821, 3036, 3161, 3286, 3383, 3502, 3626, 3731, 3828, 3946, 4043, 4140, 4251, 4342, 4432, 4543, 4640, 4766, 4870, 4946, 5078, 5154, 5293, 5383, 5494, 5592, 5696, 5779, 5925, 2552, 7660, 7411, 3605, 7145, 7779, 5401, 6416, 7665, 8172, 7323, 6043, 6739, 7440, 7637, 7881, 9271, 10341, 5004, 6829, 6527, 7427, 6154, 7576, 2900, 7479, 7296, 6705, 1931, 7526, 7392, 7243, 7109, 7745, 7782, 6968, 7913, 6573, 7626, 7621, 918, 2173, 3069, 3504, 4769, 7300, 7724, 7347, 6525, 7392, 6828, 7193, 7357, 7537, 9477, 6996, 7813, 7537, 7668, 7874, 8133, 6959, 7577, 2174, 939, 1410, 2001, 6734, 7380, 8627, 1034, 2334, 7301, 6656, 6309, 9098, 7634, 6312, 7661, 7596, 7333, 7884, 1547, 6725, 7175, 7536, 9364, 6801, 6665, 7231, 7293, 1324, 3367, 6465, 7332, 6561, 6712, 8056, 4842, 7143, 6567, 7750, 7257, 7470, 1714, 3680, 4547, 5614, 6480, 7131, 5986, 7652, 7180, 7420, 7537, 7674, 7270, 8092, 9857, 7234, 6543, 7575, 7331, 5859, 7373, 6778, 6919, 7452, 8689, 9772, 6979, 663, 1396, 5982, 7617, 6763, 8374, 7392, 7282, 7224, 7540, 5871, 1521, 8013, 6767, 7481, 7493, 7314, 7974, 3649, 5450, 7149, 7277, 7860, 7698]
		stamps[('G_ME', 'LA', 'VC')] = [7557, 8307, 9247, 6778, 6753, 7260, 2691, 6278, 6797, 7400, 6375, 7626, 6580, 2854, 5511, 6631, 7536, 6982, 5931, 7344, 7294, 3917, 471, 3331, 4232, 4814, 5081, 5697, 6582, 7164, 7381, 6317, 7262, 1216, 2316, 7353, 7305, 8256, 7760, 658, 1111, 1391, 2360, 3395, 4026, 5028, 7407, 8341, 7347, 6750, 7317, 4047, 5418, 1784, 3001, 5858, 7457, 5978, 7211, 7595, 5257, 6471, 5662, 7411, 7586, 6571, 1520, 3529, 4354, 5108, 6159, 7593, 7572, 1039, 1711, 3381, 6793, 560, 6809, 7325, 8231, 5617, 6805, 7557, 6560, 7444, 6257, 6591, 7444, 799, 1282, 4896, 4943, 1143, 1827, 3045, 4480, 4816, 5288, 6753, 4287, 6489, 8722, 2422, 4157, 5569, 6971, 6884, 6042, 6458, 7684, 5210, 10220, 1025, 1551, 3633, 9216, 7961, 5438, 6029, 6304, 7535, 6125, 628, 1596, 5097, 6746, 7162, 1323, 2332, 5871, 6304, 6113, 6996, 6518, 7392, 7393, 6594, 7441, 5426, 5754, 6969, 6807, 7111, 8195, 6891, 7384, 5202, 7581, 7169, 7282, 1083, 1717, 7397]
		stamps[('G_ME', 'LA', 'VD')] = [5183, 6349, 7548, 986, 2227, 2561, 2948, 3122, 3375, 3549, 3776, 3936, 4189, 4336, 4603, 4763, 5030, 5204, 5471, 5724, 3731, 2188, 502, 5556, 3604, 3985, 6334, 7424, 5544, 7427, 4969, 9796, 4853, 5818, 950, 7552, 1370, 1870, 2937, 3674, 4037, 4424, 5687, 7632, 7244, 2233, 5014, 4434, 5651, 7434, 512, 748, 997, 1330, 2296, 3495, 3633, 4330, 7385, 5081, 7896, 7749, 470, 5478, 7471, 6581, 6998, 7264, 7302, 3185, 6571, 1100, 1626, 1988, 2651, 2895, 3454, 3749, 4293, 8547, 6267, 1607, 6934, 6308, 7611, 6216, 7210, 5700, 699, 6709, 4074, 7523, 7663, 8663, 7249, 853, 4343, 5101, 5385, 5652, 7037, 3509, 907, 5683, 7482, 6660, 7179, 6596, 7411, 6377, 7193, 3159, 3947, 3902, 5029, 6613, 7595, 3189, 835, 1236, 4666, 5736, 5759, 4863, 2156, 4310, 7774, 4517, 6050, 7102, 7436, 6107, 5676]
		stamps[('G_ME', 'LA', 'VE')] = [7729, 6597, 7230, 5466, 9221, 3471, 7601, 6023, 7259, 7201, 4899, 4256, 5161, 6695, 6962, 6022, 7222, 6293, 7195, 6999, 5498, 6534, 7587, 828, 1462, 2244, 6811, 3410, 2599, 4516, 528, 962, 1294, 2179, 2546, 2795, 3056, 3962, 4377, 4527, 4693, 4993, 5127, 5628, 6143, 7453, 3742, 6394, 6891, 1036, 2069, 6779, 7264, 4025, 6564, 7896, 8842, 6435, 5549, 6096, 7128, 7638, 6882, 1331, 2808, 3727, 8780, 969, 6475, 7401, 4524, 7707, 6777, 6499, 1811, 7684, 5093, 6260, 7287, 8877, 7627, 8900, 7220, 1318, 2567, 4693, 6769, 2537, 9094, 1159, 5618, 6420, 646, 6672, 7217, 5929, 4172, 2252, 3817, 6448, 8557, 9772, 8750, 4517, 2541, 4969, 5998, 7026, 1587, 3603, 5504, 7270, 3213, 4460, 6345, 6916, 7777, 3573, 6340, 7226, 6076, 7226, 5911, 7614, 3190, 6843, 5193, 1288, 5072, 5862, 5171, 1715, 6798, 7610, 6989, 8003]
		stamps[('G_ME', 'LB', None)] = [7877, 7393, 6478, 7551, 8478, 6848, 7365, 7399, 5837, 6920, 5821, 7506, 1575, 2124, 2582, 348, 6468, 7427, 7075, 7243, 6953, 7537, 6359, 7326, 7739, 7445, 7054, 6611, 7104, 7572, 5919, 6891, 257, 430, 617, 751, 1124, 1498, 1885, 3927, 4141, 4288, 4915, 5075, 7018, 5034, 5771, 7620, 7116, 4323, 6318, 5530, 4359, 6014, 7271, 2179, 4207, 8194, 6717, 5383, 3879, 7150, 631, 3865, 7871, 8623, 7472, 6983, 7405, 7311, 7585, 7108, 6123, 7668, 7605, 7050, 6391, 7524, 7505, 7250, 7525, 7729, 6382, 7232, 5448, 6932, 6320, 7438, 7720, 7722, 7358, 5838, 6112, 6796, 7552, 5990, 6848, 6052, 6784, 2802, 7234, 5644, 7143, 7243, 7033, 7269, 2438, 6891, 7562, 8129, 1913, 3482, 5013, 5814, 5947, 6114, 6214, 6380, 6699, 6868, 7047, 2429, 6512, 7228, 1032, 7097, 2575, 8629, 7208, 7171, 6291, 6716, 8382, 768, 4651, 6867, 7167, 3312, 7526, 7568, 7296, 5960, 6633, 7432, 5570, 7179, 3477, 7306, 7320, 1039, 1604, 2194, 1200, 7594, 4931, 6702, 9314, 7235, 7130, 6320, 7399, 7438, 7711, 7301, 6627, 7394, 7181, 7760, 7290, 7451, 8121, 5994, 7825, 7426, 950, 1580, 2080, 2730, 3417, 4930, 6380, 7401, 6975, 7020, 6977, 7428, 5255, 6106, 6971, 7039, 6299, 1373, 1970, 2102, 2241, 2366, 2519, 2678, 2804, 2935, 3060, 3185, 3317, 3442, 3582, 3720, 3838, 3998, 4151, 4303, 4435, 4526, 4706, 4831, 4949, 5053, 5171, 5262, 5352, 5491, 5644, 5727, 5817, 5887, 6019, 6102, 6213, 6296, 6415, 6533, 6609, 6692, 6796, 6346, 7089, 2694, 6229, 7404, 7939, 7409, 3055, 4873, 6396, 7229, 6938, 7337, 7259, 7330, 7800, 6475, 7089, 7600, 6810, 6402, 2761, 563, 1214, 2647, 3531, 4048, 7163, 614, 897, 1264, 1549, 1848, 2231, 2530, 3613, 4180, 4764, 5197, 5764, 755, 5099, 6548, 2286, 7811, 8085, 7879, 7121, 6702, 5736, 5800, 7217, 7513, 6248, 7400, 7214, 7650, 7967, 8025, 7295, 5658, 7273, 7110, 3577, 7434, 7403, 7252, 6577, 7775, 7735, 5637, 7648, 6554, 6157, 8495, 10398, 7913, 6774, 4647, 5573, 7207, 6361, 7994, 5988, 5730, 7096, 6768, 7128, 8241, 8024, 6457, 7404, 1104, 6737, 2816, 470, 561, 485, 7507, 6293, 8376, 7036, 6828, 5547, 6986, 5909, 7313, 5718, 7172, 7481, 808, 1461, 3167, 701, 7984, 7337, 3807, 6872, 6325, 7008, 6048, 7315, 7511, 7341, 7385, 7294, 7304, 7761, 555, 7326, 329, 1662, 3847, 1830, 3013, 5196, 7081, 7682, 7776, 3854, 5649, 4528, 5932, 5203, 6457, 2734, 7733, 6149, 730, 4814, 5981, 6148, 6381, 6765, 6997, 7631, 7269, 5776, 7765, 8450, 7374, 6450, 6157, 7151, 7435, 7752, 8471, 6664, 7531, 1529, 7379, 1180, 7772, 963, 4897, 6990, 7851, 7347, 7379, 3893, 8682, 9606, 6558, 8009, 7577, 4897, 6680, 7141, 7432, 6900, 7700, 7455, 7256, 7298, 6427, 4189, 6922, 7201, 1565, 2560, 4241, 5369, 7470, 1287, 2516, 3139, 1174, 3983, 1360, 1376, 7272, 8065, 4626, 5960, 6847, 7399, 7242, 2704, 7280, 7611, 7065, 6722, 6397, 6222, 7145, 6062, 7213, 6294, 7661, 8647, 2857, 3425, 8825, 7239, 4638, 7487, 8481, 6969, 7920, 7521, 8391, 2715, 7079, 7378, 7487, 7021, 7425, 7398, 6052, 7372, 958, 7782, 8167, 7290, 7797, 7426, 7001, 7972, 7372, 6409, 7059, 6488, 7474, 6513, 7586, 6414, 7569, 7826, 8155, 4455, 7228, 7252, 7500, 7629, 8106, 1597, 6414, 9010, 7764, 10347, 7042, 7972, 6393, 7532, 7425, 5677, 7186, 9409, 6212, 7313, 7618, 6865, 4989, 6787, 7513, 8700, 7694, 6876, 1508, 3838, 1654, 2520, 6399, 6787, 1615, 2847, 4628, 5627, 7693, 513, 696, 1263, 2144, 2412, 2644, 4095, 4894, 7452, 2370, 2298, 8515, 5532, 7082, 6405, 7974, 6461, 7999, 976, 2304, 5890, 6472, 1266, 2083, 4887, 2471, 6522, 7438, 7540, 5841, 7128, 7096, 6819, 7453, 6220, 7237, 6123, 6392, 4506, 5525, 6792, 7052, 7615, 10006, 9401, 1079, 2967, 5526, 835, 7243, 4474, 5362, 7247, 8037, 6447, 7218, 7095, 8074, 9210, 7182, 6300, 6176, 7121, 5164, 6999, 7605, 6008, 6566, 7531, 7315, 7883, 647, 913, 1063, 1230, 1342, 1563, 1879, 2012, 2195, 2345, 2545, 2678, 2894, 3211, 3344, 3577, 3910, 964, 4294, 7424, 2657, 6111, 6810, 4101, 7099, 2143, 1521, 801, 5766, 7498, 6973, 4816, 8029, 7430, 8367, 7294, 6766, 7449, 6173, 6777, 7372, 7752, 6900, 7531, 7689, 838, 7433, 1550, 5530, 1213, 2095, 4559, 1837, 3970, 6565, 7422, 949, 6080, 6018, 7809, 7599, 7588, 5091, 6626, 3874, 6887, 7379, 7438, 7192, 8380, 8571, 7535, 7397, 6002, 7077, 5719, 7086, 3482, 5382, 7507, 7611, 9393, 6335, 6857, 6439, 7747, 7481, 7458, 771, 1139, 2088, 2671, 3271, 3938, 4637, 5388, 6122, 6422, 6871, 1089, 2222, 1344, 2510, 429, 846, 1663, 2397, 2729, 3329, 4263, 4846, 4394, 6655, 7573, 2085, 2395, 7233, 10220, 7911, 9446, 7717, 8181, 3634, 6218, 7872, 6819, 4541, 7125, 6804, 7421, 7500, 6603, 4304, 4967, 5371, 7002, 5801, 7186, 7494, 5224, 7320, 7637, 10545, 5240, 6172, 6260, 1342, 3503, 4575, 5032, 5898, 8141, 8498, 8445, 7792, 7391, 7984, 4713, 7113, 7591, 5197, 7540, 6317, 7847, 5111, 1153, 1733, 7780, 8417, 7214, 7623, 6759, 7436, 6148, 7315, 4432, 5915, 7249, 6797, 6561, 6601, 7268, 7408, 7064, 7866, 7616, 5673, 9061, 5567, 7029, 7087, 6716, 3463, 3780, 4013, 4162, 4347, 4430, 4645, 8384, 7892, 6877, 6971, 6645, 7427, 6518, 3535, 1626, 2667, 7126, 6293, 7676, 1687, 7490, 10538, 1046, 8073, 7036, 7548, 7138, 7289, 7041, 4044, 6453, 7252, 7683, 7558, 6806, 7285, 8373, 9947, 8841, 7601, 7299, 7557, 8743, 8422, 8790, 9123, 6819, 7199, 7218, 7727, 9103, 7372, 7337, 5210, 5241, 7449, 7342, 7432, 8174, 1199, 7319, 873, 6897, 8062, 2228, 7517, 4736, 3426, 7476, 5789, 8447, 7741, 7494, 8206, 7288, 7331, 7081, 7287, 7238, 7404, 7162, 7150, 6842, 7625, 1040, 1891, 3007, 4189, 6823, 7442, 957, 4907, 7043, 7572, 8240, 1540, 2357, 3256, 4675, 5274, 6924, 7923, 8726, 1004, 2337, 4763, 7041, 7294, 7250, 7666, 6447, 7272, 7303]
		stamps[('G_ME', 'LB', 'VA')] = [7399, 1575, 2124, 2582, 7739, 7104, 7572, 7116, 7871, 8623, 7405, 7605, 7505, 7729, 7720, 7358, 7552, 7269, 7562, 8129, 2575, 7171, 768, 4651, 6867, 7167, 7526, 6702, 9314, 7399, 7438, 7290, 7451, 7825, 7020, 6977, 7428, 1373, 1970, 2102, 2241, 2366, 2519, 2678, 2804, 2935, 3060, 3185, 3317, 3442, 3582, 3720, 3838, 3998, 4151, 4303, 4435, 4526, 4706, 4831, 4949, 5053, 5171, 5262, 5352, 5491, 5644, 5727, 5817, 5887, 6019, 6102, 6213, 6296, 6415, 6533, 6609, 6692, 6796, 2694, 6229, 3055, 4873, 7330, 7800, 8085, 7879, 6702, 7273, 7110, 7252, 6774, 470, 701, 7984, 7511, 329, 1662, 7776, 730, 4814, 5981, 6148, 6381, 6765, 6997, 7631, 7374, 7752, 1180, 7379, 6900, 7700, 7256, 1287, 2516, 3139, 7272, 8065, 7242, 8647, 7920, 7487, 7782, 8167, 7001, 7972, 7474, 7569, 7826, 8155, 7629, 7532, 6787, 7513, 6876, 1615, 2847, 4628, 5627, 7693, 7452, 7974, 7540, 9401, 5526, 7243, 7182, 6999, 7605, 7883, 4101, 1521, 7498, 6766, 7449, 7752, 6900, 7531, 838, 7433, 7599, 6887, 8380, 7397, 7507, 7611, 9393, 7747, 7458, 4394, 6655, 7911, 9446, 6819, 7320, 7792, 7540, 7847, 1153, 1733, 7436, 7616, 8384, 6971, 1626, 2667, 7126, 1046, 7036, 7548, 7138, 7289, 6453, 7252, 7558, 6806, 7285, 8422, 8790, 9123, 7199, 9103, 7337, 7432, 1199, 7319, 873, 7081, 7238, 957, 4907, 7043, 7572, 8240, 4763, 7250, 7666]
		stamps[('G_ME', 'LB', 'VB')] = [7551, 8478, 6468, 7427, 6953, 7537, 7445, 6611, 6891, 7620, 4359, 6014, 7271, 7150, 7311, 7585, 7050, 7524, 7250, 7525, 7722, 6848, 2802, 7234, 7033, 6291, 6716, 8382, 7568, 6633, 7432, 3477, 7306, 7320, 1039, 1604, 2194, 7594, 7711, 7181, 7760, 8121, 7426, 950, 1580, 2080, 2730, 3417, 4930, 6380, 7401, 6971, 6346, 7089, 7404, 7939, 7259, 2761, 563, 1214, 2647, 3531, 4048, 7163, 2286, 7811, 7121, 7513, 7400, 7650, 7967, 7434, 7403, 7775, 5637, 7648, 6361, 7994, 6768, 7128, 8241, 561, 7507, 6828, 7481, 1461, 3167, 7337, 7341, 7761, 3847, 7682, 6457, 7733, 7435, 6664, 7531, 7772, 6990, 7577, 6680, 7432, 7298, 1565, 2560, 4241, 5369, 7470, 7399, 7280, 7611, 7145, 7661, 2857, 3425, 7239, 7521, 7378, 7425, 7797, 7426, 7372, 6409, 7059, 7586, 7252, 7500, 9010, 7972, 7425, 7186, 7694, 1654, 2520, 6399, 6787, 2298, 8515, 6405, 6461, 7999, 1266, 2083, 4887, 6819, 7453, 10006, 835, 4474, 8037, 8074, 6566, 7531, 964, 4294, 7424, 8029, 7430, 8367, 6777, 7372, 7689, 7422, 7809, 7588, 7438, 7192, 7535, 3482, 5382, 6335, 6857, 7481, 771, 1139, 2088, 2671, 3271, 3938, 4637, 5388, 6122, 6422, 6871, 2395, 7233, 7717, 8181, 7421, 7500, 6603, 7637, 8498, 7391, 7984, 4713, 7113, 6317, 7623, 6797, 6601, 7268, 7064, 7087, 7892, 6645, 7427, 10538, 8073, 7041, 7683, 7601, 6819, 7218, 7372, 5241, 7449, 8062, 3426, 7476, 7494, 7331, 7287, 7404, 7150, 6842, 7625, 1004, 2337, 7294, 7303]
		stamps[('G_ME', 'LB', 'VC')] = [5837, 6920, 7075, 6359, 7326, 7018, 2179, 4207, 8194, 6717, 7472, 7668, 6320, 7438, 6052, 6784, 7243, 2438, 6891, 3312, 5570, 7179, 7130, 6627, 7394, 5994, 6975, 6299, 7409, 6396, 7229, 6475, 7089, 614, 897, 1264, 1549, 1848, 2231, 2530, 3613, 4180, 4764, 5197, 5764, 5736, 7295, 6157, 7913, 5730, 7096, 6457, 7404, 485, 6293, 8376, 5909, 7313, 3807, 6872, 7294, 7304, 7326, 3854, 5649, 6149, 7765, 8450, 6157, 7151, 9606, 4897, 7141, 7455, 7201, 1174, 3983, 6397, 6294, 8825, 4638, 7487, 7021, 7398, 958, 7290, 6513, 4455, 8106, 7764, 5677, 6212, 7313, 1508, 3838, 5532, 7082, 2471, 6522, 7438, 6123, 6392, 7615, 6300, 6008, 647, 913, 1063, 1230, 1342, 1563, 1879, 2012, 2195, 2345, 2545, 2678, 2894, 3211, 3344, 3577, 3910, 7099, 6973, 7294, 1550, 5530, 949, 6080, 6018, 3874, 5719, 7086, 6439, 429, 846, 1663, 2397, 2729, 3329, 4263, 4846, 10220, 3634, 6218, 7872, 7125, 5801, 7186, 10545, 1342, 3503, 4575, 5032, 5898, 8141, 8445, 7591, 5197, 7780, 8417, 6561, 7408, 9061, 7029, 7676, 7557, 8743, 5210, 8174, 5789, 8447, 1540, 2357, 3256, 4675, 5274, 6924, 7923, 8726, 6447, 7272]
		stamps[('G_ME', 'LB', 'VD')] = [6478, 5821, 7506, 348, 7243, 5919, 257, 430, 617, 751, 1124, 1498, 1885, 3927, 4141, 4288, 4915, 5075, 5771, 4323, 6318, 5383, 7108, 6123, 6391, 6382, 7232, 5838, 5990, 5644, 7143, 1913, 3482, 5013, 5814, 5947, 6114, 6214, 6380, 6699, 6868, 7047, 7097, 8629, 7296, 5960, 4931, 7235, 6320, 7039, 7600, 6402, 6248, 8025, 7735, 8495, 10398, 4647, 5573, 7207, 8024, 2816, 7036, 5718, 7172, 6325, 7008, 6048, 7315, 1830, 3013, 5196, 5203, 2734, 7269, 6450, 8471, 1529, 7379, 3893, 8682, 6427, 1376, 6722, 8481, 6969, 7372, 6414, 7228, 10347, 9409, 7618, 2370, 6472, 5841, 7128, 7096, 4506, 5525, 6792, 7052, 1079, 2967, 7247, 6447, 7218, 9210, 5164, 7315, 2143, 1213, 2095, 4559, 7379, 8571, 6002, 7077, 1344, 2510, 2085, 4541, 5224, 5240, 6172, 6759, 6148, 7315, 7866, 5673, 5567, 3535, 8373, 9947, 7727, 7288]
		stamps[('G_ME', 'LB', 'VE')] = [7877, 7393, 6848, 7365, 7054, 5034, 5530, 3879, 631, 3865, 6983, 5448, 6932, 6112, 6796, 2429, 6512, 7228, 1032, 7208, 1200, 7301, 5255, 6106, 6938, 7337, 6810, 755, 5099, 6548, 5800, 7217, 7214, 5658, 3577, 6577, 6554, 5988, 1104, 6737, 5547, 6986, 808, 7385, 555, 7081, 4528, 5932, 5776, 963, 4897, 7851, 7347, 6558, 8009, 4189, 6922, 1360, 4626, 5960, 6847, 2704, 7065, 6222, 6062, 7213, 8391, 2715, 7079, 6052, 6488, 1597, 6414, 7042, 6393, 6865, 4989, 8700, 513, 696, 1263, 2144, 2412, 2644, 4095, 4894, 976, 2304, 5890, 6220, 7237, 5362, 7095, 6176, 7121, 2657, 6111, 6810, 801, 5766, 4816, 6173, 1837, 3970, 6565, 5091, 6626, 1089, 2222, 7573, 6804, 4304, 4967, 5371, 7002, 7494, 6260, 5111, 7214, 4432, 5915, 7249, 6716, 3463, 3780, 4013, 4162, 4347, 4430, 4645, 6877, 6518, 6293, 1687, 7490, 4044, 8841, 7299, 7342, 6897, 2228, 7517, 4736, 7741, 8206, 7162, 1040, 1891, 3007, 4189, 6823, 7442, 7041]
		stamps[('G_ME', 'LC', None)] = [519, 3200, 5630, 7731, 7460, 7518, 7553, 703, 6772, 7487, 1277, 6837, 6512, 541, 7181, 6138, 8702, 7457, 7736, 7352, 6280, 7359, 8414, 6041, 7558, 6092, 7227, 6640, 6522, 9662, 1411, 7368, 7161, 4935, 7827, 7414, 8864, 1208, 7219, 6054, 2141, 5895, 7854, 4473, 5243, 7622, 6507, 7227, 7690, 2470, 5050, 8115, 4357, 7521, 1191, 7256, 7178, 7204, 7491, 3495, 4582, 6894, 6627, 7730, 7249, 7614, 7251, 5450, 4685, 6769, 7369, 7336, 6912, 7460, 7777, 8047, 7589, 5873, 7005, 5388, 5654, 6719, 7442, 5321, 6721, 2067, 2700, 6015, 7132, 7210, 5181, 7014, 7169, 3980, 8002, 8669, 6476, 6845, 999, 684, 7401, 7299, 6458, 7919, 5660, 7502, 5585, 5144, 6953, 7588, 5277, 6493, 1180, 2413, 5147, 2156, 3972, 4123, 4305, 4457, 4822, 4940, 5172, 6823, 7505, 7033, 7248, 2097, 7318, 7175, 6182, 6586, 1684, 1647, 7570, 7429, 5951, 6651, 7131, 7487, 6231, 7967, 2300, 7287, 5591, 6663, 7530, 7451, 7448, 6507, 7611, 7432, 7339, 3178, 4950, 7380, 4487, 7240, 2324, 7163, 420, 532, 739, 7857, 7934, 8968, 5208, 6110, 4635, 6313, 7031, 3299, 7284, 7538, 6768, 7201, 7377, 6823, 6097, 7379, 6530, 7311, 7753, 9007, 913, 526, 775, 1874, 2192, 2542, 3491, 4544, 5460, 6242, 6692, 7240, 1731, 2046, 2412, 989, 1640, 2073, 765, 2032, 2498, 3032, 3568, 4082, 4683, 5183, 5615, 6133, 6649, 7082, 7697, 8164, 8931, 9713, 10560, 7246, 7402, 5684, 6845, 6864, 7311, 3638, 6322, 7039, 6666, 7701, 5547, 3465, 5166, 7001, 7251, 6692, 7706, 9298, 9866, 8627, 7230, 4110, 7190, 7269, 7164, 6502, 6178, 5076, 6036, 7058, 9975, 6590, 7370, 5297, 6746, 7316, 7510, 6487, 7704, 4554, 6872, 7132, 5945, 1778, 4211, 8849, 7269, 7409, 6214, 7708, 7392, 581, 714, 1898, 2583, 5748, 7281, 928, 673, 467, 916, 1583, 3184, 4937, 7237, 7776, 5763, 7249, 6315, 7185, 7974, 7202, 7338, 8202, 7699, 8478, 7649, 7392, 7429, 1328, 7821, 6311, 4168, 7261, 7003, 7576, 1191, 1541, 1991, 2159, 2841, 3524, 4558, 5108, 5574, 6024, 6491, 7524, 1492, 2607, 3374, 4056, 4790, 5918, 1353, 2353, 6797, 2230, 8485, 7424, 5710, 7298, 5655, 7212, 7549, 8194, 7373, 7283, 7071, 7521, 3520, 8871, 8104, 7373, 9775, 7456, 7601, 7471, 7045, 7659, 6392, 7259, 7398, 8079, 5462, 7222, 7351, 6988, 7537, 6530, 2224, 7380, 7702, 1258, 7415, 2482, 1133, 5335, 994, 1865, 2795, 4372, 5522, 6593, 7398, 4048, 4814, 5931, 6981, 6994, 4869, 7017, 779, 6645, 5837, 7136, 7464, 6691, 7343, 7110, 6764, 7597, 7680, 7979, 7486, 7442, 8210, 2493, 8432, 5641, 6994, 6405, 7800, 7970, 8806, 6221, 7453, 7474, 7407, 6324, 7040, 1106, 2275, 7642, 6601, 7385, 5925, 7075, 4376, 6858, 7302, 6447, 7084, 7353, 8088, 9070, 6543, 7273, 7038, 7309, 6219, 7629, 8788, 4091, 9208, 3005, 7688, 2909, 8768, 7295, 7889, 7447, 7281, 7869, 8118, 6075, 7751, 6733, 6962, 9517, 8696, 7160, 7818, 6242, 7161, 6865, 6374, 7200, 6733, 981, 6758, 1377, 3474, 1605, 2986, 3653, 4134, 4733, 5383, 5510, 7544, 7259, 7235, 7331, 457, 8421, 7054, 4113, 7787, 5676, 7878, 5334, 7515, 6188, 7282, 6663, 7829, 7069, 4555, 7272, 5226, 6891, 7186, 7534, 5929, 5005, 6389, 6939, 7787, 7330, 8490, 8085, 989, 975, 7310, 1612, 3070, 7666, 7676, 5701, 7501, 6328, 7877, 8258, 7310, 8041, 1256, 5994, 3843, 3270, 3720, 4637, 7248, 7272, 1967, 6914, 6260, 5568, 7034, 7784, 6639, 7517, 7745, 7630, 4543, 1862, 3412, 5811, 6742, 7308, 1219, 1702, 2001, 2334, 2686, 3150, 3833, 4950, 6798, 6177, 7143, 3284, 4116, 4599, 4915, 5114, 5416, 5583, 5964, 6281, 6616, 6915, 7331, 7110, 8421, 7561, 5513, 7672, 7558, 8137, 4296, 4581, 7204, 7279, 6672, 7419, 7503, 7705, 7451, 5680, 6581, 7422, 5561, 7443, 4578, 7244, 6368, 6818, 7602, 7150, 7480, 9012, 6562, 7647, 6629, 3155, 7459, 955, 2038, 3105, 4222, 5304, 7139, 824, 1408, 1758, 2006, 2890, 3274, 4024, 4823, 582, 865, 7364, 5768, 6901, 2300, 7584, 4572, 7375, 5077, 7037, 6978, 5093, 7840, 7856, 4891, 1739, 6916, 7782, 6709, 6723, 5853, 7635, 6287, 7287, 7070, 7971, 6087, 7153, 7432, 7365, 7205, 6007, 7206, 6953, 8126, 8153, 7719, 7449, 7287, 680, 927, 1012, 1161, 1418, 2030, 2135, 2882, 3114, 3469, 3849, 4797, 5434, 5665, 6097, 6437, 6671, 6813, 7030, 7168, 6159, 4869, 6924, 7581, 7428, 4027, 5711, 6511, 7225, 7067, 7156, 7079, 7111, 7917, 8478, 8406, 6369, 7064, 6308, 7517, 7193, 7909, 7089, 4309, 4786, 5913, 7949, 7337, 8655, 2004, 8367, 3099, 4158, 7743, 6446, 7813, 7302, 7102, 7959, 5866, 7516, 6456, 7989, 7076, 8479, 5826, 7875, 6594, 7899, 7669, 1494, 6858, 6015, 5080, 6935, 7357, 7479, 7844, 1688, 6084, 7385, 5634, 7669, 5691, 8048, 6363, 7493, 6650, 7591, 5520, 5158, 6226, 7916, 8445, 7166, 6848, 7316, 6376, 7226, 6298, 6424, 1212, 2230, 1589, 2094, 3739, 4011, 4583, 7414, 1534, 1017, 2269, 3336, 4000, 4822, 7317, 7991, 7953, 6569, 7096, 7015]
		stamps[('G_ME', 'LC', 'VA')] = [7731, 7553, 541, 6138, 7352, 8414, 9662, 4935, 7827, 8864, 7219, 6507, 7227, 6894, 4685, 6769, 7369, 8047, 7442, 7210, 6845, 7502, 1180, 2413, 5147, 7175, 7570, 7131, 7487, 7451, 3178, 420, 532, 739, 7857, 7934, 8968, 3299, 7284, 7538, 989, 1640, 2073, 7311, 3465, 5166, 7001, 7251, 7706, 7190, 7164, 9975, 5297, 7392, 467, 916, 1583, 3184, 7202, 8478, 1191, 1541, 1991, 2159, 2841, 3524, 4558, 5108, 5574, 6024, 6491, 7524, 1492, 2607, 3374, 4056, 4790, 5918, 7373, 7283, 3520, 7398, 8079, 6988, 7537, 994, 1865, 2795, 4372, 5522, 6593, 7398, 6994, 7680, 7486, 6221, 7407, 1106, 2275, 6858, 8088, 9070, 7309, 2909, 7295, 7889, 8118, 9517, 7818, 7161, 7259, 6663, 7829, 7069, 975, 8258, 1256, 5994, 7784, 1219, 1702, 2001, 2334, 2686, 3150, 3833, 4950, 6798, 3284, 4116, 4599, 4915, 5114, 5416, 5583, 5964, 6281, 6616, 6915, 7331, 7558, 6672, 7419, 6368, 6818, 7480, 9012, 7459, 582, 865, 7364, 6978, 7840, 7782, 7432, 7719, 680, 927, 1012, 1161, 1418, 2030, 2135, 2882, 3114, 3469, 3849, 4797, 5434, 5665, 6097, 6437, 6671, 6813, 7030, 7168, 6511, 7917, 7193, 7909, 4309, 4786, 5913, 8367, 3099, 4158, 7743, 7102, 8479, 5080, 7357, 1688, 7669, 5158, 6226, 7166, 1017, 2269, 3336, 4000, 4822, 7317, 7953, 7015]
		stamps[('G_ME', 'LC', 'VB')] = [7460, 6837, 7181, 7457, 6280, 6640, 1411, 7368, 7414, 6054, 7690, 7521, 7204, 6627, 7730, 7249, 7614, 6912, 7460, 7777, 7589, 6721, 7169, 8002, 8669, 684, 7401, 6953, 7588, 2156, 3972, 4123, 4305, 4457, 4822, 4940, 5172, 6823, 7505, 7033, 7429, 5951, 6651, 7967, 6663, 7530, 6507, 7611, 7432, 6768, 7201, 6530, 7311, 7753, 526, 775, 1874, 2192, 2542, 3491, 4544, 5460, 6242, 6692, 7240, 10560, 7246, 7402, 6845, 7701, 6692, 9298, 9866, 7230, 7269, 6178, 6590, 7316, 7510, 6487, 7704, 7269, 6214, 7708, 673, 6315, 7185, 7338, 7699, 7392, 7821, 7576, 7549, 8194, 7071, 7521, 8104, 7373, 7471, 7045, 7659, 7351, 2224, 7380, 7702, 4869, 7017, 7464, 6764, 7597, 8210, 6994, 7474, 7642, 6601, 7385, 7302, 7353, 7273, 4091, 9208, 7447, 7281, 7751, 6865, 981, 6758, 7235, 8421, 5676, 7878, 6188, 7282, 7272, 7534, 7330, 8085, 3070, 7666, 7676, 6328, 7877, 3843, 6914, 7517, 7630, 1862, 3412, 5811, 6742, 7308, 7110, 7672, 8137, 4581, 7503, 6581, 7422, 7602, 6562, 6629, 955, 2038, 3105, 4222, 5304, 7139, 7037, 4891, 7070, 7971, 6953, 6924, 7428, 7225, 7156, 8406, 7064, 7517, 7089, 8655, 6446, 7813, 7302, 6456, 7076, 7899, 7669, 6858, 7479, 7844, 8048, 6650, 7591, 7916, 6848, 7316, 6376, 7226, 6424, 4011, 4583, 7414, 6569, 7096]
		stamps[('G_ME', 'LC', 'VC')] = [5630, 7518, 703, 6772, 7487, 2470, 5050, 8115, 7178, 7491, 7251, 5873, 7005, 2067, 2700, 6015, 7132, 6476, 999, 7919, 5585, 7248, 2097, 7318, 1684, 6231, 2300, 7287, 7339, 7380, 6313, 7031, 6097, 7379, 1731, 2046, 2412, 6864, 6502, 6746, 5945, 1778, 4211, 8849, 7409, 581, 714, 1898, 2583, 5748, 7281, 7776, 8202, 6311, 7261, 7003, 7424, 5710, 7298, 7456, 6392, 7259, 7222, 1258, 7415, 6691, 7343, 7442, 8432, 7800, 5925, 7075, 6447, 7084, 3005, 7688, 6075, 6374, 7200, 1605, 2986, 3653, 4134, 4733, 5383, 5510, 7544, 7054, 5334, 7515, 7787, 8490, 989, 1612, 7248, 4543, 6177, 7143, 7561, 7279, 7705, 7451, 5680, 7647, 5768, 6901, 2300, 7584, 1739, 6916, 6287, 7287, 6007, 7206, 8153, 7287, 6159, 7581, 7079, 7337, 5866, 7516, 1494, 6935, 6084, 5691, 6363, 7493, 6298, 1534]
		stamps[('G_ME', 'LC', 'VD')] = [519, 3200, 6041, 7558, 6092, 7227, 7161, 1208, 4473, 5243, 7622, 4357, 1191, 7256, 3495, 4582, 5450, 7336, 5388, 5654, 6719, 5321, 5181, 7014, 6458, 5660, 6586, 1647, 5591, 7448, 4950, 2324, 7163, 6110, 6823, 9007, 765, 2032, 2498, 3032, 3568, 4082, 4683, 5183, 5615, 6133, 6649, 7082, 7697, 8164, 8931, 9713, 5684, 5076, 6036, 7058, 4554, 6872, 7974, 1328, 1353, 2353, 6797, 5655, 7212, 8871, 9775, 5462, 2482, 4048, 7110, 7979, 2493, 5641, 6405, 7629, 8768, 6733, 7160, 1377, 3474, 4113, 7787, 4555, 5929, 7310, 5701, 7501, 7310, 3270, 3720, 4637, 1967, 5568, 7034, 7745, 8421, 7204, 5561, 7443, 7150, 3155, 5077, 5093, 6723, 6087, 7153, 7365, 4869, 6369, 7949, 7959, 7989, 5826, 7875, 6594, 6015, 5634, 5520, 1589, 2094, 3739, 7991]
		stamps[('G_ME', 'LC', 'VE')] = [1277, 6512, 8702, 7736, 7359, 6522, 2141, 5895, 7854, 3980, 7299, 5144, 5277, 6493, 6182, 4487, 7240, 5208, 4635, 7377, 913, 3638, 6322, 7039, 6666, 5547, 8627, 4110, 7370, 7132, 928, 4937, 7237, 5763, 7249, 7649, 7429, 4168, 2230, 8485, 7601, 6530, 1133, 5335, 4814, 5931, 6981, 779, 6645, 5837, 7136, 7970, 8806, 7453, 6324, 7040, 4376, 6543, 7038, 6219, 8788, 7869, 6962, 8696, 6242, 6733, 7331, 457, 5226, 6891, 7186, 5005, 6389, 6939, 8041, 7272, 6260, 6639, 5513, 4296, 4578, 7244, 824, 1408, 1758, 2006, 2890, 3274, 4024, 4823, 4572, 7375, 7856, 6709, 5853, 7635, 7205, 8126, 7449, 4027, 5711, 7067, 7111, 8478, 6308, 2004, 7385, 8445, 1212, 2230]
		stamps[('G_ME', 'LD', None)] = [9559, 7474, 7287, 7434, 595, 7557, 8171, 7633, 7014, 7898, 5916, 6351, 7436, 3844, 4240, 7830, 8017, 1168, 7831, 1367, 1423, 7539, 7700, 7788, 7389, 3960, 5636, 5786, 1031, 4564, 3274, 1706, 5532, 8160, 376, 670, 950, 1070, 1257, 1497, 1657, 1764, 1884, 2044, 2138, 2365, 2631, 2925, 3219, 3539, 4140, 4447, 5047, 5127, 5368, 5688, 5781, 6035, 268, 454, 628, 855, 1002, 1242, 1549, 1749, 1896, 2096, 2203, 2430, 2523, 2763, 3097, 3391, 3551, 3724, 3858, 4125, 4405, 4579, 4739, 4925, 5059, 5273, 5419, 5606, 5740, 5927, 6220, 6300, 6514, 6621, 6807, 6968, 7128, 7288, 7435, 1101, 1248, 278, 7753, 7400, 4978, 6922, 1805, 7703, 4151, 4788, 7318, 5296, 5924, 7181, 7918, 4821, 7896, 472, 8652, 9030, 6206, 7442, 7145, 7805, 6767, 5301, 6454, 6212, 7928, 8245, 7909, 7152, 7070, 6691, 6298, 5675, 6561, 7830, 5753, 6854, 7383, 7345, 2106, 1465, 4119, 3894, 6148, 7128, 2611, 3361, 3761, 5572, 6338, 894, 884, 4792, 1526, 5759, 6241, 5682, 1363, 5793, 7540, 2505, 7189, 2577, 3460, 6161, 6794, 1665, 6419, 2783, 8077, 7008, 7356, 8460, 7552, 5779, 7391, 2980, 7964, 1673, 5855, 6696, 7307, 4849, 9898, 1496, 3693, 5226, 8109, 967, 1763, 7391, 7062, 7377, 7588, 7655, 7855, 4759, 6969, 2091, 4392, 6808, 8840, 3334, 4068, 8934, 10134, 2584, 3588, 7553, 7607, 7171, 7252, 518, 833, 1067, 1518, 2952, 695, 7570, 7681, 7996, 5653, 7945, 7408, 7428, 7207, 8020, 7552, 8051, 5378, 6931, 7279, 6798, 2519, 5115, 7345, 6060, 6836, 7204, 10019, 8418, 9938, 5674, 6989, 7517, 7390, 6022, 7315, 8142, 1833, 2568, 7783, 7433, 4479, 8373, 6286, 7295, 1077, 2450, 8517, 8634, 7116, 7998, 7652, 7431, 4471, 6027, 1426, 2126, 3059, 3510, 3910, 7177, 924, 1940, 4981, 5799, 8174, 6303, 7242, 8226, 7532, 7509, 506, 725, 1125, 1337, 1491, 2270, 2834, 3033, 3420, 3617, 3791, 3981, 4194, 4579, 4757, 5188, 5389, 5553, 5751, 5955, 6172, 6535, 6718, 6940, 7135, 7515, 7719, 7904, 8119, 8300, 8521, 8717, 8932, 9170, 9317, 9606, 10267, 6777, 7253, 7601, 1133, 2089, 6737, 1270, 1947, 2552, 3374, 5075, 1381, 2857, 3402, 4504, 5432, 5711, 6410, 7362, 1329, 2400, 4399, 6158, 2101, 7267, 6577, 1661, 7230, 6294, 8327, 6595, 7430, 7682, 6458, 7191, 8793, 7540, 7121, 8275, 5349, 7063, 5623, 7274, 6793, 6265, 1585, 7573, 7153, 7414, 7986, 7185, 7278, 6253, 7033, 4672, 5611, 6511, 10048, 4353, 3802, 8597, 8318, 7151, 8884, 7645, 7763, 7735, 2314, 3230, 8429, 7304, 5088, 6188, 1033, 2415, 3965, 5628, 1694, 2377, 3193, 3875, 4642, 1001, 2199, 6095, 7077, 8508, 1478, 5749, 1130, 3181, 6880, 7595, 1473, 4056, 6708, 7439, 2526, 767, 3118, 7302, 3088, 4052, 5871, 519, 969, 1286, 2220, 7422, 7363, 6037, 6821, 6708, 7275, 5508, 6326, 4358, 765, 2096, 996, 7185, 5703, 7501, 793, 6415, 2389, 8344, 8212, 9265, 8088, 7412, 7842, 7053, 593, 775, 1125, 1592, 683, 6095, 6926, 6126, 6510, 6727, 8329, 10109, 5205, 8755, 2902, 5761, 5964, 4117, 7183, 4104, 7003, 7476, 926, 1909, 4457, 2136, 2903, 7467, 1117, 1767, 2459, 3361, 3908, 4658, 7954, 7283, 5341, 5764, 5277, 7976, 8259, 1537, 5520, 3006, 1594, 700, 4578, 8575, 2598, 5367, 6218, 7492, 6349, 5962, 7002, 6405, 9445, 5458, 880, 1880, 2864, 3766, 4729, 5447, 7096, 1644, 2611, 3828, 4811, 5178, 5677, 428, 770, 1070, 1369, 1704, 2153, 2452, 3069, 4236, 458, 1276, 2258, 5149, 8716, 7577, 936, 2130, 1124, 3008, 4387, 4272, 5738, 6522, 8522, 9889, 5024, 4712, 4295, 3010, 5669, 8071, 7598, 7737, 5999, 8210, 7323, 8087, 7315, 1449, 8937, 7683, 7760, 3803, 7048, 7865, 1411, 1932, 5666, 5802, 7679, 7909, 6423, 8384, 7972, 7246, 8343, 9345, 7367, 594, 7777, 7000, 3178, 3906, 1766, 8909, 6811, 8135, 8285, 7629, 5488, 9755, 7219, 8266, 8486, 4588, 3793, 8086, 3737, 4472, 5054, 6085, 6227, 9906, 7562, 887, 7493, 8248, 7942, 6416, 3663, 5584, 7424, 7468, 2865, 5950, 465, 3223, 7014, 7844, 9211, 7500, 6804, 9038, 5851, 6701, 7584, 8285, 7911, 7298, 7260, 5604, 7879, 2667, 4184, 1376, 7126, 3624, 7786, 7731]
		stamps[('G_ME', 'LD', 'VA')] = [7474, 8171, 3844, 4240, 7830, 7831, 1367, 1706, 5532, 8160, 1101, 1248, 7400, 1805, 7703, 7918, 7896, 9030, 7909, 7830, 7128, 884, 4792, 2577, 3460, 6161, 6794, 8077, 2980, 7964, 4849, 9898, 1763, 7588, 7855, 7607, 7681, 7996, 8020, 8051, 8418, 7517, 7433, 1426, 2126, 3059, 3510, 3910, 7177, 4981, 5799, 7532, 7601, 2101, 7267, 8793, 1585, 7573, 7986, 10048, 7304, 5088, 6188, 1130, 3181, 6880, 7595, 7422, 2096, 793, 775, 1125, 1592, 4117, 7183, 2136, 2903, 7467, 7976, 700, 8575, 1644, 2611, 3828, 4811, 5178, 5677, 4272, 5738, 6522, 8522, 9889, 4712, 8071, 7737, 8210, 8087, 7760, 7909, 7972, 7777, 3906, 1766, 8909, 8285, 3793, 8086, 8248, 7424, 3223, 5851, 6701, 7584, 8285]
		stamps[('G_ME', 'LD', 'VB')] = [9559, 7633, 7014, 7898, 1031, 4564, 4151, 4788, 7318, 7145, 7928, 8245, 7345, 894, 2505, 7189, 1496, 3693, 5226, 8109, 7655, 2584, 3588, 7553, 518, 833, 1067, 1518, 2952, 7428, 7207, 7552, 6798, 6836, 7204, 5674, 6989, 7390, 8142, 7652, 7431, 1940, 8174, 8226, 506, 725, 1125, 1337, 1491, 2270, 2834, 3033, 3420, 3617, 3791, 3981, 4194, 4579, 4757, 5188, 5389, 5553, 5751, 5955, 6172, 6535, 6718, 6940, 7135, 7515, 7719, 7904, 8119, 8300, 8521, 8717, 8932, 9170, 9317, 9606, 10267, 1329, 2400, 4399, 6577, 8327, 7682, 8275, 7063, 7153, 7414, 7151, 7763, 3230, 8429, 1033, 2415, 3965, 5628, 767, 3118, 3088, 7363, 6708, 7275, 765, 8212, 8755, 4104, 7003, 7476, 1117, 1767, 2459, 3361, 3908, 4658, 7954, 3006, 4578, 5367, 6218, 7002, 880, 1880, 2864, 3766, 4729, 5447, 7096, 7577, 3010, 7323, 1449, 8937, 7683, 7865, 1932, 5666, 8384, 8343, 7367, 594, 3178, 8135, 7629, 7562, 887, 7493, 6416, 2865, 5950, 7844, 9211, 7879, 1376, 7126, 7731]
		stamps[('G_ME', 'LD', 'VC')] = [595, 7557, 6351, 7436, 1423, 7539, 5636, 5786, 278, 7753, 5924, 7181, 472, 6206, 7805, 6767, 6212, 7152, 6691, 6561, 7383, 2106, 2611, 3361, 3761, 5682, 1665, 6419, 2783, 5779, 7391, 1673, 5855, 6696, 7307, 7391, 7377, 2091, 4392, 6808, 8840, 7171, 695, 5653, 7945, 2519, 5115, 7345, 10019, 7315, 1833, 2568, 7783, 1077, 2450, 8517, 8634, 7116, 7998, 6303, 7242, 7509, 6777, 7253, 1381, 2857, 3402, 4504, 5432, 5711, 6410, 7362, 6158, 6294, 6458, 7191, 7540, 5623, 6253, 7033, 6511, 8318, 7645, 2314, 1694, 2377, 3193, 3875, 4642, 1473, 4056, 6708, 7439, 519, 969, 1286, 2220, 6037, 6821, 4358, 996, 2389, 8344, 7842, 6126, 6510, 6727, 8329, 10109, 2902, 5964, 7283, 1537, 5520, 6349, 5458, 1124, 3008, 4387, 5669, 5488, 9755, 7942, 3663, 5584, 465, 7500, 6804, 9038, 2667, 4184]
		stamps[('G_ME', 'LD', 'VD')] = [7434, 8017, 376, 670, 950, 1070, 1257, 1497, 1657, 1764, 1884, 2044, 2138, 2365, 2631, 2925, 3219, 3539, 4140, 4447, 5047, 5127, 5368, 5688, 5781, 6035, 5296, 4821, 8652, 7070, 5675, 5753, 6854, 3894, 6148, 5572, 6338, 5793, 8460, 967, 7062, 3334, 4068, 8934, 10134, 7570, 7408, 6060, 9938, 6022, 7295, 6027, 1133, 2089, 6737, 7274, 7185, 4672, 3802, 7735, 1001, 2199, 6095, 7077, 2526, 7302, 5508, 6326, 7185, 6415, 7412, 683, 6095, 6926, 926, 1909, 4457, 5277, 7492, 5962, 9445, 458, 1276, 2258, 5149, 8716, 5024, 7048, 7246, 9345, 7000, 4588, 3737, 4472, 5054, 6085, 6227, 7014, 7911, 7786]
		stamps[('G_ME', 'LD', 'VE')] = [7287, 5916, 1168, 7700, 7788, 7389, 3960, 3274, 268, 454, 628, 855, 1002, 1242, 1549, 1749, 1896, 2096, 2203, 2430, 2523, 2763, 3097, 3391, 3551, 3724, 3858, 4125, 4405, 4579, 4739, 4925, 5059, 5273, 5419, 5606, 5740, 5927, 6220, 6300, 6514, 6621, 6807, 6968, 7128, 7288, 7435, 4978, 6922, 7442, 5301, 6454, 6298, 1465, 4119, 1526, 5759, 6241, 1363, 7540, 7008, 7356, 7552, 4759, 6969, 7252, 5378, 6931, 7279, 4479, 8373, 6286, 4471, 924, 1270, 1947, 2552, 3374, 5075, 1661, 7230, 6595, 7430, 7121, 5349, 6793, 6265, 7278, 5611, 4353, 8597, 8884, 8508, 1478, 5749, 4052, 5871, 5703, 7501, 9265, 8088, 7053, 593, 5205, 5761, 5341, 5764, 8259, 1594, 2598, 6405, 428, 770, 1070, 1369, 1704, 2153, 2452, 3069, 4236, 936, 2130, 4295, 7598, 5999, 7315, 3803, 1411, 5802, 7679, 6423, 6811, 7219, 8266, 8486, 9906, 7468, 7298, 7260, 5604, 3624]
		stamps[('G_ME', 'LE', None)] = [6351, 1786, 6475, 4749, 6000, 6048, 8665, 6374, 8832, 2075, 4130, 7213, 505, 8817, 6124, 5185, 5639, 6268, 907, 933, 4822, 3520, 4586, 6320, 7184, 6972, 5744, 7427, 7194, 8445, 7950, 4419, 6435, 2689, 6316, 8362, 3810, 7831, 1353, 7887, 7508, 7543, 5760, 7789, 4686, 482, 699, 1098, 2832, 5366, 549, 629, 8752, 6688, 4252, 4124, 5780, 6642, 5824, 7966, 8199, 8500, 4142, 4779, 7739, 5157, 6991, 6041, 6823, 2912, 8133, 4942, 7201, 10484, 5919, 7118, 8031, 3686, 5866, 6964, 485, 463, 469, 7517, 3292, 4626, 3195, 7663, 6515, 3506, 8515, 3396, 7411, 6459, 806, 7303, 1337, 10587, 1075, 5707, 8762, 7746, 7831, 8618, 7192, 1797, 7317, 7367, 6330, 7114, 5997, 3792, 6644, 6643, 4731, 5027, 5471, 5991, 3992, 3418, 4983, 3180, 7587, 6750, 6572, 2922, 4105, 5421, 3041, 3623, 4207, 4658, 5041, 5543, 6740, 9293, 6069, 6188, 9615, 4267, 1394, 5993, 6765, 8246, 4893, 4796, 6420, 7806, 4107, 5694, 5267, 5973, 6736, 7385, 7819, 5589, 4736, 7100, 7455, 8161, 668, 1617, 7835, 347, 713, 913, 5979, 4657, 5990, 7943, 2476, 4933, 889, 3213, 5780, 8964, 6245, 5043, 7339, 8318, 6919, 609, 2412, 7094, 2824, 5671, 7218, 7430, 6774, 7352, 4365, 3969, 4719, 6119, 6268, 6435, 6652, 6919, 7234, 7386, 7569, 7867, 5997, 8041, 6357, 8250, 5853, 6936, 6848, 7236, 5974, 4565, 7548, 6991, 7582, 7114, 7901, 1064, 3271, 4251, 5147, 5612, 1615, 3994, 1228, 3583, 5002, 6582, 8011, 4021, 10069, 3684, 5250, 5285, 7074, 5590, 7329, 8504, 6271, 7995, 6714, 7607, 6306, 8202, 4651, 5876, 7561, 4863, 7216, 7680, 6610, 4544, 6242, 1569, 3001, 3416, 2328, 3859, 4760, 1215, 2382, 3263, 7775, 576, 877, 1027, 1376, 1626, 2874, 3326, 3490, 4790, 5306, 6557, 837, 1573, 2336, 2702, 3670, 4135, 4951, 5432, 1216, 8250, 1558, 4324, 3507, 1354, 2451, 7423, 6846, 7391, 6626, 8516, 8527, 6748, 8097, 1201, 3449, 7028, 823, 3487, 5070, 3390, 5458, 9264, 6479, 7869, 5485, 7212, 1310, 4895, 6432, 5591, 6989, 6258, 953, 1955, 3653, 4419, 5484, 6267, 1323, 2088, 2505, 3371, 4504, 6469, 2632, 2965, 3148, 3267, 3498, 3681, 3865, 4032, 4250, 4402, 4565, 4749, 4914, 5100, 5281, 5531, 5968, 6115, 6383, 6602, 6833, 7064, 7249, 7716, 6346, 4229, 5730, 8011, 2238, 6213, 5205, 6270, 6588, 4705, 6039, 6856, 5706, 7419, 5862, 7747, 7678, 6117, 6237, 769, 1119, 1884, 2635, 3769, 4136, 6334, 7018, 302, 718, 1319, 1712, 5197, 7696, 5764, 8050, 1652, 5492, 2097, 7796, 3296, 5264, 7380, 7388, 7903, 6863, 7699, 8433, 4913, 7152, 6044, 5629, 9502, 4938, 5886, 4638, 6469, 6278, 8553, 3116, 1955, 2407, 2723, 4010, 4411, 4912, 8018, 8290, 8664, 287, 440, 607, 720, 935, 1569, 1798, 1873, 2317, 2568, 2976, 3300, 3697, 3775, 3988, 4182, 4738, 5009, 5156, 5638, 5769, 6254, 6466, 7002, 7098, 7856, 6169, 4647, 5399, 954, 7827, 3947, 6052, 2169, 7466, 7422, 7734, 9484, 6481, 5357, 4823, 957, 1018, 8721, 8494, 7662, 1349, 1597, 6575, 6202, 6956, 7502, 9494, 7553, 5463, 6654, 8072, 7748, 8875, 7946, 8630, 3698, 7736, 8423, 7617, 2466, 1332, 3670, 1201, 1047, 2714, 4246, 5515, 7630, 6770, 7212]
		stamps[('G_ME', 'LE', 'VA')] = [1786, 8665, 6972, 8445, 7950, 6435, 8362, 7831, 7543, 7789, 7966, 8199, 8500, 8133, 4942, 7201, 10484, 3195, 7663, 8515, 1075, 5707, 8618, 6330, 7114, 8246, 7806, 5973, 6736, 7385, 7819, 8161, 668, 1617, 7835, 7943, 8964, 8318, 7430, 3969, 4719, 6119, 6268, 6435, 6652, 6919, 7234, 7386, 7569, 7867, 8041, 6357, 8250, 6848, 7582, 7901, 6582, 8011, 3684, 5250, 8202, 1215, 2382, 3263, 7775, 3507, 8516, 8527, 1201, 3449, 7028, 8011, 6856, 7747, 1712, 5197, 7696, 5492, 8433, 9502, 6278, 8553, 8018, 8290, 8664, 7856, 9484, 6481, 1018, 8721, 1349, 7553, 7748, 8875, 3698, 8423, 1047, 2714, 4246, 5515, 7630]
		stamps[('G_ME', 'LE', 'VB')] = [6374, 8832, 5744, 7427, 7194, 7508, 482, 699, 1098, 2832, 5366, 7739, 6041, 6823, 8031, 6964, 469, 806, 7303, 7746, 1797, 7317, 7367, 5997, 7587, 4736, 7100, 7352, 5853, 6936, 5974, 6991, 7114, 1228, 3583, 4021, 10069, 7329, 7607, 7216, 576, 877, 1027, 1376, 1626, 2874, 3326, 3490, 4790, 5306, 6557, 1354, 7423, 7391, 9264, 7869, 2632, 2965, 3148, 3267, 3498, 3681, 3865, 4032, 4250, 4402, 4565, 4749, 4914, 5100, 5281, 5531, 5968, 6115, 6383, 6602, 6833, 7064, 7249, 5205, 7678, 769, 1119, 1884, 2635, 3769, 4136, 6334, 7018, 2097, 7796, 7380, 287, 440, 607, 720, 935, 1569, 1798, 1873, 2317, 2568, 2976, 3300, 3697, 3775, 3988, 4182, 4738, 5009, 5156, 5638, 5769, 6254, 6466, 7002, 6169, 4647, 5399, 7827, 3947, 7662, 6654, 7946, 8630, 7617, 1332, 3670]
		stamps[('G_ME', 'LE', 'VC')] = [2075, 4130, 7213, 8817, 933, 2689, 6316, 1353, 7887, 4686, 549, 6642, 5824, 2912, 485, 7411, 1337, 10587, 8762, 7192, 6643, 4731, 5027, 5471, 5991, 6572, 2922, 4105, 5421, 6188, 6765, 4107, 5694, 5589, 347, 713, 913, 889, 3213, 5780, 6919, 7995, 4863, 7680, 837, 1573, 2336, 2702, 3670, 4135, 4951, 5432, 6846, 6748, 823, 3487, 5070, 5485, 1310, 6432, 6258, 7716, 6346, 6588, 5862, 1652, 7903, 6044, 3116, 1955, 2407, 2723, 4010, 4411, 4912, 954, 6052, 7422, 7734, 8494, 9494, 5463, 7212]
		stamps[('G_ME', 'LE', 'VD')] = [6475, 6048, 907, 4822, 3520, 4586, 6320, 3810, 629, 4252, 4124, 5780, 4142, 5157, 6991, 5919, 7118, 5866, 463, 7517, 6515, 7831, 3792, 6644, 3992, 3418, 3180, 3041, 3623, 4207, 4658, 5041, 5543, 6740, 9293, 9615, 4893, 5043, 2412, 7094, 2824, 5671, 7218, 6774, 7236, 4565, 1615, 3994, 5002, 8504, 6271, 6714, 7561, 4544, 2328, 3859, 4760, 1558, 4324, 2451, 7212, 4895, 5591, 6989, 953, 1955, 3653, 4419, 5484, 6267, 4229, 5730, 6270, 4705, 6039, 5706, 7419, 6117, 302, 718, 1319, 3296, 7388, 5629, 4638, 6469, 7098, 5357, 4823, 6202, 6956, 7736, 2466]
		stamps[('G_ME', 'LE', 'VE')] = [6351, 4749, 6000, 505, 6124, 5185, 5639, 6268, 7184, 4419, 5760, 8752, 6688, 4779, 3686, 3292, 4626, 3506, 3396, 6459, 4983, 6750, 6069, 4267, 1394, 5993, 4796, 6420, 5267, 7455, 5979, 4657, 5990, 2476, 4933, 6245, 7339, 609, 4365, 5997, 7548, 1064, 3271, 4251, 5147, 5612, 5285, 7074, 5590, 6306, 4651, 5876, 6610, 6242, 1569, 3001, 3416, 1216, 8250, 6626, 8097, 3390, 5458, 6479, 1323, 2088, 2505, 3371, 4504, 6469, 2238, 6213, 6237, 5764, 8050, 5264, 6863, 7699, 4913, 7152, 4938, 5886, 2169, 7466, 957, 1597, 6575, 7502, 8072, 1201, 6770]
		stamps[('G_ME', 'LO', None)] = [898, 8400, 6666, 10234, 769, 8390, 7908, 9517, 4319, 9521, 5182, 6472, 313, 486, 606, 793, 1354, 1434, 1674, 1954, 2235, 2355, 2608, 2942, 3276, 3609, 3716, 3970, 4197, 4557, 5585, 5732, 5892, 6065, 6185, 6386, 6546, 6652, 6879, 7000, 7173, 7333, 7520, 330, 490, 583, 770, 944, 1037, 1291, 1571, 1865, 2131, 2305, 2438, 2599, 2732, 2892, 3039, 3199, 3306, 3533, 3653, 3853, 4013, 4187, 4347, 4547, 4654, 4881, 5215, 5548, 5842, 6176, 1026, 5044, 9295, 9638, 8594, 7461, 2485, 6174, 2976, 8337, 8153, 626, 9291, 8614, 2608, 2219, 8584, 4482, 8895, 8795, 9077, 8560, 9210, 7244, 8513, 8055, 3183, 9132, 6916, 551, 3617, 547, 743, 7974, 5895, 6678, 965, 8355, 5659, 9538, 4533, 9251, 864, 3859, 1385, 9068, 1001, 5084, 7294, 1022, 461, 1307, 8541, 2447, 9046, 8837, 3814, 8682, 9223, 658, 1208, 1924, 2879, 3241, 3474, 3728, 3928, 4408, 5912, 7091, 6813, 7462, 5849, 8628, 9220, 9194, 8262, 6173, 8814, 2104, 2738, 7405, 10020, 8722, 8385, 8465, 4302, 10179, 8866, 512, 1263, 2062, 9862, 10062, 7771, 5875, 8504, 2148, 2333, 2481, 2680, 2832, 5654, 9612, 5072, 7689, 8318, 3233, 7361, 9043, 9632, 8666, 8602, 9381, 1069, 7864, 6351, 7534, 2866, 9083, 643, 1478, 600, 954, 4831, 556, 911, 2844, 5211, 4141, 8785, 5004, 8770, 9379, 8772, 5575, 2097, 1942, 2459, 550, 3450, 3254, 8400, 7503, 4796, 521, 689, 2355, 2571, 3326, 1925, 5394, 8929, 6063, 6381, 7211, 8010, 8649, 1534, 9266, 8904, 7921, 3043, 10530, 9070, 444, 645, 1133, 1472, 1699, 1931, 2349, 2531, 2764, 2965, 3201, 3446, 3846, 4280, 5003, 5602, 6264, 6832, 7756, 8180, 8716, 9301, 5019, 6914, 3860, 4794, 1221, 2773, 3956, 5651, 2032, 854, 2082, 2934, 5161, 6461, 7335, 7842, 6278, 8801, 1363, 9072, 8314, 9648, 8561, 7352, 1437, 3738, 2676, 4030, 8590, 8228, 8802, 8686, 9906, 8540, 8093, 3657, 5605, 8157, 9199, 6480, 9237, 7965, 7528, 8817, 2880, 1260, 969, 924, 2821, 3670, 4704, 5335, 6302, 6951, 1154, 3684, 6949, 2210, 1462, 3778, 6080, 6597, 9582, 9156, 9774, 9400, 7764, 4893, 2464, 7988, 7001, 8318, 3634, 8832, 5304, 933, 1416, 1775, 2809, 3874, 5273, 5922, 7271, 7820, 8653, 5088, 5685, 9018, 1834, 2567, 8583, 5524, 6821, 7389, 1067, 5479, 8031, 868, 902, 7803, 9446, 8848, 439, 1039, 1705, 2853, 8633, 957, 2106, 4254, 5786, 8485, 532, 1432, 2048, 3646, 5445, 6276, 6726, 1064, 1829, 2360, 2580, 2730, 2963, 3129, 3297, 3462, 3663, 3997, 4482, 4622, 4813, 4947, 5129, 5279, 2291, 5056, 8564, 10417, 8166, 6433, 7641, 2352, 3630, 9185, 1500, 6936, 8645, 7565, 8887, 9954, 739, 5856, 6656, 7507, 8123, 299, 982, 1482, 2032, 4699, 5415, 6200, 6898, 7465, 8116, 8566, 4268, 8535, 3967, 6932, 8716, 9617, 2916, 1499, 5610, 6626, 8546, 10159, 3138, 4732, 8813, 9144, 1816, 2933, 2569, 9053, 9306, 9191, 3203, 844, 9379, 2547, 4571, 9044, 9632, 9304, 5846, 9729, 8458, 4524, 6807, 7451, 5378, 6678, 4680, 8505, 1097, 1863, 7710, 9438, 3369, 5926, 8972, 10109, 8637, 9003, 6730, 8216, 8862, 7033, 7865, 8482, 8600, 7694, 4105, 1938, 4348, 8745, 9038, 2410, 2639, 3216, 4211, 4512, 4923, 5724, 9603, 8581, 9200, 8196, 5216, 1209, 4709, 984, 1883, 2516, 9236, 9099, 9283]
		stamps[('G_ME', 'LO', 'VA')] = [9521, 9295, 9638, 8594, 8153, 8895, 8795, 9210, 8513, 3183, 965, 8355, 9538, 1022, 658, 1208, 1924, 2879, 3241, 3474, 3728, 3928, 4408, 5912, 7091, 9220, 6173, 8814, 2104, 2738, 7405, 10020, 4302, 8866, 9612, 5072, 8318, 9043, 9632, 8602, 7864, 2866, 9083, 8770, 9379, 2097, 7503, 645, 1133, 1472, 1699, 1931, 2349, 2531, 2764, 2965, 3201, 3446, 3846, 4280, 5003, 5602, 6264, 6832, 7756, 8180, 8716, 9301, 9648, 8561, 9906, 8817, 3778, 6080, 9582, 9400, 2464, 5685, 9018, 1067, 9446, 439, 1039, 1705, 2853, 8633, 10417, 6433, 8887, 9954, 739, 5856, 6656, 7507, 8123, 3967, 6932, 8716, 9617, 9144, 9053, 9306, 9191, 9044, 9632, 9304, 8972, 6730, 8216, 8862, 8600, 2410, 2639, 3216, 4211, 4512, 4923, 9603, 5216, 9099, 9283]
		stamps[('G_ME', 'LO', 'VB')] = [8400, 10234, 769, 8390, 9517, 1026, 5044, 2485, 6174, 626, 8584, 9077, 8055, 9132, 551, 3617, 7974, 5895, 6678, 1385, 9068, 1307, 8541, 8682, 9194, 8385, 512, 1263, 2062, 9862, 10062, 8504, 7361, 1069, 556, 5004, 8772, 8400, 6063, 6381, 7211, 8010, 8649, 9266, 10530, 3956, 5651, 8801, 9072, 8686, 8093, 9199, 924, 2821, 3670, 4704, 5335, 6302, 6951, 6597, 9156, 9774, 4893, 8832, 933, 1416, 1775, 2809, 3874, 5273, 5922, 7271, 7820, 8653, 1834, 2567, 8583, 868, 8848, 957, 2106, 4254, 5786, 8485, 8564, 8166, 8645, 299, 982, 1482, 2032, 4699, 5415, 6200, 6898, 7465, 8116, 8566, 3138, 844, 9379, 8505, 9438, 7694, 4348, 8745, 9038, 9200, 8196]
		stamps[('G_ME', 'LO', 'VC')] = [898, 7908, 313, 486, 606, 793, 1354, 1434, 1674, 1954, 2235, 2355, 2608, 2942, 3276, 3609, 3716, 3970, 4197, 4557, 5585, 5732, 5892, 6065, 6185, 6386, 6546, 6652, 6879, 7000, 7173, 7333, 7520, 2976, 8337, 9291, 8614, 8560, 6916, 547, 9251, 864, 3859, 9046, 3814, 9223, 8262, 8722, 8465, 7771, 5654, 7689, 8666, 643, 1478, 8785, 1942, 2459, 1925, 5394, 8929, 8904, 7921, 854, 2082, 2934, 5161, 6461, 7335, 7842, 6278, 8314, 8590, 8228, 8802, 8540, 8157, 6480, 9237, 1154, 3684, 6949, 7764, 7988, 5304, 5088, 902, 7803, 1064, 1829, 2360, 2580, 2730, 2963, 3129, 3297, 3462, 3663, 3997, 4482, 4622, 4813, 4947, 5129, 5279, 7641, 3630, 9185, 1499, 5610, 6626, 8546, 10159, 8458, 7710, 10109, 4105, 5724, 8581, 1209, 4709, 9236]
		stamps[('G_ME', 'LO', 'VD')] = [6666, 4319, 5182, 6472, 330, 490, 583, 770, 944, 1037, 1291, 1571, 1865, 2131, 2305, 2438, 2599, 2732, 2892, 3039, 3199, 3306, 3533, 3653, 3853, 4013, 4187, 4347, 4547, 4654, 4881, 5215, 5548, 5842, 6176, 2219, 4482, 5659, 4533, 1001, 5084, 7294, 8837, 5849, 8628, 2148, 2333, 2481, 2680, 2832, 911, 2844, 5211, 3254, 521, 689, 2355, 2571, 3043, 9070, 6914, 2032, 1437, 3738, 4030, 3657, 7965, 2880, 969, 1462, 7001, 8318, 3634, 6821, 7389, 8031, 532, 1432, 2048, 3646, 5445, 6276, 6726, 2291, 5056, 2352, 1500, 6936, 7565, 2916, 4732, 1816, 2933, 2569, 3203, 2547, 5846, 9729, 4524, 6807, 7451, 5378, 6678, 3369, 5926, 8637, 9003, 1938, 984, 1883, 2516]
		stamps[('G_ME', 'LO', 'VE')] = [7461, 2608, 7244, 743, 461, 2447, 6813, 7462, 10179, 5875, 3233, 9381, 6351, 7534, 600, 954, 4831, 4141, 5575, 550, 3450, 4796, 3326, 1534, 444, 5019, 3860, 4794, 1221, 2773, 1363, 7352, 2676, 5605, 7528, 1260, 2210, 5524, 5479, 4268, 8535, 8813, 4571, 4680, 1097, 1863, 7033, 7865, 8482]
		stamps[('G_AWAY', 'LA', None)] = [7504, 4794, 7042, 7184, 5882, 6418, 397, 7114, 7099, 7641, 5448, 6998, 6438, 5878, 8048, 5077, 593, 2572, 3045, 3461, 4294, 5142, 6086, 4029, 6335, 7154, 6018, 6618, 7551, 5561, 1341, 2032, 2340, 7598, 7078, 7160, 7107, 7256, 7388, 6450, 7065, 7912, 7304, 6150, 7182, 7443, 5439, 5976, 7167, 6474, 7178, 6691, 4952, 6687, 10040, 7448, 6294, 300, 460, 620, 753, 1007, 1327, 1621, 1941, 2302, 2595, 2836, 2956, 3183, 3316, 3476, 5959, 6239, 1993, 4958, 6665, 6509, 1502, 1756, 7249, 7211, 7061, 7782, 9005, 7468, 8820, 7263, 7197, 6900, 7526, 5674, 6775, 5438, 6461, 2591, 7787, 749, 8572, 7074, 7373, 7071, 4622, 5880, 7888, 6937, 7327, 7239, 3744, 6925, 6770, 7241, 7319, 7415, 6926, 7045, 6860, 7610, 6598, 5332, 6229, 7126, 6066, 6019, 5938, 6720, 7387, 7155, 7788, 7036, 7339, 3615, 6978, 5910, 6507, 7218, 7257, 4841, 7425, 8408, 6168, 6602, 6868, 7119, 7702, 1634, 7117, 7101, 781, 7359, 7716, 2704, 7739, 5311, 5844, 6192, 7044, 1771, 2255, 3088, 3505, 3874, 4109, 4291, 4724, 5291, 5641, 6508, 6088, 6294, 5608, 6722, 2682, 7148, 6622, 6791, 9137, 9932, 7688, 845, 836, 2211, 3264, 615, 3739, 6718, 6828, 7208, 6988, 7614, 6774, 7465, 6142, 6784, 3589, 7540, 8314, 7499, 7521, 7409, 7497, 7552, 6267, 5200, 6777, 1181, 2031, 3348, 4301, 5048, 6431, 7114, 7123, 7313, 2129, 7064, 7311, 7169, 3199, 6267, 7950, 7414, 8811, 5069, 5917, 1816, 3166, 5032, 7183, 8549, 2887, 5903, 8053, 5654, 7910, 7483, 4816, 6718, 5925, 8429, 1799, 2965, 6273, 7356, 8617, 7904, 8129, 7766, 7937, 6171, 1348, 1783, 2148, 2715, 3398, 4665, 4999, 6048, 6731, 7048, 982, 2004, 1493, 2276, 3277, 5326, 6426, 9007, 8139, 7835, 6136, 7533, 7229, 6316, 6641, 6979, 6719, 6192, 8592, 7419, 6942, 6666, 7685, 5503, 6503, 7885, 7890, 8985, 7186, 5865, 8197, 6965, 7524, 6839, 7725, 7048, 6852, 6784, 7251, 3215, 4146, 7095, 7082, 1793, 4660, 6055, 6529, 7038, 7035, 8919, 5337, 5946, 8664, 7782, 7213, 6374, 8458, 1225, 1825, 2942, 3541, 4641, 5491, 8741, 604, 1371, 8292, 9711, 7087, 7570, 5609, 7142, 5450, 7435, 3784, 6681, 5614, 5051, 5976, 6887, 694, 832, 1910, 6842, 5372, 6222, 7468, 7061, 6669, 7123, 7399, 7093, 1615, 2848, 3331, 6071, 7462, 7041, 7444, 6903, 3926, 7169, 6731, 1406, 2505, 3322, 3855, 4372, 4721, 5105, 5504, 7165, 4228, 5529, 6495, 7214, 2354, 3302, 4056, 4554, 6422, 7312, 6586, 8320, 7215, 7221, 7145, 6768, 461, 5540, 6878, 7274, 7458, 7280, 7565, 5918, 7098, 7429, 7117, 7120, 6204, 7123, 1089, 1943, 3568, 6139, 2124, 2668, 3121, 3626, 1254, 4079, 1295, 4098, 7177, 7650, 6617, 7025, 7139, 6903, 1105, 6092, 8036, 5501, 6317, 4709, 7554, 7070, 3158, 4533, 5524, 6072, 5612, 6379, 8104, 2065, 8169, 7768, 6929, 6917, 6511, 7715, 7600, 6644, 6250, 6733, 8641, 7830, 5873, 6258, 6610, 7488, 7377, 6096, 5905, 6740, 7147, 7307, 7368, 7256, 7210, 6710, 7285, 9331, 4358, 7622, 9152, 7431, 7604, 7281, 5283, 7791, 7487, 7795, 7860, 7495, 7663, 7942, 8374, 7311, 8461, 6750, 8816, 7417, 7008, 7824, 5909, 8792, 7402, 7636, 6452, 6891, 7251, 7741, 7012, 6695, 7931, 7557, 7105, 6986, 6580, 7049, 9796, 1924, 7399, 690, 2188, 7466, 4126, 5959, 6874, 841, 1574, 2672, 3471, 4585, 5268, 6219, 7416, 6096, 6986, 2026, 2994, 6345, 6451, 868, 6284, 6617, 7167, 6632, 7567, 7991, 7455, 7213, 6995, 7420, 7357, 4726, 6028, 5528, 7939, 6187, 7183, 7494, 7162, 6347, 9721, 7597, 7204, 6298, 7041, 7303, 6219, 6540, 5842, 6976, 7582, 7200, 4404, 7054, 8755, 862, 1440, 10520, 3413, 3880, 7110, 7074, 4908, 7106, 7065, 7189, 6730, 5769, 1104, 7204, 7896, 6784, 7653, 6813, 7260, 7608, 2938, 7151, 7153, 6701, 1378, 7283, 5388, 5921, 7004, 7796, 7083, 8538, 8093, 10526, 7634, 1430, 5458, 7278, 7175, 5935, 7747, 8445, 4593, 7211, 2710, 6922, 7048, 7613, 995, 1794, 1967, 2593, 4325, 4991, 7008, 7825, 5942, 7592, 6975, 7393, 6497, 6591, 7573, 8964, 6765, 7469, 5933, 5932, 6682, 5208, 7290, 4006, 6355, 7182, 7480, 7160, 1202, 6189, 7548, 9764, 7492, 7358, 7169, 5308, 6933, 7137, 7697, 7289, 5622, 6704, 7386, 7404, 9413, 7419, 7461, 7395, 7210, 7471, 5556, 926, 1615, 2159, 2643, 3159, 4060, 4909, 5459, 520, 1404, 2169, 3052, 3886, 4753, 6153, 339, 807, 6972, 5950, 7516, 4429, 6446, 2709, 6467, 1939, 5895, 8030, 1526, 2629, 2927, 3260, 3493, 3726, 3961, 4210, 4360, 4762, 6178, 8366, 7785, 7118, 7016, 7007, 7810, 7017, 5520, 6459, 7789, 7850, 3788, 7449, 7213, 6163, 8813, 736, 4763, 5113, 7151, 6966, 7332, 8033, 7487, 6922, 1060, 1171, 1391, 1626, 1756, 1874, 1994, 2198, 2539, 2833, 2975, 3193, 3305, 3540, 3810, 4419, 4491, 4678, 5331, 5392, 5792, 5881, 6107, 6267, 6467, 6546, 6682, 6143, 7065, 7202, 3042, 7240, 6789, 6938, 7397, 7173, 7072, 5578, 4426, 7102, 7664, 7164, 6979, 7074, 7229, 7601, 7158, 7067, 806, 8722, 6726, 7127, 9056, 9738, 7321, 7089, 7486, 5620, 7870, 8049, 4481, 7889, 7205, 7167, 7476, 7149, 6447, 6104, 6671, 4017, 5551, 6765, 5279, 7559, 6930, 7629, 7710, 8966, 7211, 6287, 803, 9115, 7629, 6179, 7455, 6497, 6492, 7409, 7349, 7257, 7163, 7461, 6394, 7029, 6713, 6163, 5240, 7106, 7230, 7356, 7640, 7298, 2622, 6252, 6972, 4750, 9061, 5066, 7552, 8104, 7064, 8740, 7198, 7215, 7294, 5511, 7078, 7650, 7514, 6352, 6170, 7403, 804, 1222, 1888, 3774, 5634, 6163, 10251, 6991, 7415, 5347, 7414, 8263, 1045, 1336, 1999, 9451, 7044, 7177, 6792, 2182, 6818, 5549, 6001, 7197, 6184, 7550, 6381, 6552, 5383, 7181, 7147, 6994, 4750, 5982, 7107, 7250, 6879, 7117, 6683, 6494, 2638, 3939, 6788, 7639, 995, 7695, 1306, 7405, 1362, 5513, 7846, 1510, 3410, 4194, 7010, 6635, 4294, 7927, 6917, 7205]
		stamps[('G_AWAY', 'LA', 'VA')] = [5448, 6998, 1341, 7256, 7182, 7443, 7178, 7249, 7211, 7782, 7526, 6775, 7074, 7327, 6598, 7126, 7036, 7218, 7257, 6622, 836, 2211, 3739, 6988, 6774, 7540, 7409, 7064, 7311, 7169, 1816, 3166, 5032, 7183, 8549, 7910, 7937, 1493, 2276, 3277, 5326, 6426, 7835, 6641, 6942, 8197, 1793, 4660, 7213, 604, 1371, 9711, 6887, 832, 1910, 7061, 7462, 6903, 461, 7098, 7120, 6903, 7600, 7830, 6740, 7368, 9331, 7417, 7557, 4126, 5959, 6874, 6986, 7074, 7189, 7608, 7153, 7283, 7393, 7573, 5933, 7548, 7137, 1939, 5895, 8030, 8366, 7007, 3788, 7332, 7202, 7072, 6979, 7089, 8049, 7167, 4017, 5551, 6765, 7559, 6713, 2622, 1045, 1336, 1999, 7177, 7181, 7107, 995, 7695, 7010]
		stamps[('G_AWAY', 'LA', 'VB')] = [6418, 7078, 6450, 7065, 300, 460, 620, 753, 1007, 1327, 1621, 1941, 2302, 2595, 2836, 2956, 3183, 3316, 3476, 5959, 6239, 7263, 6900, 5438, 6461, 749, 6926, 6229, 6507, 7716, 5311, 5844, 6192, 7044, 2682, 6791, 845, 6784, 5200, 6777, 7123, 2129, 3199, 6267, 7950, 8811, 6718, 7904, 9007, 6192, 8592, 6666, 6839, 7251, 7082, 7035, 8919, 8664, 1225, 1825, 2942, 3541, 4641, 5491, 8741, 3784, 6681, 6731, 4228, 5529, 6495, 7312, 7145, 7274, 7565, 1089, 1943, 3568, 6139, 6617, 5501, 6317, 6929, 8374, 7008, 7824, 6695, 7105, 841, 1574, 2672, 3471, 4585, 5268, 6219, 6451, 7420, 4726, 6028, 6347, 9721, 8755, 1104, 6784, 1378, 8093, 10526, 7634, 7175, 6765, 5932, 6682, 7160, 7492, 6933, 7419, 926, 1615, 2159, 2643, 3159, 4060, 4909, 5459, 4429, 6446, 1060, 1171, 1391, 1626, 1756, 1874, 1994, 2198, 2539, 2833, 2975, 3193, 3305, 3540, 3810, 4419, 4491, 4678, 5331, 5392, 5792, 5881, 6107, 6267, 6467, 6546, 6682, 7240, 7158, 6726, 7127, 6671, 6930, 803, 9115, 6497, 6492, 6394, 7029, 7356, 6170, 804, 1222, 1888, 3774, 6991, 7197, 5383, 4750, 5982, 6683, 2638, 3939, 6788, 7639]
		stamps[('G_AWAY', 'LA', 'VC')] = [397, 7114, 6438, 8048, 593, 2572, 3045, 3461, 4294, 5142, 6086, 2032, 2340, 7598, 5976, 6474, 6687, 7448, 6509, 7468, 8820, 7197, 7787, 5880, 7888, 6770, 7241, 7415, 6019, 7339, 5910, 6168, 6602, 6868, 7119, 7702, 1634, 7117, 7359, 1771, 2255, 3088, 3505, 3874, 4109, 4291, 4724, 5291, 5641, 6508, 7148, 9137, 9932, 3264, 7465, 3589, 7499, 6267, 7313, 4816, 1799, 2965, 8129, 1348, 1783, 2148, 2715, 3398, 4665, 4999, 6048, 6731, 7048, 6719, 7685, 7890, 7186, 6784, 3215, 4146, 7095, 5946, 6374, 7087, 7570, 5051, 694, 6842, 7399, 6071, 3926, 7169, 7214, 2354, 3302, 4056, 4554, 6422, 7215, 6878, 7280, 5918, 6204, 1295, 4098, 7177, 7650, 7139, 7554, 5612, 6379, 8104, 7715, 7488, 7377, 5905, 7285, 9152, 7604, 7795, 7311, 8461, 7636, 7251, 7012, 6986, 6580, 7049, 9796, 7399, 6096, 6632, 7567, 7991, 7213, 7494, 7303, 5842, 6976, 7200, 862, 7065, 6730, 7204, 7896, 7260, 2938, 7151, 7796, 5935, 8445, 995, 1794, 1967, 2593, 4325, 4991, 7008, 7825, 5942, 6497, 7469, 5208, 7290, 7182, 7480, 7169, 7697, 5622, 6704, 7386, 7395, 520, 1404, 2169, 3052, 3886, 4753, 6153, 1526, 2629, 2927, 3260, 3493, 3726, 3961, 4210, 4360, 4762, 6178, 7118, 7213, 6163, 8813, 6966, 6789, 5578, 7664, 7229, 7601, 9056, 9738, 7486, 4481, 7889, 7205, 7149, 7710, 6179, 7349, 7257, 6163, 7230, 4750, 7064, 6352, 8263, 7044, 6792, 5549, 6001, 6552, 6879, 6494, 1306, 7405, 6917]
		stamps[('G_AWAY', 'LA', 'VD')] = [7504, 7184, 7641, 5878, 4029, 6335, 7154, 5561, 7160, 7912, 7304, 5439, 7167, 4952, 10040, 1993, 4958, 6665, 7061, 9005, 5674, 2591, 7071, 4622, 6937, 7319, 7045, 5332, 6066, 7155, 7788, 4841, 7425, 8408, 7101, 781, 2704, 7739, 6088, 5608, 6722, 6828, 6142, 8314, 7497, 1181, 2031, 3348, 4301, 5048, 6431, 7114, 7414, 5069, 5917, 7483, 8429, 6273, 7356, 7766, 6171, 6136, 7229, 6979, 7419, 7885, 5865, 6965, 7048, 6852, 6055, 6529, 7038, 7782, 8292, 5609, 7142, 5976, 7468, 6669, 7123, 7093, 7041, 7444, 6586, 6768, 5540, 7429, 7123, 1254, 4079, 7025, 8036, 4709, 6072, 2065, 8169, 6917, 6511, 6250, 6733, 5873, 6258, 6610, 6096, 7307, 7256, 6710, 4358, 7622, 7431, 5283, 7791, 7487, 7495, 7942, 5909, 8792, 7402, 7416, 868, 6284, 6617, 7167, 7455, 6995, 7939, 7162, 7597, 7041, 6540, 4404, 7054, 10520, 4908, 7106, 7653, 5388, 5921, 7004, 7083, 8538, 1430, 7747, 2710, 6922, 7592, 6591, 4006, 6355, 6189, 7358, 5308, 7404, 9413, 7461, 7210, 5556, 2709, 6467, 7016, 7810, 5520, 7789, 7850, 7449, 736, 8033, 3042, 6938, 7397, 7102, 7074, 7067, 7321, 6104, 5279, 7629, 6287, 7455, 7409, 7163, 5240, 7106, 7298, 5066, 7552, 8104, 7215, 7294, 5511, 7078, 7415, 5347, 7414, 9451, 2182, 6818, 6381, 7147, 7250, 7117, 1510, 3410, 4194, 4294, 7927]
		stamps[('G_AWAY', 'LA', 'VE')] = [4794, 7042, 5882, 7099, 5077, 6018, 6618, 7551, 7107, 7388, 6150, 6691, 6294, 1502, 1756, 8572, 7373, 7239, 3744, 6925, 6860, 7610, 5938, 6720, 7387, 3615, 6978, 6294, 7688, 615, 6718, 7208, 7614, 7521, 7552, 2887, 5903, 8053, 5654, 5925, 8617, 982, 2004, 8139, 7533, 6316, 5503, 6503, 8985, 7524, 7725, 5337, 8458, 5450, 7435, 5614, 5372, 6222, 1615, 2848, 3331, 1406, 2505, 3322, 3855, 4372, 4721, 5105, 5504, 7165, 8320, 7221, 7458, 7117, 2124, 2668, 3121, 3626, 1105, 6092, 7070, 3158, 4533, 5524, 7768, 6644, 8641, 7147, 7210, 7281, 7860, 7663, 6750, 8816, 6452, 6891, 7741, 7931, 1924, 690, 2188, 7466, 2026, 2994, 6345, 7357, 5528, 6187, 7183, 7204, 6298, 6219, 7582, 1440, 3413, 3880, 7110, 5769, 6813, 6701, 5458, 7278, 4593, 7211, 7048, 7613, 6975, 8964, 1202, 9764, 7289, 7471, 339, 807, 6972, 5950, 7516, 7785, 7017, 6459, 4763, 5113, 7151, 7487, 6922, 6143, 7065, 7173, 4426, 7164, 806, 8722, 5620, 7870, 7476, 6447, 8966, 7211, 7629, 7461, 7640, 6252, 6972, 9061, 8740, 7198, 7650, 7514, 7403, 5634, 6163, 10251, 6184, 7550, 6994, 1362, 5513, 7846, 6635, 7205]
		stamps[('G_AWAY', 'LB', None)] = [7935, 8698, 3671, 7191, 3786, 6910, 4639, 5033, 6266, 4957, 7399, 5943, 7224, 7801, 7065, 3774, 5724, 7140, 7074, 7744, 7353, 6270, 7068, 8174, 5295, 6642, 7733, 7295, 6701, 6788, 6475, 7384, 7618, 7049, 7424, 7524, 7208, 7374, 5314, 7482, 3860, 7100, 4903, 6286, 6656, 7198, 6464, 6961, 1831, 6902, 7868, 3258, 7405, 6625, 9045, 2248, 7613, 7535, 5831, 6731, 7200, 7162, 2810, 3513, 4162, 7464, 9456, 7158, 7380, 7100, 413, 746, 1113, 7896, 2832, 5348, 7076, 4323, 4178, 5301, 7918, 6564, 7547, 6035, 7245, 7447, 7130, 5882, 4827, 6206, 6593, 5557, 7493, 2644, 5325, 5777, 6976, 7327, 335, 3533, 6955, 7203, 7682, 7298, 7183, 7665, 787, 2971, 7221, 7940, 5800, 566, 7815, 6378, 3747, 9315, 7338, 7385, 2196, 7798, 4614, 8164, 9830, 6728, 690, 1823, 2773, 3540, 5590, 7157, 8357, 5093, 7123, 7293, 7430, 7035, 6261, 7418, 6908, 1534, 8418, 5230, 7489, 8795, 2946, 565, 800, 3232, 4542, 5510, 6130, 7847, 7418, 7025, 8107, 7374, 5956, 7246, 6034, 6331, 6067, 7304, 6379, 7969, 7282, 6594, 5460, 7402, 5806, 6651, 1429, 2432, 3746, 6316, 8593, 7308, 7406, 5985, 7110, 6673, 8505, 8650, 2089, 6258, 7730, 7208, 5057, 7580, 6995, 6787, 5560, 7343, 8401, 2086, 1083, 1766, 2300, 7743, 7293, 7929, 5706, 7000, 4927, 5277, 5677, 8205, 5708, 7425, 3892, 5025, 5675, 6626, 7225, 7236, 7616, 8437, 7882, 934, 1165, 1431, 1649, 1830, 3479, 3880, 4298, 4582, 5064, 608, 1008, 1324, 4491, 4724, 5276, 5574, 5990, 587, 890, 1353, 1718, 2037, 2336, 2552, 3419, 3985, 5303, 6235, 6686, 7553, 7217, 6772, 6439, 7406, 6405, 6989, 7055, 7563, 7387, 6646, 7186, 8540, 1264, 8360, 7975, 7035, 7107, 6237, 7165, 7565, 6710, 6326, 7159, 7263, 3940, 5330, 7204, 5782, 3899, 1902, 3482, 6645, 6272, 7664, 6253, 7016, 4821, 5394, 7151, 7197, 497, 847, 1063, 1263, 1463, 1663, 1848, 2030, 2214, 2396, 2780, 2980, 3444, 3842, 8720, 10805, 466, 510, 5249, 6579, 7729, 7711, 7211, 4750, 6183, 6439, 1400, 3440, 6600, 7356, 6887, 7188, 5948, 5291, 6714, 4767, 6985, 8080, 6103, 838, 7528, 1468, 7252, 2830, 9211, 8611, 7339, 7159, 500, 1116, 1233, 1483, 1767, 2334, 3134, 4052, 4195, 4902, 5101, 5269, 5468, 5969, 6135, 6286, 6786, 5169, 7217, 465, 3480, 5445, 8376, 2025, 3726, 5541, 4743, 2064, 6139, 4938, 7055, 6805, 1032, 5581, 1158, 6649, 7452, 7794, 747, 1306, 1905, 2349, 2920, 3616, 7304, 8117, 5833, 7201, 7472, 7128, 7022, 7210, 5527, 7213, 6752, 9871, 6862, 7162, 8505, 6182, 5807, 882, 1834, 4739, 6268, 8793, 1021, 2553, 6825, 7713, 8602, 7911, 7125, 5817, 3623, 6423, 7257, 7122, 5873, 6778, 7744, 7444, 840, 7239, 7617, 7684, 7171, 7224, 7505, 7464, 977, 7826, 7771, 6917, 5891, 7445, 5768, 7167, 7021, 6157, 8373, 6782, 7233, 7932, 5638, 3237, 7062, 6652, 7235, 6219, 7478, 6318, 5215, 7231, 8698, 4049, 7117, 4471, 8598, 7224, 7182, 8018, 8172, 7588, 7662, 5836, 5996, 7432, 9141, 7898, 6664, 9580, 7399, 5788, 6061, 5230, 7362, 7529, 7736, 6544, 1524, 2624, 7983, 1545, 4595, 6040, 6874, 992, 2442, 3625, 5388, 9914, 832, 1033, 1687, 4887, 5570, 7363, 7215, 6432, 3937, 1871, 8784, 7540, 3657, 2654, 4740, 7715, 7198, 7115, 7792, 5450, 7453, 4573, 5991, 5315, 6800, 6950, 7212, 3974, 5176, 8096, 8487, 9337, 7800, 2295, 7501, 4175, 7609, 6141, 7464, 7193, 7202, 5951, 5547, 7919, 7604, 4479, 6483, 7548, 1322, 6898, 9331, 5849, 6381, 5719, 6169, 5597, 7360, 6079, 8796, 3385, 7761, 6141, 6507, 4587, 7219, 7837, 7495, 5617, 7819, 1247, 5926, 7325, 1079, 2476, 4408, 6322, 866, 1449, 5480, 6412, 6702, 984, 2000, 4265, 4681, 5914, 6963, 8095, 10312, 5932, 5181, 6489, 7837, 9230, 5298, 7682, 7145, 3008, 8041, 6562, 7144, 7060, 3524, 9271, 7911, 7040, 6289, 6654, 6333, 3654, 4037, 7171, 6468, 5201, 8935, 7313, 7152, 7124, 7151, 7185, 6705, 6521, 7570, 6987, 818, 1319, 2078, 3434, 4034, 4601, 410, 1011, 1378, 1777, 2210, 3161, 3412, 3912, 4177, 4444, 5012, 6295, 630, 996, 2013, 475, 959, 3525, 3948, 4941, 5325, 5758, 4668, 5101, 2660, 4905, 6570, 2660, 6276, 2264, 6779, 7347, 7964, 6574, 7174, 6703, 8048, 7234, 5702, 155, 7514, 6759, 1640, 3307, 2940, 5956, 7419, 6402, 7212, 7079, 7206, 7431, 7162, 7383, 5054, 8202, 4690, 6997, 5708, 7136, 7535, 7085, 7310, 7260, 7459, 7520, 7250, 5811, 5728, 7261, 7639, 6840, 7033, 8268, 4427, 6644, 5728, 9620, 6432, 6171, 6581, 7541, 5808, 8414, 7243, 7760, 7598, 7195, 7502, 6308, 7661, 762, 6890, 7537, 6037, 4379, 5592, 6879, 7066, 7552, 3294, 6808, 7214, 7491, 7004, 7010, 7082, 6912, 7865, 6137, 6715, 5437, 6405, 7040, 8458, 7077, 1399, 5787, 7172, 6568, 6259, 7459, 7445, 7279, 9167, 7519, 4811, 7277, 7857, 514, 6986, 7370, 6566, 8073, 7801, 8064, 6111, 7243, 9018, 8363, 9462, 10829, 2651, 6318, 7223, 6174, 6437, 5652, 7568, 6917, 5655, 9987, 7238, 6542, 8794, 6952, 815, 3615, 7513, 1199, 2003, 4873, 6355, 1393, 2194, 2893, 4145, 6261, 7661, 6987, 6838, 5278, 8134]
		stamps[('G_AWAY', 'LB', 'VA')] = [7935, 6910, 7399, 7353, 6701, 7049, 7208, 7374, 6656, 7535, 4162, 7464, 2832, 5348, 6564, 7130, 6955, 7298, 3747, 9315, 690, 1823, 2773, 3540, 5590, 7157, 8357, 7035, 6908, 1534, 2946, 3232, 7025, 7282, 8593, 7406, 6995, 8401, 1083, 1766, 2300, 8205, 7882, 6646, 6710, 1902, 3482, 6645, 7016, 7211, 1400, 3440, 6887, 6714, 500, 1116, 1233, 1483, 1767, 2334, 3134, 4052, 4195, 4902, 5101, 5269, 5468, 5969, 6135, 6286, 6786, 2025, 3726, 5541, 1158, 7794, 8117, 7472, 6752, 882, 1834, 4739, 6268, 8793, 7744, 7171, 977, 7826, 7167, 7478, 9141, 7898, 7362, 1545, 4595, 6040, 6874, 832, 7715, 7501, 7193, 7604, 8796, 7837, 6702, 7145, 3008, 8041, 7911, 3654, 4037, 7171, 7124, 475, 959, 3525, 3948, 4941, 5325, 5758, 2264, 6779, 7347, 7964, 6574, 155, 7514, 7383, 6997, 7535, 7520, 7261, 9620, 6581, 7760, 6879, 7066, 6808, 6912, 8064, 8363, 9462, 10829, 6917, 9987, 6952, 815, 3615, 7513]
		stamps[('G_AWAY', 'LB', 'VB')] = [7065, 7074, 8174, 7295, 6788, 6464, 1831, 7868, 2248, 7613, 5831, 6731, 7158, 7076, 4178, 6035, 2644, 6378, 7123, 6261, 6130, 7847, 6331, 7304, 1429, 2432, 3746, 6316, 2089, 6258, 6787, 608, 1008, 1324, 4491, 4724, 5276, 5574, 5990, 7217, 6405, 7387, 7165, 7565, 6272, 7151, 510, 6600, 838, 9211, 5169, 8376, 7452, 8505, 840, 6652, 4049, 7117, 8172, 7662, 6664, 9580, 992, 2442, 3625, 5388, 6432, 3657, 3974, 2295, 6141, 6483, 7548, 5849, 6381, 7761, 7495, 984, 2000, 4265, 4681, 5914, 6963, 8095, 6489, 6562, 6333, 6468, 6705, 6521, 630, 996, 2013, 2660, 4905, 6570, 6759, 6402, 7431, 7310, 6840, 4427, 7502, 7661, 7491, 7865, 6137, 6568, 9167, 514, 6566, 6542, 1393, 2194, 2893, 4145, 6261, 7661, 6838]
		stamps[('G_AWAY', 'LB', 'VC')] = [8698, 7191, 5943, 7801, 7744, 6642, 7733, 6475, 7524, 7482, 7100, 7405, 9045, 7200, 2810, 3513, 7100, 7918, 5882, 6206, 7493, 7327, 7940, 566, 7815, 4614, 8164, 9830, 7430, 800, 7374, 6034, 5460, 5806, 7308, 6673, 8505, 8650, 5057, 7580, 7743, 5706, 7000, 7236, 7616, 8437, 587, 890, 1353, 1718, 2037, 2336, 2552, 3419, 3985, 5303, 6235, 6686, 7055, 7563, 7186, 7975, 6237, 6326, 7159, 7664, 6253, 5394, 466, 7711, 4750, 6183, 6439, 7356, 7188, 1468, 7252, 7159, 7217, 3480, 5445, 6805, 6649, 5527, 6862, 7162, 6182, 8602, 6778, 7239, 7224, 7505, 6917, 7445, 5638, 6219, 6318, 7224, 7588, 5996, 7399, 6061, 7529, 6544, 9914, 1687, 4887, 5570, 7215, 3937, 2654, 4740, 7198, 5450, 5315, 6800, 6950, 7212, 7609, 7464, 5547, 7919, 9331, 7360, 6141, 6507, 5617, 1247, 5926, 7325, 5932, 7837, 5298, 7682, 3524, 6289, 5201, 7151, 6987, 410, 1011, 1378, 1777, 2210, 3161, 3412, 3912, 4177, 4444, 5012, 6295, 2660, 6276, 6703, 8048, 5702, 1640, 3307, 7206, 5708, 7085, 7250, 5811, 7639, 8268, 6644, 5728, 6432, 5808, 7598, 7195, 7537, 6037, 3294, 7004, 7010, 6715, 1399, 5787, 7172, 6259, 7459, 4811, 7277, 8073, 9018, 2651, 6318, 7223, 6174, 6437, 5652, 7568, 5655]
		stamps[('G_AWAY', 'LB', 'VD')] = [3786, 5033, 6266, 7224, 3774, 5724, 7140, 6270, 5295, 7384, 7618, 3860, 4903, 6286, 6961, 6625, 413, 746, 1113, 7896, 4323, 7547, 7245, 7447, 4827, 5557, 5325, 5777, 6976, 3533, 7183, 7665, 7338, 7385, 2196, 7798, 6728, 7418, 5230, 7489, 8795, 4542, 5510, 7418, 5956, 7246, 6379, 7969, 7402, 5985, 7110, 7730, 5560, 7343, 7293, 3892, 5025, 5675, 6626, 7225, 7553, 6989, 1264, 8360, 7035, 7263, 3940, 5330, 7204, 3899, 4821, 7197, 8720, 10805, 5249, 6579, 7729, 5291, 7528, 2830, 8611, 465, 4743, 6139, 1032, 5581, 747, 1306, 1905, 2349, 2920, 3616, 7304, 7128, 7022, 7210, 1021, 2553, 6825, 7713, 7125, 3623, 6423, 7257, 5873, 7444, 7617, 7684, 7464, 7771, 5891, 5768, 7021, 6782, 7233, 3237, 7062, 7235, 5215, 7231, 8698, 7182, 8018, 5836, 7432, 5230, 7736, 7363, 1871, 8784, 7115, 7453, 4573, 5991, 7800, 4175, 7202, 5951, 4479, 1322, 6898, 5719, 6169, 3385, 7219, 7819, 866, 1449, 5480, 6412, 10312, 5181, 7144, 7060, 6654, 7185, 7570, 818, 1319, 2078, 3434, 4034, 4601, 5101, 7174, 7234, 2940, 5956, 7419, 7212, 7162, 5054, 8202, 4690, 7459, 6171, 7541, 7243, 762, 6890, 4379, 5592, 7214, 7082, 5437, 6405, 7040, 8458, 7279, 7857, 6986, 7370, 6111, 7243, 7238, 2003, 4873, 6355, 5278]
		stamps[('G_AWAY', 'LB', 'VE')] = [3671, 4639, 4957, 7068, 7424, 5314, 7198, 6902, 3258, 7162, 9456, 7380, 5301, 6593, 335, 7203, 7682, 787, 2971, 7221, 5800, 5093, 7293, 8418, 565, 8107, 6067, 6594, 6651, 7208, 2086, 7929, 4927, 5277, 5677, 5708, 7425, 934, 1165, 1431, 1649, 1830, 3479, 3880, 4298, 4582, 5064, 6772, 6439, 7406, 8540, 7107, 5782, 497, 847, 1063, 1263, 1463, 1663, 1848, 2030, 2214, 2396, 2780, 2980, 3444, 3842, 5948, 4767, 6985, 8080, 6103, 7339, 2064, 4938, 7055, 5833, 7201, 7213, 9871, 5807, 7911, 5817, 7122, 6157, 8373, 7932, 4471, 8598, 5788, 1524, 2624, 7983, 1033, 7540, 7792, 5176, 8096, 8487, 9337, 5597, 6079, 4587, 1079, 2476, 4408, 6322, 9230, 9271, 7040, 8935, 7313, 7152, 4668, 7079, 7136, 7260, 5728, 7033, 8414, 6308, 7552, 7077, 7445, 7519, 7801, 8794, 1199, 6987, 8134]
		stamps[('G_AWAY', 'LC', None)] = [7416, 7684, 3649, 2810, 3223, 6186, 3884, 10198, 8454, 8351, 8106, 7484, 7868, 6496, 8071, 1824, 8637, 8052, 7205, 8116, 8461, 6118, 7388, 8255, 3691, 7191, 5920, 7496, 4834, 7845, 7257, 8619, 5994, 7177, 8116, 7887, 8128, 8171, 7105, 7856, 5078, 4638, 8131, 296, 483, 616, 763, 963, 1270, 1577, 1831, 2017, 2124, 2284, 2418, 2605, 2738, 2978, 3125, 3285, 3499, 3659, 3833, 3993, 4166, 4353, 4473, 4687, 4767, 4994, 5114, 268, 682, 2671, 7089, 7662, 8054, 4643, 6018, 5268, 5989, 6607, 2152, 7412, 8367, 7408, 8050, 7900, 634, 1401, 3317, 7518, 7068, 7834, 5997, 7280, 3485, 7704, 6301, 7599, 8094, 8534, 6918, 7845, 7548, 8075, 8012, 7827, 7380, 8380, 7365, 8048, 9361, 8153, 7150, 7883, 7327, 5986, 6704, 7805, 7385, 7935, 6809, 7393, 4978, 6848, 7983, 7161, 575, 2407, 1032, 5912, 8211, 3215, 5365, 7314, 7597, 8780, 9164, 6444, 1468, 3655, 7010, 897, 4798, 6848, 7831, 6996, 7863, 2742, 3059, 3276, 3426, 3609, 3743, 3893, 4059, 7314, 8912, 7689, 7395, 8210, 8025, 8375, 6988, 8740, 7283, 8013, 1461, 3849, 8615, 3275, 7046, 7222, 8004, 6007, 6525, 6111, 5329, 7245, 7844, 1993, 3051, 8046, 7592, 8144, 6524, 6690, 7773, 8039, 6985, 7452, 8149, 8388, 8603, 4935, 7620, 8187, 7467, 1022, 2888, 4122, 5455, 5972, 5539, 5259, 8056, 5578, 9432, 8088, 6894, 8011, 6849, 8857, 3466, 8558, 4966, 3502, 8333, 7043, 1010, 4820, 7366, 5179, 3960, 6060, 8242, 2133, 3165, 7943, 7293, 7824, 7095, 7929, 6960, 7457, 8702, 8768, 2024, 2274, 3056, 4374, 5206, 6289, 6857, 7942, 422, 639, 873, 1022, 1338, 1555, 2106, 2457, 2699, 3238, 3671, 4020, 5287, 5571, 5853, 6088, 6541, 7122, 7907, 7598, 4201, 1868, 7433, 6120, 3621, 3640, 6458, 7707, 8175, 8143, 7409, 8159, 7267, 8032, 5917, 6616, 7432, 8051, 7492, 7842, 5073, 4219, 2754, 7365, 7964, 7730, 5089, 1905, 4654, 3100, 8282, 1700, 7603, 7444, 9031, 9889, 3033, 6021, 8230, 8712, 2080, 8011, 1359, 9308, 511, 7969, 8527, 3916, 5218, 8308, 7083, 7916, 10276, 5887, 6938, 7950, 4009, 7532, 8165, 2299, 7301, 7767, 8102, 4551, 7117, 7967, 5438, 7114, 6744, 8327, 8669, 7173, 7990, 7424, 3824, 7070, 8120, 897, 3863, 7529, 7077, 8039, 2726, 3294, 4444, 4810, 5094, 6311, 6844, 8544, 5820, 7852, 8286, 7135, 8052, 520, 3087, 3203, 3419, 4170, 4566, 4800, 4999, 5199, 5367, 6149, 5727, 2587, 2804, 3054, 3185, 3418, 3570, 3768, 3985, 4207, 4335, 4502, 4601, 4719, 4919, 5023, 5168, 2468, 3100, 9267, 5823, 7556, 8123, 7321, 7654, 6811, 7793, 7960, 6612, 8078, 6915, 1657, 6905, 8287, 6881, 6155, 3762, 3104, 7730, 7448, 8176, 8542, 7641, 7899, 8782, 2132, 8336, 1031, 1482, 2508, 4067, 4590, 4986, 5435, 792, 1515, 4820, 1055, 3909, 4432, 6627, 6351, 7501, 8234, 3983, 3997, 8531, 7374, 8021, 8654, 6209, 8465, 2714, 8292, 3970, 2982, 5584, 8107, 7980, 7565, 7792, 4188, 7181, 8436, 9289, 7757, 8361, 9496, 8654, 1691, 6964, 7979, 7680, 7826, 9367, 8277, 6600, 7717, 8009, 7514, 8064, 8517, 7142, 7963, 8385, 5299, 7366, 4365, 8066, 6284, 7788, 8272, 9093, 8464, 7944, 8235, 4447, 7026, 8409, 8143, 2941, 4986, 7499, 865, 2211, 3278, 2123, 849, 1315, 2332, 4298, 2046, 3254, 5553, 7414, 5737, 6601, 7865, 7920, 7917, 7591, 7625, 7957, 8317, 2048, 3535, 8901, 7321, 8069, 7997, 8122, 4557, 6441, 9139, 6306, 8292, 2615, 6900, 8168, 7392, 8824, 9174, 6261, 6349, 7225, 3860, 1970, 6016, 5030, 7095, 7861, 7272, 7989, 2378, 8909, 7476, 8004, 4122, 4775, 5956, 8151, 3368, 4367, 8418, 7072, 7875, 3914, 7389, 7938, 4270, 7366, 7898, 9936, 7337, 8022, 7897, 7214, 1208, 7839, 8708, 8115, 8895, 1626, 8002, 1653, 3669, 4584, 1239, 2439, 1156, 3987, 6820, 8326, 7445, 4668, 7790, 8108, 6615, 8032, 5685, 8485, 965, 2540, 7318, 4987, 8304, 7996, 7376, 8043, 2508, 3858, 1811, 8772, 8122, 7314, 4084, 7252, 7686, 7885, 8475, 6000, 6694, 6868, 8067, 6545, 7226, 7704, 5234, 6617, 8381, 8103, 789, 2240, 3314, 4006, 4724, 5457, 6307, 508, 792, 1191, 1576, 2024, 2624, 2958, 280, 1631, 2119, 6614, 7163, 9338, 9637, 1828, 6962, 7967, 763, 3599, 8417, 2559, 5591, 7958, 6772, 2595, 6880, 7636, 8220, 7230, 8097, 8199, 7489, 1938, 8235, 3339, 5138, 7880, 7671, 5787, 6772, 8189, 8471, 6447, 9101, 7575, 8543, 7823, 7929, 679, 1524, 8207, 6752, 7935, 6018, 7157, 5981, 6653, 8782, 4481, 2056, 2938, 1930, 2662, 3091, 3494, 5118, 8219, 7679, 6982, 7798, 7926, 4329, 4768, 8272, 7469, 8139, 7141, 7906, 8309, 8553, 1965, 8612, 2325, 7228, 9374, 3805, 7217, 7182, 7898, 4336, 7676, 8225, 9493, 8100, 7440, 7978, 8767, 7430, 8488, 8244, 7351, 7972, 7760, 7188, 7738, 8425, 7718, 4574, 4669, 6201, 1311, 2730, 4063, 6327, 7763, 8364, 7202, 8400, 7654, 7070, 7945, 6810, 8567, 4059]
		stamps[('G_AWAY', 'LC', 'VA')] = [7416, 7684, 8351, 7484, 7868, 8637, 8052, 8461, 7496, 7257, 8171, 8131, 8054, 2152, 7412, 8367, 7408, 7280, 7704, 7599, 8094, 8012, 9361, 8153, 7327, 3655, 7010, 8912, 8025, 8375, 7222, 8004, 8046, 8144, 8388, 8603, 1022, 2888, 4122, 5455, 5972, 8056, 8088, 8558, 3502, 8333, 4201, 7433, 8175, 8143, 5917, 6616, 7432, 8051, 7730, 1700, 7444, 9889, 3033, 6021, 8230, 1359, 9308, 8527, 8102, 7077, 8039, 5727, 7960, 8287, 8176, 7899, 8782, 792, 1515, 4820, 8531, 8021, 8654, 9496, 8277, 7514, 8517, 8235, 8143, 7917, 7625, 8122, 8292, 8824, 9174, 7272, 7989, 8004, 4122, 4775, 9936, 8022, 8895, 1156, 3987, 7790, 8108, 4987, 8304, 1811, 8772, 8122, 7885, 8475, 8103, 789, 2240, 3314, 4006, 4724, 5457, 6307, 9338, 9637, 7636, 8220, 7489, 8235, 7880, 8471, 9101, 7929, 8207, 6018, 2056, 2938, 8219, 7926, 8309, 8425, 8400]
		stamps[('G_AWAY', 'LC', 'VB')] = [6186, 1824, 6118, 7388, 8255, 7845, 8619, 5994, 7177, 8116, 7105, 7856, 6018, 5268, 5989, 6607, 7900, 634, 1401, 3317, 7518, 6301, 8534, 7827, 5986, 7983, 7161, 5912, 8211, 2742, 3059, 3276, 3426, 3609, 3743, 3893, 4059, 7314, 7395, 8210, 6111, 6690, 7773, 8039, 2133, 3165, 7293, 8768, 2024, 2274, 3056, 4374, 5206, 6289, 6857, 7942, 6120, 1905, 4654, 8282, 2080, 8011, 8308, 5887, 6938, 7950, 4009, 8669, 520, 3087, 3203, 3419, 4170, 4566, 4800, 4999, 5199, 5367, 6149, 2587, 2804, 3054, 3185, 3418, 3570, 3768, 3985, 4207, 4335, 4502, 4601, 4719, 4919, 5023, 5168, 6915, 6905, 6155, 1055, 3909, 4432, 6627, 6351, 7501, 8234, 2714, 8292, 2982, 5584, 8654, 9367, 6600, 8066, 7944, 7026, 8409, 2941, 4986, 7499, 2123, 7920, 7997, 6306, 6261, 6349, 7225, 1970, 6016, 7476, 7072, 7875, 7389, 7938, 1208, 1626, 5685, 8485, 7376, 8043, 6545, 6772, 6880, 6447, 7575, 8543, 7823, 7157, 7679, 8272, 8553, 1965, 2325, 7228, 9374, 7217, 8244, 7351, 6810]
		stamps[('G_AWAY', 'LC', 'VC')] = [2810, 3223, 8106, 8116, 5920, 4834, 4643, 7380, 8380, 7385, 7935, 4978, 6848, 1032, 6996, 7863, 8740, 3849, 8615, 3275, 6007, 6525, 5329, 7245, 7844, 6524, 6985, 7452, 8149, 4935, 5259, 5578, 9432, 4966, 7043, 5179, 8242, 7824, 7095, 7929, 6960, 7457, 8702, 422, 639, 873, 1022, 1338, 1555, 2106, 2457, 2699, 3238, 3671, 4020, 5287, 5571, 5853, 6088, 6541, 7122, 7907, 7267, 8032, 5073, 5089, 5218, 5438, 7424, 5820, 7852, 8286, 7321, 7654, 1031, 1482, 2508, 4067, 4590, 4986, 5435, 7374, 7565, 7680, 7826, 8009, 7142, 7963, 5299, 4365, 6284, 7788, 8272, 7414, 3535, 5030, 7095, 7861, 3368, 4367, 8418, 7337, 7839, 8708, 1653, 3669, 4584, 6615, 8032, 965, 2508, 5234, 6617, 8381, 763, 3599, 8417, 7671, 5118, 4768, 8139, 7182, 7898, 7676, 8225, 9493, 7430, 8488, 7760, 4574, 4669, 6201, 8567]
		stamps[('G_AWAY', 'LC', 'VD')] = [3649, 3884, 10198, 6496, 8071, 3691, 7191, 268, 682, 2671, 7089, 7662, 5997, 6918, 7845, 6704, 7805, 3215, 5365, 7314, 7597, 8780, 9164, 6444, 1468, 7689, 6988, 1461, 7592, 7620, 8187, 5539, 6894, 8011, 3960, 6060, 7943, 7598, 3640, 6458, 7707, 7492, 7842, 4219, 7365, 7964, 3100, 7603, 3916, 7083, 7916, 10276, 7532, 8165, 2299, 7114, 6744, 8327, 7070, 8120, 7135, 8052, 2468, 3100, 9267, 6612, 8078, 3762, 7448, 8542, 7641, 2132, 8336, 3997, 6209, 3970, 7792, 4188, 7181, 8436, 7757, 8361, 7717, 8064, 8385, 849, 1315, 2332, 4298, 5737, 6601, 7865, 7957, 2048, 8901, 4557, 6900, 8168, 4270, 7366, 7898, 7214, 8115, 1239, 2439, 6820, 8326, 3858, 7314, 6000, 7704, 280, 1631, 2119, 6614, 7163, 2559, 5591, 7958, 8199, 1938, 6772, 8189, 6752, 7935, 5981, 6653, 8782, 1930, 2662, 3091, 7469, 8612, 8100, 7440, 7978, 8767, 7718, 1311, 2730, 4063, 6327, 7763, 8364, 7202, 7070, 7945]
		stamps[('G_AWAY', 'LC', 'VE')] = [8454, 7205, 7887, 8128, 5078, 4638, 296, 483, 616, 763, 963, 1270, 1577, 1831, 2017, 2124, 2284, 2418, 2605, 2738, 2978, 3125, 3285, 3499, 3659, 3833, 3993, 4166, 4353, 4473, 4687, 4767, 4994, 5114, 8050, 7068, 7834, 3485, 7548, 8075, 7365, 8048, 7150, 7883, 6809, 7393, 575, 2407, 897, 4798, 6848, 7831, 7283, 8013, 7046, 1993, 3051, 7467, 6849, 8857, 3466, 1010, 4820, 7366, 1868, 3621, 7409, 8159, 2754, 9031, 8712, 511, 7969, 7301, 7767, 4551, 7117, 7967, 7173, 7990, 3824, 897, 3863, 7529, 2726, 3294, 4444, 4810, 5094, 6311, 6844, 8544, 5823, 7556, 8123, 6811, 7793, 1657, 6881, 3104, 7730, 3983, 8465, 8107, 7980, 9289, 1691, 6964, 7979, 7366, 9093, 8464, 4447, 865, 2211, 3278, 2046, 3254, 5553, 7591, 8317, 7321, 8069, 6441, 9139, 2615, 7392, 3860, 2378, 8909, 5956, 8151, 3914, 7897, 8002, 7445, 4668, 2540, 7318, 7996, 4084, 7252, 7686, 6694, 6868, 8067, 7226, 508, 792, 1191, 1576, 2024, 2624, 2958, 1828, 6962, 7967, 2595, 7230, 8097, 3339, 5138, 5787, 679, 1524, 4481, 3494, 6982, 7798, 4329, 7141, 7906, 3805, 4336, 7972, 7188, 7738, 7654, 4059]
		stamps[('G_AWAY', 'LD', None)] = [7410, 7910, 2788, 2560, 8998, 7313, 8430, 10662, 4756, 7357, 2351, 1710, 5024, 8743, 8008, 4300, 6449, 7366, 8565, 7752, 4509, 8433, 7281, 9019, 7616, 3673, 3077, 7247, 8174, 7455, 8905, 7423, 7195, 7546, 8247, 1019, 7043, 1442, 3067, 9348, 2209, 4051, 277, 464, 691, 837, 998, 1211, 5389, 1480, 6209, 8193, 3859, 8130, 3811, 4631, 7334, 1102, 460, 748, 1131, 1414, 2607, 8806, 8564, 9093, 6398, 8038, 8344, 4946, 9267, 7379, 9064, 6733, 7367, 8416, 8845, 4977, 8124, 5221, 398, 6144, 9075, 4119, 8387, 4535, 7652, 466, 4660, 2752, 7202, 6854, 3583, 4395, 5905, 6983, 8167, 4970, 4597, 7672, 8972, 7846, 7101, 6794, 8955, 1929, 8378, 8870, 5163, 6189, 8075, 3331, 6955, 6514, 5826, 10396, 4963, 8822, 7945, 9306, 2325, 4992, 7326, 9575, 8500, 9031, 2437, 4804, 9167, 5812, 7615, 3201, 5968, 8017, 10184, 1450, 2600, 3634, 4717, 5650, 6433, 7251, 964, 7631, 483, 818, 1136, 7983, 8647, 6777, 8310, 6100, 7758, 7904, 7412, 6590, 9277, 8304, 9484, 7450, 8747, 4042, 8436, 8722, 7901, 7855, 7440, 1936, 7604, 7609, 8020, 6789, 8190, 9019, 6980, 8431, 9100, 9538, 9378, 7039, 8395, 8589, 4031, 6796, 7913, 9012, 7502, 6147, 7213, 3299, 8674, 5492, 7794, 6667, 730, 1197, 6031, 2630, 3629, 4446, 4763, 5063, 5346, 5546, 5846, 6412, 7479, 8628, 2134, 10260, 10605, 8712, 7917, 8468, 2519, 4755, 6236, 7969, 3878, 7728, 2323, 9408, 3780, 5218, 4589, 7150, 8783, 6717, 7787, 8154, 1091, 5573, 7753, 1173, 6051, 7376, 1092, 2240, 2816, 3644, 4615, 5142, 5644, 6342, 7867, 1129, 5283, 7108, 1668, 3224, 4796, 6372, 8027, 7541, 6735, 6327, 7436, 3929, 6554, 8789, 3942, 6424, 8507, 7086, 1493, 5744, 3952, 8715, 6877, 7809, 8713, 9073, 6453, 8537, 6663, 7848, 8208, 8122, 3852, 8219, 9335, 8492, 8232, 7595, 1710, 5751, 7415, 1791, 4888, 933, 2648, 3097, 3563, 4014, 5644, 860, 2059, 10088, 3237, 5220, 1796, 3163, 5713, 1748, 986, 6685, 8469, 6422, 6559, 7847, 4919, 10225, 874, 1958, 7297, 1235, 7468, 4604, 4232, 8610, 8505, 8640, 6397, 7447, 8083, 3515, 557, 8383, 3491, 8910, 6856, 832, 2431, 8309, 8948, 8667, 7120, 7403, 5062, 4264, 7000, 482, 964, 6119, 8101, 8750, 733, 8301, 7959, 1723, 2970, 4053, 7904, 10167, 10516, 7932, 8315, 846, 9717, 9120, 8304, 8774, 4595, 986, 1851, 3833, 4137, 7175, 8643, 7066, 7240, 8030, 8846, 1616, 3682, 8733, 8926, 2306, 8985, 5718, 4216, 9020, 9075, 2816, 7327, 8043, 3256, 5300, 8955, 6544, 6943, 6761, 8648, 6806, 720, 1304, 1692, 624, 1859, 2375, 1038, 3358, 6189, 6539, 4761, 5478, 7022, 4082, 8999, 5280, 2202, 5052, 6952, 6017, 7445, 7744, 7107, 9178, 8005, 7202, 7706, 7707, 817, 472, 8883, 8535, 7404, 8280, 8529, 8994, 3372, 8485, 2941, 8795, 6750, 7716, 9346, 5993, 7176, 7335, 6662, 7560, 9163, 8753, 7218, 2479, 8491, 7878, 6763, 8197, 4378, 8565, 8466, 1674, 2640, 7764, 8720, 7663, 8648, 2131, 4815, 9224, 733, 3484, 6317, 8092, 5750, 9785, 2148, 5998, 8406, 4222, 6828, 8704, 8662, 6147, 9406, 9449, 7743, 3000, 6484, 8642, 5335, 6068, 9664, 8225, 7729, 8629, 8692, 7514, 7856, 5253, 4748, 1614, 2799, 4283, 5100, 8865, 10130, 1088, 2970, 1227, 3268, 3945, 5691, 6392, 7925]
		stamps[('G_AWAY', 'LD', 'VA')] = [2351, 8743, 8565, 8433, 9019, 1442, 3067, 9348, 2209, 4051, 6209, 8193, 1102, 8806, 8564, 9093, 9267, 9064, 8845, 4119, 8387, 3583, 4395, 8955, 8870, 8822, 9306, 8500, 9031, 9167, 3201, 5968, 8017, 10184, 8647, 9484, 8747, 4042, 8436, 8722, 8431, 9100, 9378, 8589, 9012, 8674, 730, 1197, 2630, 3629, 4446, 4763, 5063, 5346, 5546, 5846, 6412, 7479, 8628, 8712, 8468, 1173, 6051, 7376, 8789, 1493, 5744, 3952, 8715, 9073, 8492, 1710, 5751, 7415, 1235, 4232, 8610, 8640, 3491, 8910, 8667, 10167, 10516, 846, 9120, 8643, 8030, 8846, 8926, 9075, 8955, 4761, 5478, 7022, 2202, 5052, 6952, 7107, 9178, 8883, 8535, 8529, 8994, 2941, 9346, 9163, 8753, 4378, 8565, 8720, 8648, 9224, 2148, 8406, 9406, 8642, 9664, 8629, 8692, 1614, 2799, 4283, 5100, 8865, 10130]
		stamps[('G_AWAY', 'LD', 'VB')] = [7410, 7910, 10662, 8008, 7752, 7281, 7616, 7423, 7195, 7546, 8247, 7043, 3859, 8130, 8038, 8344, 7379, 8124, 4535, 7652, 8167, 7672, 7846, 7101, 1929, 8378, 6189, 8075, 6955, 6514, 7945, 7631, 7904, 7450, 7855, 7604, 8020, 6980, 7039, 8395, 7502, 5492, 7794, 6031, 7917, 3878, 7728, 9408, 3780, 1091, 5573, 7753, 7541, 6327, 7086, 8713, 7848, 8208, 8232, 7595, 10088, 986, 6685, 8469, 4919, 10225, 7468, 8505, 557, 8383, 6856, 8948, 733, 7959, 1723, 2970, 4053, 7904, 9717, 8304, 8774, 7240, 8733, 2306, 8985, 8043, 6943, 720, 1304, 1692, 5280, 7744, 7706, 7404, 8485, 7716, 7335, 2479, 8491, 1674, 2640, 7764, 7663, 5750, 9785, 9449, 7729, 7514, 7856, 1227, 3268, 3945, 5691, 6392, 7925]
		stamps[('G_AWAY', 'LD', 'VC')] = [8998, 4756, 7357, 1710, 4509, 3077, 7247, 8174, 277, 464, 691, 837, 998, 1211, 5389, 3811, 4631, 7334, 748, 1131, 1414, 6398, 4946, 6733, 7367, 8416, 5221, 398, 4660, 6854, 5905, 6983, 8972, 4963, 2437, 4804, 483, 818, 1136, 7983, 6100, 7758, 6590, 8304, 1936, 9019, 6147, 7213, 6667, 10260, 10605, 7150, 8783, 6717, 7787, 8154, 1668, 3224, 4796, 6372, 8027, 7436, 6554, 6877, 7809, 6663, 8122, 3852, 8219, 9335, 933, 2648, 3097, 3563, 4014, 5644, 1748, 6422, 4604, 6397, 7447, 8083, 7120, 7000, 6119, 8101, 8301, 4595, 4137, 7175, 7066, 1616, 3682, 5718, 5300, 6544, 6806, 1038, 3358, 6189, 6539, 8280, 8795, 6750, 5993, 7176, 6662, 7560, 7218, 7878, 6763, 8197, 2131, 4815, 8662, 6147, 7743, 5335, 6068, 4748]
		stamps[('G_AWAY', 'LD', 'VD')] = [2560, 5024, 4300, 6449, 7366, 7455, 8905, 1480, 6144, 466, 4970, 4597, 5812, 7615, 964, 6777, 8310, 7412, 7440, 7609, 6789, 8190, 4031, 6796, 7913, 2519, 4755, 6236, 7969, 2323, 4589, 1129, 5283, 7108, 3942, 6424, 8507, 6453, 8537, 860, 2059, 3237, 5220, 7847, 3515, 7403, 4264, 8750, 7932, 8315, 986, 1851, 3833, 4216, 7327, 3256, 6017, 472, 733, 3484, 6317, 8092, 5998, 4222, 6828, 8704, 5253, 1088, 2970]
		stamps[('G_AWAY', 'LD', 'VE')] = [2788, 7313, 8430, 3673, 1019, 460, 2607, 4977, 9075, 2752, 7202, 6794, 5163, 3331, 5826, 10396, 2325, 4992, 7326, 9575, 1450, 2600, 3634, 4717, 5650, 6433, 7251, 9277, 7901, 9538, 3299, 2134, 5218, 1092, 2240, 2816, 3644, 4615, 5142, 5644, 6342, 7867, 6735, 3929, 1791, 4888, 1796, 3163, 5713, 6559, 874, 1958, 7297, 832, 2431, 8309, 5062, 482, 964, 9020, 2816, 6761, 8648, 624, 1859, 2375, 4082, 8999, 7445, 8005, 7202, 7707, 817, 3372, 8466, 3000, 6484, 8225]
		stamps[('G_AWAY', 'LE', None)] = [945, 6213, 643, 866, 3166, 7329, 7326, 8995, 8432, 5959, 7709, 6992, 3680, 4345, 3721, 860, 6232, 6068, 8697, 6362, 7681, 7695, 7460, 8295, 9777, 7649, 8128, 8085, 8656, 1574, 1974, 2290, 8024, 8685, 5624, 6630, 8609, 2393, 5740, 5306, 7191, 5390, 5907, 5875, 8269, 1445, 7705, 8724, 8610, 8137, 7697, 7144, 7701, 4585, 4920, 8436, 7418, 7787, 7868, 7383, 512, 895, 2178, 7045, 1340, 6689, 8214, 3361, 8770, 5861, 6299, 5824, 6551, 7744, 7419, 9338, 2089, 4471, 7071, 7770, 6542, 3276, 5544, 7491, 1920, 3303, 943, 7560, 5987, 8839, 2898, 4687, 5492, 6675, 1281, 8363, 655, 6078, 4644, 5112, 5429, 4233, 5979, 8232, 1837, 6419, 2323, 7906, 2978, 7637, 9564, 5724, 8785, 6691, 5421, 743, 856, 9586, 5280, 7060, 5086, 4471, 5872, 8626, 3821, 7556, 6149, 6105, 7551, 7101, 7032, 1615, 7241, 7339, 7530, 5502, 6903, 8609, 2409, 2631, 2735, 3634, 4388, 4929, 3881, 2362, 9342, 3245, 4894, 6363, 8312, 9796, 9140, 8187, 6787, 3382, 7686, 7904, 5350, 6708, 7023, 6886, 8786, 473, 872, 1238, 1672, 2372, 3154, 3603, 4071, 5489, 6737, 7572, 1914, 2931, 3432, 3897, 4347, 4797, 5683, 6779, 7348, 8080, 7421, 2209, 5978, 7011, 5091, 7309, 3139, 6050, 4440, 8339, 8079, 8444, 6020, 7204, 8900, 5972, 6642, 8799, 6975, 5937, 6961, 4974, 4344, 7782, 8170, 8712, 8329, 9893, 6084, 7781, 5686, 7299, 891, 8675, 1416, 3246, 1001, 8817, 4820, 7371, 5589, 8474, 7435, 8105, 7726, 2751, 6894, 7853, 6984, 1509, 8978, 7662, 6245, 6795, 7328, 9199, 6796, 3722, 2675, 4685, 9694, 7219, 7495, 8929, 5910, 9481, 5828, 7535, 834, 1150, 1367, 1634, 1867, 3384, 4467, 5951, 8051, 826, 5871, 6636, 8253, 8325, 859, 5324, 8672, 5439, 6875, 7983, 10459, 7169, 8139, 9021, 9227, 9894, 1013, 6396, 7229, 2535, 8368, 623, 5444, 2505, 4001, 4621, 6056, 8584, 1425, 2938, 4629, 5181, 6875, 8477, 8011, 6225, 7314, 5654, 7068, 5899, 6474, 672, 5722, 7139, 1272, 4473, 6223, 1819, 2996, 5569, 7071, 7996, 10376, 1187, 2305, 1377, 3306, 4176, 6106, 6857, 7428, 8161, 1224, 1949, 3905, 5526, 7107, 8007, 8077, 6204, 2805, 1379, 6761, 3554, 2718, 7277, 8730, 5855, 8784, 1321, 7805, 5572, 8363, 3918, 828, 6912, 7594, 1343, 3989, 8604, 1725, 2830, 8765, 8393, 8368, 6616, 6050, 6106, 8490, 7137, 7984, 6024, 8231, 7450, 8011, 5932, 7347, 7540, 7890, 5502, 7466, 4912, 7434, 5928, 2674, 5731, 7596, 1338, 3103, 3919, 4651, 5384, 6407, 7302, 1225, 4105, 5655, 7136, 1649, 5262, 9017, 2700, 5463, 826, 1928, 2283, 9042, 7793, 7175, 3221, 7430, 6927, 7304, 1214, 5128, 9239, 5796, 7755, 8396, 7070, 8053, 8088, 5612, 7362, 5580, 6915, 5489, 7685, 8654, 8941, 1396, 2296, 3282, 807, 8297, 1720, 3606, 6735, 8437, 6374, 6274, 8449, 7565, 6331, 8059, 8033, 6795, 7765, 8267, 5962, 6711, 1912, 7471, 3937, 5080, 5242, 944, 6035, 7136, 8656, 6144, 3486, 6050, 679, 2444, 3676, 1402, 5016, 6365, 9809, 6111, 8125, 9241, 7389, 4146, 7081, 8419, 9510, 6041, 8330, 8580, 4546, 7886, 8170, 4785, 5638, 8901, 6283, 3375, 6102, 7712, 5700, 4679, 7312, 5035, 8513, 7840, 8229, 8534, 8799, 7435, 7277, 2373, 3406, 4156, 5524, 6674, 7923, 7473, 9232, 8642, 1477, 2197, 6130, 6028, 7496, 1015, 4374, 4434, 1455, 8852, 5861, 8494, 2299, 3599, 5350, 3233, 8651, 6259, 1680, 6589, 7440, 8654, 594, 3441, 2721, 7666, 7146, 5632, 7464, 795, 10167, 5883, 8558, 9786, 923, 8524, 5512, 1742, 9495, 5948, 8223, 1465, 3762, 5516, 6652, 9376, 6133, 6731, 6295, 5144, 7027, 8747, 7957, 8319, 6110, 9119, 3716, 7439, 8119, 7502, 8799, 8676, 6856, 8459, 7122, 9215, 8589, 7837, 7454, 6349, 8408, 5823, 7584, 6226, 5956, 7306, 8563, 8848, 4991, 1325, 2761, 1368, 3271, 6886, 1362, 3482, 7111, 7748, 8377, 6742]
		stamps[('G_AWAY', 'LE', 'VA')] = [643, 866, 8432, 3721, 8697, 8656, 8685, 8609, 5875, 8137, 4920, 8436, 8214, 9338, 8839, 1281, 8363, 8232, 9564, 8785, 9586, 5086, 8626, 1615, 8609, 9342, 9140, 8786, 1914, 2931, 3432, 3897, 4347, 4797, 5683, 6779, 7348, 8080, 8339, 8444, 8799, 9893, 891, 8675, 8105, 8978, 9694, 8929, 834, 1150, 1367, 1634, 1867, 3384, 4467, 5951, 8051, 5871, 6636, 8253, 10459, 8139, 9227, 9894, 2505, 4001, 4621, 6056, 8584, 8477, 1187, 2305, 8077, 8730, 1343, 3989, 8604, 8765, 8368, 7137, 1338, 3103, 3919, 4651, 5384, 6407, 7302, 9017, 2283, 9042, 3221, 7430, 1214, 5128, 9239, 8396, 8654, 8297, 8449, 6331, 8656, 3486, 6050, 6111, 8125, 9241, 9510, 8330, 8580, 4785, 5638, 8901, 4679, 8513, 8229, 8799, 2373, 3406, 4156, 5524, 6674, 7923, 8642, 8852, 5861, 8494, 8651, 3441, 10167, 8558, 9495, 9376, 8747, 8319, 9119, 8676, 8459, 8408, 8563, 8848, 1325, 2761, 8377]
		stamps[('G_AWAY', 'LE', 'VB')] = [7681, 7649, 8085, 1574, 1974, 2290, 8024, 8269, 8610, 7701, 7868, 8770, 7419, 943, 7560, 2323, 7906, 7637, 7556, 7551, 7241, 5350, 473, 872, 1238, 1672, 2372, 3154, 3603, 4071, 5489, 6737, 7572, 8079, 6975, 8170, 8329, 1001, 8817, 8474, 7853, 7662, 9199, 2675, 9481, 826, 8325, 7983, 2535, 8368, 5444, 7314, 1819, 2996, 5569, 7071, 7996, 10376, 1379, 6761, 7805, 7594, 7450, 8011, 5932, 7347, 7466, 7434, 1225, 4105, 5655, 7136, 2700, 5463, 7175, 6927, 7070, 8053, 8088, 7685, 8941, 807, 8437, 7471, 3937, 7136, 6365, 9809, 7886, 7712, 7840, 8534, 7496, 1455, 8654, 594, 9786, 7957, 3716, 7439, 7454, 7584]
		stamps[('G_AWAY', 'LE', 'VC')] = [945, 6213, 7329, 5959, 7709, 4345, 6362, 7695, 8128, 5624, 6630, 5740, 5306, 7191, 1445, 7697, 7144, 4585, 7418, 7787, 512, 895, 2178, 7045, 5824, 6551, 7744, 7071, 7770, 6542, 5987, 4687, 5492, 6675, 5112, 5429, 1837, 6419, 5724, 856, 5280, 7060, 3821, 6149, 6105, 7101, 7530, 2409, 2631, 2735, 3634, 4388, 4929, 2362, 8187, 6787, 7686, 7904, 6708, 5091, 7309, 4440, 5937, 6961, 4344, 7782, 8712, 5686, 7299, 4820, 7371, 7435, 7726, 6796, 7219, 5910, 5828, 7535, 5439, 6875, 7169, 1013, 6396, 7229, 623, 8011, 5899, 672, 5722, 7139, 1377, 3306, 4176, 6106, 6857, 7428, 8161, 6204, 7277, 1321, 5572, 8363, 828, 6912, 8393, 6106, 8490, 7984, 6024, 8231, 7540, 7890, 5502, 4912, 5928, 5731, 7596, 1649, 5262, 826, 1928, 7793, 5796, 7755, 5612, 7362, 5580, 6915, 5489, 1396, 2296, 3282, 3606, 6374, 7565, 8033, 7765, 5962, 6711, 5080, 6035, 679, 2444, 3676, 7389, 6041, 6283, 5700, 7277, 7473, 9232, 1015, 4374, 4434, 5350, 7146, 5632, 7464, 5883, 923, 8524, 1742, 6133, 6731, 6295, 5144, 7027, 8119, 7502, 8589, 6349, 6226, 5956, 7306, 7748, 6742]
		stamps[('G_AWAY', 'LE', 'VD')] = [3166, 8995, 6992, 860, 6068, 7460, 8295, 9777, 2393, 5390, 5907, 8724, 1340, 6689, 3361, 5861, 3276, 5544, 655, 6078, 4644, 5979, 6691, 5421, 4471, 5872, 5502, 6903, 3245, 4894, 6363, 8312, 9796, 3382, 6886, 7421, 2209, 5978, 7011, 6020, 7204, 5972, 6642, 6084, 7781, 1416, 3246, 2751, 6894, 1509, 4685, 7495, 1425, 2938, 4629, 5181, 6875, 6225, 6474, 1224, 1949, 3905, 5526, 7107, 8007, 2718, 5855, 1725, 2830, 6616, 6735, 8267, 1912, 944, 6144, 1402, 5016, 4146, 7081, 8419, 4546, 8170, 5035, 1477, 2197, 6130, 2299, 3599, 3233, 1680, 6589, 7440, 7666, 5512, 8223, 5516, 6652, 6110, 6856, 7122, 9215, 7837, 1368, 3271, 6886]
		stamps[('G_AWAY', 'LE', 'VE')] = [7326, 3680, 6232, 7705, 7383, 6299, 2089, 4471, 7491, 1920, 3303, 2898, 4233, 2978, 743, 7032, 7339, 3881, 7023, 3139, 6050, 8900, 4974, 5589, 6984, 6245, 6795, 7328, 3722, 859, 5324, 8672, 9021, 5654, 7068, 1272, 4473, 6223, 2805, 3554, 8784, 3918, 6050, 2674, 7304, 1720, 6274, 8059, 6795, 5242, 3375, 6102, 7312, 7435, 6028, 6259, 2721, 795, 5948, 1465, 3762, 8799, 5823, 4991, 1362, 3482, 7111]
		stamps[('G_AWAY', 'LO', None)] = [8624, 7208, 9753, 9017, 8645, 8878, 980, 3820, 6956, 5690, 8568, 6154, 7513, 7480, 8681, 9566, 7482, 8256, 3928, 8606, 6851, 7498, 4567, 10375, 9749, 7995, 3768, 8827, 9311, 10188, 565, 671, 6930, 7617, 3400, 8692, 4288, 6218, 7421, 6291, 7908, 6770, 7772, 9218, 4768, 8482, 7884, 8668, 8983, 4112, 3919, 2822, 9012, 5384, 1080, 673, 6783, 5256, 4219, 9647, 5040, 8125, 2396, 2134, 2734, 3517, 5184, 7924, 8168, 7381, 2162, 3803, 6217, 7367, 1284, 968, 9650, 2749, 7941, 6602, 584, 1037, 2050, 2484, 5400, 2096, 9546, 8459, 4668, 7304, 7202, 3935, 9074, 7353, 7618, 4617, 8451, 8879, 8963, 7633, 7803, 3271, 9032, 3625, 7509, 8076, 846, 1320, 7039, 8688, 3082, 8671, 1525, 3275, 5391, 6029, 545, 846, 1196, 4462, 740, 4558, 4992, 5375, 5742, 6091, 6492, 6924, 7224, 7874, 8191, 8691, 821, 1338, 1688, 1989, 2353, 2622, 2873, 3122, 3256, 3488, 3654, 3838, 4020, 4188, 4339, 4521, 4721, 5088, 5203, 8861, 2784, 7207, 8635, 3733, 4433, 5133, 6117, 7334, 3046, 1311, 8144, 4616, 5605, 5973, 9469, 4514, 6416, 7435, 6304, 7205, 7806, 8223, 7816, 9125, 7432, 3165, 7409, 8660, 7966, 10128, 5616, 8125, 7271, 8759, 8014, 8846, 737, 3204, 557, 3931, 5534, 6542, 6959, 8199, 7476, 2100, 3175, 4396, 9142, 8752, 8860, 7626, 8914, 6473, 8139, 2283, 7132, 1156, 611, 2844, 4411, 8615, 1121, 1388, 2437, 3653, 3870, 4136, 4354, 7903, 8436, 602, 1784, 2265, 2733, 3266, 1167, 2001, 2669, 7434, 5035, 5598, 6647, 2147, 2663, 3462, 8020, 3375, 9356, 5757, 8532, 9177, 3340, 5073, 2404, 8640, 5259, 965, 2228, 3044, 3755, 4370, 5615, 6203, 6940, 1324, 8734, 4202, 5997, 8611, 8186, 8242, 7502, 1804, 5034, 1256, 3479, 8845, 3287, 6170, 7992, 3268, 2015, 5170, 8767, 5677, 7979, 5008, 6858, 7391, 8144, 8488, 9415, 6928, 8194, 6277, 7932, 8137, 9079, 7072, 7506, 7281, 3903, 1791, 8497, 7997, 8897, 8469, 5908, 8775, 7491, 7755, 4216, 7026, 5412, 5949, 7968, 4370, 9850, 8424, 8977, 6266, 5753, 7335, 1570, 7202, 2015, 1166, 6200, 8909, 8987, 5105, 4246, 8461, 8806, 2742, 7183, 4723, 4718, 6033, 7686, 7521, 6358, 9684, 7236, 1081, 3620, 890, 5787, 4138, 6686, 7403, 1036, 5770, 1961, 2868, 8824, 7017, 7809, 2244, 6961, 10105, 5375, 7607, 1014, 8663, 8847, 8556, 7771, 2330, 759, 2272, 1115, 5217, 2499, 5530, 8710, 2165, 7500, 4906, 7206, 6415, 9077, 1266, 682, 5380, 3563, 10478, 5808, 3505, 6833, 8431, 8744, 9088, 2611, 4155, 8893, 6431, 949, 1514, 2086, 2365, 3848, 789, 1438, 1655, 2088, 2338, 2605, 3304, 3971, 4539, 5120, 5689, 6271, 7172, 7671, 519, 1253, 2086, 3270, 3985, 4502, 6086, 6935, 7385, 2143, 2117, 4090, 2445, 4878, 6777, 7244, 3050, 7866, 1962, 3113, 3629, 4113, 4395, 4664, 4878, 5478, 5713, 6529, 8712, 9700, 10074, 3462, 6730, 7208, 8134, 7180, 8685, 7186, 8069, 6380, 8332, 9047, 3579, 8759, 4732, 7962, 4432, 2455, 6323, 6939, 10174, 7816, 8877, 7408, 9861, 8359, 7223, 7992, 3814, 1670, 3644, 4405, 470, 413, 8433, 2931, 7540, 3167, 1225, 6348, 7156, 6852, 8586, 8749, 3293, 4196, 8415, 8188, 8087, 5335, 1861, 9404, 9919, 5051, 4793, 4292, 8627, 8796, 8678, 8860, 8356, 3090, 3907, 1591, 8727, 4842, 936]
		stamps[('G_AWAY', 'LO', 'VA')] = [8624, 8878, 6154, 8681, 8606, 9749, 3768, 8827, 9311, 10188, 565, 8692, 8482, 8983, 9012, 1080, 9650, 2096, 9546, 3935, 9074, 8451, 8963, 3271, 9032, 7039, 8688, 8671, 545, 846, 1196, 4462, 8861, 7207, 8635, 3046, 9469, 9125, 7409, 8660, 8759, 5534, 9142, 8860, 8914, 8615, 1121, 1388, 2437, 3653, 3870, 4136, 4354, 7903, 8436, 3375, 9356, 8734, 7502, 8845, 5170, 8767, 9415, 8424, 8977, 8987, 8806, 9684, 8824, 10105, 1014, 8663, 8847, 8556, 8710, 4906, 7206, 9077, 3563, 8744, 9088, 789, 1438, 1655, 2088, 2338, 2605, 3304, 3971, 4539, 5120, 5689, 6271, 7172, 7671, 10074, 8685, 3579, 8759, 2455, 10174, 8359, 413, 9404, 9919, 8627, 8796, 8678, 8727]
		stamps[('G_AWAY', 'LO', 'VB')] = [7208, 9017, 6956, 8568, 7480, 9566, 7482, 6851, 7498, 10375, 7995, 7617, 4112, 673, 7924, 8168, 7381, 1284, 7941, 8459, 7202, 8879, 7633, 7803, 3625, 8076, 846, 1320, 4616, 7816, 7432, 7966, 10128, 8014, 8846, 8199, 7626, 1156, 4411, 1167, 2001, 2669, 7434, 5035, 5598, 6647, 8532, 965, 2228, 3044, 3755, 4370, 5615, 6203, 6940, 8186, 8242, 2015, 5677, 7979, 8488, 6928, 8194, 8137, 8497, 8469, 4216, 7968, 2015, 8909, 8461, 7183, 7521, 1081, 4138, 6686, 7403, 1961, 7017, 7809, 7607, 7771, 759, 1115, 7500, 1266, 10478, 2611, 8893, 519, 1253, 2086, 3270, 3985, 4502, 6086, 6935, 7385, 1962, 3113, 3629, 4113, 4395, 4664, 4878, 5478, 5713, 6529, 8712, 8134, 4732, 7962, 7223, 7992, 470, 8433, 7540, 3293, 8415, 8188, 8860]
		stamps[('G_AWAY', 'LO', 'VC')] = [8645, 5690, 7513, 6930, 6291, 7908, 6770, 7772, 9218, 4768, 7884, 8668, 5256, 9647, 5040, 8125, 3803, 6217, 7367, 2749, 6602, 584, 1037, 2050, 2484, 5400, 7618, 4617, 7509, 3082, 1525, 3275, 5391, 6029, 740, 4558, 4992, 5375, 5742, 6091, 6492, 6924, 7224, 7874, 8191, 8691, 3733, 4433, 5133, 6117, 7334, 5605, 5973, 4514, 3165, 7476, 4396, 2283, 7132, 1784, 2147, 2663, 3462, 8020, 5757, 9177, 3340, 5073, 2404, 8640, 5259, 1324, 5997, 8611, 3268, 8144, 6277, 7932, 7281, 5908, 8775, 5949, 5753, 7335, 1166, 6200, 4718, 6033, 6358, 5787, 1036, 5770, 6961, 5217, 2165, 6415, 5380, 4155, 6431, 949, 1514, 2086, 2365, 3848, 2445, 4878, 6777, 7244, 9700, 7208, 6323, 6939, 7816, 8877, 9861, 1670, 3644, 8749, 5051, 8356, 3090, 3907, 4842]
		stamps[('G_AWAY', 'LO', 'VD')] = [980, 3820, 3928, 671, 3919, 5384, 6783, 4219, 2396, 2162, 968, 7353, 2784, 8144, 6416, 7435, 7271, 737, 3204, 2100, 8752, 6473, 8139, 602, 4202, 1804, 5034, 3287, 6170, 7992, 5008, 6858, 7391, 1791, 7755, 5412, 6266, 1570, 7202, 5105, 2742, 7686, 7236, 3620, 5375, 2330, 2499, 5530, 3505, 8431, 2143, 3050, 7866, 7180, 6380, 8332, 9047, 4432, 7408, 4405, 2931, 3167, 6852, 8586, 4196, 8087, 1861, 4793, 1591]
		stamps[('G_AWAY', 'LO', 'VE')] = [9753, 8256, 4567, 3400, 4288, 6218, 7421, 2822, 2134, 2734, 3517, 5184, 4668, 7304, 821, 1338, 1688, 1989, 2353, 2622, 2873, 3122, 3256, 3488, 3654, 3838, 4020, 4188, 4339, 4521, 4721, 5088, 5203, 1311, 6304, 7205, 7806, 8223, 5616, 8125, 557, 3931, 6542, 6959, 3175, 611, 2844, 2265, 2733, 3266, 1256, 3479, 9079, 7072, 7506, 3903, 7997, 8897, 7491, 7026, 4370, 9850, 4246, 4723, 890, 2868, 2244, 2272, 682, 5808, 6833, 2117, 4090, 3462, 6730, 7186, 8069, 3814, 1225, 6348, 7156, 5335, 4292, 936]
		export_path_moments_confusion_for_each_goal(r, best_paths, stamps, exp_settings)

	export_path_options_for_each_goal(r, best_paths, exp_settings)
	
	get_envir_cache(r, exp_settings)

	# print(fn_export_from_exp_settings(exp_settings))

	export_table_all_viewers(r, best_paths, exp_settings)


	# print(best_paths)


def do_exp(lam, astr, rb, km):
	# Run the scenario that aligns with our use case
	restaurant = experimental_scenario_single()
	unique_key = 'exp_single'
	start = restaurant.get_start()
	all_goals = restaurant.get_goals_all()

	sample_pts = []
	# sampling_type = SAMPLE_TYPE_CENTRAL
	# sampling_type = SAMPLE_TYPE_DEMO
	# sampling_type = SAMPLE_TYPE_CENTRAL_SPARSE
	# sampling_type = SAMPLE_TYPE_FUSION
	# sampling_type = SAMPLE_TYPE_SPARSE
	# sampling_type = SAMPLE_TYPE_SYSTEMATIC
	# sampling_type = SAMPLE_TYPE_HARDCODED
	# sampling_type = SAMPLE_TYPE_VISIBLE
	# sampling_type = SAMPLE_TYPE_INZONE
	# sampling_type = SAMPLE_TYPE_CURVE_TEST
	# sampling_type = SAMPLE_TYPE_NEXUS_POINTS
	sampling_type = SAMPLE_TYPE_NEXUS_POINTS_ONLY


	OPTION_DOING_STATE_LATTICE = False
	if OPTION_DOING_STATE_LATTICE:
		for i in range(len(all_goals)):
			goal = all_goals[i]
			# lane_state_sampling_test1(resto, goal, i)
			make_path_libs(resto, goal)

	exp_settings = {}
	exp_settings['title'] 			= unique_key
	exp_settings['sampling_type'] 	= sampling_type
	exp_settings['resolution']		= 15
	exp_settings['f_vis_label']		= 'no-chunk'
	exp_settings['epsilon'] 		= 0 #1e-12 #eps #decimal.Decimal(1e-12) # eps #.000000000001
	exp_settings['lambda'] 			= lam #decimal.Decimal(1e-12) #lam #.000000000001
	exp_settings['num_chunks']		= 50
	exp_settings['chunk-by-what']	= chunkify.CHUNK_BY_DURATION
	exp_settings['chunk_type']		= chunkify.CHUNKIFY_LINEAR	# CHUNKIFY_LINEAR, CHUNKIFY_TRIANGULAR, CHUNKIFY_MINJERK
	exp_settings['angle_strength']	= astr #40
	exp_settings['min_path_length'] = {}
	exp_settings['is_denominator']	= False
	exp_settings['f_vis']			= f_exp_single_normalized
	exp_settings['kill_1']			= km
	exp_settings['angle_cutoff']	= 70
	exp_settings['fov']	= 120
	exp_settings['prob_og']			= False
	exp_settings['right-bound']		= rb
	exp_settings['waypoint_offset']	= 20

	pprint.pprint(exp_settings)
	print("---!!!---")

	# Preload envir cache for faster calculations
	# envir_cache = get_envir_cache(restaurant, exp_settings)
	# restaurant.set_envir_cache(envir_cache)

	print("Prepped environment")
	# print(envir_cache)

	# SET UP THE IMAGES FOR FUTURE DRAWINGS
	img = restaurant.get_img()
	cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_empty.png', img)

	min_paths = []
	for g in restaurant.get_goals_all():
		print("Finding min path for goal " + str(g))
		min_path_length = get_min_viable_path_length(restaurant, g, exp_settings)
		exp_settings['min_path_length'][g] = min_path_length
		
		min_path = get_min_viable_path(restaurant, g, exp_settings)
		min_paths.append(min_path)

	title = title_from_exp_settings(exp_settings)
	resto.export_raw_paths(restaurant, img, min_paths, title, fn_export_from_exp_settings(exp_settings) + "_all" + "-min")

	paths_for_analysis = {}
	#  add permutations of goals with some final-angle-wiggle
	for goal in all_goals:
		print("Generating paths for goal " + str(goal))
		paths, sample_pts_that_generated = create_systematic_path_options_for_goal(restaurant, exp_settings, start, goal, img, num_paths=10)
		print("Made paths")
		paths_for_analysis[goal] = {'paths':paths, 'sp': sample_pts_that_generated}

	# debug curvatures
	# plt.clf()
	# print(curvatures)
	# sns.histplot(data=curvatures, bins=100)
	# # plt.hist(curvatures, bins=1000) 
	# plt.title("histogram of max angles")
	# plt.tight_layout()
	# plt.savefig("path_assessment/curvatures.png") 
	# plt.clf()

	# sns.histplot(data=max_curvatures, bins=100)
	# # plt.hist(curvatures, bins=1000) 
	# plt.title("histogram of max curvatures")
	# plt.tight_layout()
	# plt.savefig("path_assessment/max-curvatures.png") 
	# exit()

	print("~~~")
	best_paths = analyze_all_paths(restaurant, paths_for_analysis, exp_settings)
		# print(best_paths.keys())

	file1 = open(fn_export_from_exp_settings(exp_settings) + "_BEST_PATHS.txt","w+")
	file1.write(str(best_paths))
	file1.close()

	# print(best_paths)



	# # Set of functions for exporting easy paths
	# title = "pts_" + str(exp_settings['num_chunks']) + "_" + str(exp_settings['angle_strength']) + "_" + str(exp_settings['chunk_type']) + " = "
	# path_a = best_paths[((1035, 307, 180), 'omniscient')]
	# path_b = best_paths[((1035, 567, 0), 'omniscient')]

	# path_a = str(path_a)
	# path_b = str(path_b)

	# print(title + path_a)
	# print(title + path_b)
	print("Number of bugs per calculation:")
	print(bug_counter)

	print("Done with experiment")

def exp_determine_lam_eps():
	lam_vals = []
	eps = 1e-7
	# eps_vals = []
	# # exit()
	# * 1e-6
	for i in range(-5, -10, -1):
	# for i in np.arange(1.1, 2, .1):
		new_val = 10 ** i 
	# 	eps_vals.append(new_val)
		lam_vals.append(new_val)
	# 	# lam_vals.append(new_val)

	# print("REMIX TIME")
	# for eps in eps_vals:
	lam = 0
	angle_strs = [520]
	rbs = [55]
	
	print("WILDIN")
	for astr in angle_strs:
		for rb in rbs:
			do_exp(lam, astr, rb, False)
	# pass

def main():
	export_best_options()
	exit()

	# lam = 1e-6
	lam = 0 #1e-6 #(10.0 / 23610)#1e-6 #0 #1.5e-8 #1e-16
	kill_mode = True
	# eps_start = decimal.Decimal(.000000000001)
	# lam_start = decimal.Decimal(.00000000001)
	astr = 500
	rb = 40

	# print("Doing main")
	do_exp(lam, astr, rb, False)
	# print("Done with main")
	# export_best_options()
	# exp_determine_lam_eps()

if __name__ == "__main__":

	# con = decimal.getcontext()
	# con.prec = 35
	# con.Emax = 1000
	# con.Emin = -1000
	main()






# result = df_vis.pivot(index='x', columns='y', values=resto.VIS_MULTI)
# inspect_heatmap(df_vis)

# print("pivoted")
# heatmap = sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
# print("made heatmap")

# fig = heatmap.get_figure()
# fig.savefig("multi-vis.png")
# print("Graphs")

# resto.draw_paths(r, paths_dict)
# resto.export(r, paths_dict)
