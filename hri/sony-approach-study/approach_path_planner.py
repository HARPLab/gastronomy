import table_path_code as resto
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import cv2
import matplotlib.pylab as plt
import math
import copy
import decimal
import random



# start 		= r.get_start()
# goals 		= r.get_goals_all()
# goal 		= r.get_current_goal()
# observers 	= r.get_observers()
# tables 		= r.get_tables()
# waypoints 	= r.get_waypoints()
# SCENARIO_IDENTIFIER = r.get_scenario_identifier()

FLAG_SAVE 			= True
FLAG_VIS_GRID 		= False
VISIBILITY_TYPES 	= resto.VIS_CHECKLIST
NUM_CONTROL_PTS 	= 3

NUMBER_STEPS = 30

PATH_TIMESTEPS = 15

resto_pickle = 'pickle_vis'
vis_pickle = 'pickle_resto'
FILENAME_PATH_ASSESS = 'path_assessment/'

FLAG_PROB_HEADING = False
FLAG_PROB_PATH = True

# PATH_COLORS = [(138,43,226), (0,255,255), (255,64,64), (0,201,87)]


def f_cost_old(t1, t2):
	return resto.dist(t1, t2)

def f_cost(t1, t2):
	a = resto.dist(t1, t2)

	return np.abs(a * a)

def f_path_cost(path):
	cost = 0
	for i in range(len(path) - 1):
		cost = cost + f_cost(path[i], path[i + 1])

	return cost


def f_audience_agnostic():
	return f_leg_personalized()
	pass

def f_leg_personalized():

	pass

def f_convolved(val_list, f_function):
	tstamps = range(len(val_list))
	ret = []
	for t in tstamps:
		ret.append(f_function(t) * val_list[t])
	return ret




# Given the observers of a given location, in terms of distance and relative heading
def f_vis4(p, df_obs):
	dist_units = 100
	angle_cone = 60
	distance_cutoff = 500

	# Given a list of entries in the format 
	# ((obsx, obsy), angle, distance)
	if len(df_obs) == 0:
		return 0
	
	vis = 0
	for obs in df_obs:
		if obs == None:
			return 0
		else:
			angle, dist = obs.get_obs_of_pt(p)

		if angle < angle_cone and dist < distance_cutoff:
			vis += (distance_cutoff - dist) * (np.abs(angle_cone - angle) / angle) 

	return vis

# Given the observers of a given location, in terms of distance and relative heading
def f_vis3(p, df_obs):
	# dist_units = 100
	angle_cone = 135.0 / 2.
	distance_cutoff = 2000


	# Given a list of entries in the format 
	# ((obsx, obsy), angle, distance)
	if len(df_obs) == 0:
		return 1
	
	vis = 0
	for obs in df_obs:	
		if obs == None:
			return 0
		else:
			angle, dist = obs.get_obs_of_pt(p)

		if angle < angle_cone and dist < distance_cutoff:
			vis += np.abs(angle_cone - angle)

	return vis

def f_vis_exp1(t, pt, aud):
	return (f_og(t) * f_vis3(pt, aud)) + .000001




def f_vis2(p, df_obs):
	dist_units = 100
	angle_cone = 60
	distance_cutoff = 500

	# Given a list of entries in the format 
	# ((obsx, obsy), angle, distance)
	if len(df_obs) == 0:
		return 0
	
	vis = 0
	for obs in df_obs:
		if obs == None:
			return 0
		else:
			angle, dist = obs.get_obs_of_pt(p)

		if angle < angle_cone and dist < distance_cutoff:
			vis += (distance_cutoff - dist)

	return vis

def f_vis1(p, df_obs):
	dist_units = 100
	angle_cone = 60
	distance_cutoff = 500


	# Given a list of entries in the format 
	# ((obsx, obsy), angle, distance)
	if len(df_obs) == 0:
		return 0
	
	vis = 0
	for obs in df_obs:
		if obs == None:
			return 0
		else:
			angle, dist = obs.get_obs_of_pt(p)
		if angle < angle_cone and dist < distance_cutoff:
			vis += 1

	return vis

# // NOT IN USE
# Given the observers of a given location, in terms of distance and relative heading
def f_visibility(df_obs):
	dist_units = 100
	angle_cone = 400
	distance_cutoff = 500

	# Given a list of entries in the format 
	# ((obsx, obsy), angle, distance)
	if len(df_obs) == 0:
		return 0
	
	vis = 0
	for obs in df_obs:	
		pt, angle, dist = obs.get_visibility_of_pt_raw(pt)
		if angle < angle_cone and dist < distance_cutoff:

			vis += (distance_cutoff - dist)

	return vis

def f_og(t, path):
	# len(path)
	return NUMBER_STEPS - t

def f_og(t):
	return PATH_TIMESTEPS - t

def f_novis(t, obs):
	return 1

# def f_vis_eqn(observers):
# 	value = 0

# 	for person in observers:
# 		if ob

# 	return value

def get_visibility_of_pt_w_observers(pt, aud):
	observers = []
	score = 0

	MAX_DISTANCE = 500
	for observer in aud:
		obs_orient 	= observer.get_orientation()
		obs_FOV 	= observer.get_FOV()

		angle 		= resto.angle_between(pt, observer.get_center())
		distance 	= resto.dist(pt, observer.get_center())

		# print(angle, distance)
		observation = (pt, angle, distance)
		observers.append(observation)


		if angle < obs_FOV:
			# full credit at the center of view
			offset_multiplier = np.abs(obs_FOV - angle) / obs_FOV

			# 1 if very close
			distance_bonus = (MAX_DISTANCE - distance) / MAX_DISTANCE
			score += (distance_bonus*offset_multiplier)

	return score


# def f_vis_t3(t, pt, aud):
# 	return (f_vis3(pt, aud)) + 1



def f_remix1(t, pt, aud):
	return (f_og(t) * f_vis1(pt, aud)) + 1

def f_remix2(t, pt, aud):
	return (f_og(t) * f_vis2(pt, aud)) + 1

def f_remix3(t, pt, aud, path):
	val = (f_og(t, path) * f_vis3(pt, aud)) 

	# if f_vis3(pt, aud) < 0:
	# 	print(val)
	# 	print(f_og(t))
	# 	print(f_vis3(pt, aud))
	# 	print(aud)
	# 	exit()

	return val


# def f_remix3(t, pt, aud):
# 	return (f_og(t) * f_vis3(pt, aud)) + .000001

def f_remix4(t, pt, aud):
	return (f_og(t) * f_vis4(pt, aud)) + 1

def f_remix_novis(t, pt, aud):
	# novis just always returns 1, so this is f_og
	return f_og(t) * f_novis(pt, aud) + 1


def f_remix(t, p1, p2, aud):
	epsilon = .0001
	multiplier = (PATH_TIMESTEPS - t)

	vis1 = get_visibility_of_pt_w_observers(p1, aud)
	vis2 = get_visibility_of_pt_w_observers(p2, aud)
	vis_aggregate = vis1 + vis2 / 2.0

	return (multiplier * vis_aggregate) + epsilon

# TODO Cache this result for a given path so far and set of goals
def prob_goal_given_path(start, p_n1, pt, goal, goals, cost_path_to_here):
	g_array = []
	g_target = 0
	for g in goals:
		p_raw = unnormalized_prob_goal_given_path(start, p_n1, pt, g, goals, cost_path_to_here)
		g_array.append(p_raw)
		if g is goal:
			g_target = p_raw

	if(sum(g_array) == 0):
		return 0

	return g_target / (sum(g_array))


def unnormalized_prob_goal_given_path(start, p_n1, pt, goal, goals, cost_path_to_here):
	decimal.getcontext().prec = 20

	c1 = decimal.Decimal(cost_path_to_here)
	c2 = decimal.Decimal(f_cost(pt, goal))
	c3 = decimal.Decimal(f_cost(start, goal))

	a = np.exp((-c1 + -c2))
	b = np.exp(-c3)
	ratio 		= a / b
	# silly_ratio = (c1 + c2) / (c3)

	# print(c1)
	# print(c2)
	# print(c3)
	# print(ratio)
	# exit()

	# ratio = silly_ratio
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
	# print(diff)

	# if diff < 90:
	# 	return diff

	# return 0.0


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


# Given a 
def f_legibility(goal, goals, path, aud, f_vis_convo):
	legibility = decimal.Decimal(0)
	divisor = decimal.Decimal(0)
	total_dist = decimal.Decimal(0)
	LAMBDA = decimal.Decimal(.0000002)

	start = path[0]
	total_cost = decimal.Decimal(0)
	aug_path = get_costs_along_path(path)

	t = 1
	p_n = path[0]
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_vis_convo(t, pt, aud, path))
		# print(f)
		# f = f_remix(t, p_n, pt, aud)
		prob_goal_given = prob_goal_given_path(start, p_n, pt, goal, goals, cost_to_here)

		legibility += prob_goal_given * f
		
		total_cost += decimal.Decimal(f_cost(p_n, pt))
		p_n = pt

		divisor = f
		t = t + 1

	if divisor == 0:
		legibility = 0
	else:
		legibility = (legibility / divisor)
		divisor += f
		t = t + 1

	legibility = (legibility / divisor)

	total_cost =  - LAMBDA*total_cost
	overall = legibility + total_cost

	# print(legibility, total_cost)

	return overall

def get_costs(path, target, obs_sets):
	vals = []

	for aud in obs_sets:
		new_val = f_cost()

	return vals

def get_visibilities(path, target, goals, obs_sets):
	vis_labels 		= ['vis1-flat', 'vis2-dist', 'vis3-angle', 'vis4-angle-dist', 'no-vis']
	# vis_functions 	= [f_vis1, 	f_vis2, f_vis3, f_vis4]
	# vis_lists 		= [[], 		[], 	[], 	[]]
	# vis_totals 		= [0, 		0, 		0, 		0]

	# v1, v2, v3, v4, v5 = [], [], [], [], []
	
	# if obs_sets == []:
	# 	return vis_labels, None

	# obs = obs_sets[-1]

	# for p in path:
	# 	v1.append(f_vis1(p, obs))
	# 	v2.append(f_vis2(p, obs))
	# 	v3.append(f_vis3(p, obs))
	# 	v4.append(f_vis4(p, obs))
	# 	v5.append(f_novis(p, obs))

	# vis_values = [v1, v2, v3, v4, v5]

	# # 'vis3-angle'
	# vis_labels = [vis_labels[2]]
	# vis_values = [vis_values[2]]

	v1, v2, v3, v4, v5 = [], [], [], [], []
	
	if obs_sets == []:
		return vis_labels, None

	obs = obs_sets[-1]

	for p in path:
		v1.append(f_vis1(p, obs))
		v2.append(f_vis2(p, obs))
		v3.append(f_vis3(p, obs))
		v4.append(f_vis4(p, obs))
		v5.append(f_novis(p, obs))

	vis_values = [v1, v2, v3, v4, v5]

	# 'vis3-angle'
	vis_labels = [vis_labels[2]]
	vis_values = [vis_values[2]]

	return vis_labels, vis_values



def get_legibilities(path, target, goals, obs_sets, f_vis):
	vals = []

	for aud in obs_sets:
		# goal, goals, path, df_obs
		new_val = f_legibility(target, goals, path, aud, f_vis)

		vals.append(new_val)

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

        print(change)
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
	blocky_path = construct_single_path(restaurant.get_start(), target, sample_pts)
	path 		= smoothed(blocky_path, restaurant)
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
	cx, cy = start
	targets = sample_pts + [end]

	# print(sample_pts)

	for target in targets:
		# print(target)
		tx, ty = target
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

def is_valid_path(restaurant, path):
	tables = restaurant.get_tables()

	for t in tables:
		for i in range(len(path) - 1):
			pt1 = path[i]
			pt2 = path[i + 1]
			
			if t.intersects_line(pt1, pt2):
				return False
	return True


def construct_single_path_bezier(start, end, sample_pts):
	points = []
	
	xys = [start] + sample_pts + [end]

	ts = [t/NUMBER_STEPS for t in range(NUMBER_STEPS + 1)]
	bezier = resto.make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points

def generate_n_paths(restaurant, num_paths, target, n_control):
	path_list = []
	vis_type = resto.VIS_OMNI

	for i in range(num_paths):
		valid_path = False
		# while (not valid_path):
		path_option = generate_single_path_grid(restaurant, target, vis_type, n_control)
		# path_option = generate_single_path(restaurant, target, vis_type, n_control)	

		path_list.append(path_option)
		# print(path_option)

	return path_list

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

def add_further_overall_stats(df):

	return df

def get_obs_sets(r):
	obs_none 	= []
	obs_a 		= [r.get_observer_back()]
	obs_b 		= [r.get_observer_towards()]
	obs_multi 	= [r.get_observer_back(), r.get_observer_towards()]

	obs_sets = [obs_none, obs_a, obs_b, obs_multi]

	return obs_sets


def get_path_analysis(all_paths, r, ti):
	target = r.get_goals_all()[ti]	


	obs_sets = get_obs_sets(r)

	goals = r.get_goals_all()
	col_labels = ['cost', 'target', 'path', 'target_index']

	# vis_labels = get_vis_labels()
	# f_list = [f_remix1, f_remix2, f_remix3, f_remix4, f_remix_novis]
	# f_list = [f_vis_t3, f_remix3]
	# f_labels = ['fvis','fcombo']

	f_list = [f_remix3]
	f_labels = ['fcombo']


	leg_labels = ['lo', 'la', 'lb', 'lm']

	data = []
	for p in all_paths:
		# Do analytics that are constant for all views of path, such as cost
		# these are the pre-listed options in col_labels
		cost = f_path_cost(p)
		vis_types, vis_values = get_visibilities(p, target, goals, obs_sets)
		# print(vis_types)
		# print(len(vis_values))

		entry = [cost, target, p, ti]
		remix_labels = []

		# Record the graphs of visibility for this path
		entry.extend(vis_values)
		remix_labels.extend(vis_types)

		#####

		# for each of the options of f functions
		for fi in range(len(f_list)):
			f_vis = f_list[fi]
			f_label = f_labels[fi]

			# For the legibility relative to each of these other audiences
			l_o, l_a, l_b, l_m = get_legibilities(p, target, goals, obs_sets, f_vis)

			max_labels = copy.copy(leg_labels)
			# ratio_labels = copy.copy(leg_labels)

			# make labels for each assessment criteria
			for i in range(len(max_labels)):
				max_labels[i] 		= "max-" + max_labels[i] + "-" + f_label


				# ratio_labels[i] = "ratio-" + ratio_labels[i] + "-" + f_label

			denominator = l_o + l_a + l_b + l_m

			remix_labels.extend(max_labels)
			entry.extend([l_o, l_a, l_b, l_m])

			# print(remix_labels)

			# if denominator == 0:
			# 	ratio_values = [0, 0, 0, 0]
			# else:
			# 	ratio_values = [(l_o / denominator), (l_a / denominator), (l_b / denominator), (l_m / denominator)]

			# remix_labels.extend(ratio_labels)
			# entry.extend(ratio_values)

		data.append(entry)

	col_labels.extend(remix_labels)
	df = pd.DataFrame(data, columns = col_labels)
	df = df.fillna(0)

	df = add_further_overall_stats(df)

	return df


def get_legib_label_combos():
	vis_labels = get_vis_labels()
	leg_labels = ['lo', 'la', 'lb', 'lm']

	all_labels = []

	for v in vis_labels:
		labels = copy.copy(leg_labels)
		for i in range(len(labels)):
			labels[i] = labels[i] + "-" + v
		all_labels.extend(labels)

	return all_labels

def get_ratio_label_combos():
	vis_labels = get_vis_labels()
	leg_labels = ['lo', 'la', 'lb', 'lm']

	all_labels = []

	for v in vis_labels:
		labels = copy.copy(leg_labels)
		for i in range(len(labels)):
			labels[i] = "ratio-" + labels[i] + "-" + v
		all_labels.extend(labels)

	return all_labels


def get_vis_labels():
	vis_labels, dummy = get_visibilities([], [], [], [])
	return vis_labels

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

# Given a set of paths, get all analysis and log it
def assess_paths(all_paths, r, ti, unique_key):
	target = r.get_goals_all()[ti]
	goal_label = resto.UNITY_GOAL_NAMES[ti]
	df = get_path_analysis(all_paths, r, ti)

	df_minmax = df.apply(minMax)
	# print(df_minmax)
	# print(df_minmax['l_agnostic'])
	# print(df_minmax['l_a'])
	# print(df_minmax['l_b'])
	# print(df_minmax['l_multi'])
	csv_title = FILENAME_PATH_ASSESS + unique_key + str(goal_label) +  '-scores.csv'
	# print(csv_title)
	df.to_csv(csv_title)

	leg_labels = ['lo', 'la', 'lb', 'lm']	
	path_key = 'path'
	path_keys = resto.VIS_CHECKLIST

	best_list 	= []
	worst_list 	= []

	paths_dict = {}
	raw_dict = {}

	inspection_labels = df.columns[5:]
	# print(inspection_labels)

	# inspection_labels = get_legib_label_combos()
	# print(inspection_labels)
	# inspection_labels.extend(get_ratio_label_combos())

	for li in range(len(inspection_labels)):
		l = inspection_labels[li]
		
		# print(l)
		# print(df[l].max())

		max_val = df[l].max()
		min_val = df[l].min()

		print(l)
		print(max_val)

		# if df[l].idxmax()

		best 	= df.loc[df[l] == max_val].iloc[0]
		worst 	= df.loc[df[l] == min_val].iloc[0]
		
		best_path 	= best[path_key]
		worst_path 	= worst[path_key]

		best_list.append(best_path)
		worst_list.append(worst_path)

		# paths_dict[path_keys[li]] = [best_path, worst_path]
		# raw_dict[path_keys[li]] = [best, worst]

		paths_dict[l] = [best_path]
		raw_dict[l] = [best]


	return paths_dict, raw_dict

def iterate_on_paths():
	path_options 		= generate_paths(NUM_PATHS, r, VISIBILITY_TYPES)
	path_dict, path_assessments 	= assess_paths(path_options)

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
	vis_labels = get_vis_labels()
	vl3 = vis_labels[2]
	options = options[0]

	for pkey in options.keys():
		print(pkey)
		path = options[pkey][0]
		# print('saving fig')


		t = range(len(path))
		v = get_vis_graph_info(path, restaurant)
		vo, va, vb, vm = v

		fig = plt.figure()
		ax1 = fig.add_subplot(111)

		print(len(t))
		print(len(vo))

		print(t)
		print(vo)

		ax1.scatter(t, vo, s=10, c='r', marker="o", label="Vis Omni")
		ax1.scatter(t, va, s=10, c='b', marker="o", label="Vis A")
		ax1.scatter(t, vb, s=10, c='y', marker="o", label="Vis B")
		ax1.scatter(t, vm, s=10, c='g', marker="o", label="Vis Multi")
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
	vals = [[], [], [], []]

	obs_sets = get_obs_sets(restaurant)

	for t in range(len(path)):
		for aud_i in range(len(obs_sets)):
			# goal, goals, path, df_obs
			new_val = f_remix3(t, path[t], obs_sets[aud_i], path)
			# print(new_val)
			# exit()

			vals[aud_i].append(new_val)

	return vals
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

	labels = ['max-lo-fcombo', 'max-la-fcombo', 'max-lb-fcombo', 'max-lm-fcombo']
	p1 = [(104, 477), (141, 459), (178, 444), (215, 430), (251, 417), (287, 405), (322, 395), (357, 386), (391, 379), (425, 373), (459, 368), (492, 365), (525, 363), (557, 363), (588, 364), (620, 366), (651, 370), (681, 375), (711, 381), (740, 389), (769, 398), (798, 409), (826, 421), (854, 434), (881, 449), (908, 465), (934, 483), (960, 502), (985, 522), (1010, 543), (1035, 567)]
	p2 = [(104, 477), (147, 447), (190, 419), (231, 394), (272, 371), (312, 350), (351, 331), (390, 315), (427, 301), (464, 289), (499, 280), (534, 273), (568, 268), (601, 265), (634, 265), (665, 267), (696, 271), (726, 277), (755, 286), (783, 297), (810, 310), (836, 325), (862, 343), (886, 363), (910, 385), (933, 410), (955, 437), (976, 466), (996, 497), (1016, 531), (1035, 567)]
	p3 = [(104, 477), (124, 447), (145, 419), (167, 394), (190, 371), (213, 350), (237, 332), (262, 315), (288, 301), (314, 290), (341, 280), (369, 273), (397, 268), (427, 266), (457, 265), (487, 267), (519, 271), (551, 278), (584, 286), (617, 297), (652, 310), (687, 326), (722, 343), (759, 363), (796, 386), (834, 410), (873, 437), (912, 466), (952, 497), (993, 531), (1035, 567)]
	p4 = [(104, 477), (146, 446), (187, 418), (228, 392), (268, 369), (307, 348), (345, 329), (383, 313), (420, 298), (456, 286), (491, 277), (525, 269), (559, 264), (592, 262), (624, 261), (656, 263), (686, 267), (716, 274), (745, 282), (774, 293), (801, 307), (828, 322), (854, 340), (879, 361), (904, 383), (928, 408), (950, 435), (973, 464), (994, 496), (1015, 530), (1035, 567)]
	p5 = [(104, 477), (98, 509), (95, 540), (95, 569), (97, 596), (101, 620), (108, 643), (118, 663), (130, 682), (145, 698), (162, 712), (182, 725), (204, 735), (229, 743), (256, 749), (286, 753), (318, 755), (353, 755), (390, 753), (430, 749), (472, 742), (517, 734), (565, 724), (615, 711), (667, 697), (722, 680), (779, 662), (839, 641), (902, 618), (967, 593), (1035, 567)]

	options = {}
	options[labels[0]] = [p5]
	options[labels[1]] = [p1]
	options[labels[2]] = [p2]
	options[labels[3]] = [p3]

	return options

# MAIN METHOD
def select_paths_and_draw(restaurant, unique_key):
	# TODO import old good paths for further analysis
	# hand-coded
	# best x of the past

	NUM_PATHS = 500

	unique_key = "" + unique_key + "_"

	img = restaurant.get_img()
	empty_img = cv2.flip(img, 0)
	cv2.imwrite(FILENAME_PATH_ASSESS + unique_key + 'empty.png', empty_img)
	goals = restaurant.get_goals_all()

	all_options = []

	# Option for exporting a specific set of saved paths
	# options = get_hardcoded()
	# fn = FILENAME_PATH_ASSESS + "presentation-drama.png"
	# resto.export_assessments_by_criteria(img, options, fn=fn)
	# exit()


	# # Reversed so most interesting done first
	# for ti in range(len(goals))[::-1]:

	# Decide how many control points to provide
	for ti in range(len(goals)):
		all_paths = []
		target = goals[ti]
		for n_control in range(1, 2):

			paths = generate_n_paths(restaurant, NUM_PATHS, target, n_control)
			fn = FILENAME_PATH_ASSESS + unique_key + "g" + str(ti) + "-pts=" + str(n_control) + "-" + "-all.png"
			resto.export_raw_paths(img, paths, fn)
			all_paths.extend(paths)

		options, details = assess_paths(all_paths, restaurant, ti, unique_key)
		all_options.append(options)
		print("Completed assessment")

		fn = FILENAME_PATH_ASSESS + "vis_" + unique_key + "g" + str(ti) + "-"
		inspect_visibility(all_options, restaurant, ti, fn)
		resto.export_assessments_by_criteria(img, options, fn=fn)


	options = combine_list_of_dicts(all_options)
	print(options)




def unity_scenario():
	generate_type = resto.TYPE_EXP_SINGLE

	# SETUP FROM SCRATCH AND SAVE
	if FLAG_SAVE:
		# Create the restaurant scene from our saved description of it
		r 	= resto.Restaurant(generate_type)
		print("PLANNER: get visibility info")

		if FLAG_VIS_GRID:
			# If we'd like to make a graph of what the visibility score is at different points
			df_vis = r.get_visibility_of_pts_pandas(f_visibility)

			dbfile = open(vis_pickle, 'ab') 
			pickle.dump(df_vis, dbfile)					  
			dbfile.close()
			print("Saved visibility map")

			df_vis.to_csv('visibility.csv')
			print("Visibility point grid created")
		
		# pickle the map for future use
		dbfile = open(resto_pickle, 'ab') 
		pickle.dump(r, dbfile)					  
		dbfile.close()
		print("Saved restaurant maps")

	# OR LOAD FROM FILE
	else:
		dbfile = open(resto_pickle, 'rb')
		r = pickle.load(dbfile)
		print("Imported pickle of restaurant")


		if FLAG_VIS_GRID:
			dbfile = open(vis_pickle, 'rb')
			df_vis = pickle.load(dbfile)
			print("Imported pickle of vis")


	select_paths_and_draw(r, "mainexp")

def main():
	# Run the scenario that aligns with our use case
	unity_scenario()





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

print("Done")