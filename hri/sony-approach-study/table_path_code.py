import numpy as np
import math
import random
import copy
import cv2
import pandas as pd
import pickle
import json
import seaborn as sns
import matplotlib.pylab as plt
import sys
from PIL import Image
from PIL import ImageDraw

from shapely.geometry import Point as fancyPoint
from shapely.geometry import box as fancyBox
from shapely.geometry import Polygon as fancyPolygon
from shapely.geometry import LineString as fancyLineString

# import custom libraries from PythonRobotics
# sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/ModelPredictiveTrajectoryGenerator/')
# sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/StateLatticePlanner/')

# print(sys.path)

from collections import defaultdict

OPTION_SHOW_VISIBILITY = True
OPTION_FORCE_GENERATE_VISIBILITY = False
OPTION_FORCE_GENERATE_OBSTACLE_MAP = True
OPTION_EXPORT = True
FLAG_MAKE_OBSTACLE_MAP = False

EXPORT_JSON = True
EXPORT_CSV = True
EXPORT_DIAGRAMS = True

# window dimensions
length = 409
width = 451
pixels_to_foot = 20

resolution_visibility = 1
resolution_planning = 30

# divide by two because these are radiuses
DIM_TABLE_RADIUS = int(1 * pixels_to_foot / 2.0)
DIM_OBSERVER_RADIUS = int(1.5 * pixels_to_foot / 2.0)
DIM_ROBOT_RADIUS = int(3 * pixels_to_foot / 2.0)
DIM_NAVIGATION_BUFFER = int(2.5 * pixels_to_foot)

num_tables = 6
num_observers = 6

# Choose the table layout for the scene
TYPE_PLOTTED = 0
TYPE_RANDOM = 1
TYPE_UNITY_ALIGNED = 2
TYPE_CUSTOM = 3
TYPE_EXP_SINGLE = 4

# Color options for visualization
COLOR_TABLE = (32, 85, 230) #(235, 64, 52) 		# dark blue
COLOR_OBSERVER = (32, 85, 230) 		# dark orange
COLOR_FOCUS_BACK = (52, 192, 235) 		# dark yellow
COLOR_PERIPHERAL_BACK = (178, 221, 235) 	# light yellow
COLOR_FOCUS_TOWARDS = (235, 64, 52)		# dark yellow
COLOR_PERIPHERAL_TOWARDS = (55, 120, 191) 	# light yellow
COLOR_GOAL = (255, 255, 255) # (50, 168, 82) 			# green
COLOR_START = (100, 100, 100) 		# white

COLOR_OBSTACLE_CLEAR = (0, 0, 0)
COLOR_OBSTACLE_BUFFER = (100, 100, 100)
COLOR_OBSTACLE_FULL = (255, 255, 255)

# Verify that these are correctly in degrees/radians
DIR_NORTH 	= 0
DIR_EAST 	= 90
DIR_SOUTH 	= 180
DIR_WEST 	= 270

VIS_INFO_RESOLUTION = -1
VIS_ALL = "VIS_ALL"
VIS_OMNI = "VIS_OMNI"
VIS_MULTI = "VIS_MULTI"
VIS_A = "VIS_A"
VIS_B = "VIS_B"
VIS_C = "VIS_C"
VIS_D = "VIS_D"
VIS_E = "VIS_E"

OBS_ALL = 'all'
OBS_NONE = 'omni'
OBS_KEY_ALL = 'all'
OBS_KEY_NONE = 'omni'
OBS_KEYS = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
OBS_KEY_A = 'a'
OBS_KEY_B = 'b'
OBS_KEY_C = 'c'
OBS_KEY_D = 'd'
OBS_KEY_E = 'e'

SUFFIX_RAW 	= "-raw"
RAW_ALL 	= VIS_ALL 	+ SUFFIX_RAW
RAW_OMNI 	= VIS_OMNI 	+ SUFFIX_RAW
RAW_MULTI 	= VIS_MULTI + SUFFIX_RAW
RAW_A 		= VIS_A 	+ SUFFIX_RAW
RAW_B 		= VIS_B 	+ SUFFIX_RAW
RAW_C 		= VIS_C 	+ SUFFIX_RAW
RAW_D 		= VIS_D 	+ SUFFIX_RAW
RAW_E 		= VIS_E 	+ SUFFIX_RAW

OBS_INDEX_A = 2
OBS_INDEX_B = 3
OBS_INDEX_C = 4
OBS_INDEX_D = 5
OBS_INDEX_E = 6

OBS_COLOR_A = (0,255,255)
OBS_COLOR_B = (0,228,171)
OBS_COLOR_C = (0,201,87)
OBS_COLOR_D = (128,106,50)
OBS_COLOR_E = (255,10,10)
OBS_COLOR_OMNISCIENT = (255,255,255)
OBS_COLOR_ALL = (138,43,226)
OBS_COLOR_NAKED = (100,100,100)

OBS_HEX_A = '#00008B'
OBS_HEX_B = '#00e4ab'
OBS_HEX_C = '#00cc00'
OBS_HEX_D = '#b2ff66'
OBS_HEX_E = '#ffeb0a'
OBS_HEX_OMNISCIENT = '#FFFFFF'
OBS_HEX_ALL = '#ff2f0a'
OBS_HEX_NAKED = '#222222'

VIS_CHECKLIST = [VIS_OMNI, VIS_ALL, VIS_A, VIS_B, VIS_C, VIS_D, VIS_E]
RAW_CHECKLIST = [RAW_OMNI, RAW_ALL, RAW_A, RAW_B, RAW_C, RAW_D, RAW_E]

# 				red 		green 		yellow 			light yellow	green 		light blue		blue			
PATH_COLORS = [(138,43,226), (0,201,87), (0,255,255), (0,228,171), (0,201,87), (128,106,50), (255,10,10)]
PATH_LABELS = ['red', 'yellow', 'blue', 'green']
# PATH_COLORS = [(138,43,226), (0,255,255), (255,64,64), (0,201,87)]
# PATH_COLORS = [(130, 95, 135), (254, 179, 8), (55, 120, 191), (123, 178, 116)]

VIS_COLOR_MAP = {}
for i in range(len(PATH_COLORS)):
	VIS_COLOR_MAP[VIS_CHECKLIST[i]] = PATH_COLORS[i]

RAW_TO_VIS_COL = {}
VIS_TO_RAW_COL = {}
for i in range(len(PATH_COLORS)):
	RAW_TO_VIS_COL[RAW_CHECKLIST[i]] = VIS_CHECKLIST[i]
	VIS_TO_RAW_COL[VIS_CHECKLIST[i]] = RAW_CHECKLIST[i]

COLOR_P_BACK 	= VIS_COLOR_MAP[VIS_A]
COLOR_P_FACING 	= VIS_COLOR_MAP[VIS_B]

goals = []
tables = []
observers = []
start = []
path = []


#lookup tables linking related objects 
goal_observers = {}
goal_obj_set = {}

visibility_maps = {}

SCENARIO_IDENTIFIER = "new_scenario"

FILENAME_OUTPUTS = 'generated/'
FILENAME_PICKLE_VIS = FILENAME_OUTPUTS + 'pickled_visibility'
FILENAME_PICKLE_OBSTACLES = FILENAME_OUTPUTS + 'pickled_obstacles'
FILENAME_VIS_PREFIX = FILENAME_OUTPUTS + "fine_fig_vis_"
FILENAME_OVERVIEW_PREFIX = FILENAME_OUTPUTS + "overview_"
FILENAME_OBSTACLE_PREFIX = FILENAME_OUTPUTS + "fig_obstacles"

FILENAME_TO_UNITY = "export/"
FILENAME_EXPORT_IMGS_PREFIX = FILENAME_TO_UNITY + "imgs/"
FILENAME_EXPORT_CSV_PREFIX = FILENAME_TO_UNITY + "csv/"

# Note: inform restaurant code of these values also
UNITY_CORNERS = [(1.23, 3.05), (11.22, -10.7)]
ux1, uy1 = UNITY_CORNERS[0]
ux2, uy2 = UNITY_CORNERS[1]

IMG_CORNERS = [(0,0), (1000, 1375)]
ix1, iy1 = IMG_CORNERS[0]
ix2, iy2 = IMG_CORNERS[1]

UNITY_OFFSET_X = (ux1 - ix1)
UNITY_OFFSET_Y = (uy1 - iy1)
UNITY_SCALE_X = (ix2 - ix1) / (ux2 - ux1)
UNITY_SCALE_Y = (iy2 - iy1) / (uy2 - uy1)

length = ix2
width = iy2

UNITY_TO_IRL_SCALE = 3

UNITY_GOAL_NAMES = ["BEFORE", "ME", "PAST", "ACROSS"]

# visibility = np.zeros((r_width, r_length))
# for x in range(r_width):
# 	for y in range(r_length):
# 		rx = x*resolution_visibility
# 		ry = y*resolution_visibility
# 		score = 0
# 		for obs in observers:
# 			score += obs.get_visibility((rx,ry))

# 		visibility[x,y] = score

# visibility = visibility.T

def to_xy(pt):
	if len(pt) == 3:
		return (pt[0], pt[1])
	return pt

def dist(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def bresenham_line(xy1, xy2):
	x0, y0 = xy1
	x1, y1 = xy2

	steep = abs(y1 - y0) > abs(x1 - x0)
	if steep:
		x0, y0 = y0, x0  
		x1, y1 = y1, x1

	switched = False
	if x0 > x1:
		switched = True
		x0, x1 = x1, x0
		y0, y1 = y1, y0

	if y0 < y1: 
		ystep = 1
	else:
		ystep = -1

	deltax = x1 - x0
	deltay = abs(y1 - y0)
	error = -deltax / 2
	y = y0

	line = []	
	for x in range(x0, x1 + 1):
		if steep:
			line.append((y,x))
		else:
			line.append((x,y))

		error = error + deltay
		if error > 0:
			y = y + ystep
			error = error - deltax
	if switched:
		line.reverse()
	return line

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

# return angle_between in degrees
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    in_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(in_rad)

def in_bounds(point):
	x, y = point

	if x > width or x < 0:
		return False

	if y > length or y < 0:
		return False

	return True

def get_cost_of_segment(pos1, pos2, obstacle_map, visibility_map):
	# time in visibility
	
	return cost

def get_cost_of_move(pos1, pos2, obstacle_map, visibility_map):
	cost = 0
	most_vis = np.amax(visibility_map)
	obstacle_max = np.amax(obstacle_map)
	obstacle_min = np.amin(obstacle_map)

	# print(obstacle_map[0][0])

	# print(most_vis)
	# print(visibility_map[0][0])
	# print(obstacle_min)
	# print(obstacle_max)
	# exit()

	line = bresenham_line(pos1, pos2)
	# print(line) 

	for point in line:
		# infinite cost to go through obstacles
		min_val = 1
		if in_bounds(point):
			# cost += 1 + (obstacle_map[point] * 500 * 0)
			vis_x = int(point[0] / resolution_visibility)
			vis_y = int(point[1] / resolution_visibility)
			# print(visibility_map.shape)
			# print(vis_x, vis_y)
			vis = visibility_map[vis_y][vis_x]
			cost_increase = 1
			# cost_increase = ((most_vis - vis) / most_vis)
			cost += cost_increase
			
			if cost_increase < min_val:
				min_val = cost_increase


			# print((vis_x, vis_y, vis, obstacle_map[point]))

			# print("point info")
			# print((vis, obstacle_map[point], cost))

			if math.isnan(cost):
				exit()

		else:
			cost += np.Inf

			# Give a bonus for being at least partially well observed in this segment
	# cost += len(line)*min_val

	return cost


def get_children_astar(node, obstacle_map, visibility_map):
	xyz, parent_node, cost_so_far = node
	children = []
	directions = [(0, resolution_planning), (resolution_planning, 0), (0, -1*resolution_planning), (-1*resolution_planning, 0)]

	for direction in directions:
		new_xyz = tuple_plus(xyz, direction)
		cost_to_add = get_cost_of_move(xyz, new_xyz, obstacle_map, visibility_map)

		# print(cost_to_add)

		new_node = (new_xyz, node, cost_so_far + cost_to_add)
		children.append(new_node)


	return children


def heuristic(node, goal, visibility_map):
	pos1 = node[0]
	pos2 = goal

	most_vis = np.amax(visibility_map)
	ratio = (1 / most_vis)
	len(bresenham_line(pos1, pos2)) * ratio

	return len(bresenham_line(pos1, pos2))

def get_paths_astar(start, goal, obstacle_map, visibility_maps):
	paths = []

	obstacle_map = obstacle_map
	vis_maps = []
	# vis_maps.append(visibility_maps[VIS_ALL][0])
	vis_maps.append(visibility_maps[VIS_OMNI][0])
	vis_maps.append(visibility_maps[VIS_MULTI][0])
	vis_maps.append(visibility_maps[VIS_INDIVIDUALS][0])
	# vis_maps.append(visibility_maps[VIS_INDIVIDUALS][1])

	# print(len(vis_maps))
	
	for visibility_map in vis_maps:
		print("math path being added")

		# node = ((x,y), parent_node, cost-so-far)

		n_0 = (start, None, 0)

		parent_grid = {}

		openset = [n_0]
		closed_set = defaultdict(lambda: float('inf'))

		i = 0
		goal_found = False
		while i < 600000 and not goal_found:
			i += 1
			openset = sorted(openset, key=lambda x: x[2] + heuristic(x, goal, visibility_map))
			# print(openset)
			# Get best bet node
			node = openset.pop(0)
			xyz, parent, cost = node
			closed_set[xyz] = (cost, parent)

			# print(len(closed_set))
			
			kids = get_children_astar(node, obstacle_map, visibility_map)
			openset.extend(kids)
			# print(len(openset))

			print(dist(xyz, goal))

			if dist(xyz, goal) < resolution_planning:
				print("Found the goal!")
				goal_found = True

				path = []
				n_path = node
				while n_path[1] is not None:
					n_path = n_path[1]
					path.append(n_path[0])
				
				print(path)
				paths.append(path[::-1])
		
	return paths

# Bezier helpers
def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result

def get_path_spoof(start, end, goal_pts, table_pts, vis_type, visibility_map):
	STEPSIZE = 15
	points = []

	goal_index = goal_pts.index(end)
	helper = goal_helper_pts[goal_index]	

	if vis_type == VIS_OMNI:
		xys = [start, end]

	elif vis_type == VIS_MULTI:
		xys = [start, (1008, 420), end]

	elif vis_type == VIS_A:
		xys = [start, (1008, 350), (1008, 350), end]

	elif vis_type == VIS_B:
		xys = [start, (822, 370), (822, 370), end]

	ts = [t/STEPSIZE for t in range(STEPSIZE + 1)]
	bezier = make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points


def get_path(start, end, obs=[]):
	path = [start, end]
	x_start, y_start = start
	x_end, y_end = end

	return path

def get_path_2(start, end, obs=[]):
	path = [start, end]
	x_start, y_start = start
	x_end, y_end = end

	x_start = x_start / 10.0
	y_start = y_start / 10.0

	x_end = x_end / 10.0
	y_end = y_end / 10.0

	target = mptj.motion_model.State(x=x_end, y=x_end, yaw=np.deg2rad(90.0))
	k0 = 0.0

	init_p = np.array([x_start, y_start, 0.0]).reshape(3, 1)
	x, y, yaw, p = mptj.optimize_trajectory(target, k0, init_p)

	# print(x)
	# print(y)
	# print(yaw)
	# print(p)
	return path


def tuple_plus(a, b):
	return (int(a[0] + b[0]), int(a[1] + b[1]))

def angle_between(p1, p2):
	ang1 = np.arctan2(*p1[::-1])
	ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	angle = math.radians(angle)

	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return (qx, qy)

def get_random_point_in_room(length, width):
	new_goal = (random.randrange(width), random.randrange(length))
	# new_goal = Point(new_goal)

	return new_goal

class Point: 
	def __init__(self, xyz):
		self.xyz = xyz

	def get_tuple(self):
		return self.xyz

	def get_x(self):
		return self.xyz[0]

	def get_y(self):
		return self.xyz[1]

class Polygon:
	def __init__(self, pt_list):
		self.points = pt_list

class Table:
	radius = DIM_TABLE_RADIUS
	shape = None
	pretty_shape = None

	def __init__(self, pt, gen_type):
		self.location = pt

		if gen_type == TYPE_UNITY_ALIGNED or gen_type == TYPE_EXP_SINGLE:
			self.radius = 1
		
		table_radius = int(.4 * UNITY_SCALE_X)
		table_radius = int(table_radius * 2)
		self.table_radius = table_radius
		# half_radius = self.radius / 2.0

		USE_RECTANGLES = False
		USE_CIRCLES = False
		USE_POLYGON = True

		tp = pt
		if USE_POLYGON:
			tx, ty, ttheta = tp
			dir_y = 0
			if ttheta == DIR_SOUTH:
				dir_y = -1
			elif ttheta == DIR_NORTH:
				dir_y = 1
			else:
				print("No def for table at this orientation")
				exit()

			# width and height
			tw = table_radius * 1.2
			th = table_radius * 1.2 # * .25

			up = -1 * dir_y * table_radius * .23

			ow_diag = tw * 0.70710
			oh_diag = th * 0.70710

			# big_depth = table_radius *

			pt_top = (tx, ty + .5*up)
			pt_top_left = (tx - (.25*tw), ty + 1.5*up)
			pt_top_right = (tx + (.25*tw), ty + 1.5*up)
			pt_left = (tx - tw, ty + up)
			pt_right = (tx + tw, ty + up)
			pt_bot = (tx, ty + (dir_y * th))
			pt_lbot = (tx - ow_diag, ty + (3  *(dir_y * oh_diag)))
			pt_rbot = (tx + ow_diag, ty + (3 * (dir_y * oh_diag)))
			tpts = [pt_top, pt_top_left, pt_left, pt_lbot, pt_bot, pt_rbot, pt_right, pt_top_right]


			tw = table_radius * .5
			th = table_radius * .5
			tan22 = 0.557851739

			ppt_left_top = (tx - tw*tan22, ty - (dir_y * tw*tan22))
			ppt_right_top = (tx + tw*tan22, ty - (dir_y * tw*tan22))

			ppt_left = (tx - tw, ty - (dir_y * tw*tan22))
			ppt_right = (tx + tw, ty - (dir_y * tw*tan22))

			ppt_lbot = (tx - tw, ty + (dir_y * tw*tan22))
			ppt_rbot = (tx + tw, ty + (dir_y * tw*tan22))

			ppt_lbot2 = (tx - tw*tan22, ty + (dir_y * th))
			ppt_rbot2 = (tx + tw*tan22, ty + (dir_y * th))

			tpts_pretty = [ppt_left_top, ppt_left, ppt_lbot, ppt_lbot2, ppt_rbot2, ppt_rbot, ppt_right, ppt_right_top]

			t = fancyPolygon(tpts)
			t_pretty = fancyPolygon(tpts_pretty)

			# t = fancyBox(tx - tw, ty - th, tx + tw, ty + th, ccw=True)

		elif USE_RECTANGLES:
			pts = [pt, pt, pt, pt]
			pts[0] = tuple_plus(pts[0], (- half_radius, - half_radius))
			pts[1] = tuple_plus(pts[1], (- half_radius, + half_radius))
			pts[2] = tuple_plus(pts[2], (+ half_radius, + half_radius))
			pts[3] = tuple_plus(pts[3], (+ half_radius, - half_radius))
			self.points = pts

			polygon_set = []
			for p in pts:
				polygon_set.append(Point(p))

			t = fancyPolygon(polygon_set)
			t_pretty = t

		elif USE_CIRCLES:
			t = fancyPoint(self.get_center()).buffer(table_radius)
			t_pretty = t

		self.shape = t
		self.pretty_shape = t_pretty

	def is_within(self, point):
		return Point(point).within(self.shape)

	def intersects_path(self, path):
		t = self.get_shape()
		for i in range(len(path) - 1):
			pt1 = path[i]
			pt2 = path[i + 1]
			
			l = fancyLineString([pt1, pt2])
			i = l.intersects(t)

			if i:
				return True

		return False


	def pt_top_left(self):
		return self.points[0]

	def pt_bottom_right(self):
		return self.points[2]

	def get_radius(self):
		return int(self.radius)

	def get_shape(self):
		return self.shape

	def get_pretty_shape(self):
		return self.pretty_shape

	def get_shape_list(self):
		pts =  list(self.get_pretty_shape().exterior.coords)[1:]
		return [(int(x[0]), int(x[1])) for x in pts]

	def get_center(self):
		return (self.location[0], self.location[1])

	def get_orientation(self):
		return self.location[2]

	def get_JSON(self):
		return (self.location, self.radius)



class Observer:
	entity_radius = DIM_OBSERVER_RADIUS
	draw_depth = 25000
	cone_depth = 2000
	focus_angle = 120.0 / 2.0
	# peripheral_angle = 120 / 2.0
	FOV_angle = 120

	orientation = 0

	color = COLOR_OBSERVER

	field_focus = []
	# field_peripheral = []


	def __init__(self, location, angle):
		# print("LOCATION")
		# print(location)
		self.location = location
		self.orientation = angle

		focus_angle = self.focus_angle
		# peripheral_angle = self.peripheral_angle
		
		# Add center of viewpoint
		focus_a = (0, self.cone_depth)
		focus_b = (0, self.cone_depth)
		# periph_a = (0, int(self.cone_depth*1.5))
		# periph_b = (0, int(self.cone_depth*1.5))

		focus_a = rotate((0,0), focus_a, focus_angle + angle)
		focus_b = rotate((0,0), focus_b, -focus_angle + angle)

		# periph_a = rotate((0,0), periph_a, peripheral_angle + angle)
		# periph_b = rotate((0,0), periph_b, -peripheral_angle + angle)
	
		focus_a = tuple_plus(focus_a, location)
		focus_b = tuple_plus(focus_b, location)

		# periph_a = tuple_plus(periph_a, location)
		# periph_b = tuple_plus(periph_b, location)

		self.field_focus = [location, focus_a, focus_b]
		# self.field_peripheral = [location, periph_a, periph_b]

		# Add center of viewpoint
		draw_focus_a = (0, self.draw_depth)
		draw_focus_b = (0, self.draw_depth)
		# draw_periph_a = (0, int(self.draw_depth*5))
		# draw_periph_b = (0, int(self.draw_depth*5))

		draw_focus_a = rotate((0,0), draw_focus_a, focus_angle + angle)
		draw_focus_b = rotate((0,0), draw_focus_b, -focus_angle + angle)

		# draw_periph_a = rotate((0,0), draw_periph_a, peripheral_angle + angle)
		# draw_periph_b = rotate((0,0), draw_periph_b, -peripheral_angle + angle)
	
		draw_focus_a = tuple_plus(draw_focus_a, location)
		draw_focus_b = tuple_plus(draw_focus_b, location)

		# draw_periph_a = tuple_plus(draw_periph_a, location)
		# draw_periph_b = tuple_plus(draw_periph_b, location)

		self.draw_field_focus = [location, draw_focus_a, draw_focus_b]
		# self.draw_field_peripheral = [location, draw_periph_a, draw_periph_b]

	def set_color(self, c):
		self.color = c

	def get_visibility(self, location):
		# print(location)
		fancy_location = fancyPoint(location)

		field_focus = fancyPolygon(self.field_focus)
		# field_peripheral = fancyPolygon(self.field_peripheral)

		if fancy_location.within(field_focus):
			return 1
		# elif fancy_location.within(field_peripheral):
		# 	return .5
		return 0

	def get_obs_to_pt_relationship(self, pt):
		obs_orient 	= self.get_orientation()
		obs_FOV 	= self.get_FOV()

		# if a point is within the field of vision of a person,
		# and within the viewing distance
		# then this observer views the current point
		angle 		= angle_between(pt, self.get_center())
		distance 	= dist(pt, self.get_center())

		observation = (angle, distance)
		return observation

	def get_center(self):
		return self.location

	def get_center_image(self):
		return self.location[1], self.location[0]

	# in degrees
	def get_orientation(self):
		return self.orientation

	# in degrees
	def get_FOV(self):
		return self.FOV_angle

	def get_field_focus(self):
		return np.int32([self.field_focus])

	# def get_field_peripheral(self):
	# 	return np.int32([self.field_peripheral])

	def get_draw_field_focus(self):
		return np.int32([self.draw_field_focus])

	# def get_draw_field_peripheral(self):
	# 	return np.int32([self.draw_field_peripheral])

	def get_radius(self):
		return int(self.entity_radius)

	def get_JSON(self):
		json_dict = {}
		json_dict['orientation'] = self.orientation
		json_dict['location'] = self.location
		return json_dict

	def get_color(self):
		return self.color

def unity_to_image_angle(theta):
	ntheta = (theta - 90) % 360
	return ntheta

def image_to_unity_angle(theta):
	ntheta = (theta + 90) % 360
	return ntheta

def unity_to_image(pt):
	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	nx = (x - UNITY_OFFSET_X) * UNITY_SCALE_X
	ny = (y - UNITY_OFFSET_Y) * UNITY_SCALE_Y

	if len(pt) == 3:
		ntheta = unity_to_image_angle(theta)
		return (int(ny), int(nx), theta)
	return (int(ny), int(nx))

def image_to_unity_list(path):
	new_path = []
	for pt in path:
		new_path.append(image_to_unity(pt))
	return new_path

def unity_to_image_list(path):
	new_path = []
	for pt in path:
		new_path.append(unity_to_image(pt))
	return new_path

def image_to_unity(pt):
	# print(UNITY_SCALE_X) = 100.1001001001001
	# print(UNITY_SCALE_Y) = -100.0
	# print(UNITY_OFFSET_X) = 1.23
	# print(UNITY_OFFSET_Y) = 3.05

	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	x = float(x)
	y = float(y)
	ny = (x / UNITY_SCALE_Y) + (UNITY_OFFSET_Y)
	nx = (y / UNITY_SCALE_X) + (UNITY_OFFSET_X)
	
	# angle conversions (image 0 = 90 unity)
	# (good range is from 30 to 150 in unity)
	
	if len(pt) == 3:
		ntheta = image_to_unity_angle(theta)
		return (nx, ny, theta)
	return (nx, ny)

def plan_to_image(pt):
	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	nx = x
	ny = y

	if len(pt) == 3:
		return (int(ny), int(nx), theta)
	return (int(ny), int(nx))


# # TESTING SUITE FOR CONVERSIONS
def verify_conversions():
	u_a = (11.22, 3.05) 
	u_b = (1.23, 3.05)
	u_c = (11.22, -10.7)
	u_d = (1.23, -10.7)
	u_e = (0, 0)

	u_pts = [u_a, u_b, u_c, u_d, u_e]
	i_pts = []

	print("TARGETS")
	print(u_pts)
	for pt in u_pts:
		ip = unity_to_image(pt)
		# print(str(pt) + "->" + str(ip))
		i_pts.append(ip)

	print(i_pts)
	n_u_pts = []
	for pt in i_pts:
		ip = image_to_unity(pt)
		# print(ip)
		n_u_pts.append(ip)

	print(n_u_pts)

	print("Validate points transform to and from correctly")


class PathPlan: 
	def __init__(self, path, generation_details):
		self.path = path
		self.generation_details = generation_details

	def get_path(self):
		return self.path

	def get_generation_details(self):
		return self.generation_details


# verify_conversions()

# nodes = width x length divided up by planning resolution\
n_width = int(width / resolution_planning) + 1
n_length = int(length / resolution_planning) + 1

goal_helper_pts = []

class Restaurant: 
	def __init__(self, generate_type, tables=None, goals=None, start=None, observers=None, dim=None):
		self.generate_type = generate_type
		print(generate_type)
		self.observers = []
		self.goals = []
		self.tables = []
		self.start = []
		self.SCENARIO_IDENTIFIER = ""
		self.waypoints = []

		self.img = None
		self.obstacle_map = None
		self.visibility_maps = None
		self.dim = None

		if generate_type == TYPE_CUSTOM:
			self.goals 	= goals
			self.tables = tables
			self.start 	= start
			self.observers = observers
			self.dim = dim
			self.length, self.width = self.dim

		elif generate_type == TYPE_PLOTTED:
			# Creates a 2x3 layout restaurant with start location in between
			self.SCENARIO_IDENTIFIER = "3x2_all_full"
			self.start = (10, 10)

			row1 = 60
			row2 = 360

			col1 = 100
			col2 = 300
			col3 = 500

			self.start = (col1 - 30, int((row1 + row2) / 2))

			table_pts = [(col1,row1), (col2,row1), (col3, row1), (col1,row2), (col2,row2), (col3, row2)]

			for pt in table_pts:
				table = Table(pt, generate_type)
				self.tables.append(table)

			for table in tables:
				obs1_pt = table.get_center()
				obs1_pt = tuple_plus(obs1_pt, (-60, 0))
				obs1_angle = 270
				obs1 = Observer(obs1_pt, obs1_angle)
				self.observers.append(obs1)


				obs2_pt = table.get_center()
				obs2_pt = tuple_plus(obs2_pt, (60, 0))
				obs2_angle = 90
				obs2 = Observer(obs2_pt, obs2_angle)
				self.observers.append(obs2)

				goal_pt = table.get_center()
				offset = (0,0)
				if (table.get_center()[1] == row1):
					offset = (0, 80)
				else: 
					offset = (0, -80)

				goal_pt = tuple_plus(goal_pt, offset)
				goal_angle = 0
				self.goals.append(goal_pt)

				# goal_observers[goal_pt] = [obs1, obs2]

		elif generate_type == TYPE_EXP_SINGLE:
			# Unity scenario created specifically for parameters of Unity restaurant

			self.SCENARIO_IDENTIFIER = "_exp_single_"

			UNITY_CORNERS = [(1.23, 3.05), (11.22, -10.7)]
			ux1, uy1 = UNITY_CORNERS[0]
			ux2, uy2 = UNITY_CORNERS[1]

			IMG_CORNERS = [(0,0), (1000, 1375)]
			ix1, iy1 = IMG_CORNERS[0]
			ix2, iy2 = IMG_CORNERS[1]

			UNITY_OFFSET_X = (ux1 - ix1)
			UNITY_OFFSET_Y = (uy1 - iy1)
			UNITY_SCALE_X = (ix2 - ix1) / (ux2 - ux1)
			UNITY_SCALE_Y = (iy2 - iy1) / (uy2 - uy1)

			self.length = ix2
			self.width = iy2

			UNITY_TO_IRL_SCALE = 3
			
			# images will be made at the scale of
			
			# x1 = 3.05
			# x2 = -10.7
			# y1 = 11.22
			# y2 = 1.23

			# length = abs(y1 - y2)
			# width = abs(x1 - x2)

			# start = (7.4, 2.37)
			y_coord_start = 5.6
			start = (5.6, 1.0, DIR_EAST)
			self.set_start(unity_to_image(start))

			# print("START")
			# print(start)
			# print(unity_to_image(start))

			length = 1000
			width = 1375

			# waypoint_kitchen_exit = (6.45477, 2.57, DIR_EAST)
			# wpt = unity_to_image(waypoint_kitchen_exit)
			# # TODO verify units on this
			# self.waypoints.append(wpt)

			# a = (.6, 0)
			a = (0, 0)
			t = (0, 0)

			# unity_goal_pt = (4.43, -7.0)

			unity_table_pts = []
			# # unity_table_pts.append((3.6, -4.0))
			# unity_table_pts.append((3.6, 	-7.0, DIR_SOUTH)) # 3.6, 	-7.5
			# # unity_table_pts.append((5.6, -10.0))
			# unity_table_pts.append((7.6 + a[0], 	-7.0  + a[1], DIR_NORTH))

			separation_table = 2.0
			unity_table_pts.append((y_coord_start - separation_table, -7.0, DIR_SOUTH))
			unity_table_pts.append((y_coord_start + separation_table + a[0], 	-7.0  + a[1], DIR_NORTH))

			# print(unity_table_pts)

			unity_goal_stop_options = []
			# # unity_goal_stop_options.append((4.3, -4.3))
			# unity_goal_stop_options.append((3.8, 	-7.0, 	DIR_SOUTH))
			# # unity_goal_stop_options.append((5.6, -9.3)
			# unity_goal_stop_options.append((7.4 + a[0], 	-7.0 + a[1], 	DIR_NORTH))

			separation = 1.8
			unity_goal_stop_options.append((y_coord_start - separation, -7.0, DIR_SOUTH))
			unity_goal_stop_options.append((y_coord_start + separation + a[0], 	-7.0  + a[1], DIR_NORTH))

			# unity_goal_options = []
			# # unity_goal_options.append((4.3, -4.0))
			# unity_goal_options.append((4.429, 	-7.0, 	DIR_SOUTH)) #(4.3, -7.0)
			# # unity_goal_options.append((5.6, -9.3))
			# unity_goal_options.append((6.9, 	-7.0, 	DIR_NORTH))

			print(unity_goal_stop_options)
			print(unity_table_pts)

			table_pts = []
			for t in unity_table_pts:
				pt = unity_to_image(t)
				# print("TABLE:" + str(t))
				# print(unity_to_image(t))

				# print(pt)
				table = Table(pt, generate_type)
				self.tables.append(table)

			# print(unity_table_pts[0])

			for g in unity_goal_stop_options:
				# print("GOAL:" + str(g))
				# print(unity_to_image(g))
				goal_helper_pts.append(unity_to_image(g))
				self.goals.append(unity_to_image(g))

			# goal = unity_to_image(unity_goal_pt)
			# self.current_goal = goal

			# Set up observers

			# unity table points are
			# unity_table_pts.append((3.6, -7.0))
			# unity_table_pts.append((7.6, -7.0))

			customer_offset = 0.67
			customer_offset_diag = customer_offset * 0.70710

			observer_table = unity_table_pts[0]
			table_x = observer_table[0]
			table_y = observer_table[1]# + customer_offset

			all_observers = []

			obs_sets = {}
			obs_sets[OBS_KEY_NONE] = []

			# person a
			obs1_pt = (table_x, table_y - customer_offset)
			# print(obs1_pt)
			obs1_pt = unity_to_image(obs1_pt)

			obs1_angle = unity_to_image_angle(150)
			obs1 = Observer(obs1_pt, obs1_angle)
			obs1.set_color(PATH_COLORS[OBS_INDEX_A])
			all_observers.append(obs1)

			# person b
			obs2_pt = (table_x - customer_offset_diag, table_y - customer_offset_diag)
			# print(obs2_pt)
			obs2_pt = unity_to_image(obs2_pt)
			
			obs2_angle = unity_to_image_angle(120)
			obs2 = Observer(obs2_pt, obs2_angle)
			obs2.set_color(PATH_COLORS[OBS_INDEX_B])
			all_observers.append(obs2)

			# person c
			obs3_pt = (table_x - customer_offset, table_y)
			# print(obs3_pt)
			obs3_pt = unity_to_image(obs3_pt)
			
			obs3_angle = unity_to_image_angle(90)
			obs3 = Observer(obs3_pt, obs3_angle)
			obs3.set_color(PATH_COLORS[OBS_INDEX_C])
			all_observers.append(obs3)
		
			# person d
			obs4_pt = (table_x - customer_offset_diag, table_y + customer_offset_diag)
			# print(obs4_pt)
			obs4_pt = unity_to_image(obs4_pt)
			
			obs4_angle = unity_to_image_angle(60)
			obs4 = Observer(obs4_pt, obs4_angle)
			obs4.set_color(PATH_COLORS[OBS_INDEX_D])
			all_observers.append(obs4)

			# person e
			obs5_pt = (table_x, table_y + customer_offset)
			# print(obs5_pt)
			obs5_pt = unity_to_image(obs5_pt)
			
			obs5_angle = unity_to_image_angle(30)
			obs5 = Observer(obs5_pt, obs5_angle)
			obs5.set_color(PATH_COLORS[OBS_INDEX_E])
			all_observers.append(obs5)
			obs_sets[OBS_KEY_A] = [obs5]
			obs_sets[OBS_KEY_B] = [obs4]
			obs_sets[OBS_KEY_C] = [obs3]
			obs_sets[OBS_KEY_D] = [obs2]
			obs_sets[OBS_KEY_E] = [obs1]
			



			# obs_sets[OBS_KEY_ALL] = all_observers
			self.obs_sets = obs_sets

			for o in self.obs_sets:
				# print("OBSERVER A")
				# print(image_to_unity(obs_sets[OBS_KEY_A][0].get_center()))
				# print("OBSERVER B")
				# print(image_to_unity(obs_sets[OBS_KEY_B][0].get_center()))
				# print("OBSERVER C")
				# print(image_to_unity(obs_sets[OBS_KEY_C][0].get_center()))
				# print("OBSERVER D")
				# print(image_to_unity(obs_sets[OBS_KEY_D][0].get_center()))
				# print("OBSERVER E")
				# print(image_to_unity(obs_sets[OBS_KEY_E][0].get_center()))
				pass
			# exit()

		elif generate_type == TYPE_RANDOM:
			# random generation of locations and objects
			# mainly useful for testing things such as vision cone impact

			self.length, self.width = 600, 800

			random_id = ''.join([random.choice(string.ascii_letters 
					+ string.digits) for n in range(10)]) 
			self.SCENARIO_IDENTIFIER = "new_scenario_" + random_id

			self.start = get_random_point_in_room(length, width)

			for i in range(num_tables):
				new_goal = get_random_point_in_room(length, width)
				self.goals.append(new_goal)

			goal = goals[0]

			for i in range(num_tables):
				table_pt = get_random_point_in_room(length, width)
				
				table = Table(table_pt)
				self.tables.append(table)

			# add customer locations
			for i in range(num_observers):
				obs_loc = get_random_point_in_room(length, width)
				angle = random.randrange(360)
				print((obs_loc, angle))

				self.observers.append(Observer(obs_loc, angle))

			print("NOTE: These values have no yaw")
			exit()


		elif generate_type == TYPE_UNITY_ALIGNED:
			print("Needs an update to have yaw values")
			exit()

			# Unity scenario created specifically for parameters of Unity restaurant

			self.SCENARIO_IDENTIFIER = "_unity_v1_"

			UNITY_CORNERS = [(1.23, 3.05), (11.22, -10.7)]
			ux1, uy1 = UNITY_CORNERS[0]
			ux2, uy2 = UNITY_CORNERS[1]

			IMG_CORNERS = [(0,0), (1000, 1375)]
			ix1, iy1 = IMG_CORNERS[0]
			ix2, iy2 = IMG_CORNERS[1]

			UNITY_OFFSET_X = (ux1 - ix1)
			UNITY_OFFSET_Y = (uy1 - iy1)
			UNITY_SCALE_X = (ix2 - ix1) / (ux2 - ux1)
			UNITY_SCALE_Y = (iy2 - iy1) / (uy2 - uy1)

			self.length = ix2
			self.width = iy2

			UNITY_TO_IRL_SCALE = 3
			
			# images will be made at the scale of
			
			# x1 = 3.05
			# x2 = -10.7
			# y1 = 11.22
			# y2 = 1.23

			start = (6.0, 2.0, DIR_EAST)
			self.start = unity_to_image(start)

			length = 1000
			width = 1375

			waypoint_kitchen_exit = (6.45477, 2.57)
			wpt = unity_to_image(waypoint_kitchen_exit)
			# TODO verify units on this
			self.waypoints.append(wpt)

			unity_goal_pt = (4.43, -7.0)

			unity_table_pts = []
			# unity_table_pts.append((3.6, -4.0))
			unity_table_pts.append((3.6, -7.0))
			# unity_table_pts.append((5.6, -10.0))
			unity_table_pts.append((7.6, -7.0))

			unity_goal_stop_options = []
			# unity_goal_stop_options.append((4.3, -4.3))
			unity_goal_stop_options.append((4.3, -7.3, DIR_SOUTH))
			# unity_goal_stop_options.append((5.6, -9.3))
			unity_goal_stop_options.append((6.9, -7.3, DIR_NORTH))

			unity_goal_options = []
			# unity_goal_options.append((4.3, -4.0))
			unity_goal_options.append((4.429, -7.0, DIR_SOUTH)) #(4.3, -7.03)
			# unity_goal_options.append((5.6, -9.3))
			unity_goal_options.append((6.9, -7.0, DIR_NORTH))

			table_pts = []
			for t in unity_table_pts:
				pt = unity_to_image(t)
				table = Table(pt, generate_type)
				self.tables.append(table)

			for g in unity_goal_stop_options:
				goal_helper_pts.append(unity_to_image(g))
				self.goals.append(unity_to_image(g))

			goal = unity_to_image(unity_goal_pt)
			self.current_goal = goal
			
			# Set up observers
			obs1_pt = (3.50, -7.71)
			obs1_pt = unity_to_image(obs1_pt)

			obs1_angle = 55
			obs1 = Observer(obs1_pt, obs1_angle)
			self.observers.append(obs1)


			obs2_pt = (3.50, -6.37)
			obs2_pt = unity_to_image(obs2_pt)
			
			obs2_angle = 305
			obs2 = Observer(obs2_pt, obs2_angle)
			self.observers.append(obs2)

			# goal_observers[goal] = [obs1, obs2]


		else:
			print("Incorrect generate_type")

		self.img = self.generate_obstacle_map_and_img(self.observers)
		self.generate_visibility_maps()

	def get_goal_index(self, goal):
		if goal not in self.goals:
			inv_map = {v: k for k, v in self.get_goal_labels().items()}
			goal = inv_map[goal]
		return self.goals.index(goal)

	def make_obstacle_map(self, obstacle_vis, img):
		obstacle_map = np.zeros((self.length, self.width), np.uint8)

		obstacle_map = cv2.cvtColor(obstacle_vis, cv2.COLOR_BGR2GRAY)
		(thresh, obstacle_map) = cv2.threshold(obstacle_map, 1, 255, cv2.THRESH_BINARY)

		self.obstacle_map = copy.copy(obstacle_map)
		self.obstacle_vis = copy.copy(obstacle_vis)

		obstacles = {}
		obstacles['map'] = copy.copy(obstacle_map)
		obstacles['vis'] = copy.copy(obstacle_vis)
		cv2.imwrite(FILENAME_OBSTACLE_PREFIX + '_map.png', obstacle_map) 
		cv2.imwrite(FILENAME_OBSTACLE_PREFIX + '_vis.png', obstacle_vis) 

		cv2.imwrite(FILENAME_OVERVIEW_PREFIX + ".png", img)
		print("Exported overview pic without paths")
		# ax = sns.heatmap(obstacle_map).set_title("Obstacle map of restaurant")
		# plt.savefig()
		# plt.clf()

		dbfile = open(FILENAME_PICKLE_OBSTACLES, 'ab') 
		pickle.dump(obstacles, dbfile)					  
		dbfile.close()
		print("Saved obstacle maps")


		print("Importing pickle of obstacle info")
		dbfile = open(FILENAME_PICKLE_OBSTACLES, 'rb')
		obstacles = pickle.load(dbfile)
		obstacle_map = obstacles['map']
		obstacle_vis = obstacles['vis']
		dbfile.close() 


	def generate_obstacle_map_and_img(self, observers):
		obstacle_vis = np.zeros((self.length, self.width,3), np.uint8)

		# DRAW the environment

		# Create a black image
		table_radius = int(.3 * UNITY_SCALE_X)
		obs_radius = int(.125 * UNITY_SCALE_X)
		goal_radius = int(.125 * UNITY_SCALE_X)
		start_radius = int(.125 * UNITY_SCALE_X)

		img = np.zeros((self.length, self.width,3), np.uint8)

		overlay = img.copy()

		obs_sets = self.get_obs_sets()
		obs_keys = obs_sets.keys()
		for o_key in obs_keys:
			# if it's a single audience member, then for each of those...
			if o_key in [OBS_KEY_A, OBS_KEY_B, OBS_KEY_C, OBS_KEY_D, OBS_KEY_E]:
				obs = obs_sets[o_key]
				if obs is not None:
					obs = obs[0]

					# cv2.fillPoly(overlay, obs.get_draw_field_peripheral(), COLOR_PERIPHERAL_AWAY)
					cv2.fillPoly(overlay, obs.get_draw_field_focus(), obs.get_color())
					alpha = 0.25  # Transparency factor.
					img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


		# Draw tables
		for table in self.tables:
			tx, ty = table.get_center()
			center_pt = (tx, ty)
			ttheta = table.get_orientation()
			radius = int(table.table_radius * .6)
			
			dir_y = 0
			if ttheta == DIR_SOUTH:
				dir_y = -1
			elif ttheta == DIR_NORTH:
				dir_y = 1

			axes = (radius,radius)
			angle = 0
			startAngle = 0
			endAngle = 180 * dir_y
			t_up = int(dir_y * table_radius * .5)

			cv2.ellipse(img, center_pt, axes, angle, startAngle, endAngle, COLOR_TABLE, -1)
			
			table_list = [(tx + radius, ty), (tx + radius, ty - t_up), (tx - radius, ty - t_up), (tx - radius, ty)]
			tpoly = np.array([table_list], np.int32)
			cv2.fillPoly(img, np.asarray(tpoly), COLOR_TABLE, lineType=cv2.LINE_AA)


			# POLYGON VIEW
			# tpoly = table.get_shape_list()
			# tpoly = np.array([tpoly], np.int32)
			# cv2.fillPoly(img, np.asarray(tpoly), COLOR_TABLE, lineType=cv2.LINE_AA)

			# table_center = table.get_center()
			# # print(table_center)
			# cv2.circle(img, table_center, table_radius, COLOR_TABLE, table_radius)

			# if FLAG_MAKE_OBSTACLE_MAP:
			# 	cv2.circle(obstacle_vis, table_center, table_radius + DIM_NAVIGATION_BUFFER, COLOR_OBSTACLE_BUFFER)
			# 	cv2.circle(obstacle_vis, table_center, table_radius, COLOR_OBSTACLE_FULL)

		for goal in self.goals:
			# if goal[0] == 1035 and goal[1] != 307:
			# Draw person
			cv2.circle(img, to_xy(goal), goal_radius, COLOR_GOAL, goal_radius)

		obs_keys = obs_sets.keys()
		for o_key in obs_keys:
			if o_key in [OBS_KEY_A, OBS_KEY_B, OBS_KEY_C, OBS_KEY_D, OBS_KEY_E]:
				obs = obs_sets[o_key]
				if obs is not None:
					obs = obs[0]
					cv2.circle(img, obs.get_center(), obs_radius, obs.get_color(), obs_radius)

		sx, sy, st = self.get_start()
		s_radius = int(obs_radius * 1.5)
		# cv2.circle(img, to_xy(self.get_start()), start_radius, COLOR_START, start_radius)
		start_list = [(sx + s_radius, sy + s_radius), (sx + s_radius, sy - s_radius), (sx - s_radius, sy - s_radius), (sx - s_radius, sy + s_radius)]
		spoly = np.array([start_list], np.int32)
		cv2.fillPoly(img, np.asarray(spoly), COLOR_START, lineType=cv2.LINE_AA)
		cv2.polylines(img, np.asarray(spoly), True, (255,255,255), lineType=cv2.LINE_AA)


		if FLAG_MAKE_OBSTACLE_MAP:
			self.make_obstacle_map(obstacle_vis, img)

		return cv2.flip(img, 0)

	def get_visibility_of_pt_raw(self, pt):
		observations = []

		for observer in observers:
			obs_orient 	= observer.get_orientation()
			obs_FOV 	= observer.get_FOV()

			angle 		= angle_between(pt, observer.get_center())
			distance 	= dist(pt, observer.get_center())

			observation = (pt, angle, distance)
			observations.append(observation)
		return observations

	def sample_points(self, num_pts, target, vis_type):
		start = self.start
		pts = []
		for i in range(num_pts):
			if False:
				pt = (random.randrange(2*width) - .5*width, random.randrange(2*length) - .5*length)
			else:
				pt = (random.randrange(width), random.randrange(length))

			pts.append(pt)

		return pts

	def get_visibility_of_pt_pandas(self, pt, f_vis):
		# Note only supports up to two observers
		a = self.get_observer_a()
		b = self.get_observer_b()


		obs_omni = []
		obs_multi = []
		obs_a = []
		obs_b = []
		obs_all = []

		for observer in observers:
			obs_orient 	= observer.get_orientation()
			obs_FOV 	= observer.get_FOV()
			obs_center 	= observer.get_center()

			angle 		= angle_between(pt, obs_center)
			distance 	= dist(pt, obs_center)

			obs_val = (obs_center, angle, distance)
			obs_all.append(obs_val)
			obs_multi.append(obs_val)

			if obs_center == a.get_center():
				obs_a.append(obs_val)
			elif obs_center == b.get_center():
				obs_b.append(obs_val)

		# print('vis omni')
		vis_omni 	= f_vis(obs_omni)
		# print('vis a')
		vis_a 		= f_vis(obs_a)
		# print('vis b')
		vis_b 		= f_vis(obs_b)
		# print('vis multi')
		vis_multi 	= f_vis(obs_multi)

		x, y = pt
		entry = [x, y, obs_all, vis_omni, vis_a, vis_b, vis_multi]
		  
		return entry

	def get_visibility_of_pts_pandas(self, f_vis):
		entries = [] 
		for x in range(length):
			for y in range(width):
				entry = self.get_visibility_of_pt_pandas((x, y), f_vis)
				entries.append(entry)

		# entry = [x, y, obs_all, obs_omni, obs_a, obs_b, obs_multi]
		df = pd.DataFrame(entries, columns = ['x', 'y', RAW_ALL, VIS_OMNI, VIS_A, VIS_B, VIS_MULTI])
		return df

	def get_obs_sets_colors(self):
		color_lookup = {}
		color_lookup['a'] = OBS_COLOR_E
		color_lookup['b'] = OBS_COLOR_D
		color_lookup['c'] = OBS_COLOR_C
		color_lookup['d'] = OBS_COLOR_B
		color_lookup['e'] = OBS_COLOR_A
		color_lookup['omni'] = OBS_COLOR_OMNISCIENT
		color_lookup['all'] = OBS_COLOR_ALL

		color_lookup['naked'] = OBS_COLOR_NAKED
		color_lookup['naked-env'] = OBS_COLOR_NAKED

		color_lookup['a-env'] = OBS_COLOR_E
		color_lookup['b-env'] = OBS_COLOR_D
		color_lookup['c-env'] = OBS_COLOR_C
		color_lookup['d-env'] = OBS_COLOR_B
		color_lookup['e-env'] = OBS_COLOR_A
		color_lookup['omni-env'] = OBS_COLOR_OMNISCIENT
		color_lookup['all-env'] = OBS_COLOR_ALL

		return color_lookup

	def get_obs_sets_hex(self):
		color_lookup = {}
		color_lookup['a'] = OBS_HEX_A
		color_lookup['b'] = OBS_HEX_B
		color_lookup['c'] = OBS_HEX_C
		color_lookup['d'] = OBS_HEX_D
		color_lookup['e'] = OBS_HEX_E
		color_lookup['omni'] = OBS_HEX_OMNISCIENT
		color_lookup['all'] = OBS_HEX_ALL

		color_lookup['naked'] = OBS_HEX_NAKED
		color_lookup['naked-env'] = OBS_HEX_NAKED

		color_lookup['a-env'] = OBS_HEX_A
		color_lookup['b-env'] = OBS_HEX_B
		color_lookup['c-env'] = OBS_HEX_C
		color_lookup['d-env'] = OBS_HEX_D
		color_lookup['e-env'] = OBS_HEX_E
		color_lookup['omni-env'] = OBS_HEX_OMNISCIENT
		color_lookup['all-env'] = OBS_HEX_ALL

		return color_lookup

	def get_obs_label(self, obs_set):
		all_sets = self.get_obs_sets()

		# if len(obs_set) == 0:
		# 	return OBS_NONE
		# if len(obs_set) > 1:
		# 	return OBS_ALL

		# for val in all_sets.values():

		key_list = list(all_sets.keys())
		val_list = list(all_sets.values())
		 
		position = val_list.index(obs_set)
		return key_list[position]

	def get_obs_sets(self):
		return self.obs_sets
		# obs_none 	= []
		# obs_all 	= self.get_observers()

		# obs_sets = {}
		# obs_sets[OBS_NONE] = obs_none

		# if self.generate_type == TYPE_UNITY_ALIGNED:	
		# 	obs_a 		= [self.get_observer_back()]
		# 	obs_b 		= [self.get_observer_towards()]

		# 	obs_sets[OBS_A] = obs_a
		# 	obs_sets[OBS_B] = obs_b
		
		# elif self.generate_type == TYPE_EXP_SINGLE:
		# 	for i in range(len(obs_all)):
		# 		key = OBS_KEYS[i]
		# 		obs_sets[key]  = [obs_all[i]]

		# else:
		# 	print("TO DO: IMPLEMENT SUBGROUPING SCENARIOS")
		# 	exit()

		# # obs_sets[OBS_ALL]  = obs_all
		# return obs_sets


	def generate_visibility_maps(self):
		visibility_maps = {}
		if OPTION_FORCE_GENERATE_VISIBILITY:
			# visibility_maps[VIS_INFO_RESOLUTION] = resolution_visibility

			r_width = int(width / resolution_visibility)
			r_length = int(length / resolution_visibility)

			visibility = np.zeros((r_width, r_length))
			omni = copy.copy(visibility)
			omni = omni.T

			ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: Omniscient")
			visibility_maps[VIS_OMNI] = [copy.copy(visibility)]
			# print(visibility.shape)
			plt.savefig(FILENAME_VIS_PREFIX + '_omni.png')
			plt.clf()

			# DEPRECATED: SAVING AN ALL VERSION
			# for x in range(r_width):
			# 	for y in range(r_length):
			# 		rx = x*resolution_visibility
			# 		ry = y*resolution_visibility
			# 		score = 0
			# 		for obs in observers:
			# 			score += obs.get_visibility((rx,ry))

			# 		visibility[x,y] = score

			# visibility = visibility.T
			# # xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
			# ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: All")
			# visibility_maps[VIS_ALL] = [copy.copy(visibility)]
			# plt.savefig(FILENAME_VIS_PREFIX + '_all.png')
			# plt.clf()
			# plt.show()

			visibility = np.zeros((r_width, r_length))
			for x in range(r_width):
				for y in range(r_length):
					rx = x*resolution_visibility
					ry = y*resolution_visibility
					score = 0
					for obs in observers:
						score += obs.get_visibility((rx,ry))

					visibility[x,y] = score

			visibility = visibility.T
			# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
			ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: Both Perspectives")
			# plt.show()
			visibility_maps[VIS_MULTI] = [copy.copy(visibility)]
			plt.savefig(FILENAME_VIS_PREFIX + '_multi.png')
			plt.clf()

			print("generated multi vis")

			indic = 0
			# visibility_maps[VIS_INDIVIDUALS] = []
			solo_paths = []

			for obs in self.observers:
				visibility = np.zeros((r_width, r_length))
				for x in range(r_width):
					for y in range(r_length):
						rx = x*resolution_visibility
						ry = y*resolution_visibility
						score = 0
						score += obs.get_visibility((rx,ry))

						visibility[x,y] = score

				visibility = visibility.T
				# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
				ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: 1 Observer #" + str(indic))
				# plt.show()
				solo_paths.append(copy.copy(visibility))
				# visibility_maps[VIS_INDIVIDUALS].append(copy.copy(visibility))
				plt.savefig(FILENAME_VIS_PREFIX + '_person_' + str(indic) + '.png')
				print("generated person vis " + str(indic))
				indic += 1
				plt.clf()

			# visibility_maps[VIS_INDIVIDUALS] = solo_paths
			visibility_maps[VIS_A] = solo_paths[0]
			visibility_maps[VIS_B] = solo_paths[1]

			# Export the new visibility maps and resolution info
			dbfile = open(FILENAME_PICKLE_VIS, 'ab') 
			pickle.dump(visibility_maps, dbfile)					  
			dbfile.close()
			# Successfully dumped pickle
		self.visibility_maps = visibility_maps

	def path_to_printable_path(self, path):
		length = self.get_length()
		new_path = []
		for p in path:
			new_path.append((p[0], length - p[1]))
		return new_path


	# observers[1] = TOWARDS
	# observers[0] = BACK

	# def get_observer_a(self):
	# 	if len(self.observers) > 0:
	# 		return self.observers[0]
	# 	return None

	# def get_observer_b(self):
	# 	if len(self.observers) > 1:
	# 		return self.observers[1]
	# 	return None


	# def get_observer_c(self):
	# 	if len(self.observers) > 0:
	# 		return self.observers[0]
	# 	return None

	# def get_observer_d(self):
	# 	if len(self.observers) > 1:
	# 		return self.observers[1]
	# 	return None

	# def get_observer_e(self):
	# 	if len(self.observers) > 0:
	# 		return self.observers[0]
	# 	return None


	# def get_observer_towards(self):
	# 	return self.get_observer_b()

	# def get_observer_back(self):
	# 	return self.get_observer_a()

	def get_scenario_identifier(self):
		return self.SCENARIO_IDENTIFIER

	def get_tables(self):
		return self.tables

	def get_goals_all(self):
		return self.goals

	def get_goal_labels(self):
		goal_labels = {}
		keys = self.get_goals_all()
		goal_labels[keys[0]] = 'GOAL-OTHER'
		goal_labels[keys[1]] = 'GOAL-ME'

		# goal_labels['GOAL-OTHER'] = keys[0]
		# goal_labels['GOAL-ME'] = keys[1]

		return goal_labels

	def get_current_goal(self):
		return self.current_goal

	def get_start(self):
		return self.start

	def set_start(self, s):
		self.start = s

	def get_waypoints(self):
		return self.waypoints

	def get_length(self):
		return self.length

	def get_width(self):
		return self.width

	# y values
	def get_sampling_length(self):
		start = self.get_start()
		min_val = start[1]
		max_val = start[1]
		for goal in self.get_goals_all():
			gy = goal[1]
			min_val = min(min_val, gy)
			max_val = max(max_val, gy)

		return range(0, self.get_length())
		return range(min_val, max_val)

	# x values
	def get_sampling_width(self):
		start = self.get_start()
		min_val = start[0]
		max_val = start[0]

		for goal in self.get_goals_all():
			gx = goal[0]
			min_val = min(min_val, gx)
			max_val = max(max_val, gx)

		return range(0, self.get_width())
		return range(min_val, max_val)

	def get_img(self):
		return copy.copy(self.img)

	def get_obs_img(self, obs_key):
		if obs_key is 'naked':
			target_obs = []
		else:
			target_obs = self.get_obs_sets()[obs_key]

		return self.generate_obstacle_map_and_img(target_obs)

	def get_envir_cache(self):
		if self.envir_cache is None:
			print("Error: No envir cache set")
			exit()

		return self.envir_cache

	def set_envir_cache(self, env_cac):
		self.envir_cache = env_cac

	def get_obstacle_map(self):
		print("DEPRECATED")
		return None
		return copy.copy(self.obstacle_map)

	def get_obstacle_vis(self):
		print("DEPRECATED")
		return None
		return copy.copy(self.obstacle_vis)

	def get_visibility_maps(self):
		return self.visibility_maps


def generate_restaurant(generate_type):
	r = Restaurant(generate_type)
	return r


FILENAME_PICKLE_VIS += SCENARIO_IDENTIFIER
FILENAME_PICKLE_OBSTACLES += SCENARIO_IDENTIFIER
FILENAME_VIS_PREFIX += SCENARIO_IDENTIFIER
FILENAME_OBSTACLE_PREFIX += SCENARIO_IDENTIFIER
FILENAME_OVERVIEW_PREFIX += SCENARIO_IDENTIFIER


# VISIBILITY Unit TEST
def visibility_unit_test():
	score = 0
	rx, ry = (133, 232)
	for obs in observers:
		this_vis = obs.get_visibility((rx,ry))
		print(this_vis)
		score += this_vis

	print(score)

	print("OR")
	score = 0
	rx, ry = (233, 133)
	for obs in observers:
		this_vis = obs.get_visibility((rx,ry))
		print(this_vis)
		score += this_vis

	print(score)



# Import the visibility info for work
# print("Importing pickle of visibility info")
# dbfile = open(FILENAME_PICKLE_VIS, 'rb')	  
# visibility_maps = pickle.load(dbfile)
# resolution_visibility = visibility_maps[VIS_INFO_RESOLUTION]
# print("Found maps at resolution " + str(resolution_visibility))
# dbfile.close() 

# # Raw coded paths
# # more vis
# p1 = [(490, 270), (490, 240), (490, 210), (460, 210), (430, 210), (400, 210), (370, 210), (340, 210), (310, 210), (280, 210), (250, 210), (220, 210), (190, 210), (160, 210), (130, 210), (100, 210), (70, 210)]
# # less vis
# p2 = [(490, 270),(490, 240),(460, 240),(460, 210),(430, 210),(400, 210),(370, 210),(340, 210),(310, 210),(280, 210),(250, 210),(220, 210),(190, 210),(160, 210),(130, 210),(100, 210),(70, 210)]
# p_names = ["first", "second"]

#TODO get real path with ASTAR or similar
# paths = []
# for visibility_map in visibility_maps:
# 	print("a map")
# 	for path in paths:
# 		print("this path")
# 		cost = 0
# 		for i in range(len(path) - 1):
# 			pos1 = path[i]
# 			pos2 = path[i + 1]
# 			cost += get_cost_of_move(pos1, pos2, obstacle_map, visibility_map)

# 		print("Cost = " + str(cost))

# paths = get_paths_astar(start, goal, obstacle_map, visibility_maps)
#p = [(490, 270), (490, 240), (490, 210), (460, 210), (430, 210), (400, 210), (370, 210), (340, 210), (310, 210), (280, 210), (250, 210), (220, 210), (190, 210), (160, 210), (130, 210), (100, 210), (70, 210)]
# p = [(490, 270),(490, 240),(460, 240),(460, 210),(430, 210),(400, 210),(370, 210),(340, 210),(310, 210),(280, 210),(250, 210),(220, 210),(190, 210),(160, 210),(130, 210),(100, 210),(70, 210)]

saved_paths = {}

# VERIFY: this is no longer where paths are chosen, just where they're drawn
# # All the types there are
# for vis_type in VIS_CHECKLIST:
# 	# print(visibility_maps.keys())
# 	print(vis_type)

# 	# UPDATE TO BETTER MAPS
# 	# vis_map = visibility_maps[vis_type]
# 	# vis_map = visibility_maps[2][0]
# 	vis_map = None

# 	for goal_index in range(len(goals)):
# 		end = goals[goal_index]
# 		# print(str(goal_index) + "->" + str(end))

# 		pkey = vis_type + "-" + str(goal_index)
# 		print(pkey)

# 		# new_path = get_path_spoof(start, end, goals, tables, vis_type, vis_map)
		
# 		# TODO fill in a more useful path here
# 		new_path = []

# 		saved_paths[pkey] = new_path


def export_diagrams_with_paths(img, saved_paths, fn=None):
	if fn is None:
		fn = FILENAME_EXPORT_IMGS_PREFIX

	print("\tExporting diagrams with paths")
	path_titles = ["OMNISCIENT", "TABLE", "Person A", "Person B"]

	# omni_paths_img = img.copy()
	# cv2.imwrite('generated/fig_path_' + "OMNISCIENT" + '.png', omni_paths_img) 

	all_paths_img = img.copy()

	img_deck = {}
	for vis_type in VIS_CHECKLIST:
		type_img = img.copy()
		img_deck[vis_type] = type_img

	for i in range(len(goals)):
		type_img = img.copy()
		img_deck[str(i)] = type_img


	for pkey in saved_paths.keys():
		path = saved_paths[pkey]
		path_img = img.copy()
		path_title = pkey
		# print()

		vis_type, goal_index = pkey.split("-")
		# print(vis_type)
		
		color = VIS_COLOR_MAP[vis_type]
		# print(color)

		by_method = img_deck[vis_type]
		by_goal = img_deck[goal_index]

		# Draw the path  
		for i in range(len(path) - 1):
			a = path[i]
			b = path[i + 1]
			
			cv2.line(path_img, a, b, color, thickness=6, lineType=8)
			cv2.line(all_paths_img, a, b, color, thickness=6, lineType=8)

			cv2.line(by_method, a, b, color, thickness=6, lineType=8)
			cv2.line(by_goal, a, b, color, thickness=6, lineType=8)		

		path_img = cv2.flip(path_img, 0)
		cv2.imwrite(fn + 'fig_path_' + path_title + '.png', path_img) 
		print("exported image of " + pkey + " for goal " + goal_index)

	all_paths_img = cv2.flip(all_paths_img, 0)
	cv2.imwrite(fn + 'ALL_CONDITIONS' + '.png', all_paths_img) 
	### END DISPLAY PATHS CODE

	for key in img_deck.keys():
		this_img = img_deck[key]
		this_img = cv2.flip(this_img, 0)
		cv2.imwrite(fn + 'total_' + key + '.png', this_img) 

	cv2.imwrite('generated/fig_tables.png', img) 


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    # import matplotlib.colors as mc
    # import colorsys
    # try:
    #     c = mc.cnames[color]
    # except:
    #     c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# DISPLAY PATHS CODE
def export_assessments_by_criteria(img, saved_paths, fn=None):
	# Given a dictionary of pairs
	# and each path with a unique set of criteria that enables them 


	if fn is None:
		fn = FILENAME_EXPORT_IMGS_PREFIX

	print("Exporting assessment diagrams")

	all_paths_img = img.copy()

	l_lookup = {} #['lo', 'la', 'lb', 'lm']
	l_lookup['omni'] = 0
	l_lookup['all'] = 1
	l_lookup['a'] = 2
	l_lookup['b'] = 3
	l_lookup['c'] = 4
	l_lookup['d'] = 5
	l_lookup['e'] = 6

	# [VIS_OMNI, VIS_A, VIS_B, VIS_MULTI]
	color_listing = [(69,21,113), (0,100,45), (52, 192, 235), (32, 85, 230),(52, 0, 235), (0, 85, 230),(52, 192, 0), (0, 0, 230),]

	f_lookup = ['fvis', 'fcombo']
	# f_lookup = ['fcombo']

	colors_2 = [(69,21,113), (52, 192, 235), (32, 85, 230), (0,100,45)]

	# VIS_CHECKLIST = [VIS_OMNI, VIS_A, VIS_B, VIS_MULTI]
	# RAW_CHECKLIST = [RAW_OMNI, RAW_A, RAW_B, RAW_MULTI]
	# PATH_COLORS = [(138,43,226), (0,255,255), (255,64,64), (0,201,87)]
	# PATH_LABELS = ['red', 'yellow', 'blue', 'green']


	# COLOR_TABLE = (32, 85, 230) #(235, 64, 52) 		# dark blue
	# COLOR_OBSERVER = (32, 85, 230) 		# dark orange
	# COLOR_FOCUS_BACK = (52, 192, 235) 		# dark yellow
	# COLOR_PERIPHERAL_BACK = (178, 221, 235) 	# light yellow
	# COLOR_FOCUS_TOWARDS = (235, 64, 52)		# dark yellow
	# COLOR_PERIPHERAL_TOWARDS = (55, 120, 191) 	# light yellow
	# COLOR_GOAL = (255, 255, 255) # (50, 168, 82) 			# green
	# COLOR_START = (100, 100, 100) 		# white

	img_deck = {}
	for aud_type in l_lookup:
		type_img = img.copy()
		img_deck[aud_type] = type_img

	for vis_type in f_lookup:
		type_img = img.copy()
		img_deck[vis_type] = type_img

	# for i in range(len(goals)):
	# 	type_img = img.copy()
	# 	img_deck[str(i)] = type_img

	# print(img_deck.keys())


	for pkey in saved_paths.keys():
		paths = saved_paths[pkey]
		# print(paths)

		# !!! KEY difference: this one can have multiple options for a path
		counter = 0
		for path in paths:
			print(path)
			path_img = img.copy()
			path_title = pkey
			crit, l, f = pkey.split("-")

			ci = l_lookup[l]			
			color = PATH_COLORS[ci]

			by_f = img_deck[f]
			by_aud = img_deck[l]

			# print("PATH IS ")
			# print(path)

			# Draw the path  
			for i in range(len(path) - 1):
				a = path[i]
				b = path[i + 1]
				
				cv2.line(path_img, a, b, color, thickness=6, lineType=8)
				cv2.line(all_paths_img, a, b, color, thickness=6, lineType=8)

				cv2.line(by_f, a, b, color, thickness=6, lineType=8)
				cv2.line(by_aud, a, b, color, thickness=6, lineType=8)		

			path_img = cv2.flip(path_img, 0)
			cv2.imwrite(fn + 'fig_path_' + path_title + '.png', path_img) 
			print("exported image of " + pkey)

	all_paths_img = cv2.flip(all_paths_img, 0)
	cv2.imwrite(fn + 'ALL_CONDITIONS' + '.png', all_paths_img) 
	### END DISPLAY PATHS CODE

	# print(img_deck.keys())

	for key in img_deck.keys():
		this_img = img_deck[key]
		this_img = cv2.flip(this_img, 0)
		cv2.imwrite(fn + 'total_' + key + '.png', this_img) 

	cv2.imwrite('generated/fig_tables.png', img) 

# DISPLAY PATHS CODE
def export_goal_options_from_assessment(img, target_index, saved_paths, fn=None):
	if fn is None:
		fn = FILENAME_EXPORT_IMGS_PREFIX

	print("\tExporting diagrams with paths from assessment")
	path_titles = ["OMNISCIENT", "TABLE", "Person A", "Person B"]

	omni_paths_img = img.copy()
	cv2.imwrite('generated/fig_path_' + "OMNISCIENT" + '.png', omni_paths_img) 

	all_paths_img = img.copy()

	img_deck = {}
	for vis_type in VIS_CHECKLIST:
		type_img = img.copy()
		img_deck[vis_type] = type_img

	for i in range(len(goals)):
		type_img = img.copy()
		img_deck[str(i)] = type_img

	print(img_deck.keys())

	for pkey in saved_paths.keys():
		paths = saved_paths[pkey]

		# !!! KEY difference: this one can have multiple options for a path
		counter = 0
		for path in paths:
			path_img = img.copy()
			path_title = pkey

			vis_type = pkey
			goal_index = str(target_index)

			color = VIS_COLOR_MAP[vis_type]
			# color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))

			by_method = img_deck[vis_type]
			by_goal = img_deck[goal_index]

			# Draw the path  
			for i in range(len(path) - 1):
				a = path[i]
				b = path[i + 1]
				
				cv2.line(path_img, a, b, color, thickness=6, lineType=8)
				cv2.line(all_paths_img, a, b, color, thickness=6, lineType=8)

				cv2.line(by_method, a, b, color, thickness=6, lineType=8)
				cv2.line(by_goal, a, b, color, thickness=6, lineType=8)		

			path_img = cv2.flip(path_img, 0)
			cv2.imwrite(fn + goal_index + 'fig_path_' + path_title + '.png', path_img) 
			print("exported image of " + pkey)

	all_paths_img = cv2.flip(all_paths_img, 0)
	cv2.imwrite(fn + 'ALL_CONDITIONS-' + goal_index + '.png', all_paths_img) 
	### END DISPLAY PATHS CODE

	# print(img_deck.keys())

	for key in img_deck.keys():
		if key == goal_index:
			this_img = img_deck[key]
			this_img = cv2.flip(this_img, 0)
			cv2.imwrite(fn  + goal_index + 'total_' + key + '.png', this_img) 

	cv2.imwrite('generated/fig_tables.png', img) 


def export_raw_paths(r, img, saved_paths_list, title, fn, sample_points_list=[]):
	print("\tExporting raw diagrams")
	all_paths_img = img.copy()

	for pi in range(len(saved_paths_list)):
		path = saved_paths_list[pi]
		
		path_img = img.copy()
		path = r.path_to_printable_path(path)

		color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))

		
		# Draw the path  
		for i in range(len(path) - 1):
			a = path[i]
			b = path[i + 1]
			
			# cv2.line(path_img, a, b, color, thickness=6, lineType=8)
			cv2.circle(all_paths_img, a, 4, color, 4)
			cv2.line(all_paths_img, a, b, color, thickness=3, lineType=8)
			

		if sample_points_list is not None and len(sample_points_list) == len(saved_paths_list):
			sample_points = sample_points_list[pi]
			for i in range(len(sample_points)):
				sp = sample_points[i]
				# cv2.circle(all_paths_img, sp, 8, (0,0,0), 8)

		# path_img = cv2.flip(path_img, 0)
		# cv2.imwrite(FILENAME_EXPORT_IMGS_PREFIX + 'fig_path_' + path_title + '.png', path_img) 
		# print("exported image of " + pkey)


	# position = (10,50)
	font_size = 1
	y0, dy = 50, 50
	for i, line in enumerate(title.split('\n')):
	    y = y0 + i*dy
	    cv2.putText(all_paths_img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (209, 80, 0, 255), 3)

	cv2.imwrite(fn + "_raw.png", all_paths_img) 
	### END DISPLAY PATHS CODE


def export_json(r, saved_paths):
	print("Exporting JSON")
	# Ready JSON Export
	table_json = []
	for table in r.get_tables():
		table_json.append(table.get_JSON())

	obs_json = []
	for obs in r.get_observers():
		obs_json.append(obs.get_JSON())

	print("Exporting path")
	file1 = open(FILENAME_OUTPUTS + "path_v1.json","a")
	data = {}
	data['paths'] = saved_paths
	data['start'] = r.get_start()
	data['goals'] = r.get_goals_all()
	data['tables'] = table_json
	data['observers'] = table_json

	# print(json.dumps(data, indent=4))

	file1.write(json.dumps(data, indent=4)) 

def export_paths_csv(saved_paths):
	print("Exporting JSON")
	for pkey in saved_paths.keys():
		csv_name = FILENAME_EXPORT_CSV_PREFIX + pkey + ".csv"
		csv_file  = open(csv_name, "w")

		output_string = ""

		path = saved_paths[pkey]
		#TODO verify the waypoint added in path creation?
		waypoint = (6.45477, 2.57)
		output_string += str(waypoint[0]) + "," + str(waypoint[1]) + "\r\n"
		unity_path = []
		for p in path:
			up = image_to_unity(p)
			unity_path.append(up)
			output_string += str(up[0]) + "," + str(up[1]) + "\r\n"


		csv_file.write(output_string)
		csv_file.close()
		print("exported csv path to " + csv_name)


def export(r, saved_paths, export_all=False):
	img = r.get_img()

	if EXPORT_DIAGRAMS or export_all:
		print('ok')
		export_diagrams_with_paths(img, saved_paths)

	if EXPORT_JSON or export_all:
		export_json(r, saved_paths)
	
	if EXPORT_CSV or export_all:
		export_paths_csv(saved_paths)


def main():
	r = generate_restaurant(TYPE_EXP_SINGLE)

	start 		= r.get_start()
	goals 		= r.get_goals_all()
	goal 		= r.get_current_goal()
	observers 	= r.get_obs_sets()
	tables 		= r.get_tables()
	waypoints 	= r.get_waypoints()
	SCENARIO_IDENTIFIER = r.get_scenario_identifier()


	# Get paths
	path = get_path(start, goal)

# if EXPO
# export(r, saved_paths)
print("Done")




