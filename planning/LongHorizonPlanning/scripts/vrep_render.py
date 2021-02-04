import random
try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import math
import random
import os
from ipdb import set_trace
import numpy as np
import threading

import rospy
from geometry_msgs.msg import Twist

env_margin = 3
table_goal_margin = 1.5
class Robot:

	def __init__(self, clientID, models_path):
		self.models_path = models_path
		self.clientID = clientID
		self.robot_sim = None

	def render (self, position):
		margin = -0.35
		# print ("robot pose: ", position)
		if self.robot_sim is None:
			# x, self.robot_sim = vrep.simxGetObjectHandle(self.clientID, "Quadricopter", vrep.simx_opmode_blocking)
			# x, self.robot_sim = vrep.simxLoadModel(self.clientID, self.models_path + "Quadricopter.ttm", 0, vrep.simx_opmode_blocking) 

			x, self.robot_sim = vrep.simxGetObjectHandle(self.clientID, "youBot", vrep.simx_opmode_blocking)
			# x, self.target = vrep.simxLoadModel(self.clientID, self.models_path + "KUKA YouBot.ttm", 0, vrep.simx_opmode_blocking)

			vrep.simxSetObjectPosition(self.clientID, self.robot_sim, -1, (position[0], position[1], position[2] + margin + \
				math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.robot_sim, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])), vrep.simx_opmode_oneshot)

			#x, self.target = vrep.simxGetObjectHandle(self.clientID, "Quadricopter_target", vrep.simx_opmode_blocking) ##quad
			x, self.target = vrep.simxGetObjectHandle(self.clientID, "target", vrep.simx_opmode_blocking)
			vrep.simxSetObjectPosition(self.clientID, self.target, -1, (position[0], position[1], position[2] + margin + \
				math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.robot_sim, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])), vrep.simx_opmode_oneshot)

			x, self.target_hall = vrep.simxGetObjectHandle(self.clientID, "inflated_convex_hall_target", vrep.simx_opmode_blocking)
			# vrep.simxSetObjectPosition(self.clientID, self.target, -1, (position[0], position[1] + 1, position[2] + margin + \
			#	math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.robot_sim, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])), vrep.simx_opmode_oneshot)

			# vrep.simxSetObjectParent(self.clientID,self.target,-1,True, vrep.simx_opmode_blocking)
			# x, self.goal_robot = vrep.simxLoadModel(self.clientID, self.models_path + "KUKA YouBot.ttm", 0, vrep.simx_opmode_blocking)
			# x, self.goal_robot = vrep.simxGetObjectHandle(self.clientID, "Quadricopter#0", vrep.simx_opmode_blocking)
			# simxInt simxSetModelProperty(self.clientID,self.goal_robot,vrep.sim_modelproperty_not_collidable,vrep.simx_opmode_oneshot)
			##sim.modelproperty_not_visible
		else:
			pass
			# vrep.simxSetObjectPosition(self.clientID, self.robot_sim, -1, (position[0], position[1], position[2] + margin + \
			# 	math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.robot_sim, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])), vrep.simx_opmode_oneshot)
		# vrep.simxSetObjectPosition(self.clientID, self.goal_robot, -1, (position[0], position[1]-1, position[2] + 10 + margin + \
		# 	math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.goal_robot, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])), vrep.simx_opmode_oneshot)
		
class Table:
	def __init__ (self, clientID, models_path, id):
		self.models_path = models_path
		self.id = id
		self.clientID = clientID
		self.chairs = []
		self.humans = []
		self.sim_table = None

	def render (self, table, sim_env):
		position = [table.get_feature("x").value-env_margin, table.get_feature("y").value-env_margin, 0]

		x, sim_table = vrep.simxLoadModel(self.clientID, self.models_path + "diningTable.ttm", 0, vrep.simx_opmode_blocking) 
		# x = vrep.simxSetObjectParent(self.clientID, sim_table, sim_env, True, vrep.simx_opmode_blocking)
		vrep.simxSetObjectPosition(self.clientID, sim_table, -1, (position[0], position[1], position[2] + \
			vrep.simxGetObjectFloatParameter (self.clientID, sim_table, vrep.sim_objfloatparam_modelbbox_max_x, vrep.simx_opmode_blocking)[1]), vrep.simx_opmode_oneshot)
		# vrep.simxSetModelProperty(self.clientID,sim_table,vrep.sim_modelproperty_not_collidable,vrep.simx_opmode_oneshot)

		#returnCode, prop = vrep.simxGetModelProperty(self.clientID,sim_table,vrep.simx_opmode_blocking)
		#returnCode2 = vrep.simxSetModelProperty(self.clientID, sim_table, prop + vrep.sim_modelproperty_not_respondable, vrep.simx_opmode_blocking)
		#returnCode3, prop = vrep.simxGetModelProperty(self.clientID, sim_table, vrep.simx_opmode_blocking)

		#print (prop)
		

		offset_z = math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, sim_table, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])
		offset_y = math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, sim_table, vrep.sim_objfloatparam_modelbbox_min_y, vrep.simx_opmode_blocking)[1])
		margin = 0.2
		bill_margin = 0.05
		
		self.chairs_offset = [(0,offset_y+margin,0,0,0,0), (0,-offset_y-margin,0,0,0,math.pi), (offset_z+margin,0,0,0,0,-math.pi/2), (-offset_z-margin,0,0,0,0,math.pi/2)]
		self.humans_offset = [(0,offset_y-margin,bill_margin,0,0,-math.pi/2), (0,-offset_y+margin,bill_margin,0,0,math.pi/2), (offset_z-margin,0,bill_margin,0,0,math.pi), (-offset_z+margin,0,bill_margin,0,0,0)]

		for i in range(len(table.humans)):
			sim_chair = self.render_chair (sim_table, position, self.chairs_offset[i])
			self.chairs.append(sim_chair)
			self.humans.append(self.render_human (sim_chair, position, self.humans_offset[i]))

		self.sim_table = sim_table
		return sim_table

	def render_chair (self, table, position, offset):
		x, chair = vrep.simxLoadModel(self.clientID, self.models_path + "dining chair.ttm", 0, vrep.simx_opmode_blocking)
		x = vrep.simxSetObjectParent(self.clientID, chair, table, True, vrep.simx_opmode_blocking)
		offset_z = math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, chair, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1])
		vrep.simxSetObjectPosition(self.clientID, chair, -1, (position[0] + offset[0], position[1] + offset[1], position[2] + offset_z + offset[2]), vrep.simx_opmode_oneshot)
		vrep.simxSetObjectOrientation(self.clientID, chair, -1, (offset[3], offset[4], offset[5]), vrep.simx_opmode_oneshot)

		#returnCode, prop = vrep.simxGetModelProperty(self.clientID,chair,vrep.simx_opmode_blocking)
		# print (prop, prop + vrep.sim_modelproperty_not_respondable + vrep.sim_modelproperty_not_collidable)
		#returnCode2 = vrep.simxSetModelProperty(self.clientID, chair, prop  + vrep.sim_modelproperty_not_respondable, vrep.simx_opmode_blocking)
		#returnCode3, prop = vrep.simxGetModelProperty(self.clientID,chair,vrep.simx_opmode_blocking)
		# print (returnCode, returnCode2, returnCode3, prop)
		# vrep.simxSetModelProperty(self.clientID,chair,vrep.sim_modelproperty_not_collidable,vrep.simx_opmode_oneshot)
		
		return chair

	def render_human (self, chair, position, offset):
		x, person = vrep.simxLoadModel(self.clientID, self.models_path + "Sitting Bill.ttm", 0, vrep.simx_opmode_blocking)
		x = vrep.simxSetObjectParent(self.clientID, person, chair, True, vrep.simx_opmode_blocking)
		offset_z = vrep.simxGetObjectFloatParameter (self.clientID, person, vrep.sim_objfloatparam_modelbbox_max_z, vrep.simx_opmode_blocking)[1]
		vrep.simxSetObjectPosition(self.clientID, person, -1, (position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]), vrep.simx_opmode_oneshot)
		vrep.simxSetObjectOrientation(self.clientID, person, -1, (offset[3], offset[4], offset[5]), vrep.simx_opmode_oneshot)
		# vrep.simxSetModelProperty(self.clientID,person,vrep.sim_modelproperty_not_collidable,vrep.simx_opmode_oneshot)


		# x, objs = vrep.simxGetObjects(self.clientID, vrep.sim_handle_all, vrep.simx_opmode_oneshot_wait)
		# x, joint = vrep.simxGetObjectHandle(self.clientID, 'Bill_leftElbowJoint', vrep.simx_opmode_oneshot_wait)
		# x, shoulder = vrep.simxGetObjectHandle(self.clientID, 'Bill_leftShoulderJoint', vrep.simx_opmode_oneshot_wait)
		# rotationMatrix = (-1,0,0,0,0,-1,0,0,0,0,-1,0)
		# x = vrep.simxSetSphericalJointMatrix(self.clientID, shoulder, rotationMatrix,vrep.simx_opmode_streaming)

		#Moving Rotational Joint
		# ang = -math.pi/2
		# x = vrep.simxSetJointPosition(self.clientID, joint, ang, vrep.simx_opmode_oneshot)

		# x, pos = vrep.simxGetJointPosition(self.clientID, joint, vrep.simx_opmode_streaming)

		return person

	def wave_human(self, num=0):
		"""Put num as -1 if name is just 'Bill', but put num as n if name is in form 'Bill#n' """
		if num == 0:
			# _,_,_,_,stringData = vrep.simxGetObjectGroupData(self.clientID,self.humans[0],0,vrep.simx_opmode_blocking)
			# print (stringData)
			# set_trace()
			human_num = ""
		else:
			human_num = str(num)

		e, leftShoulder = vrep.simxGetObjectHandle(self.clientID, 'Bill_leftShoulderJoint#' + human_num, vrep.simx_opmode_oneshot_wait)
		e, rightShoulder = vrep.simxGetObjectHandle(self.clientID, 'Bill_rightShoulderJoint#' + human_num, vrep.simx_opmode_oneshot_wait)
		e, leftElbow = vrep.simxGetObjectHandle(self.clientID, 'Bill_leftElbowJoint#' + human_num, vrep.simx_opmode_oneshot_wait)
		e, rightElbow = vrep.simxGetObjectHandle(self.clientID, 'Bill_rightElbowJoint#' + human_num, vrep.simx_opmode_oneshot_wait)

		for i in range (0,10):
			matrix = Rz(math.pi / 2).dot(Ry(-math.pi/1.3*(i/100)));
			e = vrep.simxSetSphericalJointMatrix(self.clientID, leftShoulder, np.asarray(matrix).reshape(-1)[:12], vrep.simx_opmode_streaming)
			time.sleep(0.005)

		for i in range(2):
			for i in range(0, 100):
				matrix = Rz(math.pi / 2).dot(Ry(-math.pi / 1.3 + abs(-math.pi / 1.3 + math.pi/1.4) * (i / 100)))
				ang = -0.7293655872344971 + (0.7293655872344971)*(i/100)
				e = vrep.simxSetSphericalJointMatrix(self.clientID, leftShoulder, np.asarray(matrix).reshape(-1)[:12], \
					vrep.simx_opmode_streaming)
				e = vrep.simxSetJointPosition(self.clientID,leftElbow,ang,vrep.simx_opmode_streaming)
				time.sleep(0.005)

			for i in range(0, 100):
				matrix = Rz(math.pi / 2).dot(Ry(-math.pi / 1.4 - abs(-math.pi / 1.3 + math.pi/1.4) * (i / 100)))
				ang = 0 - (0.7293655872344971)*(i/100)
				e = vrep.simxSetSphericalJointMatrix(self.clientID, leftShoulder, np.asarray(matrix).reshape(-1)[:12], \
					vrep.simx_opmode_streaming)
				e = vrep.simxSetJointPosition(self.clientID,leftElbow,ang,vrep.simx_opmode_streaming)
				time.sleep(0.005)
		

	# def render_cutlery (self, position, plate_offset, fork_offset, spoon_offset):
	# 	x, plate = vrep.simxLoadModel(self.clientID, self.models_path + "plate.ttm", 0, vrep.simx_opmode_blocking)
	# 	vrep.simxSetObjectPosition(self.clientID, plate, -1, (position[0], position[1] - 0.3, position[2] + 0.77), vrep.simx_opmode_oneshot)
	# 	x, spoon = vrep.simxLoadModel(self.clientID, self.models_path + "spoon.ttm", 0, vrep.simx_opmode_blocking)
	# 	vrep.simxSetObjectPosition(self.clientID, spoon, -1, (position[0] + 0.2, position[1] - 0.3, position[2] + 0.77), vrep.simx_opmode_oneshot)
	# 	x, fork = vrep.simxLoadModel(self.clientID, self.models_path + "fork.ttm", 0, vrep.simx_opmode_blocking)
	# 	vrep.simxSetObjectPosition(self.clientID, fork, -1, (position[0] - 0.2, position[1] - 0.3, position[2] + 0.77), vrep.simx_opmode_oneshot)
	# 	return plate, spoon, fork

class Vrep_Restaurant (object):
	"""docstring for vrep"""
	def __init__(self):
		print ('Program started')
		self.models_path = os.getcwd()  + "/../models/"

		vrep.simxFinish(-1) # just in case, close all opened connections
		self.clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
		vrep.simxSynchronous(self.clientID,True)

		self.youbot_cmd = rospy.Publisher('cmd_vel', Twist, queue_size=100)
		if self.clientID!=-1:
			print ('Connected to remote API server')
			vrep.simxLoadScene(self.clientID, os.getcwd()  + '/../scene/remote api.ttt',0xFF,vrep.simx_opmode_blocking)
			self.tables = []
			self.robot = Robot(self.clientID, self.models_path)
			self.reset = True
			self.planner = Planner()
		else:
			print ('Failed connecting to remote API server')
		print ('Program ended')

	
	def render_tables (self, tables, sim_env):
		if len(self.tables) == 0:
			for table in tables:
				new_table = Table (self.clientID, self.models_path, table.id)
				self.tables.append(new_table)
				t = new_table.render(table, sim_env)



			
	def render (self, robot, tables, done = False):
		if self.reset:
			# self.robot.render([robot.get_feature("x").value-3, robot.get_feature("y").value-3, 2])
			self.robot.render([robot.get_feature("x").value-env_margin, robot.get_feature("y").value-env_margin, 0]) #quad

			returnCode, envHandle = vrep.simxGetObjectHandle(self.clientID, "Environment", vrep.simx_opmode_blocking)
			self.render_tables (tables, envHandle)

			vrep.simxGetPingTime(self.clientID)
			self.reset = False
			dt = .01
   			# vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, dt, vrep.simx_opmode_oneshot)
			# vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
			
			# set_trace()
		else:
			# self.robot.render([robot.get_feature("x").value-3, robot.get_feature("y").value-3, 2])
			pass

		if done:
			# vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
			vrep.simxFinish(self.clientID)

	def done(self):
		vrep.simxFinish(self.clientID)

	def go_to (self, start_pos, end_pos, table_id):	
		# x = start_pos[0]-3; y = start_pos[1]-3; z = 2; #quad
		# new_x = end_pos[0]-3; new_y = end_pos[1]-3; new_z = 2 ##quad
		human_id = 0
		for i in range(0,table_id):
			human_id += len(self.tables[i].humans)

		wave = threading.Thread(target=self.tables[table_id].wave_human(human_id),daemon=True, args=())
		wave.start()

		_,position = vrep.simxGetObjectPosition(self.clientID,self.robot.robot_sim,-1,vrep.simx_opmode_blocking)
		_,orientation = vrep.simxGetObjectOrientation(self.clientID,self.robot.robot_sim,-1,vrep.simx_opmode_blocking)
		x = position[0]; y = position[1]; z = position[2];
		print (orientation)
		#z = math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.robot.robot_sim, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1]);
		
		vrep.simxSetObjectPosition(self.clientID, self.robot.target, -1, (x,y,z), vrep.simx_opmode_oneshot_wait)

		new_x = end_pos[0]-env_margin+table_goal_margin; new_y = end_pos[1]-env_margin+table_goal_margin; 
		new_z = position[2]
		
		# math.fabs(vrep.simxGetObjectFloatParameter (self.clientID, self.robot.robot_sim, vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1]);

		_,orientation = vrep.simxGetObjectOrientation(self.clientID,self.robot.robot_sim,-1,vrep.simx_opmode_blocking)
		print (orientation)
		# threshold = 0.5 

		print ("go to... start:", (x, y, z), orientation, ", goal: ", (new_x, new_y, new_z))
		# if start_pos != end_pos:

		

		# 	_,_,_,_,_ = vrep.simxCallScriptFunction(self.clientID,"youBot",vrep.sim_scripttype_childscript,"goto",[],[new_x,new_y,new_z],[],bytearray(),vrep.simx_opmode_oneshot_wait)
			
		# 	code, xyz = vrep.simxGetObjectPosition(self.clientID, self.robot.robot_sim, -1, vrep.simx_opmode_blocking)
		# 	x = xyz[0]; y = xyz[1]; z = xyz[2]
		# 	count = 0
		# 	while  count < 100:
		# 		vrep.simxSynchronousTrigger(self.clientID)
		# 		code, xyz = vrep.simxGetObjectPosition(self.clientID, self.robot.robot_sim, -1, vrep.simx_opmode_blocking)
		# 		x = xyz[0]; y = xyz[1]; z = xyz[2]			
		# 		count += 1

		# 	vrep.simxGetPingTime(self.clientID)	
		came_from, cost_so_far = self.planner.a_star_search(self, (x,y,z), (new_x,new_y,new_z))

		if cost_so_far != -1:
			path = self.planner.reconstruct_path(came_from, (x,y,z), (new_x,new_y,new_z))
			set_trace()
			for p in path:
				v0rep.simxSetObjectPosition(self.clientID, self.robot.target, -1, p, vrep.simx_opmode_oneshot_wait)

		# returnCode,collision,_,_,_ = vrep.simxCallScriptFunction(self.clientID,"youBot",vrep.sim_scripttype_childscript,"isInCollision", \
		# 	table_handles,[],[],bytearray(),vrep.simx_opmode_oneshot_wait)
		# print (collision)
		# if len(collision) > 0 and collision[0] == 1:
		# 	print ("inCollision")
		# else: 
		# 	vel_msg = Twist()
		# 	speed = 1
		# 	isForward = False
		# 	if(isForward):
		# 		vel_msg.linear.x = abs(speed)
		# 		vel_msg.linear.y = -abs(speed)
		# 	else:
		# 		vel_msg.linear.x = -abs(speed)
		# 		vel_msg.linear.y = abs(speed)
		
		# 	vel_msg.linear.z = 0	
		# 	vel_msg.angular.x = 0
		# 	vel_msg.angular.y = 0
		# 	vel_msg.angular.z = 0

		# 	self.youbot_cmd.publish(vel_msg)
		
	def checkCollision(self, start, goal):

		# m=simGetObjectMatrix(handle,-1)
		# mRot=simBuildMatrix({0,0,0},{0,0,rot})
		# newM=simMultiplyMatrices(m,mRot)
		# simSetObjectMatrix(handle,-1,newM)

		# print ("check collision: ", start, goal)
		vrep.simxSetObjectPosition(self.clientID, self.robot.target, -1, goal, vrep.simx_opmode_oneshot_wait)

		collision_handles = []
		for table in self.tables:
			collision_handles.append(table.sim_table)
			for chair in table.chairs:
				collision_handles.append(chair)

			for human in table.humans:
				collision_handles.append(human)

		returnCode,collision,_,_,_ = vrep.simxCallScriptFunction(self.clientID,"target",vrep.sim_scripttype_childscript,"isInCollision", \
			collision_handles,[],[],bytearray(),vrep.simx_opmode_oneshot_wait)

		#vrep.simxSetObjectPosition(self.clientID, self.robot.target, -1, start, vrep.simx_opmode_oneshot_wait)

		if len(collision) > 0 and collision[0] == 1:
			print ("inCollision")
			return 1

		return 0

######################################## planning
def Ry(y):
	return np.array([[math.cos(y), 0, math.sin(y), 0],
				[0, 1, 0, 0],
				[-math.sin(y), 0, math.cos(y), 0],
				[0, 0, 0, 1]])

def Rx(x):
	return np.array([[1, 0, 0, 0],
				[0, math.cos(x), -math.sin(x), 0],
				[0, math.sin(x), math.cos(x), 0],
				[0, 0, 0, 1]])

def Rz(z):
	return np.array([[math.cos(z), -math.sin(z), 0, 0],
				[math.sin(z), math.cos(z),0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])
def Rt(xt,yt,zt):
	return np.asarray(zt.dot(yt)).reshape(-1)[:12]

import collections
import heapq

class Queue:
	def __init__(self):
		self.elements = collections.deque()

	def empty(self):
		return len(self.elements) == 0

	def put(self, x):
		self.elements.append(x)

	def get(self):
		return self.elements.popleft()

class PriorityQueue:
	def __init__(self):
		self.elements = []

	def empty(self):
		return len(self.elements) == 0

	def put(self, item, priority):
		heapq.heappush(self.elements, (priority, item))

	def get(self):
		return heapq.heappop(self.elements)[1]


def divide_tuple_by_int(tup,div):
	return tuple(int(t/div) for t in tup)

def multiply_tuple_by_int(tup,mul):
	return tuple(t*mul for t in tup)

class Planner:

	def __init__(self):
		self.discretization = 0.5

	def neighbors(self, env, current):
		(x,y,z) = current
		neighbors = []
		for (n_x,n_y) in [(x+1, y), (x, y-1), (x-1, y), (x, y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1), (x+1,y+1)]:
			if not env.checkCollision(multiply_tuple_by_int((x,y,z),self.discretization), \
				multiply_tuple_by_int((n_x,n_y,z),self.discretization)):
				neighbors.append((n_x,n_y,z))
		return neighbors

	def cost(self, current, next):
		return self.distance(current,next)

	def distance(self, a, b):
		(x1, y1, z) = multiply_tuple_by_int(a, self.discretization)
		(x2, y2, z) = multiply_tuple_by_int(b, self.discretization)
		return np.sqrt(np.power((x1 - x2),2) + np.power((y1 - y2),2))

	def heuristic(self, a, b):
		return self.distance(a,b)

	def a_star_search(self, env, start, goal):
		start = divide_tuple_by_int(start,self.discretization)
		goal = divide_tuple_by_int(goal,self.discretization)
		print ("grid start: ", start, ", goal: ", goal)
		frontier = PriorityQueue()
		frontier.put(start, 0)
		came_from = {}
		cost_so_far = {}
		came_from[start] = None
		cost_so_far[start] = 0
		path_found = False
		while not frontier.empty():
			current = frontier.get()

			if current == goal:
				# print ("new goal: ", current)
				path_found = True
				break

			for next in self.neighbors(env, current):
				new_cost = cost_so_far[current] + self.cost(current, next)
				#print ("neighor: ", next, new_cost)
				if next not in cost_so_far or new_cost < cost_so_far[next]:
					cost_so_far[next] = new_cost
					priority = new_cost + self.heuristic(goal, next)
					frontier.put(next, priority)
					came_from[next] = current

		if not path_found:
			print ("path not found!")
			return {},-1
		return came_from, cost_so_far

	def reconstruct_path(self, came_from, start, goal):
		start = divide_tuple_by_int(start,self.discretization)
		goal = divide_tuple_by_int(goal,self.discretization)
		# print ("grid start: ", start, ", goal: ", goal)
		current = goal
		path = []
		while current != start:
			path.append(multiply_tuple_by_int(current, self.discretization))
			current = came_from[current]
		path.append(start) # optional
		path.reverse() # optional
		return path