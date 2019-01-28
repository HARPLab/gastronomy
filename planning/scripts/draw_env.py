import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from ipdb import set_trace

def drawTables(coords, tables):
	lw=2
	c = [0,0,0]
	tables_color = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.9, 0.6, 0.32]]

	colors = []

	xs = [coords[0], coords[1], coords[1], coords[0], coords[0]]
	ys = [coords[3], coords[3], coords[2], coords[2], coords[3]]
	plt.plot(xs,ys,color=c, linewidth=lw)
	plt.axis(coords)

	ax = plt.gca()

	patches = []
	count = 0
	margin = 0.5
	text_margin = 0.15
	for table in tables:
		x = table.get_feature("x").value - margin
		y = table.get_feature("y").value - margin
		
		# plt.plot(r*np.cos(an) + x, r*np.sin(an) + y, color = c[count])
		# ray = agents[i, :2] + np.asarray([(agents[i, 2]) * np.cos(agents[i, 3]), (agents[i, 2]) * np.sin(agents[i, 3])])

		# plt.plot([agents[i, 0], ray[0]], [agents[i, 1], ray[1]], color = c[i] )
		rect = Rectangle((x,y), 1, 1) # facecolor doesnt work with collection?, facecolor=c[i])

		label = ax.annotate("T" + str(table.id), xy=(table.get_feature("x").value,table.get_feature("y").value - text_margin), fontsize=15, ha="center")

		rect.set_facecolor(tables_color[int(table.id)])
		colors.append(tables_color[int(table.id)])
		patches.append(rect)
		count += 1

	p = PatchCollection(patches, alpha=0.4, facecolor=colors)

	ax.add_collection(p)


def drawRobot(robot, r):
	an = np.linspace(0, 2 * np.pi, 100)
	patches = []
	ax = plt.gca()
	
	c = [[0.5, 0.45, 0.5] for i in range(4)]
	x = robot.get_feature("x").value
	y = robot.get_feature("y").value
	plt.plot(r*np.cos(an) + x, r*np.sin(an) + y, color = c[0])
	# ray = agents[i, :2] + np.asarray([(agents[i, 2]) * np.cos(agents[i, 3]), (agents[i, 2]) * np.sin(agents[i, 3])])

	# plt.plot([agents[i, 0], ray[0]], [agents[i, 1], ray[1]], color = c[i] )
	circle = Circle((x,y), r) # facecolor doesnt work with collection?, facecolor=c[i])
	circle.set_facecolor(c[0])
	text_margin = 0.15
	if robot.curr_task != None:
		label = ax.annotate("T" + str(robot.curr_task.table.id), xy=(x,y - text_margin), fontsize=15, ha="center")

	patches.append(circle)
	wheel_margin = 0.35
	for wheel in [(-1,-1),(-1,1),(1,-1),(1,1)]:
		circle = Circle((x + wheel[0] * wheel_margin, y + wheel[1] * wheel_margin), 0.15) # facecolor doesnt work with collection?, facecolor=c[i])
		circle.set_facecolor(c[0])
		patches.append(circle)

	p = PatchCollection(patches, alpha=0.4, facecolor=c)

	ax.add_collection(p)

# def drawPath(history, count):
# 	markersize = 3
# 	path_shape = ['-','--','-','--','-','--']
# 	c = [[0.0, 0.0, 1.0], [0.0, 0.7, 0.0], [1.0,0.0,0.0]]
# 	num_agents = int(history[0,:].shape[0]/2)
# 	if type(count) is int: 
# 		# c = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [0.5, 0.45, 0.5],[0.9, 0.6, 0.32]]

# 		for i in range(num_agents):
# 			plt.plot(history[:count,0+i*2], history[:count,1+i*2], path_shape[i*2], linewidth=markersize, color=c[i])
# 	else:
# 		# c = [[0.0, 0.1, 0.5], [0.7, 0.4, 0.12], [0.3, 0.3, 0.3],[0.9, 0.6, 0.32]]

# 		for i in range(count.shape[0]):
# 			if i == 0:
# 				plt.plot(history[:count[i],0+i*2], history[:count[i],1+i*2], path_shape[i*2+1], linewidth=markersize, color=c[i])
# 			else:
# 				plt.plot(history[count[i-1]:count[i],0+i*2], history[count[i-1]:count[i],1+i*2], path_shape[i*2+1], linewidth=markersize, color=c[i])


def drawTasks(tasks):
	c = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.9, 0.6, 0.32]]
	margin = 0.45
	for i in range(len(tasks)):
		color = c[int(tasks[i].table.id)]
		plt.plot(tasks[i].get_feature("x").value + margin, tasks[i].get_feature("y").value + margin, '*', markersize=12, color=color)

