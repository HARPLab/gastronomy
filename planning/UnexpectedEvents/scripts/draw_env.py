import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,8)
import seaborn as sns
import pandas as pd
from ipdb import set_trace
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

def drawTables_beliefs(coords, tables, pomdp_tasks, actions, pomdp_solvers, final_rew):
	lw=2
	c = [0,0,0]
	sat_colors = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5], \
					[0.2, 0.4, 0.7], [0.9, 0.7, 0.32], [1,0.1,0],[0.2, 0.7, 0.7],[0.7, 0.2, 0.5]]

	count = 0
	data = []
	data.append(['','0','0'])
	data.append(['','1','0'])
	data.append(['','2','0'])
	data.append(['','3','0'])
	data.append(['','4','0'])
	data.append(['','5','0'])

	filter_data = np.full((len(pomdp_tasks)+1,),True)
	filter_data[0] = False

	for task in pomdp_tasks:
		solver_prob = pomdp_solvers[count].belief.prob
		for p in solver_prob:
			dat = []
			dat.append("Table " + str(task.task.table.id))
			st = task.get_state_tuple(p[1])			
			dat.append(str(st[task.feature_indices['customer_satisfaction']]))
			dat.append(round(p[0],2))
			data.append(dat)
		count += 1

	df = pd.DataFrame(data=data, columns = ['Table', 'Satisfaction_level','Probability'])
	df['Probability']=df['Probability'].astype(float)
	df_plot = df.groupby(['Satisfaction_level', 'Table','Probability']).size().reset_index().pivot(columns='Satisfaction_level', index='Table', values='Probability')
	df_plot = df_plot[filter_data]
	if actions == "0: ":
		df_plot.plot(kind='bar', stacked=True, figsize=(6,6), width=1.0, edgecolor = "k", colormap="bwr_r")
		plt.legend(bbox_to_anchor=(-0.00, 1., 1.00,.102), loc='lower left', ncol=6, mode="expand", borderaxespad=0., title="Satisfaction level")
	else:
		df_plot.plot(kind='bar', stacked=True, figsize=(6,6), width=1.0, edgecolor = "k", legend=False, colormap="bwr_r")

	ax.set_xlabel('Table',fontsize=26)
	ax.set_ylabel('Probability',fontsize=26)
	fig = plt.gcf()
	x_min, x_max = ax.get_xlim()
	ticks = [(tick - x_min)/(x_max - x_min) for tick in ax.get_xticks()]

	count = 0
	for task in pomdp_tasks:
		solver_prob = pomdp_solvers[count].belief.prob
		for p in solver_prob:
			st = task.get_state_tuple(p[1])			
			time_steps = str(st[task.feature_indices['time_since_hand_raise']])
			hand_raise = st[task.feature_indices['hand_raise']]

		if hand_raise == 1:
			plt.text(x=x_pos+0.1, y=0.86, s="t=" + str(time_steps), fontsize=15, ha="center", transform=fig.transFigure)
		req = st[task.feature_indices['current_request']]
		req_text = ""
		# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
		hand_raise = st[task.feature_indices['hand_raise']]
		if hand_raise == 0:
			req_text += "table\ndone"
		else:
			if req == 1:
				req_text += "want\nmenu"
			elif req == 2:
				req_text += "ready\nto\norder"
			elif req == 3 and st[task.feature_indices['cooking_status']] != 2:
				req_text += "want\nfood"
			elif req == 3 and st[task.feature_indices['cooking_status']] == 2:
				req_text += "food\nready"
			elif req == 4 and st[task.feature_indices['food']] != 3:
				req_text += "food\nserved"
			elif req == 4 and st[task.feature_indices['food']] == 3:
				req_text += "want\ndessert"
			elif req == 5 and st[task.feature_indices['water']] != 3:
				req_text += "dessert\nserved"
			elif req == 5 and st[task.feature_indices['water']] == 3:
				req_text += "want\nbill"
			elif req == 6:
				req_text += "get\ncards"
			elif req == 7:
				req_text += "want\ncards"
			elif req == 8:
				req_text += "collect\nbill"

		plt.text(x=x_pos+0.1, y=0.78, s=req_text, fontsize=12, ha="center", transform=fig.transFigure)
		count += 1
	


def drawRobot_beliefs(robot, tables, actions):
	an = np.linspace(0, 2 * np.pi, 100)
	patches = []
	ax = plt.gca()
	fig = plt.gcf()
	
	c = [[0.5, 0.45, 0.5] for i in range(4)]
	x = robot.get_feature("x").value
	y = robot.get_feature("y").value
	text = "robot at start position"
	for table in range(len(tables)):
		goal_x = tables[table].goal_x
		goal_y = tables[table].goal_y
		if goal_x == x and goal_y == y:
			text = "at table " + str(table)
			break

	if actions != "0: ":
		plt.text(x=0.53, y=0.94, s="action " + actions + ' --- ' + text, fontsize=15, ha="center", transform=fig.transFigure)
	else:
		pass

def drawTasks_beliefs(tasks):
	c = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5], \
					[0.2, 0.4, 0.7], [0.9, 0.7, 0.32], [1,0.1,0],[0.2, 0.7, 0.7],[0.7, 0.2, 0.5]]
	margin = 0.45
	for i in range(len(tasks)):
		color = c[int(tasks[i].table.id)]
		plt.plot(tasks[i].get_feature("x").value + margin, tasks[i].get_feature("y").value + margin, '*', markersize=12, color=color)



def drawTables(coords, tables, pomdp_tasks, pomdp_solvers, actions, final_rew):
	feature_name = "current_request"
	lw=2
	c = [0,0,0]
	tables_color = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5], \
					[0.2, 0.4, 0.7], [0.9, 0.7, 0.32], [1,0.1,0],[0.2, 0.7, 0.7],[0.7, 0.2, 0.5], \
					[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5]]

	colors = []

	fig, ax = plt.subplots()

	xs = [coords[0], coords[1], coords[1], coords[0], coords[0]]
	ys = [coords[3], coords[3], coords[2], coords[2], coords[3]]
	ax.plot(xs,ys,color=c, linewidth=0)
	ax.axis(coords)

	ax.set_xticks([])
	ax.set_yticks([])


	ax.spines['bottom'].set_color(None)
	ax.spines['top'].set_color(None) 
	ax.spines['right'].set_color(None)
	ax.spines['left'].set_color(None)


	patches = []
	count = 0
	margin = 0.6
	text_margin = 0.15

	wheel_margin = 0.35

	for table in tables:
		x = table.goal_x - margin
		y = table.goal_y - margin
		
		rect = Circle((x+margin,y+margin), .5)

		label = ax.annotate("T" + str(table.id), xy=(table.goal_x,table.goal_y - text_margin), fontsize=18, ha="center")

		rect.set_facecolor(tables_color[int(table.id)])
		colors.append(tables_color[int(table.id)])
		patches.append(rect)

		wheel_margin = 0.35
		for wheel in [(-1,-1),(-1,1),(1,-1),(1,1)]:
			circle = Circle((x + margin + wheel[0] * wheel_margin, y + margin + wheel[1] * wheel_margin), 0.15) # facecolor doesnt work with collection?, facecolor=c[i])
			circle.set_facecolor(tables_color[int(table.id)])
			patches.append(circle)
			colors.append(tables_color[int(table.id)])

		count += 1

	##################### KITCHEN
	if pomdp_tasks[0].KITCHEN:
		x = pomdp_tasks[0].kitchen_pos[0] - margin
		y = pomdp_tasks[0].kitchen_pos[1] - margin
		rect = Circle((x+margin,y+margin), .5)

		label = ax.annotate("K", xy=(pomdp_tasks[0].kitchen_pos[0],pomdp_tasks[0].kitchen_pos[1] - text_margin), fontsize=18, ha="center")

		rect.set_facecolor([0,0,0])
		colors.append([0,0,0])
		patches.append(rect)

	############################################# KITCHEN

	p = PatchCollection(patches, alpha=0.3, facecolor=colors)
	ax.add_collection(p)


	ax.add_collection(p)

	ax.add_patch(Rectangle((0.5, 0.5), 8.6, 9,alpha=1,fill=None))

	count = 0
	req_text = ""
	for task in pomdp_tasks:
		solver_prob = pomdp_solvers[count].belief.prob
		x = tables[count].goal_x
		y = tables[count].goal_y
		feature_high = task.task.get_feature(feature_name).high + 1
		sats = np.full((feature_high,2),0.0); 
		sats[0:feature_high,0] = np.arange(0,feature_high);
		for p in solver_prob:
			st = task.get_state_tuple(p[1])			
			time_steps = str(st[task.feature_indices['time_since_hand_raise']])
			time_food_ready = str(st[task.feature_indices['time_since_food_ready']])
			hand_raise = st[task.feature_indices['hand_raise']]
			cooking_status = st[task.feature_indices['cooking_status']]
			sat = st[task.feature_indices['customer_satisfaction']]
			req = st[task.feature_indices['current_request']]

			#################################### 
			# max_time = task.state_space[task.feature_indices['time_since_hand_raise']][1]
			# max_sat = task.state_space[task.feature_indices['customer_satisfaction']][1]
			# if req == 3 and (cooking_status == 2):			
			# 	if st[task.feature_indices['time_since_food_ready']]-1 >= 0:
			# 		time_index = st[task.feature_indices['time_since_food_ready']]-1
			# 	else:
			# 		time_index = 0
			# 	threshold = int(max_time/(max_sat+1))
			# else:
			# 	if st[task.feature_indices['time_since_hand_raise']]-1 >= 0:
			# 		time_index = st[task.feature_indices['time_since_hand_raise']]-1
			# 	else:
			# 		time_index = 0
			# 	threshold = int(max_time/(max_sat))
			#######################################

			sat_req = st[task.feature_indices[feature_name]]
			sats[sat_req,1] += round(p[0],2)

			if req == 3:
				if cooking_status == 2: # + str(time_food_ready)
					label = ax.annotate("food ready", xy=(x-0.2,y+1.17), fontsize=22, ha="center")
					if hand_raise == 1:
						label = ax.annotate("t=" + str(time_food_ready), xy=(x,y+1.7), fontsize=22, ha="center")
				elif cooking_status == 1:
					label = ax.annotate("food half", xy=(x-0.2,y+1.55), fontsize=22, ha="center")
					label = ax.annotate("cooked", xy=(x-0.2,y+1.17), fontsize=22, ha="center")
					if hand_raise == 1:
						label = ax.annotate("t=" + str(0), xy=(x,y+2), fontsize=22, ha="center")
				elif cooking_status == 0:
					label = ax.annotate("food being", xy=(x-0.2,y+1.55), fontsize=22, ha="center")
					label = ax.annotate("cooked", xy=(x-0.2,y+1.17), fontsize=22, ha="center")
					if hand_raise == 1:
						label = ax.annotate("t=" + str(0), xy=(x,y+2), fontsize=22, ha="center")


			else:
				if hand_raise == 1:
					label = ax.annotate("t=" + str(time_steps), xy=(x,y+1.17), fontsize=22, ha="center")
			
			req_text = ""
			# no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
			hand_raise = st[task.feature_indices['hand_raise']]
			if hand_raise == 0:
				req_text += "table done"
			else:
				if req == 1:
					req_text += "want menu"
				elif req == 2:
					req_text += "ready to order"
				elif req == 3 and st[task.feature_indices['cooking_status']] != 2:
					req_text += "want food"
				elif req == 3 and st[task.feature_indices['cooking_status']] == 2:
					req_text += "food ready"
				elif req == 4 and st[task.feature_indices['food']] != 3:
					req_text += "eating"
				elif req == 4 and st[task.feature_indices['food']] == 3:
					req_text += "want dessert"
				elif req == 5 and st[task.feature_indices['water']] != 3:
					req_text += "eating dessert"
				elif req == 5 and st[task.feature_indices['water']] == 3:
					req_text += "want bill"
				elif req == 6:
					req_text += "cash ready" ##payment ready
				elif req == 7:
					req_text += "cash collected"
				elif req == 8:
					req_text += "clean table"

		# print ("time",time_index)
		# mod = np.mod(time_index,threshold)
		# deduct_sat = (mod*1.0)/(threshold)
		# for sat in reversed(range(0,6)):
		# 	print ("*************************")
		# 	if sats[sat,1] > 0:
		# 		d = sats[sat,1] - deduct_sat
		# 		if sat != 0:
		# 			print (sat, sats[sat-1,1], sats[sat,1], deduct_sat, d)
		# 			if d < 0:
		# 				sats[sat-1,1] += sats[sat,1]
		# 				sats[sat,1] = 0
		# 			else:
		# 				sats[sat-1,1] += deduct_sat
		# 				sats[sat,1] = d

		# 			deduct_sat = 0
		# 			if d < 0 or sats[sat-1,1] > 1:
		# 				if d < 0:
		# 					deduct_sat = -d
		# 			else:	
		# 				break

		# 	print (sat, sats[sat-1,1], sats[sat,1])
		# #####################################

		df = pd.DataFrame(data=sats, columns = [feature_name,'Probability'])
		if count ==0:
			label = ax.annotate(req_text, xy=(x-0.05,y+0.7), fontsize=22, ha="center")
		elif count == 4:
			label = ax.annotate(req_text, xy=(x-0.5,y+0.7), fontsize=22, ha="center")
		else:
			label = ax.annotate(req_text, xy=(x-0.3,y+0.7), fontsize=22, ha="center")


		if len(pomdp_tasks) == 5:
			margin = 2
			if count == 0:
				ax2 = ax.inset_axes([(x-0.5)/10.0,(y-2.3)/10.0,0.15,0.15])
			if count == 1:
				ax2 = ax.inset_axes([(x-1.2)/10.0,(y-margin)/10.0,0.15,0.15])
			if count == 2:
				ax2 = ax.inset_axes([(x+0.8)/10.0,(y-0.65)/10.0,0.15,0.15])
			if count == 3:
				ax2 = ax.inset_axes([(x-0.7)/10.0,(y-margin)/10.0,0.15,0.15])
			if count == 4:
				ax2 = ax.inset_axes([(x-3.7)/10.0,(y-.5)/10.0,0.15,0.15])

		elif len(pomdp_tasks) == 3:
			margin = 2
			if count == 0:
				ax2 = ax.inset_axes([(x+1.1)/10.0,(y-0.75)/10.0,0.15,0.15])
			if count == 1:
				ax2 = ax.inset_axes([(x+1.1)/10.0,(y-0.65)/10.0,0.15,0.15])
			if count == 2:
				ax2 = ax.inset_axes([(x-0.5)/10.0,(y-2.3)/10.0,0.15,0.15])
		else:
			ax2 = ax.inset_axes([(x)/10.0,(y)/10.0,0.15,0.15])

		ax2.bar(sats[:,0],sats[:,1], align='center', alpha=1.0, color=tables_color[count])		
		ax2.set_xticks(np.arange(0,feature_high))
		ax2.set_yticks([])
		ax2.set_ylim((0,1))
		count += 1
	
	label = ax.annotate("a" + actions.replace('- ', ''), xy=(5,9.7), fontsize=10, ha="center")



def drawRobot(tables, pomdp_tasks, robot, r):
	an = np.linspace(0, 2 * np.pi, 100)
	patches = []
	ax = plt.gca()
	
	c = [[0.5, 0.45, 0.5] for i in range(4)]
	x = robot.get_feature("x").value
	y = robot.get_feature("y").value


	img = mpimg.imread('robot.png')

	count = 0
	found = False
	for table in range(len(tables)):
		goal_x = tables[table].goal_x
		goal_y = tables[table].goal_y
		if pomdp_tasks[0].KITCHEN:
			if goal_x == x and goal_y == y:
				if count == 0:
					x += -1.7; y += -1.0;
				if count == 1:
					x += -2; y += -1.1;
				if count == 2:
					x += -1.6; y += -1.6;
				if count == 3:
					x += -1.7; y += -1.1;
				if count == 4:
					x += -1.9; y += -1.4;
				break
		else:
			if goal_x == x and goal_y == y:
				if count == 0:
					x += -1.5; y += -1.4;
				if count == 1:
					x += -2; y += -1.1;
				if count == 2:
					x += -1.3; y += -0.8;
				if count == 3:
					x += -1.7; y += -1.1;
				if count == 4:
					x += -1.9; y += -1.4;
				break
		count += 1

	if pomdp_tasks[0].KITCHEN and pomdp_tasks[0].kitchen_pos[0] == x and pomdp_tasks[0].kitchen_pos[1] == y:
		x += -1.5; y += -1.5;

	ax2 = ax.inset_axes([(x)/10.0,(y)/10.0,0.15,0.15])
	ax2.patch.set_alpha(0)
	ax2.imshow(img)

	ax2.set_xticks([])
	ax2.set_yticks([])

	ax2.spines['bottom'].set_color(None)
	ax2.spines['top'].set_color(None) 
	ax2.spines['right'].set_color(None)
	ax2.spines['left'].set_color(None)
	# ax2.axis('off')
	count = 0



def drawTasks(tasks):
	c = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5], \
					[0.2, 0.4, 0.7], [0.9, 0.7, 0.32], [1,0.1,0],[0.2, 0.7, 0.7],[0.7, 0.2, 0.5]]
	margin = 0.45
	for i in range(len(tasks)):
		color = c[int(tasks[i].table.id)]
		plt.plot(tasks[i].get_feature("x").value + margin, tasks[i].get_feature("y").value + margin, '*', markersize=12, color=color)

