import _pickle as cPickle
import seaborn as sns
import pandas as pd
import os
from matplotlib.cbook import boxplot_stats
import numpy as np
import pylab as plt
import matplotlib.pyplot as matplt
from tabulate import tabulate
from ipdb import set_trace

total_execs = 30

class Plot():
	def __init__(self, POMDPTasks):
		self.POMDPTasks = POMDPTasks
		self.opened_file = False

	def plot_statistics(self):		
		global total_execs
		frames = []
		tables = [2,3,4,5,6,7,8,9,10,11,12]
		horizons = [2,3,4,5,6]
		simples = [False, True]
		greedys = [False, True]
		# total_execs = 10 #self.POMDPTasks.num_random_executions #20 #[i*100 for i in range(1,11)]
		max_num_steps = 20 #self.POMDPTasks.max_steps
		label_gr = "optimal"
		label_le = "greedy"

		for table in tables:
			for horizon in horizons:
				for simple in simples:
					for greedy in greedys:

						filename = self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
							 + '_greedy-' + str(greedy) +  '_horizon-' + str(horizon) + '_#execs-' + \
							 str(self.POMDPTasks.num_random_executions) + '_seed-' + str(self.POMDPTasks.seed);

						if greedy:
							filename += "_noop-" + str(self.POMDPTasks.no_op)
							if self.POMDPTasks.hybrid_3T:
								filename += "_hybrid_3T-" + str(self.POMDPTasks.greedy_hybrid)
								label_gr = "optimal"
								label_le = "greedy_wo_pairs"
							else:
								filename += "_hybrid-" + str(self.POMDPTasks.greedy_hybrid)
								if self.POMDPTasks.greedy_hybrid:
									label_gr = "optimal"
									label_le = "hybrid"
								if self.POMDPTasks.hierarchical_baseline:
									label_gr = "HPOMDP"
									label_le = "hybrid"

							if self.POMDPTasks.shani_baseline:
								filename += "_shani-" + str(self.POMDPTasks.shani_baseline)
								label_gr = "optimal"
								label_le = "shani"

						elif self.POMDPTasks.hierarchical_baseline:
							label_gr = "HPOMDP"
							label_le = "hybrid"
							filename += "_H_POMDP-" + str(self.POMDPTasks.hierarchical_baseline)

						pkl_filename = filename + '.pkl'
						exists = os.path.isfile(pkl_filename)

						if simple:
							if not exists and total_execs == 30:
								pkl_filename = pkl_filename.replace("#execs-100", "#execs-30")
								exists = os.path.isfile(pkl_filename)
								if exists:
									total_execs = 30
						else:
							if not exists and total_execs == 10:
								pkl_filename = pkl_filename.replace("#execs-30", "#execs-20")
								exists = os.path.isfile(pkl_filename)
								if exists:
									total_execs = 10

							if not exists and total_execs == 10:
								pkl_filename = pkl_filename.replace("#execs-30", "#execs-10")
								exists = os.path.isfile(pkl_filename)
								if exists:
									total_execs = 10


						if exists:
							data = cPickle.load(open(pkl_filename,'rb'))
							final_num_steps = np.minimum(np.full((total_execs),max_num_steps),data['final_num_steps'][0:total_execs])
							data['final_num_steps'] = final_num_steps
							data['final_total_reward'] = np.sum(data['final_total_reward'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							# set_trace()
							data['final_total_max_Q'] = np.sum(data['final_total_max_Q'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							data['final_satisfaction'] = np.sum(data['final_satisfaction'][0:total_execs,0:max_num_steps],axis=1)/np.sum(data['final_total_satisfaction'][0:total_execs,0:max_num_steps],axis=1)
							data.pop('final_total_satisfaction')
							data['final_unsatisfaction'] = np.sum(data['final_unsatisfaction'][0:total_execs,0:max_num_steps],axis=1)/np.sum(data['final_total_unsatisfaction'][0:total_execs,0:max_num_steps],axis=1)
							data.pop('final_total_unsatisfaction')
							data['planning_time'] = np.sum(data['planning_time'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							data['tree_sizes'] = np.sum(data['tree_sizes'][0:total_execs,0:max_num_steps],axis=1)
							data['horizon'] = data['horizon'][0:total_execs]
							data['greedy'] = data['greedy'][0:total_execs]
							data['simple'] = data['simple'][0:total_execs]
							data['tables'] = data['tables'][0:total_execs]
							data['hybrid'] = data['hybrid'][0:total_execs]
							data['no_op'] = data['no_op'][0:total_execs]
							data['shani'] = data['shani'][0:total_execs]
							data['H_POMDP'] = data['H_POMDP'][0:total_execs]

							if 'hybrid_3T' not in data.keys():
								hybrid_arr = np.empty(total_execs,dtype=int)
								hybrid_arr.fill(noop_hybrid)
								data['hybrid_3T'] = hybrid_arr
							elif 'hybrid_3T' in data.keys():
								data['hybrid_3T'] = data['hybrid_3T'][0:total_execs]

							# if 'shani' in data.keys():
							# 	data['shani'] = data['shani'][0:total_execs]
							# else:
							# 	shani = np.empty(total_execs,dtype=int)
							# 	shani.fill(self.POMDPTasks.shani_baseline)
							# 	data['shani'] = shani

							# if 'H_POMDP' in data.keys():
							# 	data['H_POMDP'] = data['H_POMDP'][0:total_execs]
							# else:
							# 	hpomdp = np.empty(total_execs,dtype=int)
							# 	hpomdp.fill(self.POMDPTasks.hierarchical_baseline)
							# 	data['H_POMDP'] = hpomdp

							# print (table, horizon, simple, greedy)
							df = pd.DataFrame(data=data)
							df['id'] = range(0, df.shape[0])
							frames.append(df) 	
						# else: ## hack for now
						# 	filename = self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
						# 	 + '_greedy-' + str(greedy) +  '_horizon-' + str(horizon) + '_#execs-' + \
						# 	 str(1000) + '_seed-' + str(self.POMDPTasks.seed);
						# 	pkl_filename = filename + '.pkl'
						# 	exists = os.path.isfile(pkl_filename)
						# 	if exists:
						# 		df = pd.read_pickle(pkl_filename)
						# 		df['id'] = range(0, df.shape[0])
						# 		frames.append(df[0:self.POMDPTasks.num_random_executions])	


		
		# for ex in execs:
			
		# 	new_frames = []
		# 	for l in range(len(frames)):
		# 		new_frames.append(frames[l][0:ex])

			# result = pd.concat(new_frames)
		result = pd.concat(frames)
		for simple in simples:
			simple = float(simple)
			res = result.loc[result['simple'] == simple]
			if not res.empty:									
				for table in tables:
					table = int(table)
					res2 = res.loc[res['tables'] == table]
					if not res2.empty:	
						print ("******************** table: ", table)					
						self.plot_tables_reward(res2, horizons, simple, greedys, table, "horizon", "final_total_belief_reward", "greedy", True, label_gr, label_le)
						self.plot_tables_reward(res2, horizons, simple, greedys, table, "horizon", "planning_time", "greedy", False, label_gr, label_le)
						print ("********************")								
						# print ("************************************************ table: ", table)					
						# self.plot_horizons_reward(res2, horizons, simple, greedys, table, "greedy", "final_total_belief_reward", "horizon", label_gr, label_le)
						# self.plot_horizons_reward(res2, horizons, simple, greedys, table, "greedy", "planning_time", "horizon", label_gr, label_le)
						# print ("************************************************")	
						

		for simple in simples:
			simple = float(simple)
			res = result.loc[result['simple'] == simple]
			if not res.empty:
				for horizon in horizons:
					horizon = int(horizon)
					res2 = res.loc[res['horizon'] == horizon]
					if not res2.empty:	
						self.plot_tables_planning_time(res2, tables, simple, greedys, horizon, "tables", "planning_time", "greedy", label_gr, label_le)
						# self.plot_tables_planning_time_curve(res2, tables, simple, greedys, horizon, "tables", "planning_time", "greedy", label_gr, label_le)
						
						# self.plot_tables_planning_time(res2, tables, simple, greedys, horizon, "tables", "final_total_belief_reward", "greedy", label_gr, label_le)
						# self.plot_tables_planning_time(res2, tables, simple, greedys, horizon, "tables", "tree_sizes", "greedy", label_gr, label_le)

	def plot_tables_planning_time_curve(self, res2, tables, simple, greedys, horizon, x, y, hue, label_gr, label_le):
		global total_execs
		execs = total_execs
		print ("----- ", y)
		data = res2[['id',x,y,hue]]

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions
		temp_data = None
		for table in tables:
			for gre in greedys:		
				temp_data = data[(data.greedy == gre) & (data.tables == table)]		
				execs = temp_data.shape[0]
				if gre:
					# ol, bp = self.draw_plot(count, ax, temp_data[y], -offset+tables[0], "black", "tomato")
					# outliers_len += ol
					bp = ax.plot(temp_data[y], 'r-')
					labels.append(label_le)
				else:
					# ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+tables[0], "black", "skyblue")
					# outliers_len += ol
					bp = ax.plot(temp_data[y], 'b--')
					labels.append(label_gr)
				# bps.append(bp["boxes"][0])
				bps.append(bp)
 
			count += 1
			# set_trace()0
		
		ax.legend(bps[0:2], labels[0:2], loc='upper right')
		matplt.xticks(np.arange(temp_data.shape[0]))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("horizon: " + str(horizon))
		matplt.xlabel(x)
		matplt.ylabel(y)
		set_trace()
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_" + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_horizon-' + str(horizon) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')



	def plot_tables_planning_time(self, res2, tables, simple, greedys, horizon, x, y, hue, label_gr, label_le):
		global total_execs
		execs = total_execs
		print ("----- ", y)
		data = res2[['id',x,y,hue]]

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions

		for table in tables:
			for gre in greedys:		
				temp_data = data[(data.greedy == gre) & (data.tables == table)]		
				execs = temp_data.shape[0]
				if gre:
					ol, bp = self.draw_plot(count, ax, temp_data[y], -offset+tables[0], "black", "tomato")
					outliers_len += ol
					labels.append(label_le)
				else:
					ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+tables[0], "black", "skyblue")
					outliers_len += ol
					labels.append(label_gr)
				bps.append(bp["boxes"][0])

			count += 1

		
		ax.legend(bps[0:2], labels[0:2], loc='upper right')
		matplt.xticks(tables[0]+np.arange(count+2))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("horizon: " + str(horizon))
		matplt.xlabel(x)
		matplt.ylabel(y)
		# set_trace()
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_" + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_horizon-' + str(horizon) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


	def plot_tables_reward(self, res2, horizons, simple, greedys, table, x, y, hue, greater_than_plot=False, label_gr="optimal", label_le="greedy"):
		print ("----- ", y)
		global total_execs
		data = res2[['id',x,y,hue]]

		if self.opened_file:
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','a+')
		else:
			self.opened_file = True
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','w')

		print ("************************************************ table: ", table, file=f)
		print ("------------------------------ ", y, file=f)

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions
		table_h_gr = np.zeros(horizons[len(horizons)-1]+1)
		table_h_eq = np.zeros(horizons[len(horizons)-1]+1)

		for horizon in horizons:
			for gre in greedys:		
				temp_data = data[(data.greedy == gre) & (data.horizon == horizon)]		
				execs = temp_data.shape[0]
				if gre:
					ol, bp = self.draw_plot(count, ax, temp_data[y], -offset+horizons[0], "black", "tomato")
					outliers_len += ol
				else:
					ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+horizons[0], "black", "skyblue")
					outliers_len += ol
				labels.append("greedy = " + str(gre))
				bps.append(bp["boxes"][0])

			temp_data_F = data[(data.greedy == False) & (data.horizon == horizon)]
			temp_data_T = data[(data.greedy == True) & (data.horizon == horizon)]
			# print ("***********  Table = ", table)
			# if horizon == 3 and table == 2 and y == "final_total_belief_reward":
			# 	set_trace()
			if len(temp_data_T) != 0 and len(temp_data_F) != 0:	
				# set_trace()
				pos = [count+horizons[0]+offset/2,temp_data_T[y].mean()] 
				# print (str(float(sum(temp_data_F[y]>=temp_data_T[y]))/self.num_random_executions))
				# set_trace()
				# if float(sum(temp_data_F[y]>=temp_data_T[y])) < 100:
				# 	set_trace()

				gr_eq = np.round(temp_data_F[y],2)>=np.round(temp_data_T[y],2)
				eq = np.round(temp_data_F[y],2)==np.round(temp_data_T[y],2)

				great = np.round(temp_data_F[y],2)>np.round(temp_data_T[y],2)
				less = np.round(temp_data_F[y],2)<np.round(temp_data_T[y],2)


				percent_gr_eq = float(sum(gr_eq))/execs
				percent_eq = float(sum(eq))/execs

				table_h_gr[horizon] = percent_gr_eq - percent_eq
				table_h_eq[horizon] = percent_eq
				# print ("F >= T %=", percent)
				# print ("F == T %=", float(sum(eq))/execs)

				# print ("mean F >:", np.mean(temp_data_F[y][great]), " std: ", np.std(temp_data_F[y][great]))
				# print ("        :", np.mean(temp_data_T[y][great]), " std: ", np.std(temp_data_T[y][great]))
				# print ("mean F <:", np.mean(temp_data_T[y][less]), " std: ", np.std(temp_data_T[y][less]))
				# print ("        :", np.mean(temp_data_F[y][less]), " std: ", np.std(temp_data_F[y][less]))
				ax.text(pos[0], pos[1], '% = '+str(percent_gr_eq), horizontalalignment='center', size='small', color='b', weight='semibold') 
			count += 1

		for h in horizons:		
			print ("F >= T %=", " horizon=", str(h), ": ", str(np.round(table_h_gr[h],2))+ "+" + str(np.round(table_h_eq[h],2)))
			print ("F >= T %=", " horizon=", str(h), ": ", str(np.round(table_h_gr[h],2))+ "+" + str(np.round(table_h_eq[h],2)), file=f) 

		# set_trace()
		f.close()
		ax.legend(bps[0:2], labels[0:2], loc='upper right')
		matplt.xticks(horizons[0]+np.arange(count+2))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("table: " + str(table) + ", num of outliers: " + str(outliers_len))
		matplt.xlabel(x)
		matplt.ylabel(y)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_" + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')

		if greater_than_plot:
			self.plot_bars(table, simple, horizons, x, y, table_h_gr, table_h_eq, gr=label_gr, le=label_le)

	def plot_horizons_reward(self, res2, horizons, simple, greedys, table, x, y, hue, label_gr, label_le):
		global total_execs
		data = res2[['id',x,y,hue]]
		print ("----- ", y)
		if self.opened_file:
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','a+')
		else:
			self.opened_file = True
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','w')
		
		print ("************************************************ table: ", table, file=f)
		print ("------------------------------ ", y, file=f)

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.3
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions
		steps = 2
		table_h_gr = np.zeros((2,horizons[len(horizons)-1]+1,horizons[len(horizons)-1]+1))
		table_h_eq = np.zeros((2,horizons[len(horizons)-1]+1,horizons[len(horizons)-1]+1))
		for gre in greedys:	
			for h in horizons:		
				# print ("greedy: ", gre, " horizon: ", h)		
				temp_data = data[(data.greedy == gre) & (data.horizon == h)]		
				execs = temp_data.shape[0]
				if h==2:
					color = "tomato"
				elif h==3:
					color = "pink"
				elif h==4:
					color = "gray"
				else:
					color = "orange"

				loc = (h-horizons[0])*offset+horizons[0]
				ol, bp = self.draw_plot(count*steps, ax, temp_data[y], loc, "black", color)
				outliers_len += ol
				print ("greedy=", gre,", h=", h, " mean=", np.mean(temp_data[y]), " std=", np.std(temp_data[y]), file=f)
			
				temp_data_H = data[(data.greedy == gre) & (data.horizon == h)]
				for h_p in range(h+1,horizons[len(horizons)-1]+1):					
					temp_data_H_1 = data[(data.greedy == gre) & (data.horizon == h_p)]
					if len(temp_data_H) != 0 and len(temp_data_H_1) != 0:	
						# set_trace()
						pos = [count*steps+loc+offset/2+(h_p-h-1)*offset,temp_data_H[y].mean()] 
						# print (str(float(sum(temp_data_F[y]>=temp_data_T[y]))/self.num_random_executions))
						# set_trace()
						# if float(sum(temp_data_F[y]>=temp_data_T[y])) < 100:
						# 	set_trace()
						gr_eq = np.round(temp_data_H_1[y],2) >= np.round(temp_data_H[y],2)
						eq = np.round(temp_data_H_1[y],2) == np.round(temp_data_H[y],2)
						great = np.round(temp_data_H_1[y],2) > np.round(temp_data_H[y],2)
						less = np.round(temp_data_H_1[y],2) < np.round(temp_data_H[y],2)
						percent_gr_eq = float(sum(gr_eq))/execs
						percent_eq = float(sum(eq))/execs
						# print ("h: ", h_p, ">=", h, " % = ", percent_gr_eq - percent_eq, " + ",percent_eq)
						table_h_gr[int(gre),h_p,h] = percent_gr_eq - percent_eq
						table_h_eq[int(gre),h_p,h] = percent_eq
						# print ("mean h\' >: ", np.mean(temp_data_H_1[y][great]), " std: ", np.std(temp_data_H_1[y][great]))
						# print ("          : ", np.mean(temp_data_H[y][great]), " std: ", np.std(temp_data_H[y][great]))
						# print ("mean h > : ", np.mean(temp_data_H[y][less]), " std: ", np.std(temp_data_H[y][less]))
						# print ("         : ", np.mean(temp_data_H_1[y][less]), " std: ", np.std(temp_data_H_1[y][less]))
						# if h_p == 6 and h==5:
						# 	print (temp_data_H_1)
						# 	print (temp_data_H)
						# 	set_trace()
						ax.text(pos[0], pos[1], '% = '+str(percent_gr_eq), horizontalalignment='center', size='small', color='b', weight='semibold') 
				
				labels.append("h = " + str(h))
				bps.append(bp["boxes"][0])

			count += 1

		# set_trace()
		
		for gre in greedys:	
			print ("greedy: ",gre)
			print ("greedy: ",gre, file=f)
			headers = []
			table_str = []
			for h_p in horizons:	
				headers.append(str(h_p))
				t_str = [str(h_p)]
				for h in horizons:	
					per = ""	
					if h_p > h:
						per = str(np.round(table_h_gr[int(gre),h_p,h],2)) + "+" + str(np.round(table_h_eq[int(gre),h_p,h],2))
						
						# print ("h: "+str(h_p)+">="+str(h)+" % = "+per)
					t_str.append(per)
				table_str.append(t_str)
			pretty_table = tabulate(table_str, headers=headers,tablefmt="rst")
			print (pretty_table)
			print(pretty_table,file=f)
			print ("--- \n")
			print ("--- \n", file=f)
		f.close()
		ax.legend(bps[0:len(horizons)], labels[0:len(horizons)], loc='upper right')
		matplt.xticks(range(2,2*len(greedys)+1,steps),greedys)
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("tables: " + str(table) + ", num of outliers: " + str(outliers_len))
		matplt.xlabel(x)
		matplt.ylabel(y)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_"  + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


##################################################################################  no_op

	def plot_statistics_no_op(self, shani=None, hybrid=None, hybrid_3T=None):	
		global total_execs
		# assert(noop != hybrid)	
		frames = []
		tables = [2,3,4,5,6,7,8,9,10,11,12]
		horizons = [2,3,4,5,6]
		simples = [self.POMDPTasks.simple]
		noops_or_hybrids = [False, True]
		# total_execs = 30 #self.POMDPTasks.num_random_executions #[i*100 for i in range(1,11)]
		max_num_steps = 20 #self.POMDPTasks.max_steps
		greedy = True
		label = ""

		for table in tables:
			for horizon in horizons:
				for simple in simples:
					for noop_hybrid in noops_or_hybrids:
						filename = self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
							 + '_greedy-' + str(greedy) +  '_horizon-' + str(horizon) + '_#execs-' + \
							 str(self.POMDPTasks.num_random_executions) + '_seed-' + str(self.POMDPTasks.seed);

						if shani:
							filename += "_noop-" + str(self.POMDPTasks.no_op)
							if self.POMDPTasks.hybrid_3T:
								filename += "_hybrid_3T-" + str(True)
							else:
								filename += "_hybrid-" + str(True)
							if noop_hybrid:
								filename += "_shani-" + str(True)
							label = "shani"
						elif hybrid:
							# set_trace()
							filename += "_noop-" + str(self.POMDPTasks.no_op)
							filename += "_hybrid-" + str(noop_hybrid)
							label = "hybrid"
						elif hybrid_3T:
							filename += "_noop-" + str(self.POMDPTasks.no_op)
							if noop_hybrid:
								filename += "_hybrid_3T-" + str(True)
							else:
								filename += "_hybrid-" + str(True)
							label = "hybrid_3T"

						pkl_filename = filename + '.pkl'
						exists = os.path.isfile(pkl_filename)

						if not exists and total_execs == 30:
							pkl_filename = pkl_filename.replace("#execs-100", "#execs-30")
							exists = os.path.isfile(pkl_filename)
							if exists:
								total_execs = 30

						if not exists and total_execs == 10:
							pkl_filename = pkl_filename.replace("#execs-30", "#execs-10")
							exists = os.path.isfile(pkl_filename)
							if exists:
								total_execs = 10


						if exists:
							data = cPickle.load(open(pkl_filename,'rb'))
							final_num_steps = np.minimum(np.full((total_execs),max_num_steps),data['final_num_steps'][0:total_execs])
							data['final_num_steps'] = final_num_steps
							data['final_total_reward'] = np.sum(data['final_total_reward'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							# set_trace()
							data['final_total_max_Q'] = np.sum(data['final_total_max_Q'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							data['final_satisfaction'] = np.sum(data['final_satisfaction'][0:total_execs,0:max_num_steps],axis=1)/np.sum(data['final_total_satisfaction'][0:total_execs,0:max_num_steps],axis=1)
							data.pop('final_total_satisfaction')
							data['final_unsatisfaction'] = np.sum(data['final_unsatisfaction'][0:total_execs,0:max_num_steps],axis=1)/np.sum(data['final_total_unsatisfaction'][0:total_execs,0:max_num_steps],axis=1)
							data.pop('final_total_unsatisfaction')
							data['planning_time'] = np.sum(data['planning_time'][0:total_execs,0:max_num_steps],axis=1)/final_num_steps
							data['tree_sizes'] = np.sum(data['tree_sizes'][0:total_execs,0:max_num_steps],axis=1)
							data['horizon'] = data['horizon'][0:total_execs]
							data['greedy'] = data['greedy'][0:total_execs]
							data['simple'] = data['simple'][0:total_execs]
							data['tables'] = data['tables'][0:total_execs]
							data['hybrid'] = data['hybrid'][0:total_execs]
							data['no_op'] = data['no_op'][0:total_execs]
							data['shani'] = data['shani'][0:total_execs]
							data['H_POMDP'] = data['H_POMDP'][0:total_execs]
							data['hybrid'] = data['hybrid'][0:total_execs]
							# data['hybrid_3T'] = data['hybrid_3T'][0:total_execs]

							

							if 'hybrid_3T' not in data.keys():
								hybrid_arr = np.empty(total_execs,dtype=int)
								hybrid_arr.fill(noop_hybrid)
								data['hybrid_3T'] = hybrid_arr
							elif 'hybrid_3T' in data.keys():
								data['hybrid_3T'] = data['hybrid_3T'][0:total_execs]

							df = pd.DataFrame(data=data)
							df['id'] = range(0, df.shape[0])
							frames.append(df) 	
							# set_trace()
		# set_trace()
		result = pd.concat(frames)
		for simple in simples:
			simple = float(simple)
			res = result.loc[result['simple'] == simple]
			if not res.empty:									
				for table in tables:
					table = int(table)
					res2 = res.loc[res['tables'] == table]
					if not res2.empty:	
						print ("******************** table: ", table)					
						self.plot_tables_reward_no_op(res2, horizons, simple, noops_or_hybrids, table, "horizon", "final_total_belief_reward", label, label, True)
						self.plot_tables_reward_no_op(res2, horizons, simple, noops_or_hybrids, table, "horizon", "planning_time", label, label, False)
						print ("********************")								
						# print ("************************************************ table: ", table)					
						# self.plot_horizons_reward_no_op(res2, horizons, simple, noops_or_hybrids, table, label, "final_total_belief_reward", "horizon", label)
						# self.plot_horizons_reward_no_op(res2, horizons, simple, noops_or_hybrids, table, label, "planning_time", "horizon", label)
						# print ("************************************************")	
						

		for simple in simples:
			simple = float(simple)
			res = result.loc[result['simple'] == simple]
			if not res.empty:
				for horizon in horizons:
					horizon = int(horizon)
					res2 = res.loc[res['horizon'] == horizon]
					if not res2.empty:	
						self.plot_tables_planning_time_no_op(res2, tables, simple, noops_or_hybrids, horizon, "tables", "planning_time", label, label)
						

	def plot_tables_planning_time_no_op(self, res2, tables, simple, no_ops, horizon, x, y, hue, label):
		global total_execs
		execs = total_execs
		print ("----- ", y)
		data = res2[['id',x,y,hue]]

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions

		for table in tables:
			for no in no_ops:		
				temp_data = data[(data[label] == no) & (data.tables == table)]		
				execs = temp_data.shape[0]
				if no:
					ol, bp = self.draw_plot(count, ax, temp_data[y], -offset+tables[0], "black", "tomato")
					outliers_len += ol
				else:
					ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+tables[0], "black", "skyblue")
					outliers_len += ol
				labels.append(label +" = " + str(no))
				bps.append(bp["boxes"][0])

			count += 1

		# set_trace()
		ax.legend(bps[0:2], labels[0:2], loc='upper right')
		matplt.xticks(tables[0]+np.arange(count+2))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("horizon: " + str(horizon))
		matplt.xlabel(x)
		matplt.ylabel(y)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_" + label + "_" + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_horizon-' + str(horizon) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


	def plot_tables_reward_no_op(self, res2, horizons, simple, no_ops, table, x, y, hue, label, greater_than_plot=False):
		global total_execs
		execs = total_execs
		print ("----- ", y)
		data = res2[['id',x,y,hue]]

		if self.opened_file:
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','a+')
		else:
			self.opened_file = True
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','w')

		print ("************************************************ table: ", table, file=f)
		print ("------------------------------ ", y, file=f)

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions
		table_h_gr = np.zeros(horizons[len(horizons)-1]+1)
		table_h_eq = np.zeros(horizons[len(horizons)-1]+1)

		for horizon in horizons:
			for no in no_ops:		
				temp_data = data[(data[label] == no) & (data.horizon == horizon)]		
				execs = temp_data.shape[0]
				if no:
					ol, bp = self.draw_plot(count, ax, temp_data[y], -offset+horizons[0], "black", "tomato")
					outliers_len += ol
				else:
					ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+horizons[0], "black", "skyblue")
					outliers_len += ol
				labels.append(label + " = " + str(no))
				bps.append(bp["boxes"][0])

			temp_data_F = data[(data[label] == False) & (data.horizon == horizon)]
			temp_data_T = data[(data[label] == True) & (data.horizon == horizon)]
			# print ("***********  Table = ", table)
			if len(temp_data_T) != 0 and len(temp_data_F) != 0:	
				# set_trace()
				pos = [count+horizons[0]+offset/2,temp_data_T[y].mean()] 

				gr_eq = np.round(temp_data_F[y],2)>=np.round(temp_data_T[y],2)
				eq = np.round(temp_data_F[y],2)==np.round(temp_data_T[y],2)

				great = np.round(temp_data_F[y],2)>np.round(temp_data_T[y],2)
				less = np.round(temp_data_F[y],2)<np.round(temp_data_T[y],2)


				percent_gr_eq = float(sum(gr_eq))/execs
				percent_eq = float(sum(eq))/execs

				table_h_gr[horizon] = percent_gr_eq - percent_eq
				table_h_eq[horizon] = percent_eq

				ax.text(pos[0], pos[1], '% = '+str(percent_gr_eq), horizontalalignment='center', size='small', color='b', weight='semibold') 
			count += 1

		for h in horizons:		
			print ("F >= T %=", " horizon=", str(h), ": ", str(np.round(table_h_gr[h],2))+ "+" + str(np.round(table_h_eq[h],2))) 
			print ("F >= T %=", " horizon=", str(h), ": ", str(np.round(table_h_gr[h],2))+ "+" + str(np.round(table_h_eq[h],2)), file=f) 

		# set_trace()
		f.close()
		ax.legend(bps[0:2], labels[0:2], loc='upper right')
		matplt.xticks(horizons[0]+np.arange(count+2))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("table: " + str(table) + ", num of outliers: " + str(outliers_len))
		matplt.xlabel(x)
		matplt.ylabel(y)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_" + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')

		if greater_than_plot:
			self.plot_bars(table, simple, horizons, x, y, table_h_gr, table_h_eq, label+"=F", label+"=T")

	def plot_horizons_reward_no_op(self, res2, horizons, simple, no_ops, table, x, y, hue, label):
		global total_execs
		execs = total_execs
		data = res2[['id',x,y,hue]]
		print ("----- ", y)
		if self.opened_file:
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','a+')
		else:
			self.opened_file = True
			f = open(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + '.txt','w')
		
			
		print ("************************************************ table: ", table, file=f)
		print ("------------------------------ ", y, file=f)

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.3
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions
		steps = 2
		table_h_gr = np.zeros((2,horizons[len(horizons)-1]+1,horizons[len(horizons)-1]+1))
		table_h_eq = np.zeros((2,horizons[len(horizons)-1]+1,horizons[len(horizons)-1]+1))
		for no in no_ops:	
			for h in horizons:		
				# print ("greedy: ", gre, " horizon: ", h)		
				temp_data = data[(data[label] == no) & (data.horizon == h)]		
				execs = temp_data.shape[0]
				if h==2:
					color = "tomato"
				elif h==3:
					color = "pink"
				elif h==4:
					color = "gray"
				else:
					color = "orange"

				loc = (h-horizons[0])*offset+horizons[0]
				ol, bp = self.draw_plot(count*steps, ax, temp_data[y], loc, "black", color)
				outliers_len += ol
				print (label, "=", no,", h=", h, " mean=", np.mean(temp_data[y]), " std=", np.std(temp_data[y]), file=f)
			
				temp_data_H = data[(data[label] == no) & (data.horizon == h)]
				for h_p in range(h+1,horizons[len(horizons)-1]+1):					
					temp_data_H_1 = data[(data[label] == no) & (data.horizon == h_p)]
					if len(temp_data_H) != 0 and len(temp_data_H_1) != 0:	
						pos = [count*steps+loc+offset/2+(h_p-h-1)*offset,temp_data_H[y].mean()] 

						gr_eq = np.round(temp_data_H_1[y],2) >= np.round(temp_data_H[y],2)
						eq = np.round(temp_data_H_1[y],2) == np.round(temp_data_H[y],2)
						great = np.round(temp_data_H_1[y],2) > np.round(temp_data_H[y],2)
						less = np.round(temp_data_H_1[y],2) < np.round(temp_data_H[y],2)
						percent_gr_eq = float(sum(gr_eq))/execs
						percent_eq = float(sum(eq))/execs
						# print ("h: ", h_p, ">=", h, " % = ", percent_gr_eq - percent_eq, " + ",percent_eq)
						table_h_gr[int(no),h_p,h] = percent_gr_eq - percent_eq
						table_h_eq[int(no),h_p,h] = percent_eq

						ax.text(pos[0], pos[1], '% = '+str(percent_gr_eq), horizontalalignment='center', size='small', color='b', weight='semibold') 
				
				labels.append("h = " + str(h))
				bps.append(bp["boxes"][0])

			count += 1

		# set_trace()
		
		for no in no_ops:	
			print (label, ": ",no)
			print (label, ": ",no, file=f)
			headers = []
			table_str = []
			for h_p in horizons:	
				headers.append(str(h_p))
				t_str = [str(h_p)]
				for h in horizons:	
					per = ""	
					if h_p > h:
						per = str(np.round(table_h_gr[int(no),h_p,h],2)) + "+" + str(np.round(table_h_eq[int(no),h_p,h],2))
						
						# print ("h: "+str(h_p)+">="+str(h)+" % = "+per)
					t_str.append(per)
				table_str.append(t_str)
			pretty_table = tabulate(table_str, headers=headers,tablefmt="rst")
			print (pretty_table)
			print(pretty_table,file=f)
			print ("--- \n")
			print ("--- \n", file=f)
		f.close()
		ax.legend(bps[0:len(horizons)], labels[0:len(horizons)], loc='upper right')
		matplt.xticks(range(2,2*len(no_ops)+1,steps),no_ops)
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("tables: " + str(table) + ", num of outliers: " + str(outliers_len))
		matplt.xlabel(x)
		matplt.ylabel(y)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/' + self.POMDPTasks.model_folder + "_" + label + "_"  + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')

##################################################################################  no_op

	def plot_2_vars (self, res2, tables, simple, greedy, greedys, horizon, x, y):
		data = res2[[x,y,'tables','greedy']]

		count = 0
		outliers_len = 0
		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		for table in tables:
			for gre in greedys:		
				temp_data = data[(data.greedy == greedy) & (data.tables == table)]	
				if gre:
					ol, bp = self.draw_plot(count, ax, temp_data[x], -offset+tables[0], "black", "tomato")
					outliers_len += ol
					labels.append("greedy = " + str(gre))
					bps.append(bp["boxes"][0])

					ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+tables[0], "black", "skyblue")
					outliers_len += ol
					labels.append("greedy = " + str(not gre))
					bps.append(bp["boxes"][0])
				else:
					ol, bp = self.draw_plot(count, ax, temp_data[y], +offset+tables[0], "black", "skyblue")
					outliers_len += ol
					labels.append("greedy = " + str(gre))
					bps.append(bp["boxes"][0])

					ol, bp = self.draw_plot(count, ax, temp_data[x], -offset+tables[0], "black", "tomato")
					outliers_len += ol
					labels.append("greedy = " + str(not gre))
					bps.append(bp["boxes"][0])

			count += 1
		

		ax.legend(bps[0:2], labels[0:2], loc='upper right')
		matplt.xticks(tables[0]+np.arange(count+2))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("num of outliers: " + str(outliers_len))
		matplt.xlabel(x)
		matplt.ylabel(y)

		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/z_' + x + "_vs_" + y + '_simple-' + str(simple) + '_horizon-' + str(horizon) + '_greedy-' + str(greedy) + '.png')
		matplt.clf()

	def draw_bar(self, num, ax, gr, eq, offset, edge_color, gr_label, le_label):
		pos = [num+offset] 		
		bp = ax.bar(pos, gr, width=0.3, color="blue", bottom=eq)		
		bp2 = ax.bar(pos, eq, width=0.3, color="red")

		# ax.text(pos[0], median + 100, 'o = '+str(outliers_len), horizontalalignment='center', size='small', color='b', weight='semibold') 
		ax.legend([bp,bp2],[gr_label+" greater than "+le_label,gr_label+" equal to "+le_label],loc='upper right')

	def plot_bars(self, table, simple, horizons, x, y, table_h_gr, table_h_eq, gr, le):	
		global total_execs
		execs = total_execs	
		# execs = self.POMDPTasks.num_random_executions
		count = 0
		fig, ax = plt.subplots()
		for h in horizons:
			self.draw_bar(int(count), ax, table_h_gr[h], table_h_eq[h], int(horizons[0]), "black", gr, le)
			count += 1

		ylabel = gr + " > " + le
		ax.set_title("tables: " + str(table) + "___" + ylabel)
		matplt.xticks(horizons)
		matplt.xlabel("horizons")
		matplt.ylim(0,1)
		matplt.ylabel(y + " ... " + ylabel)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(total_execs) + '/bar_' + self.POMDPTasks.model_folder + "_" + x + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


	def draw_plot(self, num, ax, data, offset,edge_color, fill_color, show_outliers=False):
		pos = [num+offset]
		meanlineprops = dict(linestyle='-', linewidth=2.5, color='purple') 
		if show_outliers:
			bp = ax.boxplot(data, positions= pos, notch=True, sym=".", meanline=True, showmeans=True, meanprops=meanlineprops, widths=0.3, patch_artist=True, manage_xticks=False, showfliers=True)
			outliers = [flier.get_ydata() for flier in bp["fliers"]]
			outliers_len = len(outliers[0])
		else:
			bp = ax.boxplot(data, positions= pos, notch=True, sym=".", meanline=True, showmeans=True, meanprops=meanlineprops, widths=0.3, patch_artist=True, manage_xticks=False, showfliers=False)
			outliers_len = 0

		median = data.median()	

		if outliers_len != 0:
			ax.text(pos[0], median + 100, 'o = '+str(outliers_len), horizontalalignment='center', size='small', color='b', weight='semibold') 

		for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
				plt.setp(bp[element], color=edge_color)
		for patch in bp['boxes']:
			patch.set(facecolor=fill_color)

		return outliers_len, bp