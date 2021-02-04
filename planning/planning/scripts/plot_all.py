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
from matplotlib.table import table
import matplotlib.lines as mlines

class Plot_All():
	def __init__(self, POMDPTasks):
		self.POMDPTasks = POMDPTasks
		self.opened_file = False
		# self.simples = [False, True]
		self.simples = [False]
		self.greedys = [False, True]

		self.compare_3T = False
		self.max_num_steps = 20 #self.POMDPTasks.max_steps
		self.total_execs = 30

		if not self.compare_3T:
			# self.algorithms = ["A:Method-2","B:Agent POMDP","C:N-samples-2","D:HPOMDP","E:Greedy"]
			self.algorithms = ["A:Method-2"]
			# self.tables = [2,3,4,5,6,7,8,9,10,11,12]
			self.tables = [10,11,12]
			self.horizons = [2,3,4,5,6]
		else:
			self.tables = [4,5,6,7,8]
			self.horizons = [5,6]
			self.algorithms = ["A:Method-2","B:Agent POMDP","C:N-samples-2","F:Method-3","G:N-samples-3"]

	def get_data(self, execs):
		frames = []
		colors = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5], \
					[0.2, 0.4, 0.7], [0.9, 0.7, 0.32], [1,0.1,0],[0.2, 0.7, 0.7],[0.7, 0.2, 0.5]]
		for table in self.tables:
			for simple in self.simples:
				prev_data_horizon = {}
				for horizon in self.horizons:
					optimal_belief_reward = None
					for alg_label in self.algorithms:
						if alg_label not in prev_data_horizon.keys():
							prev_data_horizon[alg_label] = {}
						if simple:
							model_folder = "simple"
						else:
							model_folder = "complex"
						if alg_label == "E:Greedy":
							model_folder += "_no_op_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid-" + str(False)

						elif alg_label == "A:Method-2":
							model_folder += "_no_op_hybrid_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid-" + str(True)

						elif alg_label == "F:Method-3":
							model_folder += "_no_op_hybrid_3T_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid_3T-" + str(True)

						elif alg_label == "C:N-samples-2":
							model_folder += "_no_op_hybrid_shani_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid-" + str(True) + "_shani-" + str(True)

						elif alg_label == "G:N-samples-3":
							model_folder += "_no_op_hybrid_3T_shani_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid_3T-" + str(True) + "_shani-" + str(True)

						elif alg_label == "D:HPOMDP":
							model_folder += "_H_POMDP_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(False) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
									"_H_POMDP-" + str(True)

						elif alg_label == "B:Agent POMDP":
							model_folder += "_r_model"
							filename = self.POMDPTasks.test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(False) +  '_horizon-' + str(horizon) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed)

						# print (filename)
						# set_trace()
						pkl_filename = filename + '.pkl'
						exists = os.path.isfile(pkl_filename)

						if exists:
							data = cPickle.load(open(pkl_filename,'rb'))
							final_num_steps = np.minimum(np.full((execs),self.max_num_steps),data['final_num_steps'][0:execs])
							data['final_num_steps'] = final_num_steps
							data['final_total_reward'] = np.sum(data['final_total_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							
							if alg_label == "A:Method-2" and optimal_belief_reward == None:
								optimal_belief_reward = data['final_total_belief_reward'][0:execs,0:self.max_num_steps]

							# print ("horizon: ", horizon, " - algorithm: ", alg_label)
							prev_data_horizon[alg_label][horizon] = data['final_total_belief_reward'][0:execs,0:self.max_num_steps]
							if horizon == self.horizons[0]: ##  or horizon == self.horizons[1]
								continue						

							if execs != 1:
								if optimal_belief_reward is None:
									continue
								# data['final_total_belief_reward'] = np.sum(-optimal_belief_reward[0:execs,0:self.max_num_steps]+data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
								# data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
								prev_data_placeholder = data['final_total_belief_reward'][0:execs,0:self.max_num_steps]
								data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps]-prev_data_horizon[alg_label][horizon-1],axis=1)/final_num_steps
								plot_data = prev_data_placeholder

								if table == 11 and horizon == 2:
									plt.figure()

								if table == 11 and horizon != 2:
									print ("table: ", table, " horizon: ", horizon)
									ax = plt.gca()
									for e in range(0,execs):
										print (" exec: ", e)
										if data['final_total_belief_reward'][e] < 0:
											matplt.plot(range(0,self.max_num_steps),prev_data_horizon[alg_label][horizon][e,0:self.max_num_steps],label='h='+str(horizon),color=colors[horizon])
											matplt.plot(range(0,self.max_num_steps),prev_data_horizon[alg_label][horizon-1][e,0:self.max_num_steps],label='h='+str(horizon-1),color=colors[horizon],linestyle='dashed')
											ax.set_xlabel('Steps',fontsize=26)
											ax.set_ylabel('Reward',fontsize=26)
											ax.set_ylabel('Reward',fontsize=26)
											ax.set_title("table: " + str(table) + " - horizon: " + str(horizon) + " - execs=" + str(e))
											matplt.xticks(np.arange(0,self.max_num_steps))
											ax.legend()
											# plt.show(block=False)
											# plt.pause(0.5)
											plt.tight_layout()
											# set_trace()
											dirct = self.POMDPTasks.test_folder + 'results_' + str(execs) + '/' + 'tables_' + str(table) + '/' + 'horizon_' + str(horizon) + '/'
											if not os.path.exists(dirct):
												os.makedirs(dirct)
											matplt.savefig(dirct + str(e) + '_exec' + '.png')
											# set_trace()
											plt.cla()
							else:
								data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps

							data['final_total_max_Q'] = np.sum(data['final_total_max_Q'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							data['final_satisfaction'] = np.sum(data['final_satisfaction'][0:execs,0:self.max_num_steps],axis=1)/np.sum(data['final_total_satisfaction'][0:execs,0:self.max_num_steps],axis=1)
							data.pop('final_total_satisfaction')
							data['final_unsatisfaction'] = np.sum(data['final_unsatisfaction'][0:execs,0:self.max_num_steps],axis=1)/np.sum(data['final_total_unsatisfaction'][0:execs,0:self.max_num_steps],axis=1)
							data.pop('final_total_unsatisfaction')
							data['planning_time'] = np.sum(data['planning_time'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							data['tree_sizes'] = np.sum(data['tree_sizes'][0:execs,0:self.max_num_steps],axis=1)
							data['horizon'] = data['horizon'][0:execs]
							data['greedy'] = data['greedy'][0:execs]
							data['simple'] = data['simple'][0:execs]
							data['tables'] = data['tables'][0:execs]
							data['hybrid'] = data['hybrid'][0:execs]
							data['no_op'] = data['no_op'][0:execs]
							data['shani'] = data['shani'][0:execs]
							data['H_POMDP'] = data['H_POMDP'][0:execs]
							algorithm = np.full((execs,),alg_label,dtype=object)
							# algorithm.fill(alg_label)
							data['algorithm'] = algorithm
							# set_trace()

							df = pd.DataFrame(data=data)
							df['id'] = range(0, df.shape[0])
							frames.append(df) 	

		
		result = None
		if len(frames) > 0:
			result = pd.concat(frames)
		return result

	def plot_statistics(self):		
		# total_execs = 10 #self.POMDPTasks.num_random_executions #20 #[i*100 for i in range(1,11)]
		execs = self.total_execs
		result = self.get_data(execs)	

		result_1 = self.get_data(1)
		ax = None

		res_1_horizon = None
		

		for simple in self.simples:
			simple = int(simple)
			res_simple = result.loc[result['simple'] == simple]
			if result_1 is not None: res_1_simple = result_1.loc[result_1['simple'] == simple];
			if not res_simple.empty:
				# for title in self.horizons:
				# 	title_name = 'horizon'
				# 	title = int(title)
				# 	res_title = res_simple.loc[res_simple[title_name] == title]

				# 	if result_1 is not None and not res_1_simple.empty:
				# 		res_1_title = res_1_simple.loc[res_1_simple[title_name] == title]

				# 	if not res_title.empty:	
				# 		if not self.compare_3T:
				# 			# self.plot_tables_planning_time(res2, tables, simple, greedys, horizon, "tables", "planning_time", "greedy", label_gr, label_le)
				# 			x = "tables"; y = "planning_time"; hue = "algorithm"
				# 			ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)

				# 			x = "tables"; y = "final_total_belief_reward"; hue = "algorithm"
				# 			ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)
				# 			# ax = self.plot_tables_reward_curve(res_horizon, res_1_horizon, simple, horizon, x, y, hue, execs)
							
							
				# 			# self.plot_tables_reward_curve(res2, self.tables, simple, self.greedys, horizon, "tables", "final_total_belief_reward", "greedy", label_gr, label_le)
				# 			# self.plot_tables_planning_time(res2, tables, simple, greedys, horizon, "tables", "tree_sizes", "greedy", label_gr, label_le)

				# 		else:
				# 			x = "tables"; y = "planning_time"; hue = "algorithm"
				# 			ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)

				# 			x = "tables"; y = "final_total_belief_reward"; hue = "algorithm"
				# 			ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)


				for title in self.tables:
					title_name = 'tables'
					title = int(title)
					res_title = res_simple.loc[res_simple[title_name] == title]

					if result_1 is not None and not res_1_simple.empty:
						res_1_title = res_1_simple.loc[res_1_simple[title_name] == title]
					else:
						res_1_title = None

					if not res_title.empty:	
						if not self.compare_3T:
							x = "horizon"; y = "planning_time"; hue = "algorithm"
							ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)

							x = "horizon"; y = "final_total_belief_reward"; hue = "algorithm"
							ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)
						else:
							x = "horizon"; y = "planning_time"; hue = "algorithm"
							ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)

							x = "horizon"; y = "final_total_belief_reward"; hue = "algorithm"
							ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)



	def plot_tables_planning_time_curve(self, res, res_1, simple, title, title_name, x, y, hue, execs):
		print ("----- ", y)
		data = res[['id',x,y,hue]]
		if not self.compare_3T:
			if len(data[x].unique()) < 8:
				fig = plt.figure(figsize=(max(len(data[x].unique()),8)-3,7))
			else:
				fig = plt.figure(figsize=(max(len(data[x].unique()),8)-4,7))
		else:
			fig = plt.figure(figsize=(max(len(data[x].unique()),8),8))
		ax = plt.gca()
		# sns.set_context("paper", rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16})   

		if title == min(self.horizons) or self.compare_3T:
			ax = sns.lineplot(x=x, y=y, hue=hue, style="algorithm", markers=True, data=data)
			plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
			plt.setp(ax.get_legend().get_title(), fontsize='22') # for legend title
		else:
			ax = sns.lineplot(x=x, y=y, hue=hue, style="algorithm", markers=True, data=data,legend=False) #, dashes=False

		l = mlines.Line2D([self.horizons[0],self.horizons[-1]], [0,0], color='red', linestyle="--")
		ax.add_line(l)

		if y == "final_total_belief_reward":
			prev_pos = None
			for horizon in data[x].unique():
				sample = data.loc[data[x] == horizon]
				mean_x = np.mean(sample[y])
				sample_gr = sample.loc[sample[y] > 0]
				sample_le = sample.loc[sample[y] < 0]
				sample_ee = sample.loc[sample[y] == 0]
				str_stat = "ge: " + str(round(len(sample_gr)/len(sample)+len(sample_ee)/len(sample),2))
				str_stat += " l: " + str(round(len(sample_le)/len(sample),2))
				print ("horizon=",horizon,len(sample_gr)/len(sample), len(sample_le)/len(sample), len(sample_ee)/len(sample))
				pos = [horizon,mean_x] 
				# rand = np.random.randint(2) + np.random.randint(2)
				# pos[0] += 0.1*rand
				# pos[1] += 5*rand
				if prev_pos is not None and np.fabs(prev_pos[1]-pos[1]) < 18:
					pos[1] -= 18
				plt.annotate(str_stat, xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]))
				prev_pos = pos


		# if title != 2:
		# ax._legend.remove()
		# plt.annotate('actual group', xy=(x+0.2,y), xytext=(x+0.3, 300), arrowprops=dict(facecolor='black', shrink=0.05, headwidth=20, width=7))
		# tab_names = data[x].unique()
		# alg_names = data[hue].unique()
		# for alg in range(len(alg_names)):
		# 	dat_temp = data.loc[data[hue] == alg_names[alg]]
		# 	mean_x = np.mean(dat_temp.loc[dat_temp[x] == tab_names[len(tab_names)-alg-1]])
		# 	pos = (tab_names[len(tab_names)-alg-1],mean_x[y])
		# 	set_trace()
		# 	plt.annotate(alg_names[alg][0], xy=(pos[0], pos[1]), xytext=(pos[0]+0.5, pos[1]+0.5), arrowprops=dict(facecolor='black', shrink=0.05),)

		if y != "final_total_belief_reward" and res_1 is not None and not res_1.empty:
			data_1 = res_1[[hue,x,y]]
			# table(ax, data_1, loc='upper left')
			# set_trace()
			
			# set_trace()
			tab_names = data_1[x].unique()
			alg_names = data_1[hue].unique()
			table_arr = np.full((len(tab_names)+1,len(alg_names)+1),"",dtype=object)
			# print ("algorithms: ", alg_names)
			# print ("tables: ", tab_names)
			for t in range(len(tab_names)):
				table_arr[t+1,0] = "tables = " + str(tab_names[t])
				for alg in range(len(alg_names)):		
					table_arr[0,alg+1] = alg_names[alg][0]		
					sample = data_1.loc[data_1[hue] == alg_names[alg]]
					sample = sample.loc[sample[x] == tab_names[t]]
					# print(sample)
					if not sample.empty:
						table_arr[t+1,alg+1] = str(np.round(float(sample[y]),1)) + "s"

			if not self.compare_3T or simple == 1:
				if title == 2 or (simple == 1 and title ==5):
					tab = table(ax, cellText=table_arr, cellLoc = 'center', rowLoc = 'center', loc='upper left', bbox=[0.1,.2,.7,.3]) ## left, bottom, width, height
				else:
					if len(data[x].unique()) < 10:
						tab = table(ax, cellText=table_arr, cellLoc = 'center', rowLoc = 'center', loc='upper left', bbox=[0.0,.7,1.0,.3]) ## left, bottom, width, height
					else:
						tab = table(ax, cellText=table_arr, cellLoc = 'center', rowLoc = 'center', loc='upper left', bbox=[0.0,.7,1.0,.3]) ## left, bottom, width, height
			else:
				tab = table(ax, cellText=table_arr, cellLoc = 'center', rowLoc = 'center', loc='upper left', bbox=[0.1,.4,.9,.2]) ## left, bottom, width, height
			
			tab.auto_set_font_size(False)
			tab.set_fontsize(24)
			# tab.scale(5, 5) 
		
		ax.set_title(title_name+": " + str(title),fontsize=26)
		# set_trace()
		matplt.xticks(np.arange(self.tables[0],max(max(data[x].unique())+1,9)))

		if "time" in y:
			ylab = "planning time (s)"
		if "reward" in y:
			ylab = "reward"
		ax.set_xlabel(x,fontsize=26)
		ax.set_ylabel(ylab,fontsize=26)
		ax.tick_params(labelsize=22)

		agent_pomdp_max_table = data.loc[data[hue] == "B:Agent POMDP"]
		ax.get_xticklabels()[len(agent_pomdp_max_table[x].unique())-1].set_color("orange")

		# ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)

		plt.tight_layout()
		# set_trace()
		matplt.savefig(self.POMDPTasks.test_folder + '/results_' + str(execs) + '/' + "3T_" + str(self.compare_3T) + "_" + x + "_" + hue + "_" + y + '_simple-' \
			+ str(simple) + '_'+title_name+'-' + str(title) + "_execs-" + str(execs) + '.png')


	def plot_tables_reward_curve(self, res, res_1, simple, horizon, x, y, hue, execs):
		print ("----- ", y)
		data = res[[x,y,hue]]

		ax = None
		offset = 0.2
		fig, ax = plt.subplots()
		labels = []
		bps = []
		# execs = self.POMDPTasks.num_random_executions
		temp_data = None

		for table in self.tables:	
			fig = plt.figure()	
			for alg in self.algorithms:			
				temp_data = data[(data.algorithm == alg) & (data.tables == table)]		
				execs = temp_data.shape[0]
				# if alg == "greedy":
				# 	cl_border = "black"
				# 	cl_fill = "cyan"
				# 	pos = -2 * offset + tables[0]
				# elif alg == "H_POMDP":
				# 	cl_border = "black"
				# 	cl_fill = "green"
				# 	pos = -1 * offset + tables[0]
				# elif alg == "shani":
				# 	cl_border = "black"
				# 	cl_fill = "purple"
				# 	pos = tables[0]
				# elif alg == "our_approach":
				# 	cl_border = "black"
				# 	cl_fill = "tomato"
				# 	pos = 1 * offset + tables[0]
				# elif alg == "agent_POMDP":
				# 	cl_border = "black"
				# 	cl_fill = "skyblue"
				# 	pos = 2 * offset + tables[0]

				# ol, bp = self.draw_plot(count, ax, temp_data[y], pos, cl_border, cl_fill)
				bp = ax.plot(temp_data[y])
				labels.append(alg)
				
				# bps.append(bp["boxes"][0])
				bps.append(bp)
				set_trace()
			set_trace()	
			plt.close(fig)
		
		set_trace()
		ax.legend(bps[0:len(self.algorithms)], labels[0:len(self.algorithms)], loc='upper right')
		# matplt.xticks(tables[0]+np.arange(len(tables)+2))
		matplt.xticks(range(0,len(temp_data)))
		# print ( 'Simple-' + str(simple) + ', horizon-' + str(horizon) + ", stats outliers size-", outliers_len)
		ax.set_title("horizon: " + str(horizon))
		matplt.xlabel("episode")
		matplt.ylabel(y)
		set_trace()
		matplt.savefig(self.POMDPTasks.test_folder + '/results_' + str(execs) + '/' + x + "_" + hue + "_" + y + '_simple-' + str(simple) + '_horizon-' + str(horizon) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


	


##################################################################################  no_op
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
		execs = execs	
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
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(execs) + '/bar_' + self.POMDPTasks.model_folder + "_" + x + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


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