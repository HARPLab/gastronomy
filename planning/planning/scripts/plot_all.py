import _pickle as cPickle
import seaborn as sns
sns.set_palette("colorblind") # ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
sns.set(font_scale=1,style="ticks", rc={'font.weight':800,'text.usetex' : True,'legend.framealpha' : 0.2,'legend.edgecolor':'gray'})
# sns.set_style("ticks")
# rc('font.weight', 'bold')
# rc('text', usetex=True)
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage[dvipsnames]{xcolor}']
# matplotlib.rc('text', usetex=True) 
# matplotlib.use('pgf')
# matplotlib.rc('pgf', texsystem='pdflatex') 
# matplotlib.rcParams['font.weight']= 'bold'
from matplotlib.cbook import boxplot_stats
import numpy as np
import pylab as plt
# plt.rc('text', usetex=True) 
import matplotlib.pyplot as matplt
# matplt.rc('text', usetex=True) 
from tabulate import tabulate
from ipdb import set_trace
from matplotlib.table import table
import matplotlib.lines as mlines

# filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

class Plot_All():
	def __init__(self, POMDPTasks):
		self.POMDPTasks = POMDPTasks
		self.opened_file = False
		# self.simples = [False, True]
		self.simples = [self.POMDPTasks.simple]
		self.greedys = [False, True]

		self.palette = {}
		self.compare_3T = self.POMDPTasks.hybrid_3T
		self.max_num_steps = 20 #self.POMDPTasks.max_steps
		self.total_execs = 10

		if not self.compare_3T:
			# self.algorithms = ["A:Method-2","B:Agent POMDP","C:N-samples-2","D:HPOMDP","E:Greedy"]
			self.algorithms = ["A:Method-2"]
			# self.tables = [2,3,4,5,6]
			# self.tables = [2,3,4,5,6,7,8,9,10,11,12]
			self.tables = [3,4,5,6,7,8,9,10,11,12]
			# self.tables = [3,5,7,9,11]
			# self.tables = [10,11,12]
			if self.POMDPTasks.package:
				self.horizons = [3,4,5,6,7,8]
				self.max_horizons = [3,4,5,6,7,8]
				self.tables = [2,3,4,5,6,7,8]
			else:
				self.horizons = [3,4,5,6,7,8]
				self.max_horizons = [3,4,5,6,7,8]
				self.tables = [3,4,5,6,7,8,9,10,11,12]
			# self.max_horizons = [5,6,7]
			# self.test_folders = ["../tests_fixedH/","../tests_beliefLib/"]
			# self.test_folders = ["../tests_fixedH/","../tests_beliefLib/","../tests_nocash/"]
			# self.test_folders = ["../tests_fixedH_3T/","../tests_nocash_3T/","../tests_beliefLib_3T/", \
			# "../tests_nocash_mix/","../tests_belief_mix/"]

			# self.test_folders = ["../cluster/tests_fixedH/","../cluster/tests_beliefLib/","../cluster/tests_nocash/"]

			# self.test_folders = ["../cluster/tests_fixedH/","../cluster/tests_beliefLib/","../cluster/tests_nocash/", \
			# "../cluster/tests_nocash_mix_nobug/","../cluster/tests_belief_mix_nobug/"]

			# self.test_folders = ["../cluster/tests_fixedH_3T/","../cluster/tests_nocash_3T/","../cluster/tests_beliefLib_3T/"]

			# self.test_folders = ["../cluster/tests_fixedH_4T/","../cluster/tests_nocash_4T/","../cluster/tests_beliefLib_4T/"]

			# self.test_folders = ["../cluster/tests_fixedH_3T/","../cluster/tests_nocash_3T/","../cluster/tests_beliefLib_3T/", \
			# "../cluster/tests_fixedH_4T/","../cluster/tests_nocash_4T/","../cluster/tests_beliefLib_4T/",
			# "../cluster/tests_nocash_mix_nobug/","../cluster/tests_belief_mix_nobug/"]

			# self.test_folders = ["../cluster/tests_fixedH_4T/","../cluster/tests_nocash_4T/","../cluster/tests_beliefLib_4T/", \
			# "../cluster/tests_nocash_mix_nobug/","../cluster/tests_belief_mix_nobug/"]

			# self.test_folders = ["../tests_agent_fixed/","../tests_agent_nocash/","../tests_agent_belief/",\
			# "../tests_fixedH/","../tests_nocash/","../tests_beliefLib/",\
			# "../tests_fixedH_3T/",\
			# "../tests_fixedH_4T/", \
			# "../tests_nocash_mix_nobug/","../tests_belief_mix_nobug/","../tests_shani_fixed/",\
			# "../tests_hpomdp_fixed/"]
			# self.avg = ""

			# self.test_folders = ["../cluster/tests_agent_fixed/","../cluster/tests_agent_nocash/","../cluster/tests_agent_belief/",\
			# "../cluster/tests_fixedH/","../cluster/tests_nocash/","../cluster/tests_beliefLib/",\
			# "../cluster/tests_fixedH_3T/",\
			# "../cluster/tests_fixedH_4T/", \
			# "../cluster/tests_nocash_mix_nobug/","../cluster/tests_belief_mix_nobug/","../cluster/tests_shani_fixed/",\
			# "../cluster/tests_hpomdp_fixed/"]
			# self.avg = ""

			self.test_folders = ["../cluster/tests_belief_mix_nobug/", \
			"../cluster/tests_belief_mix_nobug/",  \
			"../cluster/tests_nocash_mix_nobug/",\
			"../cluster/tests_fixedH/",\
			"../cluster/tests_fixedH_3T/", \
			"../cluster/tests_fixedH_4T/", \
			"../cluster/tests_agent_belief/", "../cluster/tests_agent_fixed/","../cluster/tests_agent_nocash/", \
			"../cluster/tests_shani_fixed/"] ## "../cluster/tests_hpomdp_fixed/",
			self.avg = "_avg"


			# self.test_folders = ["../cluster/tests_belief_mix_nobug/","../cluster/tests_belief_mix_nobug/",\
			# "../cluster/tests_fixedH/",\
			# "../cluster/tests_fixedH_3T/", \
			# "../cluster/tests_fixedH_4T/"]
			# self.avg = "_avg"

			# self.test_folders = ["../tests_belief_mix_nobug/","../tests_belief_mix_nobug/",\
			# "../tests_fixedH/",\
			# "../tests_fixedH_3T/", \
			# "../tests_fixedH_4T/"]
			# self.avg = "_avg"


		else:
			self.tables = [3,4,5,6,7,8]
			self.horizons = [1005,1006,5,6]
			# self.algorithms = ["A:Method-2","B:Agent POMDP","C:N-samples-2","F:Method-3","G:N-samples-3"]
			self.algorithms = ["F:Method-3"]
			self.max_horizons = [3,4,5,6]

	def get_data(self, execs):
		frames = []
		colors = [[0.2, 0.3, 0.7], [0.9, 0.6, 0.32], [1,0,0],[0.2, 0.6, 0.7],[0.7, 0.1, 0.5], \
					[0.2, 0.4, 0.7], [0.9, 0.7, 0.32], [1,0.1,0],[0.2, 0.7, 0.7],[0.7, 0.2, 0.5]]
		execs = self.total_execs
		test_folder = self.POMDPTasks.test_folder
		alg_label = ""
		for table in self.tables:
			for simple in self.simples:
				prev_data_horizon = {}
				for horizon in self.horizons:
					optimal_belief_reward = None
					optimal_belief_steps = None
					optimal_belief_execs = None
					for test_folder in self.test_folders:
						execs = self.total_execs
						if not self.POMDPTasks.package:
							if simple:
								model_folder = "simple"
							else:
								model_folder = "complex"
						else:
							simple = True
							model_folder = "package"

						if "agent" in test_folder:
							model_folder += "_no_op_hybrid"+ self.avg + "_model"								
							filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
							 + '_greedy-' + str(False) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
							 str(execs) + '_seed-' + str(self.POMDPTasks.seed)

							if "fixed" in test_folder:
								alg_label = "E: Agent-POMDP-FH"	
								color = '#FFC300'#'#949494'
							elif "belief" in test_folder:
								alg_label = "D: Agent-POMDP-AD "
								color = '#81c784'#'#0d47a1'
							else:
								alg_label = "F: Agent-POMDP-AD wo reuse "
								color = '#4fc3f7'#'#85c1e9' #
								continue

						elif "shani" in test_folder:
							# set_trace()
							model_folder += "_no_op_hybrid"+ self.avg + "_model"								
							filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
							 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
							 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + "_noop-" + str(True)

							alg_label = "N: N-samples"	
							color = '#4A3F34' # #515151  #50214A
							# print (filename)
							# set_trace()
							if horizon >=7:
								filename +=  "_hybrid_4T-" + str(True) + "_shani-" + str(True)
							elif horizon >=5:
								filename +=  "_hybrid_3T-" + str(True) + "_shani-" + str(True)
								# print (filename)
								# set_trace()
							else:
								filename +=  "_hybrid-" + str(True) + "_shani-" + str(True)

						elif "hpomdp" in test_folder:
							# set_trace()
							model_folder += "_no_op_hybrid"+ self.avg + "_model"								
							filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
							 + '_greedy-' + str(False) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
							 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + "_H_POMDP-" + str(True)

							alg_label = "P: HPOMDP"	
							color = '#808b96' #50214A
							# set_trace()

						elif "fixed" in test_folder:
							color = '#e67e22'#'#ca9161'
							if "4T" in test_folder:
								alg_label = "FH 4T"
								model_folder += "_no_op_hybrid_4T"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid_4T-" + str(True)
							elif "3T" in test_folder:
								alg_label = "FH 3T"
								model_folder += "_no_op_hybrid_3T"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid_3T-" + str(True)
							else:
								alg_label = "FH 2T"
								model_folder += "_no_op_hybrid"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid-" + str(True)

							if not ((horizon >= 2 and horizon <= 4 and "2T" in alg_label) or (horizon >= 5 and "2T" in alg_label and table == 2)) and \
								not ((horizon >= 5 and horizon <= 6 and "3T" in alg_label) or (horizon >= 7 and "3T" in alg_label and table == 3)) and \
								not (horizon >= 7 and "4T" in alg_label):
								continue
							else:
								alg_label = "C: Multi-task-FH"

						else: 
							if "belief" in test_folder:
								color =   '#43a047'  #'#029e73'
								alg_label = "A: Multi-task-AD "
							else:
								alg_label = "B: Multi-task-AD wo reuse "
								color =  '#1976d2'#'#81c784'  #'#82E0AA'

							if "mix" in test_folder:
								ext_alg_label = "2 & 3 & 4"
								model_folder += "_no_op_hybrid"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid-" + str(True)
							elif "4T" in test_folder:
								ext_alg_label = "4T"
								model_folder += "_no_op_hybrid_4T"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid_4T-" + str(True)
							elif "3T" in test_folder:
								ext_alg_label = "3T"
								model_folder += "_no_op_hybrid_3T"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid_3T-" + str(True)
							else:
								ext_alg_label = "2T"
								model_folder += "_no_op_hybrid"+ self.avg + "_model"
								filename = test_folder + model_folder + '/tables-' + str(table) + '_simple-' + str(simple) \
								 + '_greedy-' + str(True) +  '_horizon-' + str(horizon+1000) + '_#execs-' + \
								 str(execs) + '_seed-' + str(self.POMDPTasks.seed) + \
								  "_noop-" + str(True) + "_hybrid-" + str(True)

							if not ((horizon >= 2 and horizon <= 4 and "2T" in ext_alg_label) or (horizon >= 5 and "2T" in ext_alg_label and table == 2)) \
							 and not ("mix" in test_folder):
								alg_label += ext_alg_label
								continue
							else:
								alg_label += ""

						if alg_label not in prev_data_horizon.keys():
							prev_data_horizon[alg_label] = {}

						pkl_filename = filename + '.pkl'
						exists = os.path.isfile(pkl_filename)

						if not exists:
							# print ("---- ", pkl_filename)

							pkl_filename = pkl_filename.replace("#execs-10_","#execs-1_")
							execs = 1
							exists = os.path.isfile(pkl_filename)

							# pkl_filename = pkl_filename.replace("#execs-1_","#execs-10_")
							# exists = os.path.isfile(pkl_filename)

							if not exists:
								# print ("---- ", pkl_filename)
								pkl_filename = pkl_filename.replace("#execs-10_","#execs-30_")
								execs = 30
								exists = os.path.isfile(pkl_filename)
								if not exists:
									print ("---- ", pkl_filename)

						if exists:
							# print (pkl_filename)
							# set_trace()								

							self.palette[alg_label] = color
							data = cPickle.load(open(pkl_filename,'rb'))
							final_num_steps = np.minimum(np.full((execs),self.max_num_steps),data['final_num_steps'][0:execs])
							data['final_num_steps'] = final_num_steps
							data['final_total_reward'] = np.sum(data['final_total_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							
							if optimal_belief_reward is None and "mix" in test_folder and "belief" in test_folder:
								optimal_belief_reward = data['final_total_belief_reward'][0:execs,0:self.max_num_steps]
								optimal_belief_steps = data['final_total_time_steps'][0:execs,0:self.max_num_steps]
								optimal_belief_execs = execs
								continue

							
							# print ("horizon: ", horizon, " - algorithm: ", alg_label)
							prev_data_horizon[alg_label][horizon] = {}
							prev_data_horizon[alg_label][horizon]['reward'] = data['final_total_belief_reward'][0:execs,0:self.max_num_steps]
							prev_data_horizon[alg_label][horizon]['timesteps'] = data['final_total_time_steps'][0:execs,0:self.max_num_steps]
							# if horizon == self.horizons[0]: ## ?????
							# 	continue						

							# if execs != 1:
							if optimal_belief_reward is None:
								continue
								# data['final_total_belief_reward'] = np.sum(-optimal_belief_reward[0:execs,0:self.max_num_steps]+data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
								# data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							# 	next_horizon = horizon - 1
							# 	print (horizon)
							# 	# set_trace()
							# 	if next_horizon >= 1002:
							# 		# np.sum(data['final_total_time_steps'][0:execs,0:self.max_num_steps],axis=1)
							# 		prev_data_placeholder = data['final_total_belief_reward'][0:execs,0:self.max_num_steps] ### WRONGGGG??? DIVIDE By num of time steps
							# 		data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/np.sum(data['final_total_time_steps'][0:execs,0:self.max_num_steps],axis=1) - \
							# 		np.sum(prev_data_horizon[alg_label][next_horizon]['reward'],axis=1)/np.sum(prev_data_horizon[alg_label][next_horizon]['timesteps'],axis=1)
							# 		plot_data = prev_data_placeholder

							# 		# if table == 11 and horizon == 2:
							# 		# 	plt.figure()

							# 		# if table == 11:
							# 		# 	set_trace()

							# 		# if table == 11 and horizon != 2:
							# 		# 	print ("table: ", table, " horizon: ", horizon)
							# 		# 	set_trace()
							# 		# 	ax = plt.gca()
							# 		# 	for e in range(0,execs):
							# 		# 		print (" exec: ", e)
							# 		# 		if data['final_total_belief_reward'][e] < 0:
							# 		# 			matplt.plot(range(0,self.max_num_steps),prev_data_horizon[alg_label][horizon]['reward'][e,0:self.max_num_steps],label='h='+str(horizon),color=colors[horizon])
							# 		# 			matplt.plot(range(0,self.max_num_steps),prev_data_horizon[alg_label][next_horizon]['reward'][e,0:self.max_num_steps],label='h='+str(next_horizon),color=colors[horizon],linestyle='dashed')
							# 		# 			ax.set_xlabel('Steps',fontsize=26)
							# 		# 			ax.set_ylabel('Reward',fontsize=26)
							# 		# 			ax.set_ylabel('Reward',fontsize=26)
							# 		# 			ax.set_title("table: " + str(table) + " - horizon: " + str(horizon) + " - execs=" + str(e))
							# 		# 			matplt.xticks(np.arange(0,self.max_num_steps))
							# 		# 			ax.legend()
							# 		# 			# plt.show(block=False)
							# 		# 			# plt.pause(0.5)
							# 		# 			plt.tight_layout()
							# 		# 			# set_trace()
							# 		# 			dirct = test_folder + 'results_' + str(execs) + '/' + 'tables_' + str(table) + '/' + 'h_' + str(next_horizon) + '_h_' + str(horizon) + '/'
							# 		# 			if not os.path.exists(dirct):
							# 		# 				os.makedirs(dirct)
							# 		# 			matplt.savefig(dirct + str(e) + '_exec' + '.png')
							# 		# 			# set_trace()
							# 		# 			plt.cla()
							# 	else:
							# 		data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/np.sum(data['final_total_time_steps'][0:execs,0:self.max_num_steps],axis=1)
							# else:
							# 	data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/np.sum(data['final_total_time_steps'][0:execs,0:self.max_num_steps],axis=1)
							# Simple 3T: table == 10 and horizon == 5 
							# if table == 10 and horizon == 5:
							# 	print ("-------------------------------------------------------------")
							# 	print ("algorithm: ",alg_label)
							# 	print ("reward", data['final_total_belief_reward'])
							# 	print ("max_Q", data['final_total_max_Q'])
							# 	print ("final_total_UB_minus_LB", data['final_total_UB_minus_LB'])
							# 	print ("final_total_horizon", data['final_total_horizon'])
							# print (data['final_total_belief_reward'][0:execs,0:self.max_num_steps])
							# print (optimal_belief_reward[0:execs,0:self.max_num_steps])
							
							new_data = np.zeros((execs,))
							# print ("label: ", alg_label, optimal_belief_execs)
							for i in range(0,min(execs,optimal_belief_execs)):
								non_zero_length = self.max_num_steps 
								non_zero_length_optimal = self.max_num_steps 
								for j in range(0,self.max_num_steps):
									# print ("final time steps: ", data['final_total_time_steps'][i,j],i,j,execs)
									if data['final_total_time_steps'][i,j] == 0:
										non_zero_length = j
										break
								for k in range(0,self.max_num_steps):
									# print ("optimal time steps: ", optimal_belief_reward,i,k,execs)
									if optimal_belief_steps[i,k] == 0:
										non_zero_length_optimal = k
										break

								# print (non_zero_length,non_zero_length_optimal)
								new_data[i] = np.sum(data['final_total_belief_reward'][i,0:non_zero_length]/data['final_total_time_steps'][i,0:non_zero_length]) + \
								-np.sum(optimal_belief_reward[i,0:non_zero_length_optimal]/optimal_belief_steps[i,0:non_zero_length_optimal])

							# data['final_total_belief_reward'] = np.sum(data['final_total_belief_reward'][0:execs,0:self.max_num_steps],axis=1)/np.sum(data['final_total_time_steps'][0:execs,0:self.max_num_steps],axis=1)
							if table == 3 and horizon == 4:
								print ("table: ", table, " horizon: ", horizon, " execs: ", execs)
								if "mix" in test_folder and "belief" in test_folder:
									print ("belief")
									print (data['final_total_belief_reward'][0:execs,0:self.max_num_steps])
									print (data['final_total_time_steps'][0:execs,0:self.max_num_steps])
									print (new_data)
									# set_trace()
								elif "mix" in test_folder and "nocash" in test_folder:
									print ("nocash")
									print (data['final_total_belief_reward'][0:execs,0:self.max_num_steps])
									print (data['final_total_time_steps'][0:execs,0:self.max_num_steps])
									print (new_data)
									# set_trace()
								elif alg_label == "C: Multi-task-FH":
									print ("fixed")
									print (data['final_total_belief_reward'][0:execs,0:self.max_num_steps])
									print (data['final_total_time_steps'][0:execs,0:self.max_num_steps])
									print (new_data)
									# set_trace()

							data['final_total_belief_reward'] = new_data

							data['final_total_max_Q'] = np.sum(data['final_total_max_Q'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							data['final_total_time_steps'] = np.sum(data['final_total_time_steps'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
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
							data['algorithm'] = algorithm
							data['max_horizon'] = data['max_horizon'][0:execs]
							data['final_total_horizon'] = np.sum(data['final_total_horizon'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							data['final_total_UB_minus_LB'] = np.sum(data['final_total_UB_minus_LB'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps
							data['horizon'] = data['max_horizon'][0:execs]
							# set_trace()
							# if "fixed" in alg_label:
							# 	data['max_horizon'] = -1
							# 	total_horizon = np.full((execs,),data['horizon'][0],dtype=float)
							# 	data['final_total_horizon'] = total_horizon
							# 	total_UB_minus_LB = np.full((execs,),0,dtype=float)
							# 	data['final_total_UB_minus_LB'] = total_UB_minus_LB
							# 	data['algorithm'] = "fixed horizon"
							# else:
							# 	# set_trace()
							# 	# data['algorithm'] = "adaptive horizon"
							# 	if horizon == 1005:
							# 		print ("++++++++++++++++++ table: ", table)
							# 		print ("2s: ", sum(sum(data['final_total_horizon'][0:execs,0:self.max_num_steps] == 2)))
							# 		print ("3s: ", sum(sum(data['final_total_horizon'][0:execs,0:self.max_num_steps] == 3)))
							# 		print ("4s: ", sum(sum(data['final_total_horizon'][0:execs,0:self.max_num_steps] == 4)))
							# 		print ("5s: ", sum(sum(data['final_total_horizon'][0:execs,0:self.max_num_steps] == 5)))

								
								# set_trace()

							if "num_pairs" not in data.keys():
								num_pairs = np.full((execs,),0,dtype=float)
								data['final_total_num_pairs'] = num_pairs
							else:
								# set_trace()
								data['final_total_num_pairs'] = np.sum(data['final_total_num_pairs'][0:execs,0:self.max_num_steps],axis=1)/final_num_steps

							# print(data)
							df = pd.DataFrame(data=data)
							df['id'] = range(0, df.shape[0])
							frames.append(df) 	

							if (not self.POMDPTasks.package and alg_label == "E: Agent-POMDP-FH") or \
								(self.POMDPTasks.package and alg_label == "D: Agent-POMDP-AD "):
								data['final_num_steps'] = np.full((execs,),0,dtype=int)
								data['final_total_reward'] = np.full((execs,),0,dtype=int)
								data['final_total_belief_reward'] = np.full((execs,),0,dtype=int)
								data['final_total_max_Q'] = np.full((execs,),0,dtype=int)
								data['final_total_time_steps'] = np.full((execs,),0,dtype=int)
								data['final_satisfaction'] = np.full((execs,),0,dtype=int)
								data['final_unsatisfaction'] = np.full((execs,),0,dtype=int)
								data['planning_time'] = np.full((execs,),0,dtype=int)
								data['tree_sizes'] = np.full((execs,),0,dtype=int)
								data['horizon'] = data['horizon'][0:execs]
								data['greedy'] = data['greedy'][0:execs]
								data['simple'] = data['simple'][0:execs]
								data['tables'] = data['tables'][0:execs]
								data['hybrid'] = data['hybrid'][0:execs]
								data['no_op'] = data['no_op'][0:execs]
								data['shani'] = data['shani'][0:execs]
								data['H_POMDP'] = data['H_POMDP'][0:execs]
								alg_label = "-----------------"
								algorithm = np.full((execs,),alg_label,dtype=object)
								self.palette[alg_label] = "#FFFFFF"
								data['algorithm'] = algorithm
								data['max_horizon'] = data['max_horizon'][0:execs]
								data['final_total_horizon'] = np.full((execs,),0,dtype=int)
								data['final_total_UB_minus_LB'] = np.full((execs,),0,dtype=int)
								data['horizon'] = data['max_horizon'][0:execs]
								num_pairs = np.full((execs,),0,dtype=float)
								data['final_total_num_pairs'] = np.full((execs,),0,dtype=int)

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
		# self.palette = None
		# print (self.palette)
		# set_trace()
		# result_1 = self.get_data(1)
		result_1 = None
		ax = None

		res_1_horizon = None
		
		simple = self.POMDPTasks.simple
		res_simple = result.loc[result['simple'] == simple]
		if result_1 is not None: res_1_simple = result_1.loc[result_1['simple'] == simple];

		if not res_simple.empty:
			for title in self.tables:
				title_name = 'tables'
				title = int(title)
				res_title = res_simple.loc[res_simple[title_name] == title]

				if result_1 is not None and not res_1_simple.empty:
					res_1_title = res_1_simple.loc[res_1_simple[title_name] == title]
				else:
					res_1_title = None

				if not res_title.empty:	
					x = "max_horizon"; y = "planning_time"; hue = "algorithm"
					ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs, title)
						
				for max_h in self.max_horizons:
					title_name = 'tables'
					title = int(title)
					res_title_base = res_simple.loc[res_simple[title_name] == title]
					res_title_base = res_title_base.loc[res_title_base["max_horizon"] == max_h]			

					res_title = res_simple.loc[res_simple[title_name] == title]
					res_title = res_title[(res_title["max_horizon"] == -1) & (res_title["horizon"] <= max_h)]
					title_name = 'max_horizon: ' + str(max_h) + ', tables'

					# if not res_title.empty:	
					# 	if not self.compare_3T:
					# 		x = "horizon"; y = "final_total_belief_reward"; hue = "algorithm"
					# 		ax = self.plot_max_horizon(res_title, res_title_base, simple, title, title_name, x, y, hue, execs)

					

		for simple in self.simples:
			simple = int(simple)
			res_simple = result.loc[result['simple'] == simple]
			if result_1 is not None: res_1_simple = result_1.loc[result_1['simple'] == simple];
			if not res_simple.empty:
				for title in self.horizons:
					title_name = 'max_horizon'
					title = int(title)
					res_title = res_simple.loc[res_simple[title_name] == title]

					if result_1 is not None and not res_1_simple.empty:
						res_1_title = res_1_simple.loc[res_1_simple[title_name] == title]
					else:
						res_1_title = res_title

					# if not res_title.empty:	
					# 	# if not self.compare_3T:
					# 	x = "tables"; y = "final_total_horizon"; hue = "algorithm"
					# 	ax = self.plot_horizon(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)


				for title in self.tables:
					title_name = 'tables'
					title = int(title)
					res_title = res_simple.loc[res_simple[title_name] == title]

					if result_1 is not None and not res_1_simple.empty:
						res_1_title = res_1_simple.loc[res_1_simple[title_name] == title]
					else:
						res_1_title = None

					if not res_title.empty:	

						x = "max_horizon"; y = "final_total_belief_reward"; hue = "algorithm"
						ax = self.plot_tables_planning_time_curve(res_title, res_1_title, simple, title, title_name, x, y, hue, execs, title)

						# x = "max_horizon"; y = "final_total_horizon"; hue = "algorithm"
						# ax = self.plot_horizon(res_title, res_1_title, simple, title, title_name, x, y, hue, execs)

		for simple in self.simples:
			simple = int(simple)
			res_simple = result.loc[result['simple'] == simple]
			if not res_simple.empty:	
				# if not self.compare_3T:
				title_name = "algorithm"
				title1 = "A: Multi-task-AD "
				title2 = "D: Agent-POMDP-AD "
				# set_trace()
				# res_simple = res_simple[(res_simple[title_name] == title1) | (res_simple[title_name] == title2)]				
				res_simple_1 = res_simple.loc[res_simple[title_name] == title1]
				hue = "max_horizon"; y = "final_total_horizon"; x = "tables"
				ax = self.plot_horizon_all(res_simple_1, res_1_title, simple, title, title_name, x, y, hue, execs, "ac_pomdp")

				res_simple_2 = res_simple.loc[res_simple[title_name] == title2]
				hue = "max_horizon"; y = "final_total_horizon"; x = "tables"
				ax = self.plot_horizon_all(res_simple_2, res_1_title, simple, title, title_name, x, y, hue, execs, "agent")


	def plot_max_horizon(self, res, res_base, simple, title, title_name, x, y, hue, execs):
		print ("----- ", y)
		data = res[['id',x,y,hue]]
		data_base = res_base[['id',x,y,hue,"max_horizon"]]

		fig = plt.figure()
		ax = plt.gca()

		data[y] = data.groupby(x)[y].apply(lambda x: data_base[y]-x)
		# for horizon in data[x].unique():
		# 	sample = data.loc[data[x] == horizon]
		# 	data['diff'] = sample[y] - data_base[y] 
		# 	print (data)
		# 	print (data_base)

		# data_base['horizon'] = -1 * data_base['max_horizon']
		ax = sns.lineplot(x=x, y=y, hue=hue, style="algorithm", palette=self.palette, markers=False, dashes=False, data=data,legend=False) #, dashes=False
		# ax = sns.lineplot(x=x, y=y, hue=hue, style="algorithm", markers=True, data=data_base,legend=False) #, dashes=False

		pppp = [1,2,3,4,5,6,7]
		if y == "final_total_belief_reward":
			prev_pos = None
			count = 0
			for horizon in data[x].unique():
				sample = data.loc[data[x] == horizon]
				# sample[y] = sample[y] - data_base[y] 
				mean_x = np.mean(sample[y])
				sample_gr = sample.loc[sample[y] > 0]
				sample_le = sample.loc[sample[y] < 0]
				sample_ee = sample.loc[sample[y] == 0]
				# str_stat = "ge: " + str(round(len(sample_gr)/len(sample)+len(sample_ee)/len(sample),2))
				str_stat = "gr: " + str(round(len(sample_gr)/len(sample),2))
				str_stat += ", e: " + str(round(len(sample_ee)/len(sample),2))
				str_stat += ", l: " + str(round(len(sample_le)/len(sample),2))
				print ("max horizon=",horizon,len(sample_gr)/len(sample), len(sample_le)/len(sample), len(sample_ee)/len(sample))
				pos = [horizon,mean_x] 
				# rand = np.random.randint(10) + np.random.randint(10)
				# pos[0] += 0.1*rand
				pos[1] += pppp[count] * 75-150
				# if prev_pos is not None and np.fabs(prev_pos[1]-pos[1]) < 18:
				# 	pos[1] -= 18
				plt.annotate(str_stat, xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=14)
				prev_pos = pos
				count += 1


		if self.POMDPTasks.package and title_name == "tables":
			title_name = "packages"
		ax.set_title(title_name+": " + str(title),fontsize=26)
		# set_trace()
		matplt.xticks(np.arange(min(data[x].unique()),max(data[x].unique())+1))

		if "time" in y:
			ylab = "planning time (s)"
		if "reward" in y:
			ylab = "reward"
		if "horizon" in y:
			ylab = "final horizon"
		ax.set_xlabel(x,fontsize=26)
		ax.set_ylabel(ylab,fontsize=26)
		ax.tick_params(labelsize=22)
		plt.tight_layout()
		matplt.savefig(self.POMDPTasks.test_folder + '/results_' + str(execs) + '/' + "3T_" + str(self.compare_3T) + "_" + x + "_" + hue + "_" + y + '_simple-' \
			+ str(simple) + '_'+title_name+'-' + str(title) + "_execs-" + str(execs) + '.png')


	def plot_tables_planning_time_curve(self, res, res_1, simple, title, title_name, x, y, hue, execs, table = None):
		print ("----- ", y)
		data = res[['id',x,y,hue]]
		# set_trace()
		# if not self.compare_3T:
		# 	if len(data[x].unique()) < 8:
		# 		fig = plt.figure(figsize=(max(len(data[x].unique()),8)-3,7))
		# 	else:
		# 		fig = plt.figure(figsize=(max(len(data[x].unique()),8)-4,7))
		# else:
		# 	fig = plt.figure(figsize=(max(len(data[x].unique()),8),8))
		weight = 800
		fig = plt.figure(figsize=(7,10))
		ax = plt.gca()
		# sns.set_context("paper", rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16})   

		# if title == min(self.horizons) or self.compare_3T:
		# print(data)
		if y == "final_total_belief_reward":
			width = 1
		else:
			width = 3
		ax = sns.lineplot(x=x, y=y, hue=hue, style="algorithm", palette=self.palette, markers=False, dashes=False, data=data, lw=width)
		# plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
		# plt.setp(ax.get_legend().get_title(), fontsize='22') # for legend title
		# else:
		# 	ax = sns.lineplot(x=x, y=y, hue=hue, style="algorithm", markers=True, data=data,legend=False) #, dashes=False

		# l = mlines.Line2D([2,self.horizons[-1]], [0,0], color='red', linestyle="--")
		# ax.add_line(l)
		pppp = [1,2,3,4,5,6,7,8,9]
		if y == "final_total_belief_reward":
			prev_pos = None
			count = 0
			for alg in data[hue].unique():
				# set_trace()
				dd = data.loc[data[hue] == alg]
				horizon = None
				if alg[0] != "N" and alg[0] != "P":
					# if alg[0] == "E" and (horizon == 5 or horizon == 7):
					# 	if table == 5: horizon = 3;
					# 	if table == 7: horizon = 3;

					# if alg[0] == "D" and (horizon == 5 or horizon == 7):
					# 	if table == 5: horizon = 3;
					# 	if table == 7: horizon = 3;

					# if alg[0] == "C":
					# 	if table == 3: horizon = 5;
					# 	if table == 5: horizon = 5;
					# 	if table == 7: horizon = 5;
					# 	if table == 9: horizon = 5;
					# 	if table == 11: horizon = 5;

					# if alg[0] == "B":
					# 	# set_trace()
					# 	if table == 3: horizon = 6;
					# 	if table == 5: horizon = 6;
					# 	if table == 7: horizon = 6;
					# 	if table == 9: horizon = 6;
					# 	if table == 11: horizon = 6;

					# if alg[0] == "A":
					# 	if table == 3: horizon = 6;
					# 	if table == 5: horizon = 6;
					# 	if table == 7: horizon = 6;
					# 	if table == 9: horizon = 6;
					# 	if table == 11: horizon = 6;
					# horizon = dd[x].unique()
					# if len(horizon) == 0:
					# 	continue
					# horizon = max(horizon)
					# print (horizon, table, alg)
					# if horizon is not None:
					# 	sample = dd.loc[dd[x] == horizon]
					# 	if not sample.empty: 
					# 		mean_x = np.mean(sample[y])
					# 		pos = [horizon,mean_x] 
					if table == 5 or table == 7:
						algs = "E/D/C/B/A"
					else:
						algs = "C/B/A"
					mean_x = 200
					horizon = 4
					pos = [horizon,mean_x] 
				else:
					algs = alg[0]
					if alg[0] == "P":
						horizon = 5
					if alg[0] == "N":
						horizon = 4
						if table == 5:
							horizon = 6	
						if table == 9:
							horizon = 5						
					dd = data.loc[data[hue] == alg]
					if not dd.empty:
						sample = dd.loc[dd[x] == horizon]
						mean_x = np.mean(sample[y])
						pos = [horizon,mean_x] 
						pos[1] = pos[1] - 1000

				if "--" not in alg and horizon is not None:
					text="\\textcolor[HTML]{" + self.palette[alg][1:len(self.palette[alg])] + "}{" + "\\textbf{" + algs + "}" + "}"
					# print (text)
					pos[0] = pos[0] - 0.2
					an = plt.annotate(text, xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=28, zorder=2)
					# x_pos = pos[0]
					# y_pos = pos[1]
					# plt.annotate(text, xy=(x_pos+0.2,y_pos), xytext=(x_pos+0.3, y_pos), arrowprops=dict(facecolor='black', shrink=0.05, headwidth=20, width=7))
					# an.set_fontweight(weight='heavy')
		else:
			prev_pos = None
			count = 0
			for alg in data[hue].unique():
				# set_trace()
				dd = data.loc[data[hue] == alg]
				for horizon in dd[x].unique():
					sample = dd.loc[dd[x] == horizon]
					mean_x = np.mean(sample[y])
					pos = [horizon,mean_x] 
					rand = np.random.randint(10) + np.random.randint(10)

					if len(sample) == 1 and "----" not in alg:
						x_data = np.full(1,int(sample["max_horizon"]))
						y_data = np.full(1,float(sample["planning_time"]))
						colors_data = np.full(1,self.palette[alg])
						ax.scatter(x_data, y_data, c=colors_data, edgecolors="gray", zorder=1)

					# pos[0] += 0.1*rand
					# pos[1] += pppp[count] * 75-150
					# if prev_pos is not None and np.fabs(prev_pos[1]-pos[1]) < 18:
					# 	pos[1] -= 18
					if self.POMDPTasks.package:
						pass
						# if horizon >= 4:
						# 	pos[0] = pos[0] - 0.2
						# 	pos[1] = pos[1] + 0.1
						# 	if "agent POMDP AD " == alg or "multi-task AD " == alg or "multi-task FH" == alg:
						# 		plt.annotate(str(int(mean_x)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=24, zorder=2) ##round(mean_x,0)
					else:
						pass
						# if horizon > 4:
						# 	if not (table == 3 and horizon == 5):
						# 		if horizon == 5:
						# 			pos[0] = pos[0] - 0.4
						# 		else:
						# 			pos[0] = pos[0] - 0.6
						# 		pos[1] = pos[1] + 0.1

						# 		if (horizon == 5 and (table == 3 or table == 4)):
						# 			if "multi-task AD " == alg:
						# 				pos[1] = pos[1] - 0.4 * mean_x
						# 			if "multi-task FH" == alg:
						# 				pos[1] = pos[1] + 0.7 * mean_x

						# 		if (horizon == 6 and (table == 3 or table == 4)):
						# 			if "multi-task AD " == alg:
						# 				pos[1] = pos[1] - 0.4 * mean_x
						# 			if "multi-task FH" == alg:
						# 				pos[1] = pos[1] + 0.7 * mean_x

						# 		if (horizon == 5 and (table == 5)):
						# 			if "multi-task AD " == alg:
						# 				pos[1] = pos[1] - 0.4 * mean_x
						# 			if "multi-task FH" == alg:
						# 				pos[1] = pos[1] + 0.3 * mean_x

						# 		if horizon <= 6: 
						# 			if table is not None and table <= 4:
						# 				if "multi-task AD " == alg or "multi-task FH" == alg:
						# 					plt.annotate(str(int(mean_x)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=24, zorder=2)
						# 			elif table is not None and table == 5 and horizon == 5:
						# 				if "multi-task AD " == alg or "multi-task FH" == alg:
						# 					plt.annotate(str(int(mean_x)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=24, zorder=2)
						# 			else:
						# 				# plt.annotate(str(round(mean_x,2)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=16)
						# 				if "multi-task AD " == alg or "multi-task FH" == alg:
						# 					plt.annotate(str(int(mean_x)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=24, zorder=2)
						# 		else:
						# 			# plt.annotate(str(round(mean_x,2)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=16)
						# 			if "multi-task AD " == alg or "multi-task FH" == alg:
						# 				plt.annotate(str(int(mean_x)), xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=24, zorder=2) ##round(mean_x,0)

					prev_pos = pos
					count += 1
				## here
				pos[0] = pos[0] - 0.1
				if table == 7:
					if alg[0] == "E":
						pos[0] = pos[0] + 0.2
						pos[1] = pos[1] + 250
					elif alg[0] == "D":
						pos[0] = pos[0] + 0.2
						pos[1] = pos[1] - 500
				if "--" not in alg:
					text="\\textcolor[HTML]{" + self.palette[alg][1:len(self.palette[alg])] + "}{" + "\\textbf{" + alg[0] + "}" + "}"
					print (text)
					an = plt.annotate(text, xy=(pos[0], pos[1]), xytext=(pos[0], pos[1]), size=28, zorder=2)
					# an.set_fontweight(weight='heavy')
		legend = False
		if self.POMDPTasks.package:
			# if title == 4:
			ax.legend(prop={'size': 22})
			# pass
			# else:
			# 	ax.get_legend().remove()
		elif not self.POMDPTasks.package:
			if title == 5 and y == "final_total_belief_reward":
				ax.legend(prop={'size': 22,'weight':weight})
				legend = True
				pass
			elif title == 3:
				ax.legend(prop={'size': 21,'weight':weight},loc="upper left")
				legend = True
				pass
			else:
				ax.get_legend().remove()
		else:
			ax.legend(prop={'size': 22,'weight':weight})
			legend = True

		if legend:
			for t in ax.get_legend().texts: t.set_text("\\textbf{" + t.get_text() + "}")

		# if y == "final_total_belief_reward":
		# 	plt.annotate('actual group', xy=(x+0.2,y), xytext=(x+0.3, 300), arrowprops=dict(facecolor='black', shrink=0.05, headwidth=20, width=7))
		# 	tab_names = data[x].unique()
		# 	alg_names = data[hue].unique()
		# 	for alg in range(len(alg_names)):
		# 		dat_temp = data.loc[data[hue] == alg_names[alg]]
		# 		mean_x = np.mean(dat_temp.loc[dat_temp[x] == tab_names[len(tab_names)-alg-1]])
		# 		pos = (tab_names[len(tab_names)-alg-1],mean_x[y])
		# 		set_trace()
		# 		plt.annotate(alg_names[alg][0], xy=(pos[0], pos[1]), xytext=(pos[0]+0.5, pos[1]+0.5), arrowprops=dict(facecolor='black', shrink=0.05),)

			# tab.scale(5, 5) 
		if self.POMDPTasks.package and title_name == "tables":
			title_name = "packages"
		ax.set_title("\\textbf{" + title_name+": " + str(title) + "}",fontsize=30,fontweight=weight)
		# set_trace()
		ll = []
		# for i in :
		# 	ll += ["\\textbf{" + str(i) + "}"]
		matplt.xticks(np.arange(min(data[x].unique()),max(data[x].unique())+1),fontsize=22)

		pack = self.POMDPTasks.package
		def func_y(x, pos):  # formatter function takes tick label and tick position
			if pack:
				d = x/1000
				s = "\\textbf{" + str(d) + "k" + "}"
			else:
				d = x/1000
				s = "\\textbf{" + str(d) + "k" + "}"
			return s

		ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_y))

		def func_x(x, pos):  # formatter function takes tick label and tick position
			return "\\textbf{" + str(x) + "}"

		ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_x))

		if "time" in y:
			ylab = "\\textbf{" + "planning time (s)" + "}"
		if "reward" in y:
			ylab = "\\textbf{" + "reward" + "}"
		if "horizon" in y:
			ylab = "\\textbf{" + "final horizon" + "}"

		xlab = x
		if "max_horizon" in x:
			xlab = "\\textbf{" + "max horizon (H)" + "}"
		ax.set_xlabel(xlab,fontsize=30,fontweight=weight)
		ax.set_ylabel(ylab,fontsize=30,fontweight=weight)
		ax.tick_params(labelsize=22,width=2)

		# agent_pomdp_max_table = data.loc[data[hue] == "B:Agent POMDP"]
		# ax.get_xticklabels()[len(agent_pomdp_max_table[x].unique())-1].set_color("orange")

		# ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)

		plt.tight_layout()
		# set_trace()

		matplt.savefig(self.POMDPTasks.test_folder + '/results_' + str(execs) + '/' + "3T_" + str(self.compare_3T) + "_" + x + "_" + hue + "_" + y + '_simple-' \
			+ str(simple) + '_'+title_name+'-' + str(title) + "_execs-" + str(execs) + '.png')

	def plot_horizon (self, res, res_1, simple, title, title_name, x, y, hue, execs):
		print ("----- ", y)
		data = res[['id',x,y,hue]]
		fig = plt.figure()
		ax = plt.gca()
		# set_trace()
		# matplt.xticks(np.arange(self.tables[0],max(max(data[x].unique())+1,9)))
		# data = res[[x,y]]
		# set_trace()
		sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, palette=self.palette, linewidth=2.5)
		fig.suptitle('')

		if "horizon" in y:
			ylab = "final horizon"

		ax.set_xlabel(x,fontsize=26)
		ax.set_ylabel(ylab,fontsize=26)
		ax.tick_params(labelsize=22)
		# ax.set_title(title_name+": " + str(title)+", max horizon: "+str(res['max_horizon'].unique()[0]),fontsize=26)
		if self.POMDPTasks.package and title_name == "tables":
			title_name = "packages"
		ax.set_title(title_name+": " + str(title),fontsize=26)
		matplt.yticks(range(0,7))

		

		# agent_pomdp_max_table = data.loc[data[hue] == "B:Agent POMDP"]
		# ax.get_xticklabels()[len(agent_pomdp_max_table[x].unique())-1].set_color("orange")

		# ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)

		plt.tight_layout()
		# set_trace()
		matplt.savefig(self.POMDPTasks.test_folder + '/results_' + str(execs) + '/' + "3T_" + str(self.compare_3T) + "_" + x + "_" + hue + "_" + y + '_simple-' \
			+ str(simple) + '_'+title_name+'-' + str(title) + "_execs-" + str(execs) + '.png')

	def plot_horizon_all (self, res, res_1, simple, title, title_name, x, y, hue, execs, alg=None):
		print ("----- ", y)
		if self.POMDPTasks.package:
			fig = plt.figure(figsize=(8,10))
		else:
			fig = plt.figure(figsize=(10,10))
		ax = plt.gca()
		data = res[['id',x,y,hue]]
		palette = {}
		# set_trace()
		# matplt.xticks(np.arange(self.tables[0],max(max(data[x].unique())+1,9)))
		# data = res[[x,y]]
		# set_trace()
		# for title in self.horizons:
		# 	title = int(title)
		# 	res_title = res.loc[res['max_horizon'] == title]
			
		# 	sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, palette=self.palette, linewidth=2.5)

		labels = []
		bps = []
		temp_data = None
		if self.POMDPTasks.package:
			offset = 0.2
		else:
			offset = 0.4

		# for alg in self.algorithms:
		count = 0
		if self.POMDPTasks.package:
			tables = [2,3,4,5]
		else:
			tables = self.tables
			# tables = [num for num in self.tables if num % 2 == 1] 
		for horizon in self.max_horizons:											
			# if horizon == 2:
			# 	cl_border = "black"
			# 	cl_fill = "#b2dfdb"
			# 	pos = -3 * offset + table
			text = "\\textbf{H=" + str(horizon) + "}"
			if horizon == 3:
				cl_border = "black"
				cl_fill = "#b2dfdb"
				xy = 8
			elif horizon == 4:
				cl_border = "black"
				cl_fill = "#d4e157"
				pos = -1 * offset + tables[0]
				xy = 8
			elif horizon == 5:
				cl_border = "black"
				cl_fill = "#f4d03f"
				pos = tables[0]
				xy = 8
			elif horizon == 6:
				cl_border = "black"
				cl_fill = "#f57c00"
				pos = 1 * offset + tables[0]
				xy = 7
			elif horizon == 7:
				cl_border = "black"
				cl_fill = "#d84315"
				pos = 2 * offset + tables[0]
				xy = 5
			elif horizon == 8:
				cl_border = "black"
				cl_fill = "#3e2723"
				pos = 3 * offset + tables[0]
				xy = 3


			temp_data = data[(data.tables == xy)]	
			temp_data = temp_data[(temp_data.max_horizon == horizon)]	
			mean = np.mean(temp_data[y])
			if horizon != 8:
				xy = [xy-0.15,mean+0.1]
			else:
				xy = [xy-0.15,mean+0.45]

			an = plt.annotate(text, xy=(xy[0], xy[1]), xytext=(xy[0], xy[1]), size=28, zorder=2)
			palette[horizon] = cl_fill
		# set_trace()
		# temp_data = data.loc[(data["tables"] == 3) | (data["tables"] == 5) | (data["tables"] == 7) | (data["tables"] == 9) | (data["tables"] == 11)]
		# set_trace()
		ax = sns.lineplot(x=x, y=y, hue=hue, style="max_horizon",err_style="bars",err_kws={'capsize':10,'elinewidth':5}, palette=palette, markers=True, dashes=True, data=data, lw=5)
		

		count += 1
		# plt.close(fig)
		
		# set_trace()
		# fig.suptitle('')

		if "horizon" in y:
			ylab = "final horizon"

		if self.POMDPTasks.package and x == "tables":
			x = "packages"
		ax.set_xlabel("\\textbf{" + x + "}",fontsize=44)
		ax.set_ylabel("\\textbf{" + ylab + "}",fontsize=44)
		ax.tick_params(labelsize=44)

		def func_x(x, pos):  # formatter function takes tick label and tick position
			return "\\textbf{" + str(x) + "}"
		ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_x))
		ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_x))

		# ax.get_legend().get_texts()[0].set_text('H')
		# for l in range(0,len(ax.get_legend().get_texts())):
		# 	ax.get_legend().get_texts()[l].set_text('H')
		new_labels = ['H', '3', '4', '5', '6', '7', '8']
		ax.legend(loc='upper right',prop={'size': 26})
		for t, l in zip(ax.get_legend().texts, new_labels): t.set_text("\\textbf{" + l + "}")

		# set_trace()
		
		# ax.legend(bps[0:len(self.max_horizons)], labels[0:len(self.horizons)], loc='upper right', prop={'size': 20})


		# ax.set_title(title_name+": " + str(title)+", max horizon: "+str(res['max_horizon'].unique()[0]),fontsize=26)
		# ax.set_title(title_name+": " + str(title),fontsize=26)
		matplt.xticks(tables)
		matplt.yticks(range(2,8))
	
		plt.tight_layout()
		# set_trace()
		matplt.savefig(self.POMDPTasks.test_folder + '/results_' + str(execs) + '/' + "3T_" + str(self.compare_3T) + "_" + x + "_" + hue + "_" + y + '_simple-' \
			+ str(simple) + "_execs-" + str(execs) + "_" + alg + '.png')

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
		if not self.POMDPTasks.package:
			ax.set_title("tables: " + str(table) + "___" + ylabel)
		else:
			ax.set_title("packages: " + str(table) + "___" + ylabel)
			
		matplt.xticks(horizons)
		matplt.xlabel("horizons")
		matplt.ylim(0,1)
		matplt.ylabel(y + " ... " + ylabel)
		matplt.savefig(self.POMDPTasks.test_folder + self.POMDPTasks.model_folder + '/imgs_' + str(execs) + '/bar_' + self.POMDPTasks.model_folder + "_" + x + "_" + y + '_simple-' + str(simple) + '_table-' + str(table) + "_execs-" + str(self.POMDPTasks.num_random_executions) + '.png')


	def draw_plot(self, num, ax, data, offset, edge_color, fill_color, show_outliers=False, horizon=None, table = None):
		pos = [num+offset]
		if not self.POMDPTasks.package and horizon == 3 and table == 3:
			pos[0] = pos[0] - 0.2


		if self.POMDPTasks.package:
			width = 0.3
			size='10'
		else:
			width=0.7
			size='14'

		bp = None
		meanlineprops = dict(linestyle='-', linewidth=2.5, color='purple') 
		if show_outliers:
			bp = ax.boxplot(data, positions= pos, sym=".", meanline=True, showmeans=False, meanprops=meanlineprops, widths=width, patch_artist=True, manage_xticks=False, showfliers=True, zorder=1)
			outliers = [flier.get_ydata() for flier in bp["fliers"]]
			outliers_len = len(outliers[0])
		else:
			bp = ax.boxplot(data, positions= pos, sym=".", meanline=True, showmeans=False, meanprops=meanlineprops, widths=width, patch_artist=True, manage_xticks=False, showfliers=False, zorder=1)
			outliers_len = 0

		# median = data.median()	

		ax.text(bp['caps'][0]._x[0]+0.1, bp['caps'][1]._y[0]+0.05, 'H='+str(horizon), horizontalalignment='center', size=size, color='black', weight='semibold') 

		for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
				plt.setp(bp[element], color=edge_color)
		for patch in bp['boxes']:
			patch.set(facecolor=fill_color)

		
		x = np.full(data.shape[0],pos)
		colors = np.full(data.shape[0],fill_color)
		ax.scatter(x, data, c=colors, edgecolors="black", zorder=2)
		# set_trace()

		return outliers_len, bp