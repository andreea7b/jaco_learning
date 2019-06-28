import numpy as np

import math
import copy
import os
import pickle
import data_io
import time

from matplotlib import rc
from matplotlib import gridspec
import matplotlib.pyplot as plt

from planners import trajopt_planner
from utils.openrave_utils import *

# UPDATE THIS WITH THE CORRECT NUMBER OF PEOPLE
NUM_PPL = 12

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

EXP_TASK = 2


def get_pickled_metrics(filename):
	"""
	Parses cleaned pickle file with metrics per participant.
	filename can be "metrics_obj.p" or "metrics_subj.p"
	"""

	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/study/"
	datapath = here + subdir + filename

	metrics = pickle.load( open( datapath, "rb" ) )
	return metrics

# ---- Plotting Functionality ---- #

def plotting(avgA, avgB, stdA, stdB, xlabel, ylabel, title, maxY, twostar=False, pval=[0.01,0.01]):
	ind = np.arange(2)  # the x locations for the groups
	width = 0.45       # the width of the bars
	offset = 0.15

	# colors
	blackC = "black"	#(214/255., 39/255., 40/255.)
	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	# fonts
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	fig, ax = plt.subplots()


	rectsA = ax.bar(ind+offset, avgA, width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsB = ax.bar(ind+width+offset, avgB, width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))


	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%.2f' % 
					height,ha='center', va='bottom', fontsize=15)

	def autolabel_star(rects, std, p, second=False, twostar=False):
		"""
		Attach a text label above each bar displaying its height
		"""
		for i in range(0,len(rects),2):
			height1 = rects[i].get_height()
			height2 = rects[i+1].get_height()
			height = max(height1,height2)

			stdh = max(std[i],std[i+1])*1.5
			print stdh
			x = (rects[i].get_x() + rects[i].get_width())			
			if second:
				x = (rects[i].get_x() + rects[i].get_width())*1.5
			y = stdh+height+2

			widthB = "widthB="+str((rects[i].get_width()+rects[i+1].get_width())*7)

			if twostar:
				ax.annotate(r'\textbf{**}', xy=(x, y), xytext=(x, y), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, '+widthB+', lengthB=1.2', lw=1.5))
			else:			
				ax.annotate(r'\textbf{*}', xy=(x, y), xytext=(x, y), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, '+widthB+', lengthB=1.2', lw=1.5))

			ax.text(x,y+1,r"p$<$"+str(p[i]),ha='center', va='bottom', fontsize=30)

	ptime = []
	peffort = []
	autolabel_star(rectsA,stdA,pval)
	autolabel_star(rectsB,stdB,pval, second=True)
	#autolabel(rectsB)

	# add some text for labels, title and axes ticks
	ax.set_ylabel(r'\textbf{'+ylabel+'}',fontsize=50,labelpad=15)
	ax.set_xlabel(r'\textbf{'+xlabel+'}',fontsize=50,labelpad=15)
	
	plt.text(0.5, 1.08, r'\textbf{'+title+'}',
			 horizontalalignment='center',
			 fontsize=55,
			 transform = ax.transAxes)
	
	ax.set_xticks(ind+width+offset)

	xlabels = ["Table","Table + Cup"] #["T"+str(t+1) for t in range(3)]
	ax.set_xticklabels(xlabels,10,fontsize=50)

	# remove the plot frame lines
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)      
	
	# set max y-limit 
	ax.set_ylim([0,maxY])
	ax.tick_params(labelsize=50)

	# set padding for x and y tick labels
	ax.tick_params(direction='out', pad=2)

	# ensure that the axis ticks only show up on left of the plot.  
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 		

	
	leg = ax.legend((rectsA[0], rectsB[0]), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), fontsize=40)
	leg.get_frame().set_linewidth(0.0)

	plt.show()

	return fig


def augment_weights(time, weights):
	"""
	Augments the weight data with 0.1 sec timesteps
	"""
	#print "time: " + str(time)
	feat_w = weights[:,0]
	#print "feat_w: " + str(feat_w)

	aug_time = [0.0]*200 # traj is 20 sec, sampling at 0.1 sec
	aug_feat = [0.0]*200

	aug_idx = 0
	idx = 0
	prev_feat = 0.0
	times = np.arange(0.1, 20.0, 0.1)
	for t in times:
		aug_time[aug_idx] = t
		#clipped_t = round(time[idx][0],1)
		if idx < len(feat_w) and np.isclose(round(time[idx][0],1), t, rtol=1e-05, atol=1e-08, equal_nan=False):
			aug_feat[aug_idx] = feat_w[idx]
			prev_feat = feat_w[idx]
			idx += 1
		else:
			aug_feat[aug_idx] = prev_feat
		aug_idx += 1

	aug_time[-1] = 20.0
	aug_feat[-1] = prev_feat
	return (aug_time, aug_feat)

def plot_weights(task, saveFig=False):
	"""
	For the current task, makes 4 plots of all the weight updates over time
	for all the participants. 
	Two sets of plots: 
		- left are for Update ALL 
		- right are for Update BETA 
	"""
	weightData = data_io.parse_exp_data("weights")

	f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')

	ax1.set_title('Update BETA-fixed')
	ax2.set_title('Update BETA-adaptive')
	ax1.set_xlabel('time (s)')
	ax2.set_xlabel('time (s)')
	ax1.set_ylabel('weight')
	ax2.set_ylabel('weight')

	ax1.set_ylim([-1.5, 1.5])
	ax2.set_ylim([-1.5, 1.5])

	ax1.set_xlim([0, 20])
	ax2.set_xlim([0, 20])

	#ax2.set_title('Task 2: Table+Cup')

	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	trial = 2

	if task == 1:
		f.suptitle("Task 1: R knows table; H corrects table",fontsize=20)
	elif task == 2:
		f.suptitle("Task 2: R knows cup orientation; H corrects distance to table",fontsize=20)
	elif task == 3:
		f.suptitle("Task 3: R knows cup orientation; H corrects cup orientation",fontsize=20)
	elif task == 4:
		f.suptitle("Task 2: R knows distance to table; H corrects cup orientation",fontsize=20)
	Acount = 0
	Bcount = 0

	for ID in weightData.keys():
		#for task in weightData[ID]:
		# trial can take values 1 or 2
		#for trial in weightData[ID][task]:
		for method in weightData[ID][task][trial]:
			
			data = weightData[ID][task][trial][method]
			timestamp = data[:,0:1]
			weights = data[:,1:]

			(aug_time, aug_feat) = augment_weights(timestamp, weights)
			if method == "A":
				if Acount != 0:
					ax1.plot(aug_time,aug_feat,linewidth=4.0, color=greyC)
				else:
					ax1.plot(aug_time,aug_feat,linewidth=4.0, color=greyC, label='feat')
					Acount += 1
					ax1.axhline(y=0, color='k', linestyle='-')
					ax1.legend()
			elif method == "B":
				if Bcount != 0:
					ax2.plot(aug_time,aug_feat,linewidth=4.0, color=orangeC)
				else:
					ax2.plot(aug_time,aug_feat,linewidth=4.0, color=orangeC, label='feat')
					Bcount += 1
					ax2.axhline(y=0, color='k', linestyle='-')
					ax2.legend()				
				
	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		f.savefig(datapath+"task"+str(task)+"Weights.pdf", bbox_inches="tight")
		print "Saved weights figure."

def plot_weights_average(task, saveFig=False):
	"""
	For the current task, makes 2 plots of all the weight updates over time
	for all the participants averaged. 
	Two sets of plots: 
		- left are for Update ALL 
		- right are for Update BETA 
	"""
	weightData = data_io.parse_exp_data("weights")

	f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')

	ax1.set_title('Update BETA-fixed')
	ax2.set_title('Update BETA-adaptive')
	ax1.set_xlabel('time (s)')
	ax2.set_xlabel('time (s)')
	ax1.set_ylabel('Average abs(weight)')
	ax2.set_ylabel('weight')

	ax1.set_ylim([-1.5, 1.5])
	ax2.set_ylim([-1.5, 1.5])

	ax1.set_xlim([0, 20])
	ax2.set_xlim([0, 20])

	#ax2.set_title('Task 2: Table+Cup')

	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	trial = 2

	if task == 1:
		f.suptitle("Task 1: R knows table; H corrects table",fontsize=20)
	elif task == 2:
		f.suptitle("Task 2: R knows cup orientation; H corrects distance to table",fontsize=20)
	elif task == 3:
		f.suptitle("Task 3: R knows cup orientation; H corrects cup orientation",fontsize=20)
	elif task == 4:
		f.suptitle("Task 4: R knows distance to table; H corrects cup orientation",fontsize=20)
	Acount = 0
	Bcount = 0
	A_weights = [None]*NUM_PPL
	B_weights = [None]*NUM_PPL

	for ID in weightData.keys():
		#for task in weightData[ID]:
		# trial can take values 1 or 2
		#for trial in weightData[ID][task]:
		for method in weightData[ID][task][trial]:
			
			data = weightData[ID][task][trial][method]
			timestamp = data[:,0:1]
			weights = data[:,1:]

			(aug_time, aug_feat) = augment_weights(timestamp, weights)
			if method == "A":
				A_weights[Acount] = aug_feat
				Acount += 1
			elif method == "B":
				B_weights[Bcount] = aug_feat
				Bcount += 1
	A_weights = abs(np.array(A_weights))
	B_weights = abs(np.array(B_weights))
	A_avg = np.mean(A_weights,0)
	A_std = np.std(A_weights,0)/np.sqrt(NUM_PPL)
	B_avg = np.mean(B_weights,0)
	B_std = np.std(B_weights,0)/np.sqrt(NUM_PPL)
	ax1.errorbar(aug_time,A_avg,A_std,linewidth=4.0, color=greyC)
	ax1.axhline(y=0, color='k', linestyle='-')
	ax1.legend()
	ax2.errorbar(aug_time,B_avg,B_std,linewidth=4.0, color=orangeC)
	ax2.axhline(y=0, color='k', linestyle='-')
	ax2.legend()	
	
	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		f.savefig(datapath+"task"+str(task)+"Weights.pdf", bbox_inches="tight")
		print "Saved weights figure."

def plot_cupTableDiffFinal(saveFig=False):
	"""
	Produces two plots, one for the cupDiff and one for the tableDiff metric
	"""
	filename = "metrics_obj.p"
	obj = get_pickled_metrics(filename)

	# for keeping average of each feature, for each method, and for each task
	cupDiffAvg = np.array([[0.0, 0.0],[0.0,0.0]]) # [method all --> [avg for task 1, task 2], method one --> [avg for task 1, task 2]]
	tableDiffAvg = np.array([[0.0, 0.0],[0.0,0.0]]) 

	# for computing stddev 
	pplCupALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL) # trial 1 --> [task 1, task 2], trial 2 --> [task 1, task 2]
	pplCupONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplTableALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplTableONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)

	stdCup = np.array([[0.0,0.0]]*2)
	stdTable = np.array([[0.0,0.0]]*2)

	for ID in obj.keys():
		for task in obj[ID]:
			for trial in [1,2]:
				cup_all = obj[ID][task][trial]["A"][17]
				cup_one = obj[ID][task][trial]["B"][17]
				table_all = obj[ID][task][trial]["A"][18]
				table_one = obj[ID][task][trial]["B"][18]

				cupDiffAvg[0][task-1] += cup_all
				cupDiffAvg[1][task-1] += cup_one

				tableDiffAvg[0][task-1] += table_all
				tableDiffAvg[1][task-1] += table_one

				pplCupALL[ID][trial-1][task-1] = cup_all
				pplCupONE[ID][trial-1][task-1] = cup_one
				pplTableALL[ID][trial-1][task-1] = table_all
				pplTableONE[ID][trial-1][task-1] = table_one

	#print "pplCup: " + str(pplCup)
	# average by number of participants
	for method in range(2):
		for task in range(2):
			cupDiffAvg[method][task] /= NUM_PPL*2 # because 2 trials
			tableDiffAvg[method][task] /= NUM_PPL*2
			
			#c = np.reshape(pplCup[:,method,:,task], NUM_PPL)
			if method == 0: # all method
				stdCup[method][task] = np.std(pplCupALL[:,:,task])/np.sqrt(NUM_PPL*2*2) # because 2 tasks, 2 trials (per method)
				stdTable[method][task] = np.std(pplTableALL[:,:,task])/np.sqrt(NUM_PPL*2*2)
			else: # one method
				stdCup[method][task] = np.std(pplCupONE[:,:,task])/np.sqrt(NUM_PPL*2*2)
				stdTable[method][task] = np.std(pplTableONE[:,:,task])/np.sqrt(NUM_PPL*2*2)
		
	print stdCup
	xlabel = "Task"
	ylabel = r"CupDiffFinal"
	title = r"\textit{CupDiffFinal}: Cup Feature Difference from Ideal"	
	maxY = 50.0
	fig = plotting(cupDiffAvg[0], cupDiffAvg[1], stdCup[0], stdCup[1], xlabel, ylabel, title, maxY, pval=[0.01,0.01])

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		fig.savefig(datapath+"cupDiff.pdf", bbox_inches="tight")
		print "Saved cupDiff figure."

	xlabel = "Task"
	ylabel = r"TableDiffFinal"
	title = r"\textit{TableDiffFinal}: Table Feature Difference from Ideal"	
	maxY = 20.0
	fig = plotting(tableDiffAvg[0], tableDiffAvg[1], stdTable[0], stdTable[1], xlabel, ylabel, title, maxY, twostar=True, pval=[0.001,0.001])

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		fig.savefig(datapath+"tableDiff.pdf", bbox_inches="tight")
		print "Saved tableDiff figure."

def plot_dotOverTime(T1=True,saveFig=False):
	"""
	Produces plot over time of the dot product
	Specify if you want to plot Task 1, or Task 2 and if u want to save
	"""
	weightData = data_io.parse_exp_data("weights")

	times = np.arange(0.0, 20.0, 0.1)
	#print "times: " + str(times)
	dot_ALL = {}
	dot_ONE = {}
	for task in [1,2]:
		dot_ALL[task] = {}
		dot_ONE[task] = {}
		for ID in range(NUM_PPL):
			dot_ALL[task][ID] = {}
			dot_ONE[task][ID] = {}
			#for t in times:
			#	dot_ALL[task][ID][t] = None#[0.0,0.0] #trial 1, trial 2
			#	dot_ONE[task][ID][t] = None#[0.0,0.0] 

	aug_times = None
	# get weight data
	for ID in weightData.keys():
		for task in weightData[ID]:
			if task == 1:
				ideal_w = np.array([0.0,1.0])
			elif task == 2:
				ideal_w = np.array([1.0,1.0])

			#for trial in weightData[ID][task]:
			for method in ["A", "B"]:
				data1 = weightData[ID][task][1][method]
				data2 = weightData[ID][task][2][method]

				timestamp1 = data1[:,0:1]
				timestamp2 = data2[:,0:1]
				weights1 = data1[:,1:len(data1)+1]
				weights2 = data2[:,1:len(data2)+1]

				(aug_time1, aug_cup1, aug_table1) = augment_weights(timestamp1, weights1)
				(aug_time2, aug_cup2, aug_table2) = augment_weights(timestamp2, weights2)
				#print "aug_time1: " + str(aug_time1)
				#print "aug_time2: " + str(aug_time2)
				for t in range(len(aug_time1)):
					if aug_time1[t] not in dot_ALL[task][ID]:
						dot_ALL[task][ID][aug_time1[t]] = [0.0,0.0]
					if aug_time1[t] not in dot_ONE[task][ID]:
						dot_ONE[task][ID][aug_time1[t]] = [0.0,0.0]

					w1 = np.array([aug_cup1[t],aug_table1[t]]) 
					w2 = np.array([aug_cup2[t],aug_table2[t]]) 
					curr_dot = [np.dot(w1,ideal_w), np.dot(w2,ideal_w)] # dot prod for trial 1 and trial 2
					if method == "A":
						#print "w1: " + str(w1)
						#print "w2: " + str(w2)
						dot_ALL[task][ID][aug_time1[t]] = curr_dot
						#print "dot all: " + str(curr_dot)
					else:
						dot_ONE[task][ID][aug_time1[t]] = curr_dot
						#print "dot one: " + str(curr_dot)

				aug_times = aug_time1

	# now do the averaging over time across each participant
	avgALLT1 = np.array([0.0]*len(aug_times))
	avgONET1 = np.array([0.0]*len(aug_times))
	avgALLT2 = np.array([0.0]*len(aug_times))
	avgONET2 = np.array([0.0]*len(aug_times))

	ALLT1 = np.array([[0.0]*NUM_PPL*2]*len(aug_times)) # time1 --> [p1/trial1, p1/trial2, p2/trial1, p2/trial2, ...], time2 --> [p1/trial1, p1/trial2, p2/trial1, p2/trial2, ...]
	ONET1 = np.array([[0.0]*NUM_PPL*2]*len(aug_times))
	ALLT2 = np.array([[0.0]*NUM_PPL*2]*len(aug_times))
	ONET2 = np.array([[0.0]*NUM_PPL*2]*len(aug_times))
	
	idx = 0
	for t in aug_times:
		aAllt1 = 0.0
		aOnet1 = 0.0
		aAllt2 = 0.0
		aOnet2 = 0.0
		idx2 = 0
		for ID in range(NUM_PPL):
			aAllt1 += dot_ALL[1][ID][t][0] + dot_ALL[1][ID][t][1]
			aOnet1 += dot_ONE[1][ID][t][0] + dot_ONE[1][ID][t][1]
			aAllt2 += dot_ALL[2][ID][t][0] + dot_ALL[2][ID][t][1]
			aOnet2 += dot_ONE[2][ID][t][0] + dot_ONE[2][ID][t][1]

			ALLT1[idx][idx2] = dot_ALL[1][ID][t][0]
			ALLT1[idx][idx2+1] = dot_ALL[1][ID][t][1]
			ONET1[idx][idx2] = dot_ONE[1][ID][t][0]
			ONET1[idx][idx2+1] = dot_ONE[1][ID][t][1]
			ALLT2[idx][idx2] = dot_ALL[2][ID][t][0]
			ALLT2[idx][idx2+1] = dot_ALL[2][ID][t][1]
			ONET2[idx][idx2] = dot_ONE[2][ID][t][0]
			ONET2[idx][idx2+1] = dot_ONE[2][ID][t][1]
			print "aALLt1: " + str(aAllt1)
			print "ALLT1[t][idx2]: " + str(ALLT1[t][idx2]) + ", " + str(ALLT1[t][idx2+1])
			idx2 += 2
		
		#print "ALLT1["+str(t)+"]: " + str(ALLT1[t])

		avgALLT1[idx] = aAllt1/(NUM_PPL*2)
		avgONET1[idx] = aOnet1/(NUM_PPL*2)
		avgALLT2[idx] = aAllt2/(NUM_PPL*2)
		avgONET2[idx] = aOnet2/(NUM_PPL*2)
		idx += 1

	#print "ALLT1: " + str(ALLT1)
	allT1error = np.array([0.0]*len(aug_times))
	oneT1error = np.array([0.0]*len(aug_times))
	allT2error = np.array([0.0]*len(aug_times))
	oneT2error = np.array([0.0]*len(aug_times))

	# compute std error
	for t in range(len(aug_times)):
		print "ALLT1["+str(t)+"]: " + str(ALLT1[t])
		allT1error[t] = np.std(ALLT1[t])/np.sqrt(NUM_PPL*2)
		oneT1error[t] = np.std(ONET1[t])/np.sqrt(NUM_PPL*2)
		allT2error[t] = np.std(ALLT2[t])/np.sqrt(NUM_PPL*2)
		oneT2error[t] = np.std(ONET2[t])/np.sqrt(NUM_PPL*2)
	print "allT1error: " + str(allT1error)

	# colors
	blackC = "black"	#(214/255., 39/255., 40/255.)
	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	# fonts
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)
	
	fig, ax = plt.subplots(figsize=(11,9))

	#ax.set_xticks(ind+width+offset)

	#xlabels = "Time (s)" #["T"+str(t+1) for t in range(3)]
	#ax.set_xticklabels(xlabels,10,fontsize=50)

	plt.ylabel(r'\textbf{\textit{DotAvg}}',fontsize=35)
	plt.xlabel(r'\textbf{Time (s)}',fontsize=35)

	# remove the plot frame lines
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)      
	
	ax.tick_params(labelsize=30)
	# set padding for x and y tick labels
	ax.tick_params(axis='both', which='major', pad=15)
	#ax.tick_params(direction='out', pad=2)

	# ensure that the axis ticks only show up on left of the plot.  
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 		

	ax.set_ylim([0.0,2.4])
	ax.set_xlim([0.0,19])

	if T1:
		plt.text(0.5, 1.08, r'\textbf{\textit{DotAvg} During Task 1: Table}',
			 horizontalalignment='center',
			 fontsize=40,
			 transform = ax.transAxes)

		# plot the average dot product over time
		allT1 = ax.plot(times, avgALLT1, '-', color=greyC, label=r'\textbf{All-at-Once}', lw=7)
		oneT1 = ax.plot(times, avgONET1, '-', color=orangeC, label=r'\textbf{One-at-a-Time}', lw=7)

		ax.fill_between(times, avgALLT1-allT1error, avgALLT1+allT1error, color=greyC, alpha=0.5, lw=0)
		ax.fill_between(times, avgONET1-oneT1error, avgONET1+oneT1error, color=orangeC, alpha=0.5, lw=0)

	else:
		plt.text(0.5, 1.08, r'\textbf{\textit{DotAvg} During Task 2: Table + Cup}',
			 horizontalalignment='center',
			 fontsize=40,
			 transform = ax.transAxes)

		# plot the average dot product over time
		allT1 = ax.plot(times, avgALLT2, '-', color=greyC, label=r'\textbf{All-at-Once}', lw=7)
		oneT1 = ax.plot(times, avgONET2, '-', color=orangeC, label=r'\textbf{One-at-a-Time}', lw=7)

	
		ax.fill_between(times, avgALLT2-allT2error, avgALLT2+allT2error, color=greyC, alpha=0.5, lw=0)
		ax.fill_between(times, avgONET2-oneT2error, avgONET2+oneT2error, color=orangeC, alpha=0.5, lw=0)

	leg = ax.legend(fontsize=30, frameon=False, loc="upper left") #ax.legend((allT1, oneT1), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), fontsize=40)

	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		if T1:
			fig.savefig(datapath+"dotAvgTimeT1.pdf", bbox_inches="tight")
		else:
			fig.savefig(datapath+"dotAvgTimeT2.pdf", bbox_inches="tight")
		print "Saved dotAvgTime figure."

def plot_undoingObjSubj(saveFig=False):
	"""
	Plots side-by-side figure of cup/table away metric with undoing subj metric 
	"""
	# ------ COMPUTE OBJ METRICS NOW ----- #

	filename = "metrics_obj.p"
	obj = get_pickled_metrics(filename)

	# for keeping average of each feature, for each method, and for each task
	cupAwayAvg = np.array([[0.0, 0.0],[0.0,0.0]]) # [method all --> [avg for task 1, task 2], method one --> [avg for task 1, task 2]]
	tableAwayAvg = np.array([[0.0, 0.0],[0.0,0.0]]) 

	# for computing stddev 
	pplCupALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL) # trial 1 --> [task 1, task 2], trial 2 --> [task 1, task 2]
	pplCupONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplTableALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplTableONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)

	stdCup = np.array([[0.0,0.0]]*2)
	stdTable = np.array([[0.0,0.0]]*2)

	for ID in obj.keys():
		for task in obj[ID]:
			for trial in [1,2]:
				cup_all = obj[ID][task][trial]["A"][15]
				cup_one = obj[ID][task][trial]["B"][15]
				table_all = obj[ID][task][trial]["A"][16]
				table_one = obj[ID][task][trial]["B"][16]

				cupAwayAvg[0][task-1] += cup_all
				cupAwayAvg[1][task-1] += cup_one

				tableAwayAvg[0][task-1] += table_all
				tableAwayAvg[1][task-1] += table_one

				pplCupALL[ID][trial-1][task-1] = cup_all
				pplCupONE[ID][trial-1][task-1] = cup_one
				pplTableALL[ID][trial-1][task-1] = table_all
				pplTableONE[ID][trial-1][task-1] = table_one

	# average by number of participants
	for method in range(2):
		for task in range(2):
			cupAwayAvg[method][task] /= NUM_PPL*2 # because 2 trials
			tableAwayAvg[method][task] /= NUM_PPL*2
			
			if method == 0: # all method
				stdCup[method][task] = np.std(pplCupALL[:,:,task])/np.sqrt(NUM_PPL*2*2) # because 2 tasks, 2 trials (per method)
				stdTable[method][task] = np.std(pplTableALL[:,:,task])/np.sqrt(NUM_PPL*2*2)
			else: # one method
				stdCup[method][task] = np.std(pplCupONE[:,:,task])/np.sqrt(NUM_PPL*2*2)
				stdTable[method][task] = np.std(pplTableONE[:,:,task])/np.sqrt(NUM_PPL*2*2)

	# -------------------------------- #

	# ------ COMPUTE SUBJ METRICS NOW ----- #
	filename = "metrics_subj.p"
	subj = get_pickled_metrics(filename)

	t1All = [0.0]*NUM_PPL 
	t2All = [0.0]*NUM_PPL
	t1One = [0.0]*NUM_PPL
	t2One = [0.0]*NUM_PPL

	undoingT1avg = [0.0, 0.0] # [all, one method]
	undoingT2avg = [0.0, 0.0] # [all, one method]

	undoingT1err = [0.0, 0.0] # [all, one method]
	undoingT2err = [0.0, 0.0] # [all, one method]

	idxoffset = 2
	for ID in range(NUM_PPL):
		for task in [1, 2]:
			Q6All = float(subj[ID][task]["A"][idxoffset+5])
			Q7All = float(subj[ID][task]["A"][idxoffset+6])
			Q8All = float(subj[ID][task]["A"][idxoffset+7])
			avgAll = (Q6All+Q7All+Q8All)/3.0

			Q6One = float(subj[ID][task]["B"][idxoffset+5])
			Q7One = float(subj[ID][task]["B"][idxoffset+6])
			Q8One = float(subj[ID][task]["B"][idxoffset+7])
			avgOne = (Q6One+Q7One+Q8One)/3.0

			if task == 1:
				t1All[ID] = avgAll
				t1One[ID] = avgOne
				undoingT1avg[0] += avgAll
				undoingT1avg[1] += avgOne
			else:
				t2All[ID] = avgAll
				t2One[ID] = avgOne
				undoingT2avg[0] += avgAll
				undoingT2avg[1] += avgOne

	#average by num ppl		
	undoingT1avg[0] /= NUM_PPL
	undoingT1avg[1] /= NUM_PPL
	undoingT2avg[0] /= NUM_PPL
	undoingT2avg[1] /= NUM_PPL
			
	undoingT1err[0] = np.std(t1All)/np.sqrt(NUM_PPL) 	
	undoingT1err[1] = np.std(t1One)/np.sqrt(NUM_PPL) 	
	undoingT2err[0] = np.std(t2All)/np.sqrt(NUM_PPL) 	
	undoingT2err[1] = np.std(t2One)/np.sqrt(NUM_PPL) 	

	# -------------------------------- #

	# ------- PLOTTING ---------#

	ind = np.arange(2)  # the x locations for the groups
	width = 0.45       # the width of the bars
	offset = 0.15

	# colors
	blackC = "black"	#(214/255., 39/255., 40/255.)
	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	# fonts
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	fig = plt.figure()
	#fig, ((t1away, t1subj), (t2away, t2subj)) = plt.subplots(2, 2)

	gs = gridspec.GridSpec(4, 2, width_ratios=[3, 1]) 
	t1away = plt.subplot(gs[0])
	t1subj = plt.subplot(gs[1])
	t2away = plt.subplot(gs[2])
	t2subj = plt.subplot(gs[3])

	avgAllT1 = [cupAwayAvg[0][0],tableAwayAvg[0][0]]
	avgOneT1 = [cupAwayAvg[1][0],tableAwayAvg[1][0]]
	stdT1 = [[stdCup[0][0], stdTable[0][0]], [stdCup[1][0], stdTable[1][0]]]# all-->[stdCup, stdTable], one-->[stdCup, stdTable]
	stdT2 = [[stdCup[0][1], stdTable[0][1]], [stdCup[1][1], stdTable[1][1]]]

	avgAllT2 = [cupAwayAvg[0][1],tableAwayAvg[0][1]]
	avgOneT2 = [cupAwayAvg[1][1],tableAwayAvg[1][1]]

	# plot the OBJECTIVE METRICS
	ticks = np.arange(0,12,2)
	t1away.set_ylim([0,12])
	t2away.set_ylim([0,12])
	#t1away.yaxis.set_ticks(ticks)#ylim([0,15])
	#t2away.yaxis.set_ticks(ticks)#ylim([0,15])
	rectsALLT1 = t1away.bar(ind+offset, avgAllT1, width, color=greyC, yerr=stdT1[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsONET1 = t1away.bar(ind+width+offset, avgOneT1, width, color=orangeC, yerr=stdT1[1], ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsALLT2 = t2away.bar(ind+offset, avgAllT2, width, color=greyC, yerr=stdT2[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsONET2 = t2away.bar(ind+width+offset, avgOneT2, width, color=orangeC, yerr=stdT2[1], ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))

	# plot the SUBJECTIVE METRICS
	t1subj.set_ylim([0,7])
	t2subj.set_ylim([0,7])
	offset = 0.5
	width = 0.8
	# task 1 subj
	rectsSubjT1ALL = t1subj.bar(offset, undoingT1avg[0], width, color=greyC, yerr=undoingT1err[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsSubjT1ONE = t1subj.bar(offset+width, undoingT1avg[1], width, color=orangeC, yerr=undoingT1err[1], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))

	# task 2 subj
	rectsSubjT2ALL = t2subj.bar(offset, undoingT2avg[0], width, color=greyC, yerr=undoingT2err[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsSubjT2ONE = t2subj.bar(offset+width, undoingT2avg[1], width, color=orangeC, yerr=undoingT2err[1], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))

	pos1 = t1away.get_position() # get the original position 
	pos2 = [pos1.x0, pos1.y0,  pos1.width , pos1.height*1.5] 
	t1away.set_position(pos2)

	pos1 = t2away.get_position() # get the original position 
	pos2 = [pos1.x0, pos1.y0 - 0.2,  pos1.width , pos1.height*1.5] 
	t2away.set_position(pos2)

	pos1 = t1subj.get_position() # get the original position 
	pos2 = [pos1.x0, pos1.y0,  pos1.width , pos1.height*1.5] 
	t1subj.set_position(pos2)

	pos1 = t2subj.get_position() # get the original position 
	pos2 = [pos1.x0, pos1.y0 - 0.2,  pos1.width , pos1.height*1.5] 
	t2subj.set_position(pos2)


	t1away.set_title(r'\textbf{\textit{CupAway} and \textit{TableAway} Metrics}', fontsize=43, y=1.48)
	#t2away.set_title(r'Task 2 \textit{CupAway} and \textit{TableAway} Metrics')
	t1subj.set_title(r'\textbf{Required \textit{Undoing}}', fontsize=42, y=1.48)
	#t2subj.set_title(r'Task 2 \textit{Undoing} Subjective Metrics')

	# add some text for labels, title and axes ticks
	sz = 38
	t1away.set_ylabel(r'\textbf{Away Measure}',fontsize=sz,labelpad=15)
	t1away.set_xlabel(r'\textbf{Task 1: Table}',fontsize=sz,labelpad=15)

	t2away.set_ylabel(r'\textbf{Away Measure}',fontsize=sz,labelpad=15)
	t2away.set_xlabel(r'\textbf{Task 2: Table + Cup }',fontsize=sz,labelpad=15)
	
	t1subj.set_ylabel(r'\textbf{Likert Score}',fontsize=sz,labelpad=15)
	t1subj.set_xlabel(r'\textbf{Task 1: Table}',fontsize=sz,labelpad=15)

	t2subj.set_ylabel(r'\textbf{Likert Score}',fontsize=sz,labelpad=15)
	t2subj.set_xlabel(r'\textbf{Task 2: Table + Cup}',fontsize=sz,labelpad=15)
	
	# set x-axis  tick marks
	t1away.set_xticks(ind+width/2+offset/2)
	t2away.set_xticks(ind+width/2+offset/2)
	t1subj.set_xticks(ind+width+offset)
	t2subj.set_xticks(ind+width+offset)

	xlabels = [r'\textit{CupAway}',r'\textit{TableAway}'] 
	t1away.set_xticklabels(xlabels,10,fontsize=sz)
	t2away.set_xticklabels(xlabels,10,fontsize=sz)

	t1subj.set_xticklabels(["",""],10,fontsize=sz)
	t2subj.set_xticklabels(["",""],10,fontsize=sz)

	# remove the plot frame lines
	t1away.spines["top"].set_visible(False)   
	t1away.spines["right"].set_visible(False)    
	t2away.spines["top"].set_visible(False)    
	t2away.spines["right"].set_visible(False)    

	t1subj.spines["top"].set_visible(False)   
	t1subj.spines["right"].set_visible(False)    
	t2subj.spines["top"].set_visible(False)    
	t2subj.spines["right"].set_visible(False)      
	
	# set padding for x and y tick labels
	t1away.tick_params(axis='both', which='major', pad=15, labelsize=30)
	t2away.tick_params(axis='both', which='major', pad=15, labelsize=30)
	t1subj.tick_params(axis='both', which='major', pad=15, labelsize=30)
	t2subj.tick_params(axis='both', which='major', pad=25, labelsize=30)

	# ensure that the axis ticks only show up on left of the plot.  
	t1away.xaxis.set_ticks_position('none') 
	t1away.yaxis.set_ticks_position('none') 		
	t2away.xaxis.set_ticks_position('none') 
	t2away.yaxis.set_ticks_position('none') 

	t1subj.xaxis.set_ticks_position('none') 
	t1subj.yaxis.set_ticks_position('none') 		
	t2subj.xaxis.set_ticks_position('none') 
	t2subj.yaxis.set_ticks_position('none') 		

	#plt.subplots_adjust(hspace = 1)
	#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

	leg1 = t1away.legend((rectsALLT1, rectsONET1), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), ncol=2, fontsize=sz, loc='upper center',bbox_to_anchor=(0.5, 1.5))
	#leg2 = t1subj.legend((rectsSubjT1ALL, rectsSubjT1ONE), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), ncol=2, fontsize=25, loc='upper center',bbox_to_anchor=(0.5, 1.3))

	leg1.get_frame().set_linewidth(0.0)
	leg1.get_frame().set_alpha(0)
	#leg2.get_frame().set_linewidth(0.0)
	#leg2.get_frame().set_alpha(0)

	# --------------------------#
	
	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		fig.savefig(datapath+"awayUndoing.pdf", bbox_inches="tight")
		print "Saved awayUndoing figure."

def plot_dotF_cupDiff_tableDiff(saveFig=False):
	"""
	Makes 3 side-by-side bar charts of dotfinal, cupdiff, tablediff plots
	"""
	idxCup = 17
	idxTable = 18
	(cupAvg,tableAvg,stdCup,stdTable) = getAvgs_Stds(idxCup, idxTable, oneMetric=False)

	idxDotF = 12
	(dotFAvg,stdDotF) = getAvgs_Stds(idxDotF, oneMetric=True)

	# ------- PLOTTING ---------#

	ind = np.arange(2)  # the x locations for the groups
	ind[1] = 2
	width = 0.8       # the width of the bars
	offset = 0.2

	# colors
	blackC = "black"	#(214/255., 39/255., 40/255.)
	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	fig, (dotF, cupDiffF, tableDiffF) = plt.subplots(1, 3, figsize=(26,15))

	# plot cupDiffFinal bar chart
	cupDiffFALL = cupDiffF.bar(ind+offset, cupAvg[0], width, color=greyC, yerr=stdCup[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	cupDiffFONE = cupDiffF.bar(ind+width+offset, cupAvg[1], width, color=orangeC, yerr=stdCup[1], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))

	tableDiffFALL = tableDiffF.bar(ind+offset, tableAvg[0], width, color=greyC, yerr=stdTable[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	tableDiffFONE = tableDiffF.bar(ind+width+offset, tableAvg[1], width, color=orangeC, yerr=stdTable[1], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))

	dotFALL = dotF.bar(ind+offset, dotFAvg[0], width, color=greyC, yerr=stdDotF[0], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	dotFONE = dotF.bar(ind+width+offset, dotFAvg[1], width, color=orangeC, yerr=stdDotF[1], ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))

	plt.subplots_adjust(left=0.12, bottom=None, right=1.2, top=None, wspace=0.3, hspace=None)

	format_subplot(cupDiffF, cupDiffFALL, cupDiffFONE, xlabel="Tasks", ylabel="CupDiffFinal", title="", indX=ind)
	format_subplot(tableDiffF, tableDiffFALL, tableDiffFONE, xlabel="Tasks", ylabel="TableDiffFinal", title="", indX=ind)
	format_subplot(dotF, dotFALL, dotFONE, xlabel="Tasks", ylabel="DotFinal", title="", indX=ind)


	leg = fig.legend((cupDiffFALL[0], cupDiffFONE[0]), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), ncol=2, fontsize=40, loc='upper center',bbox_to_anchor=(0.6, 0.68))
	leg.get_frame().set_linewidth(0.0)
	leg.get_frame().set_alpha(0)

	"""
	pval1 = [0.01,0.01]
	pval2 = [0.001,0.001]
	autolabel_star(cupDiffF, cupDiffFALL, stdCup[0], pval1)
	autolabel_star(cupDiffF, cupDiffFONE, stdCup[1], pval1, second=True)
	autolabel_star(tableDiffF, tableDiffFALL, stdTable[0],pval2, twostar=True)
	autolabel_star(tableDiffF, tableDiffFONE, stdTable[1],pval2, twostar=True, second=True)
	autolabel_star(dotF, dotFALL, stdDotF[0], pval2, twostar=True)
	autolabel_star(dotF, dotFONE, stdDotF[1], pval2, twostar=True, second=True)
	"""

	plt.title(r'\textbf{Final Learned Reward Metrics}', fontsize=50, x=-0.86, y=1.35)

	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/study/"
		datapath = here + subdir
		fig.savefig(datapath+"dotDiffFinal.pdf", bbox_inches="tight")
		print "Saved dotDiffFinal figure."

def autolabel_star(ax, rects, std, p, twostar=False, second=False):
	"""
	Attach a text label above each bar displaying its height
	second means the second error plot label
	"""
	for i in range(0,len(rects),2):
		height1 = rects[i].get_height()
		height2 = rects[i+1].get_height()
		height = max(height1,height2)
		print "height: " + str(height)
		stdh = max(std[i],std[i+1])*1.5
		print stdh
		x = (rects[i].get_x() + rects[i].get_width())			
		if second:
			x = (rects[i].get_x() + rects[i].get_width())*2.5

		y = stdh+height
		widthB = "widthB="+str((rects[i].get_width()+rects[i+1].get_width())*3)

		if twostar:
			yoffset = 1.2
			ax.annotate(r'\textbf{**}', xy=(x, y+yoffset), xytext=(x, y+yoffset), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, '+widthB+', lengthB=1.2', lw=1.5))
		else:		
			yoffset = 1.2
			ax.annotate(r'\textbf{*}', xy=(x, y), xytext=(x, y), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, '+widthB+', lengthB=1.2', lw=1.5))

		ax.text(x,y+yoffset,r"p$<$"+str(p[i]), ha='center', va='bottom', fontsize=30)


def autolabel(ax, rects):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%.2f' % 
				height,ha='center', va='bottom', fontsize=15)


def format_subplot(ax, rectsA, rectsB, xlabel, ylabel, title, indX=[0,0]):
	"""
	Sets up subplot (named ax) with al of the standard font n stuff
	"""
	width = 0.45       # the width of the bars
	offset = 0.15

	# fonts
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	# add some text for labels, title and axes ticks
	ax.set_ylabel(r'\textit{\textbf{'+ylabel+'}}',fontsize=40,labelpad=15)
	ax.set_xlabel(r'\textbf{'+xlabel+'}',fontsize=40,labelpad=15)
	
	plt.text(0.5, 1.08, r'\textbf{'+title+'}',
			 horizontalalignment='center',
			 fontsize=50,
			 transform = ax.transAxes)
	
	ax.set_xticks(indX+width+offset)

	xlabels = ["Table","Table + Cup"] #["T"+str(t+1) for t in range(3)]
	ax.set_xticklabels(xlabels,10,fontsize=40)

	# remove the plot frame lines
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)      
	
	# set max y-limit 
	#ax.set_ylim([0,maxY])
#	ax.tick_params(labelsize=40)
	ax.tick_params(axis='both', which='major', pad=15, labelsize=40)

	# set padding for x and y tick labels
	#ax.tick_params(direction='out', pad=2)

	# ensure that the axis ticks only show up on left of the plot.  
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 		
	
	#leg = ax.legend((rectsA[0], rectsB[0]), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), fontsize=30)
	#leg.get_frame().set_linewidth(0.0)
	#leg.get_frame().set_alpha(0)

	pos1 = ax.get_position() # get the original position 
	pos2 = [pos1.x0, pos1.y0,  pos1.width , pos1.height*0.5] 
	ax.set_position(pos2)



def getAvgs_Stds(idxCup, idxTable=None, oneMetric=False):
	"""
	Returns the averages and standard deviations for the Cup and Table metrics at 
	idxCup index and idxTable index in metrics_obj. 
	If oneMetric = True, then we are only considering one of the metrics, and 
	return just data on that. Assume that the idxCup has the right index, and idxTable = None
	"""
	filename = "metrics_obj.p"
	obj = get_pickled_metrics(filename)

	if oneMetric is False:
		# for keeping average of each feature, for each method, and for each task
		cupAvg = np.array([[0.0, 0.0],[0.0,0.0]]) # [method all --> [avg for task 1, task 2], method one --> [avg for task 1, task 2]]
		tableAvg = np.array([[0.0, 0.0],[0.0,0.0]]) 

		# for computing stddev 
		pplCupALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL) # trial 1 --> [task 1, task 2], trial 2 --> [task 1, task 2]
		pplCupONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
		pplTableALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
		pplTableONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)

		stdCup = np.array([[0.0,0.0]]*2)
		stdTable = np.array([[0.0,0.0]]*2)

		for ID in obj.keys():
			for task in obj[ID]:
				for trial in [1,2]:
					cup_all = obj[ID][task][trial]["A"][idxCup]
					cup_one = obj[ID][task][trial]["B"][idxCup]
					table_all = obj[ID][task][trial]["A"][idxTable]
					table_one = obj[ID][task][trial]["B"][idxTable]

					cupAvg[0][task-1] += cup_all
					cupAvg[1][task-1] += cup_one

					tableAvg[0][task-1] += table_all
					tableAvg[1][task-1] += table_one

					pplCupALL[ID][trial-1][task-1] = cup_all
					pplCupONE[ID][trial-1][task-1] = cup_one
					pplTableALL[ID][trial-1][task-1] = table_all
					pplTableONE[ID][trial-1][task-1] = table_one

		# average by number of participants
		for method in range(2):
			for task in range(2):
				cupAvg[method][task] /= NUM_PPL*2 # because 2 trials
				tableAvg[method][task] /= NUM_PPL*2
			
				if method == 0: # all method
					stdCup[method][task] = np.std(pplCupALL[:,:,task])/np.sqrt(NUM_PPL*2*2) # because 2 tasks, 2 trials (per method)
					stdTable[method][task] = np.std(pplTableALL[:,:,task])/np.sqrt(NUM_PPL*2*2)
				else: # one method
					stdCup[method][task] = np.std(pplCupONE[:,:,task])/np.sqrt(NUM_PPL*2*2)
					stdTable[method][task] = np.std(pplTableONE[:,:,task])/np.sqrt(NUM_PPL*2*2)

		return (cupAvg,tableAvg,stdCup,stdTable)
	else:
		# for keeping average of each feature, for each method, and for each task
		cupAvg = np.array([[0.0, 0.0],[0.0,0.0]]) # [method all --> [avg for task 1, task 2], method one --> [avg for task 1, task 2]]

		# for computing stddev 
		pplCupALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL) # trial 1 --> [task 1, task 2], trial 2 --> [task 1, task 2]
		pplCupONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)

		stdCup = np.array([[0.0,0.0]]*2)

		for ID in obj.keys():
			for task in obj[ID]:
				for trial in [1,2]:
					cup_all = obj[ID][task][trial]["A"][idxCup]
					cup_one = obj[ID][task][trial]["B"][idxCup]

					cupAvg[0][task-1] += cup_all
					cupAvg[1][task-1] += cup_one

					pplCupALL[ID][trial-1][task-1] = cup_all
					pplCupONE[ID][trial-1][task-1] = cup_one

		# average by number of participants
		for method in range(2):
			for task in range(2):
				cupAvg[method][task] /= NUM_PPL*2 # because 2 trials
			
				if method == 0: # all method
					stdCup[method][task] = np.std(pplCupALL[:,:,task])/np.sqrt(NUM_PPL*2*2) # because 2 tasks, 2 trials (per method)
				else: # one method
					stdCup[method][task] = np.std(pplCupONE[:,:,task])/np.sqrt(NUM_PPL*2*2)

		return (cupAvg,stdCup)
		

#--------- OprenRAVE plotting --------#

def plot_taskOpenrave(task):
	"""
	Plots the default and ideal trajectory for the task.
	"""

	blackC = [0.2,0.2,0.2] #[0,0,0] 	
	greyC =  [0.5,0.5,0.5]		
	blueC = [75/255.0,171/255.0,197/255.]
	orangeC = [247/255.,149/255.,69/255.0] 

	# total traj time
	T = 20.0

	ideal_w = [0.0,0.0]
	if task == 1:
		ideal_w = [0.0,1.0]
	elif task == 2:
		ideal_w = [1.0,1.0]

	if task == 2:
		# if two features are wrong, initialize the starting config badly (tilted cup)
		pick = copy.copy(pick_basic)
		pick[-1] = 200.0
	else:
		pick = copy.copy(pick_basic) 
	place = copy.copy(place_lower)

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod="MAX", numFeat=task)

	# --- ideal traj --- #
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, ideal_w, 0.0, T, 0.1, seed=None)	
	# plot ideal trajectory
	plotCupTraj(plan.env,plan.robot,plan.bodies,plan.waypts,color=blueC,increment=20)

	# --- default traj --- #
	default_w = [0.0,0.0]
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, default_w, 0.0, T, 0.1, seed=None)	
	# plot ideal trajectory
	plotCupTraj(plan.env,plan.robot,plan.bodies,plan.waypts,color=blackC,increment=20)

	time.sleep(30)

if __name__ == '__main__':

	# --- for plotting trajectory default and ideal ---- #
	#task = 1
	#plot_taskOpenrave(task)

	# --- for plotting objective metrics --- #
	#plot_cupTableDiffFinal(True)
	#plot_dotOverTime(T1=True, saveFig=True) 		# DONE
	#plot_undoingObjSubj(saveFig=True)				# DONE
	#plot_dotF_cupDiff_tableDiff(True)				# DONE
	plot_weights_average(2)
