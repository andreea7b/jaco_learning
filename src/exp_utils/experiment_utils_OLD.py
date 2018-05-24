import numpy as np
from numpy import linalg
from numpy import linspace
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import time
import scipy
import math

import logging
import copy

import csv
import os

import openrave_utils
from openrave_utils import *

import stats
from stats import *

NUM_PPL = 6
# if numbering of participants doesn't start at 0 (ex. starts at 2) then set the offset here for proper indexing
PPL_OFFSET = 2

class ExperimentUtils(object):
	
	def __init__(self):
		# stores waypoints of originally planned trajectory
		self.original_traj = None
		# stores trajectory as it is deformed
		self.deformed_traj = None
		# stores the list of positions as the robot is moving 
		# in the form [timestamp, j1, j2, ... , j7]
		self.tracked_traj = None

		# stores start and end time of the interaction
		self.startT = 0.0
		self.endT = 0.0

		# stores weights over time 
		# always in the form [timestamp, weight1, weight2, weight3]
		self.weights = None

		# stores running list of forces applied by human
		# in the form [timestamp, j1, j2, ... , j7]
		self.tauH = None 

	def update_original_traj(self, waypts):
		"""
		Updates the original trajectory
		"""
		self.original_traj = waypts

	def update_deformed_traj(self, waypts):
		"""
		Updates the deformed trajectory
		"""
		self.deformed_traj = waypts

	def update_tracked_traj(self, timestamp, curr_pos):
		"""
		Uses current position read from the robot to update the trajectory
		Saves timestamp when this position was read
		""" 
		currTraj = np.append([timestamp], curr_pos.reshape(7))
		if self.tracked_traj is None:
			self.tracked_traj = np.array([currTraj])
		else:	
			self.tracked_traj = np.vstack([self.tracked_traj, currTraj])
		
	def update_tauH(self, timestamp, tau_h):
		"""
		Uses current joint torque reading from the robot during interaction
		Saves timestamp when this torque was read
		""" 
		currTau = np.append([timestamp], tau_h.reshape(7))
		if self.tauH is None:
			self.tauH = np.array([currTau])
		else:
			self.tauH = np.vstack([self.tauH, currTau])

	def update_weights(self, timestamp, new_weight):
		"""
		Updates list of timestamped weights
		"""
		new_w = np.array([timestamp, new_weight[0], new_weight[1]])
		if self.weights is None:
			self.weights = np.array([new_w])
		else:
			self.weights = np.vstack([self.weights, new_w])
	
	def set_startT(self,start_t):
		"""
		Records start time for experiment
		"""
		self.startT = start_t

	def set_endT(self,end_t):
		"""
		Records end time for experiment
		"""
		self.endT = end_t

	# ---- Plotting Functionality ---- #

	def plotting(self, avgA, avgB, stdA, stdB, xlabel, ylabel, title, maxY, avgOpt=None, plotType="effort"):
		"""
		Basic plotting functionality
		----
		avgOpt		plots optimal in addition to experimental
		plotType 		"effort" or "time" or "cost"
			- effort		sets up graphing for task effort comparison
			- time			sets up graphing for task time comparison
			- cost 			sets up graphing for task cost comparison	
			- weight 		sets up graphing for weight comparison
		"""
		ind = np.arange(3)  # the x locations for the groups
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

		# plots with stdev
		if avgOpt is None:
			rectsA = ax.bar(ind+offset, avgA, width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
			rectsB = ax.bar(ind+width+offset, avgB, width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
		else: 
			width = 0.25
			rectsA = ax.bar(ind+offset, avgA, width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
			rectsB = ax.bar(ind+offset+width, avgB, width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
			rectsOpt = ax.bar(ind+offset+width*2, avgOpt, width, color=blueC,ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))		

		def autolabel(rects):
			"""
			Attach a text label above each bar displaying its height
			"""
			for rect in rects:
				height = rect.get_height()
				ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%.2f' % 
						height,ha='center', va='bottom', fontsize=15)

		def autolabel_star(rects,std,p,plotType):
			"""
			Attach a text label above each bar displaying its height
			"""

			for i in range(len(rects)):
				height = rects[i].get_height()

				# set up autolabel based on type=effort,time,cost
				if plotType is "effort":
					x = rects[i].get_x() + rects[i].get_width()
					y = std[i]+height+50
					xoffset = 0.0
					yoffset = 10.0
					astyle = '-[, widthB=2.5, lengthB=1.2'
					ax_xoffset = 0
					ax_yoffset = 60
				elif plotType is "time":
					x = rects[i].get_x() + rects[i].get_width()
					y = std[i]+height+1
					xoffset = 0.0
					yoffset = 0.1
					astyle = '-[, widthB=2.5, lengthB=1.2'
					ax_xoffset = 0
					ax_yoffset = 0.8
				elif plotType is "cost":
					x = rects[i].get_x() + rects[i].get_width()*1.5
					y = std[i]+height+10
					xoffset = -0.12
					yoffset = 0.8
					astyle = '-[, widthB=1.8, lengthB=1.2'
					ax_xoffset = -0.12
					ax_yoffset = 6
				elif plotType is "weight":
					x = rects[i].get_x() + rects[i].get_width()
					y = std[i]+height+0.5
					xoffset = 0.0
					yoffset = 1.0
					astyle = '-[, widthB=2.5, lengthB=1.2'
					ax_xoffset = 0
					ax_yoffset = 0.8

				ax.annotate(r'\textbf{*}', xy=(x,y), xytext=(x+xoffset,y+yoffset), xycoords='data', fontsize=18, ha='center', va='bottom',arrowprops=dict(arrowstyle=astyle, lw=1.5))
				ax.text(x+ax_xoffset,y+ax_yoffset,r"p$<$"+str(p[i]),ha='center', va='bottom', fontsize=15)

		#ptime = []
		#peffort = []
		#pcost = [0.001,0.001,0.001]
		#autolabel_star(rectsA,stdA,pcost,plotType)
		#autolabel(rectsB)

		# add some text for labels, title and axes ticks
		ax.set_ylabel(r'\textbf{'+ylabel+'}',fontsize=20,labelpad=15)
		ax.set_xlabel(r'\textbf{'+xlabel+'}',fontsize=20,labelpad=15)
		
		plt.text(0.5, 1.08, r'\textbf{'+title+'}',
				 horizontalalignment='center',
				 fontsize=25,
				 transform = ax.transAxes)
		
		ax.set_xticks(ind+width+offset)
		if avgOpt is not None:
			ax.set_xticks(ind+width+width/2+offset)

		xlabels = ["Cup","Table","Laptop"]
		ax.set_xticklabels(xlabels,10,fontsize=18)
 
		# remove the plot frame lines
		ax.spines["top"].set_visible(False)    
		ax.spines["right"].set_visible(False)      
		
		# set max y-limit 
		ax.set_ylim([0,maxY])
		ax.tick_params(labelsize=18)

		# set padding for x and y tick labels
		ax.tick_params(direction='out', pad=2)

		# ensure that the axis ticks only show up on left of the plot.  
		ax.xaxis.set_ticks_position('none') 
		ax.yaxis.set_ticks_position('none') 		


		leg = ax.legend((rectsA[0], rectsB[0]), (r'\textbf{Grav Comp}', r'\textbf{Impedance}'), fontsize=18)
		if avgOpt is not None:		
			leg = ax.legend((rectsA[0], rectsB[0], rectsOpt[0]), (r'\textbf{Grav Comp}', r'\textbf{Impedance}', r'\textbf{Desired}'), fontsize=18)

		leg.get_frame().set_linewidth(0.0)
		plt.show()

		return fig

	def plot_forceOverTime(self, filename, method="A",saveFig=False):
		"""
		Takes force file and plots human forces over time.
		"""
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/force/"
		datapath = here + subdir + filename		
		tauH = None
		with open(datapath, 'r') as f:
			methodData = [None]*8
			i = 0
			for line in f:
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				methodData[i] = final_values
				i += 1
			data = np.array(methodData)
			tauH = data

		(h,w) = np.shape(tauH)
		sumTau = np.zeros(w)
		for i in range(w):
			sumTau[i] = np.sum(np.abs(tauH[1:,i]))
		
		x = np.arange(0.0, 15.0, 0.1)
		y = [0.0]*len(x)
		count = 0
		time = tauH[0,:]
		print time
		t = time[count]
		for i in range(len(time)):
			val = int(np.floor(time[i]*10))
			y[val] = sumTau[i]

		# fonts
		rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
		rc('text', usetex=True)

		fig, ax = plt.subplots()
		fig.set_size_inches(30.0, 5.5, forward=True)

		title_text = r'\textbf{Human Interaction Effort With Impedance}'

		if method is "B":
			title_text = r'\textbf{Human Interaction Effort With Learning}'
		plt.text(0.5, 1.03, title_text,
				 horizontalalignment='center',
				 fontsize=25,
				 transform = ax.transAxes)
		
		# remove the plot frame lines
		ax.spines["top"].set_visible(False)    
		ax.spines["right"].set_visible(False)      
		
		# set max y-limit 
		ax.set_ylim([0,20])
		ax.tick_params(labelsize=18)

		# set padding for x and y tick labels
		ax.tick_params(direction='out', pad=2)

		# ensure that the axis ticks only show up on left of the plot.  
		ax.xaxis.set_ticks_position('none') 
		ax.yaxis.set_ticks_position('none') 		

		ax.set_xlabel(r'\textbf{Time (s)}',fontsize=20,labelpad=15)
		ax.set_ylabel(r'\textbf{Total Force (Nm)}',fontsize=20,labelpad=15)
		
		greyC = "grey"		
		orangeC = "#F79545"
		c = greyC
		if method is "B":
			c = orangeC

		plt.plot(x,y,color=c, linewidth=5)
		plt.show()

		if saveFig:
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/"
			datapath = here + subdir
			if method is "A":
				fig.savefig(datapath+"force_over_timeA.pdf", bbox_inches="tight")
			else:
				fig.savefig(datapath+"force_over_timeB.pdf", bbox_inches="tight")
			print "Saved effort figure." 


	def plot_taskEffort(self, filename, saveFig=False, saveName="taskEffort"):
		"""
		Takes all participant data files and produces bar chart
		comparing average force exerted by each participant for each task
		----
		saveFig 	if True, saves final plot
		saveName	name of saved output file (will be saved as .pdf)
		"""
		metrics = self.parse_metrics(filename)
		# store avg for trial 1,2,3
		sumA = [0.0,0.0,0.0]
		sumB = [0.0,0.0,0.0]

		pplA = [[0.0]*NUM_PPL, [0.0]*NUM_PPL, [0.0]*NUM_PPL]
		pplB = [[0.0]*NUM_PPL, [0.0]*NUM_PPL, [0.0]*NUM_PPL]
		for ID in metrics.keys():
			for task in metrics[ID]:
				trialAvgA = 0.0
				trialAvgB = 0.0
				for trial in metrics[ID][task]:
					trialAvgA += metrics[ID][task][trial]["A"][6]
					trialAvgB += metrics[ID][task][trial]["B"][6]
				trialAvgA /= 2.0
				trialAvgB /= 2.0
				sumA[task-1] += trialAvgA
				sumB[task-1] += trialAvgB
	
				pplA[task-1][ID-PPL_OFFSET] = trialAvgA
				pplB[task-1][ID-PPL_OFFSET] = trialAvgB
		avgA = [a/NUM_PPL for a in sumA]
		stdA = [np.std(pplA[0]), np.std(pplA[1]), np.std(pplA[2])]
		avgB = [b/NUM_PPL for b in sumB]
		stdB = [np.std(pplB[0]), np.std(pplB[1]), np.std(pplB[2])]

		# plot data
		xlabel = "Task"
		ylabel = "Total Effort (Nm)"
		title = "Average Total Human Effort"	
		maxY = 900	
		fig = self.plotting(avgA,avgB,stdA,stdB,xlabel,ylabel,title,maxY)

		if saveFig:
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/"
			datapath = here + subdir
			fig.savefig(datapath+saveName+".pdf", bbox_inches="tight")
			print "Saved effort figure." 

	def plot_taskEffortTime(self, filename, saveFig=False, saveName="taskEffortTime"):
		"""
		Takes all participant data files and produces bar chart
		comparing average interaction time for each task
		----
		saveFig 	if True, saves final plot
		saveName	name of saved output file (will be saved as .pdf)
		"""
		metrics = self.parse_metrics(filename)
		# store avg for trial 1,2,3
		sumA = [0.0,0.0,0.0]
		sumB = [0.0,0.0,0.0]

		pplA = [[0.0]*NUM_PPL, [0.0]*NUM_PPL, [0.0]*NUM_PPL]
		pplB = [[0.0]*NUM_PPL, [0.0]*NUM_PPL, [0.0]*NUM_PPL]
		for ID in metrics.keys():
			for task in metrics[ID]:
				trialAvgA = 0.0
				trialAvgB = 0.0
				for trial in metrics[ID][task]:
					trialAvgA += metrics[ID][task][trial]["A"][7]
					trialAvgB += metrics[ID][task][trial]["B"][7]
				trialAvgA /= 2.0
				trialAvgB /= 2.0
				sumA[task-1] += trialAvgA
				sumB[task-1] += trialAvgB
	
				pplA[task-1][ID-PPL_OFFSET] = trialAvgA
				pplB[task-1][ID-PPL_OFFSET] = trialAvgB
		avgA = [a/NUM_PPL for a in sumA]
		stdA = [np.std(pplA[0]), np.std(pplA[1]), np.std(pplA[2])]
		avgB = [b/NUM_PPL for b in sumB]
		stdB = [np.std(pplB[0]), np.std(pplB[1]), np.std(pplB[2])]

		# plot data
		xlabel = "Task"
		ylabel = "Interact Time (s)"
		title = "Average Total Interaction Time"	
		maxY = 20.0
		fig = self.plotting(avgA,avgB,stdA,stdB,xlabel,ylabel,title,maxY)

		if saveFig:
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/"
			datapath = here + subdir
			fig.savefig(datapath+saveName+".pdf", bbox_inches="tight")
			print "Saved time figure."


	def plot_taskCost(self, filename, saveFig=False, saveName="taskCost"):
		"""
		Takes all participant data files and produces bar chart
		comparing cumulative cost for each task
		----
		saveFig 	if True, saves final plot
		saveName	name of saved output file (will be saved as .pdf)
		"""
		metrics = self.parse_metrics(filename)
		# store avg for trial 1,2,3
	
		optimal = [0.0,0.0,0.0]
		sumA = [0.0,0.0,0.0]
		sumB = [0.0,0.0,0.0]

		pplA = [[0.0]*NUM_PPL, [0.0]*NUM_PPL, [0.0]*NUM_PPL]
		pplB = [[0.0]*NUM_PPL, [0.0]*NUM_PPL, [0.0]*NUM_PPL]
		for ID in metrics.keys():
			for task in metrics[ID]:
				trialAvgA = 0.0
				trialAvgB = 0.0
				for trial in metrics[ID][task]:
					trialAvgA += metrics[ID][task][trial]["A"][1]
					trialAvgB += metrics[ID][task][trial]["B"][1]
				# doesn't matter which one you choose, optimal is always opt
				optimal[task-1] = metrics[ID][task][1]["B"][3]

				trialAvgA /= 2.0
				trialAvgB /= 2.0
				sumA[task-1] += trialAvgA
				sumB[task-1] += trialAvgB
	
				pplA[task-1][ID-PPL_OFFSET] = trialAvgA
				pplB[task-1][ID-PPL_OFFSET] = trialAvgB
		avgA = [a/NUM_PPL for a in sumA]
		stdA = [np.std(pplA[0]), np.std(pplA[1]), np.std(pplA[2])]
		avgB = [b/NUM_PPL for b in sumB]
		stdB = [np.std(pplB[0]), np.std(pplB[1]), np.std(pplB[2])]

		xlabel = "Task"
		ylabel = r"Cost Value"
		title = r"Average Cost Across Tasks"	
		maxY = 80.0
		fig = self.plotting(avgA,avgB,stdA,stdB,xlabel,ylabel,title,maxY,optimal)

		if saveFig:
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/"
			datapath = here + subdir
			fig.savefig(datapath+saveName+".pdf", bbox_inches="tight")
			print "Saved cost figure."


	# ---- OpenRAVE Plotting ---- #

	def upsample(self, waypts_plan, start_time, final_time, step_time):
		"""
		Put waypoints along trajectory at step_time increments.
		---
		input desired time increment, update upsampled trajectory
		"""
		num_waypts = int(math.ceil((final_time - start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts

		num_waypts_plan = 4 # len(waypts_plan)
		step_time_plan = (final_time - start_time)/(num_waypts_plan - 1)
		t = start_time
		for i in range(num_waypts):
			if t >= final_time:
				waypts_time[i] = final_time
				waypts[i,:] = waypts_plan[num_waypts_plan - 1]
			else:
				deltaT = t - start_time
				prev_idx = int(deltaT/step_time_plan)
				print prev_idx	
				print num_waypts			
				prev = waypts_plan[prev_idx]
				next = waypts_plan[prev_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = prev+((t-prev_idx*step_time_plan)/step_time_plan)*(next-prev)
			t += step_time
		return waypts

	def plot_traj(self, dataType, filename):

		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + dataType

		validTypes = ["deformed", "tracked", "original"]
		if dataType not in validTypes:
			print dataType + " is not a valid trajectory data type!"
			return None

		data = {}
		with open(os.path.join(datapath, filename), 'r') as f:
			# account for differently structured data between trajectories
			if dataType is not "tracked":
				lines = f.readlines()
			elif dataType is "tracked":
				 lines = f.readlines()[2:]
			jnum = 0
			for line in lines:
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				data[jnum] = final_values
				jnum += 1

		# get waypoints from file data
		waypts = np.zeros((len(data[0]),7))
		for i in range(len(data[0])):
			jangles = []
			for j in range(7):
				jangles.append(data[j][i])
			waypts[i] = np.array(jangles)		

		print waypts		
		# load openrave
		model_filename = 'jaco_dynamics'
		env, robot = initialize(model_filename)

		# plot saved waypoints
		bodies = []
		plotTraj(env,robot,bodies,waypts)
		plotTable(env)
		plotTableMount(env,bodies)
		plotLaptop(env,bodies)
		plotCabinet(env)

		time.sleep(25)

	def plot_allTraj(self, ID, task, method, trial):
		"""
		Plots all three trajectories (tracked, deformed, original)
		for specified participant and trial and task with method
		"""
	
		#if method is "B":
		#	file1 = "deformed" + str(ID) + str(task) + method + str(trial) + ".csv"
		#	# parse waypts from file
		#	waypts = self.parse_traj("deformed", file1)
		#	waypts = self.upsample(waypts, 0.0, 15.0, 0.1)

		# parse waypts from file
		file2 = "tracked" + str(ID) + str(task) + method + str(trial) + ".csv"
		waypts2 = self.parse_traj("tracked", file2)

		file3 = "original" + str(ID) + str(task) + method + str(trial) + ".csv"
		# parse waypts from file
		waypts3 = self.parse_traj("original", file3)
		waypts3 = self.upsample(waypts3, 0.0, 15.0, 0.1)

		# load openrave
		model_filename = 'jaco_dynamics'
		env, robot = initialize(model_filename)

		# plot saved waypoints
		bodies = []		
		#if method is "B":
		#	plotTraj(env,robot,bodies,waypts)
		plotTraj(env,robot,bodies,waypts2,color=[1, 0, 0])
		plotTraj(env,robot,bodies,waypts3,color=[0, 0, 1])
		plotTable(env)
		plotTableMount(env,bodies)
		plotLaptop(env,bodies)
		#plotCabinet(env)

		time.sleep(30)

	def plot_ABOptTraj(self, waypts_opt, filenameA, filenameB, tasknum):
		"""
		Plots measured trajectory for Method A, Method B and optimal trajectory
		given certain task
		"""
		# get ground truth for task 2 only!!!
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/tracked/"
		datapathA = here + subdir + filenameA
		datapathB = here + subdir + filenameB
		firstline = True
		wayptsA = None
		with open(datapathA, 'r') as f:
			methodData = [None]*8
			i = 0
			for line in f:
				# skip first line in tracked that has totalT
				if firstline:
					firstline = False
					continue
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				methodData[i] = final_values
				i += 1
			data = np.array(methodData)
			wayptsA = data

		firstline = True
		wayptsB = None
		with open(datapathB, 'r') as f:
			methodData = [None]*8
			i = 0
			for line in f:
				# skip first line in tracked that has totalT
				if firstline:
					firstline = False
					continue
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				methodData[i] = final_values
				i += 1
			data = np.array(methodData)
			wayptsB = data

		# plot saved waypoints
		bodies = []	
		plan = Planner(tasknum)

		blackC = [0,0,0] 	
		greyC =  [0.5,0.5,0.5]		
		blueC = [75/255.0,171/255.0,197/255.]
		orangeC = [247/255.,149/255.,69/255.0] 

		# impedance
		#plotTraj(plan.env,plan.robot,bodies,wayptsA[1:8].T, color=greyC)
		# learning
		plotTraj(plan.env,plan.robot,bodies,wayptsB[1:8].T, color=orangeC)
		# optimal
		plotTraj(plan.env,plan.robot,bodies,waypts_opt,color=blueC)
		# original
		#file3 = "original0" + str(tasknum) + "A1.csv"
		# parse waypts from file
		#waypts_orig = self.parse_traj("original", file3)
		#waypts_orig = self.upsample(waypts_orig, 0.0, 15.0, 0.1)
		#plotTraj(plan.env,plan.robot,bodies,waypts_orig,color=blackC)

		time.sleep(50)

	# ---- Trajectory Analysis Functionality ----#

	def featurize_traj(self, dataType, filename):
		"""
		Takes trajectory file and featurizes it. 
		"""
		# parse waypts from file
		waypts = self.parse_traj(dataType, filename)

		# get task number
		values = filename.split(dataType)
		taskNum = int(values[1][1])

		plan = Planner(taskNum)		
		features = plan.featurize(waypts)
		return features	

	# ---- I/O Functionality ---- #

	def parse_traj(self, dataType, filename):
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + dataType

		validTypes = ["deformed", "tracked", "original"]
		if dataType not in validTypes:
			print dataType + " is not a valid trajectory data type!"
			return None

		data = {}
		with open(os.path.join(datapath, filename), 'r') as f:
			# account for differently structured data between trajectories
			if dataType is not "tracked":
				lines = f.readlines()
			elif dataType is "tracked":
				 lines = f.readlines()[2:]
			jnum = 0
			for line in lines:
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				data[jnum] = final_values
				jnum += 1

		# get waypoints from file data
		waypts = np.zeros((len(data[0]),7))
		for i in range(len(data[0])):
			jangles = []
			for j in range(7):
				jangles.append(data[j][i])
			waypts[i] = np.array(jangles)		

		print "parsed trajectory: " + str(waypts)
		return waypts

	def parse_metrics(self, filename):
		"""
		Parses cleaned CSV file with metrics per participant.
		filename can be "metrics_obj.csv" or "metrics_subj.csv"
		"""

		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + filename

		data = {}
		firstline = True
		with open(datapath, 'r') as f:
			for line in f:
				# skip first line in tracked that has totalT
				if firstline:
					firstline = False
					continue
				values = line.split(',')
				ID = int(values[0][1])
				task = int(values[1][1])
				trial = int(values[2])
				method = values[3]
				# sanity check if participant ID and experi# exist already
				if ID not in data:
					data[ID] = {}
				if task not in data[ID]:
					data[ID][task] = {}
				if trial not in data[ID][task]:
					data[ID][task][trial] = {}
				#if method not in data[ID][task][trial]:
				data[ID][task][trial][method] = [float(v) for v in values[4:len(values)]]
		return data

	def parse_data(self, dataType):
		"""
		Parses a set of CSV files from multiple participants and returns
		an aggregate array of the data for all participants across all trials

		Returns:
		ID 1 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		ID 2 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		...
		ID N --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		---
		dataType - can be 'force' or 'traj'
		"""

		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + dataType

		validTypes = ["force", "weights", "deformed", "tracked", "original"]
		if dataType not in validTypes:
			print dataType + " is not a valid data type!"
			return None

		data = {}
		for filename in os.listdir(datapath):
			print filename
			info = filename.split(dataType)[1]
			ID = int(info[0])
			task = int(info[1])
			methodType = info[2]
			trial = int(info[3])

			# sanity check if participant ID and experi# exist already
			if ID not in data:
				data[ID] = {}
			if task not in data[ID]:
				data[ID][task] = {}
			if trial not in data[ID][task]:
				data[ID][task][trial] = {}
			with open(os.path.join(datapath, filename), 'r') as f:
				trajTypes = ["deformed", "original"]

				methodData = [None]*8
				if dataType == "weights": 
					methodData = [None]*2
				elif dataType in ["deformed", "original"]:
					methodData = [None]*7
				elif dataType is "tracked":
					methodData = [None]*8
				firstLine = True
				i = 0
				for line in f:
					# skip first line in tracked that has totalT
					if dataType is "tracked" and firstLine:
						firstLine = False
						continue
					values = line.split(',')
					final_values = [float(v) for v in values[1:len(values)]]
					methodData[i] = final_values
					i += 1
				#print methodData
				data[ID][task][trial][methodType] = np.array(methodData)

		return data				
		

	def save_tauH(self, filename):
		"""
		Saves the human-applied force data to CSV file. 
		"""	
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/force/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

	
		print "tau: " + str(self.tauH)
		#print "len(tau): " + str(len(self.tauH))

		with open(filepath, 'w') as out:
			if self.tauH is not None:
				for i in range(len(self.tauH.T)):
					if i == 0:
						out.write('time')
					else:
						out.write('tau_j'+str(i))
					for j in range(len(self.tauH.T[i])):
						out.write(',%f' % self.tauH.T[i][j])
					out.write('\n')
		out.close()

	def save_weights(self, filename):
		"""
		Saves the weights over time to CSV file. 
		"""	
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/weights/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

		print "weights: " + str(self.weights[:,0])
		#print "len(w): " + str(len(self.weights[:,0]))

		with open(filepath, 'w') as out:
			out.write('time')
			for i in range(len(self.weights[:,0])):
				out.write(',%f' % self.weights[i][0])		
			out.write('\n')					
			out.write('weights1')
			for i in range(len(self.weights[:,1])):
				out.write(',%f' % self.weights[i][1])
			out.write('\n')	
			out.write('weights2')
			for i in range(len(self.weights[:,2])):
				out.write(',%f' % self.weights[i][2])
			out.write('\n')
		out.close()

	def save_original_traj(self, filename):
		"""
		Saves the originally planned trajectory to CSV file.
		"""

		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/original/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

		with open(filepath, 'w') as out:
			for j in range(7):
				out.write('j%d' % j)
				for pt in range(len(self.original_traj)):
					out.write(',%f' % self.original_traj[pt][j])
				out.write('\n')
		out.close()

	def save_deformed_traj(self, filename):
		"""
		Saves the deformed trajectory to CSV file.
		"""

		if self.deformed_traj is not None:
			# get the current script path
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/deformed/"
			filepath = here + subdir + filename + "1.csv"

			i = 2
			while os.path.exists(filepath):
				filepath = here+subdir+filename+str(i)+".csv"
				i+=1

			with open(filepath, 'w') as out:
				for j in range(7):
					out.write('j%d' % j)
					for pt in range(len(self.deformed_traj)):
						out.write(',%f' % self.deformed_traj[pt][j])
					out.write('\n')
			out.close()
		else:
			print "No deformed trajectory to write."

	def save_tracked_traj(self, filename):
		"""
		Saves the measured positions of the trajectory to CSV file. 
		"""	

		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/tracked/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

		print "sT: " + str(self.startT)
		print "eT: " + str(self.endT)
		with open(filepath, 'w') as out:
			out.write('total_trajT: %f\n' % (self.endT-self.startT))
			out.write('time')
			for t in range(len(self.tracked_traj[:,0])):
				out.write(',%f' % self.tracked_traj[t][0])		
			out.write('\n')					
			for j in range(1,8):
				out.write('j%d' % j)
				for waypt in self.tracked_traj[:,j]:
					out.write(',%f' % waypt)
				out.write('\n')
		out.close()

if __name__ == '__main__':
	"""
	NOTE: Make sure to change the number of participants at the top! and the offset!
	"""

	experi = ExperimentUtils()
	#task = 2
	#opt_waypts = get_opt_waypts("task2.csv")
	#filenameA = "tracked12A1.csv"
	#filenameB = "tracked12B1.csv"
	#experi.plot_ABOptTraj(opt_waypts,filenameA,filenameB, task)

	#filename1 = "force12A1.csv"
	#filename2 = "force12B1.csv"
	#experi.plot_forceOverTime(filename2, method="B", saveFig=True)
	
	filename = "pilot_metrics_obj_ALL.csv"
	#experi.plot_taskEffort(filename, saveFig=True,saveName="taskEffort_ALL")
	#experi.plot_taskEffortTime(filename, saveFig=True,saveName="taskEffortTime_ALL")	
	experi.plot_taskCost(filename, saveFig=True,saveName="taskCost_ALL")

	#dataType = "tracked"	
	#filename = "tracked53B1.csv"
	#experi.plot_traj(dataType, filename)

