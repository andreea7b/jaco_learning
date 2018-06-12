import numpy as np
from numpy import linalg
from numpy import linspace
from matplotlib import rc
import matplotlib.pyplot as plt
import time
import scipy
import math
import logging
import copy

import os
import pickle
import data_io

import openrave_utils
from openrave_utils import *

import experiment_utils
from experiment_utils import *

import trajopt_planner

# possible range of feature values for each feature
CUP_RANGE = 1.87608702
TABLE_RANGE = 0.6918574
LAPTOP_RANGE = 1.00476554

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

TABLE_TABLE = 1
COFFEE_TABLE = 2
COFFEE_COFFEE = 3
TABLE_COFFEE = 4

ALL = "ALL"
BETA = "BETA"
EXP_TASK = 2

NUM_PPL = 12					# number of participants

# ------- Saves out cleaned and computed statistics ------# 

def	save_parsed_data(filename, csvData=True, pickleData=False):
	"""
	Restructures all data from participants into a single file for
	objective measures and subjective measures.
	-----
	Give the filename you want to save it as
	If you want to pickle or not
	"""
	obj_metrics = compute_obj_metrics()
	subj_metrics = compute_subj_metrics()

	# write to file
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/study/"

	if pickleData:
		filepath_obj = here + subdir + filename + "_obj.p"
		pickle.dump(obj_metrics, open( filepath_obj, "wb" ) )
		filepath_subj = here + subdir + filename + "_subj.p"
		pickle.dump(subj_metrics, open( filepath_subj, "wb" ) )

	if csvData:
		filepath_obj = here + subdir + filename + "_obj.csv"
		filepath_subj = here + subdir + filename + "_subj.csv"

        # write objective metrics
		with open(filepath_obj, 'w') as out_obj:
			header = "participant,task,attempt,method,"
			header += "Rcup,Rtable,Rcup*,Rtable*,RcD,RtD,iactForce,iactTime,"
			header += "DotAvg,DotFinal,AngleFinal,L2Final,CupAway,TableAway,CupDiffFinal,TableDiffFinal,"
			header +=  "CupWeightPath,TableWeightPath,WeightPath,CupDiff,TableDiff,RegretFinal,AngleAvg,Regret\n"
			out_obj.write(header)
			# participant ID can take values 0 - 11
			for ID in obj_metrics.keys():
				for task in obj_metrics[ID]:
					# trial can take values 1 or 2
					for trial in obj_metrics[ID][task]:
						for method in obj_metrics[ID][task][trial]:
							row = "P"+str(ID)+",T"+str(task)+","+str(trial)+","+method
							out_obj.write(row)
							for num in obj_metrics[ID][task][trial][method]:
								out_obj.write(","+str(num))
							out_obj.write('\n')
		out_obj.close()

        # write subjective metrics
		with open(filepath_subj, 'w') as out_subj:
			header = "participant,task,method,age,gender,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8\n"
			out_subj.write(header)
			for ID in subj_metrics.keys():
				for task in subj_metrics[ID]:
					for method in subj_metrics[ID][task]:
						row = "P"+str(ID)+","+str(task)+","+method
						out_subj.write(row)
						for num in subj_metrics[ID][task][method]:
							out_subj.write(","+str(num))
			out_subj.write('\n')
		out_subj.close()
	

# ------ Creates large dictionary of all relevant statistics -----------#

def compute_obj_metrics():
	"""
	Computes the optimal reward, force, and total interaction time for all
	participants across all trials and all experimental conditions.
	"""
	# each participant does task 1 (TABLE_TABLE), 2 (COFFEE_TABLE), 3 (COFFEE_COFFEE), 4 (TABLE_COFFEE) 
	# and has attempt 1,2 with Method A (ALL), B (BETA) = 4*4*N
	# objective metrics: optimal reward, avg_force, weight_metric, total iact time = 4 
	# participant ID can take values 0 - 11

	effortData = data_io.parse_exp_data("force")
	trackedData = data_io.parse_exp_data("tracked")
	weightData = data_io.parse_exp_data("weights")

	obj_metrics = {}

	# compute effort metrics
	for ID in effortData.keys():
		for task in effortData[ID]:
			# trial can take values 1 or 2
			for trial in effortData[ID][task]:
				for method in effortData[ID][task][trial]:
					# sanity checks
						if ID not in obj_metrics:
							obj_metrics[ID] = {}
						if task not in obj_metrics[ID]:
							obj_metrics[ID][task] = {}
						if trial not in obj_metrics[ID][task]:
							obj_metrics[ID][task][trial] = {}
						if method not in obj_metrics[ID][task][trial]:
							# stores: 
							# Rvel,Rcup,Rtable,Rvel*,Rcup*,Rtable*,RvD,RcD,RtD,iactForce,iactTime,
							# DotAvg,DotFinal,AngleFinal,L2Final,CupAway,TableAway,CupDiffFinal,TableDiffFinal,WeightPath,
							# CupWeightPath,TableWeightPath,CupDiff,TableDiff,RegretFinal,AngleAvg,Regret
							obj_metrics[ID][task][trial][method] = np.array([0.0]*27)

						# --- Compute Effort & Interact Time Metrics ---#
						data = effortData[ID][task][trial][method]
						effort = compute_effort(data,ID)
						iactT = compute_iactT(data,ID)
						obj_metrics[ID][task][trial][method][9] = effort
						obj_metrics[ID][task][trial][method][10] = iactT

						# --- Compute Weight Metrics ---#
						wdata = weightData[ID][task][trial][method]
						weight_dotAvg = compute_weightDot(wdata,task)
						weight_dotF = compute_weightFinalDot(wdata,task)
						weight_angleF = compute_weightFinalAngle(wdata,task)
						weight_l2diff = compute_weightL2FinalDiff(wdata,task)
						(cup_score, table_score) = compute_awayScore(wdata,task)

						obj_metrics[ID][task][trial][method][11] = weight_dotAvg
						obj_metrics[ID][task][trial][method][12] = weight_dotF
						obj_metrics[ID][task][trial][method][13] = weight_angleF
						obj_metrics[ID][task][trial][method][14] = weight_l2diff
						obj_metrics[ID][task][trial][method][15] = cup_score
						obj_metrics[ID][task][trial][method][16] = table_score

						# --- Compute difference between the traj from learned weights and optimal --- #
						(diffFinal_cup, diffFinal_table) = compute_rewardFinalDiff(wdata, task)
						obj_metrics[ID][task][trial][method][17] = diffFinal_cup
						obj_metrics[ID][task][trial][method][18] = diffFinal_table

						# --- Compute weight path length --- #
						(cup_pathLength, table_pathLength) = compute_weightFeatPathLength(wdata)
						weight_path = compute_weightPathLength(wdata)
						obj_metrics[ID][task][trial][method][19] = cup_pathLength
						obj_metrics[ID][task][trial][method][20] = table_pathLength
						obj_metrics[ID][task][trial][method][21] = weight_path

						# --- Compute total final regret --- #
						regret_final = compute_rewardRegretFinal(wdata, task)
						obj_metrics[ID][task][trial][method][24] = regret_final

						# --- Weight angle avg over time --- #
						weight_angleAvg = compute_weightAngleAvg(wdata,task)
						obj_metrics[ID][task][trial][method][25] = weight_angleAvg
	
	# compute tracked trajectory metrics
	for ID in trackedData.keys():
		for task in trackedData[ID]:
			# compute optimal reward
			(Rvel_opt, Rcup_opt, Rtable_opt) = get_optimalReward(task)
			for trial in trackedData[ID][task]:
				for method in trackedData[ID][task][trial]:
					# --- Compute Reward ---#
					if method == "A":
						fmethod = ALL
					elif method == "B":
						fmethod = MAX
					plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod=fmethod, numFeat=task)	
					data = trackedData[ID][task][trial][method]
					# data is: [[time1, j1, j2, j3, ... j7], [timeN, j1, j2, j3, ... j7]]
					(Rvel, Rcup, Rtable) = compute_reward(data, plan)
					# --- Store metrics ---#
					obj_metrics[ID][task][trial][method][0] = Rvel
					obj_metrics[ID][task][trial][method][1] = Rcup
					obj_metrics[ID][task][trial][method][2] = Rtable
					obj_metrics[ID][task][trial][method][3] = Rvel_opt
					obj_metrics[ID][task][trial][method][4] = Rcup_opt
					obj_metrics[ID][task][trial][method][5] = Rtable_opt
					obj_metrics[ID][task][trial][method][6] = np.fabs(Rvel_opt - Rvel)
					obj_metrics[ID][task][trial][method][7] = np.fabs(Rcup_opt - Rcup)
					obj_metrics[ID][task][trial][method][8] = np.fabs(Rtable_opt - Rtable)

					# --- Compute difference between the traj from tracked traj --- #
					(diffTracked_cup, diffTracked_table) = compute_rewardTrackedDiff(Rcup, Rtable, Rcup_opt, Rtable_opt)
					obj_metrics[ID][task][trial][method][22] = diffTracked_cup
					obj_metrics[ID][task][trial][method][23] = diffTracked_table

					# --- Compute regret of traj executed by robot --- #
					tracked_regret = compute_rewardRegretTracked(Rcup, Rtable, Rcup_opt, Rtable_opt, task)
					obj_metrics[ID][task][trial][method][26] = tracked_regret

				
					plan.kill_planner()

	return obj_metrics

def compute_subj_metrics():
	"""
	Computes all subjective metric Likert data.
	"""
	# each participant does task 1,2 with method A,B = 2*2*N
	# likert data includes age, gender, Q1 - Q10 = 2+10 = 12

	# set up data structure
	subj_metrics = {}
	for ID in range(NUM_PPL):
		for task in [ONE_FEAT, TWO_FEAT]:
			for method in ["A","B"]:
				# sanity checks
				if ID not in subj_metrics:
					subj_metrics[ID] = {}
				if task not in subj_metrics[ID]:
					subj_metrics[ID][task] = {}
				if method not in subj_metrics[ID][task]:
					subj_metrics[ID][task][method] = [None]*12

	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	datapath = here + subdir + "likert_responses_clean.csv"

	data = {}
	firstline = True
	with open(datapath, 'r') as f:
		for line in f:
			if firstline:
				firstline = False
				continue
			values = line.split(',')
			ID = int(values[0])
			task = int(values[2].split(' ')[1]) # Comes as 'Task 1', split on space, take number
			if values[3] == "ONE":
				method = "B"
			elif values[3] == "ALL":
				method = "A"
			age = int(values[4])
			gender = values[5]
			techbg = values[6]
			# store age
			subj_metrics[ID][task][method][0] = age
			subj_metrics[ID][task][method][1] = gender
			# parse likert data
			for i in range(10):
				subj_metrics[ID][task][method][i+2] = values[i+7]
			
	return subj_metrics

# ------ Utils ------ #

def get_optimalReward(task):
	"""
	Returns the optimal reward for given task. Precomputed. 
	"""
	if task == ONE_FEAT: # one feature task
		return (0.0252667811742, 67.9221456532, 41.5027278248)  
	elif task == TWO_FEAT: # two feature task
		return (0.0521496254811, 117.782620548, 61.2207211307)  
	else:
		print "wrong task number!"
		return (0,0,0)

def compute_optimalReward(task):
	"""
	Computes optimal reward from scratch, given task. 
	"""
	T = 20.0
	weights = [0.0,0.0]
	if task == ONE_FEAT:
		weights = [0.0,1.0]
	elif task == TWO_FEAT:
		weights = [1.0,1.0]

	if task == TWO_FEAT:
		# if two features are wrong, initialize the starting config badly (tilted cup)
		pick = copy.copy(pick_basic)
		pick[-1] = 200.0
	else:
		pick = copy.copy(pick_basic) 
	place = copy.copy(place_lower)

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	print "computing optimal reward"
	plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=task)
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, weights, 0.0, T, 0.1, seed=None)	
	# use the upsampled traj from planner
	r = plan.featurize(plan.waypts)
	Rvel = r[0]
	Rfeat1 = np.sum(r[1])
	Rfeat2 = np.sum(r[2])
	print Rvel, Rfeat1, Rfeat2
	plan.kill_planner()
	return (Rvel, Rfeat1, Rfeat2)

def compute_effort(data,ID):
	"""
	Given one participant's force measurements for one trial of one experiment
	computes the total interaction time
	-----
	IMPT NOTE: 
	Participants with ID 0-7 had a script where it accidentally recorded the same
	interaction measurement 2 times! If you are analyzing one of those participants
	just count every other measurement. 
	"""
	# get only the data (no timestamps)
	edata = data[:,1:8]
	effort = 0.0
	if ID in range(0,8):
		#print "incr 2 id is " + str(ID)
		incr = 2
	else:
		#print "incr 1 id is " + str(ID)
		incr = 1
	for t in range(0,len(edata),incr):
		joint = edata[t]
		total = np.sum(np.abs(joint))
		effort += total

	return effort

def compute_iactT(data, ID):
	"""
	Given one participant's force measurements for one trial of one experiment
	computes the total interaction time	
	-----
	IMPT NOTE: 
	Participants with ID 0-7 had a script where it accidentally recorded the same
	interaction measurement 2 times! If you are analyzing one of those participants
	just count every other measurement. 
	"""
	time = data[:,0:1]
	if ID in range(0,8):
		count = len(time)/2.0
	else:
		count = len(time)
	# get only the timestamps
	totalT = count*0.1

	return totalT

def compute_reward(data, planner):
	"""
	Given one participant's tracked trajectory for one trial of one experiment
	computes the rewards from this trajectory
	"""
	# get only waypt data (no timestamps)
	waypts = data[:,1:8]

	r = planner.featurize(waypts)
	Rvel = r[0]
	Rcup = np.sum(r[1])
	Rtable = np.sum(r[2])
	#print "Rvel:" + str(Rvel)
	#print "Rcup:" + str(Rcup)
	return (Rvel, Rcup, Rtable)

def compute_rewardTrackedDiff(Rcup, Rtable, Rcup_opt, Rtable_opt):
	"""
	Given the tracked trajectory, abs value of reward difference between 
	that trajectory and the	optimal trajectory.
	"""

	diffTracked_cup = np.fabs(Rcup - Rcup_opt)
	diffTracked_table = np.fabs(Rtable - Rtable_opt)

	return (diffTracked_cup, diffTracked_table)

def compute_rewardRegretTracked(Rcup, Rtable, Rcup_opt, Rtable_opt, task):
	"""
	Given the rewards of the tracked trajectory and the optimal one, 
	compute the total regret by weighting the feature count diff by ideal weights
	"""
	ideal_w = np.array([0.0,0.0])
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	feat_des = np.array([Rcup_opt, Rtable_opt])
	feat_tracked = np.array([Rcup, Rtable])

	regret = np.dot(ideal_w, feat_tracked) - np.dot(ideal_w, feat_des)

	return regret

def compute_rewardRegretFinal(weightdata, task):
	"""
	Given the final learned weight, compute the trajectory with those weights
	then compute the abs value of reward difference between that trajectory and the
	optimal trajectory.
	"""
	ideal_w = [0.0,0.0]
	if task == ONE_FEAT:
		ideal_w = [0.0,1.0]
	elif task == TWO_FEAT:
		ideal_w = [1.0,1.0]

	timestamp = weightdata[:,0:1]
	weights = weightdata[:,1:len(weightdata)+1]

	final_w = weights[-1]

	#print "in compute_rewardRegret - task: " + str(task)

	T = 20.0
	if task == TWO_FEAT:
		# if two features are wrong, initialize the starting config badly (tilted cup)
		pick = copy.copy(pick_basic)
		pick[-1] = 200.0
	else:
		pick = copy.copy(pick_basic) 

	place = copy.copy(place_lower)

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	#print "in rewardRegret - start: " + str(start)
	plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=task)
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, final_w, 0.0, T, 0.1, seed=None)	
	# compute reward of current traj with final learned weights
	r = plan.featurize(plan.waypts)
	Rvel = r[0]
	Rcup = np.sum(r[1])
	Rtable = np.sum(r[2])

	# get optimal reward
	(Rvel_opt, Rcup_opt, Rtable_opt) = get_optimalReward(task)

	theta = np.array(ideal_w)
	feat_ideal = np.array([Rcup_opt, Rtable_opt])
	feat_final = np.array([Rcup, Rtable])

	# compute regret of final learned weight
	regret_final = np.dot(theta,feat_final) - np.dot(theta,feat_ideal)

	plan.kill_planner()

	return regret_final

def compute_rewardFinalDiff(weightdata, task):
	"""
	Given the final learned weight, compute the trajectory with those weights
	then compute the abs value of reward difference between that trajectory and the
	optimal trajectory for both cup and table feature.
	"""
	ideal_w = [0.0,0.0]
	if task == ONE_FEAT:
		ideal_w = [0.0,1.0]
	elif task == TWO_FEAT:
		ideal_w = [1.0,1.0]

	timestamp = weightdata[:,0:1]
	weights = weightdata[:,1:len(weightdata)+1]

	final_w = weights[-1]

	#print "in compute_rewardRegret - task: " + str(task)

	T = 20.0
	if task == TWO_FEAT:
		# if two features are wrong, initialize the starting config badly (tilted cup)
		pick = copy.copy(pick_basic)
		pick[-1] = 200.0
	else:
		pick = copy.copy(pick_basic) 

	place = copy.copy(place_lower)

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	#print "in rewardRegret - start: " + str(start)
	plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=task)
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, final_w, 0.0, T, 0.1, seed=None)	
	# compute reward of current traj with final learned weights
	r = plan.featurize(plan.waypts)
	Rvel = r[0]
	Rcup = np.sum(r[1])
	Rtable = np.sum(r[2])

	# get optimal reward
	(Rvel_opt, Rcup_opt, Rtable_opt) = get_optimalReward(task)

	diff_cup = np.fabs(Rcup - Rcup_opt)
	diff_table = np.fabs(Rtable - Rtable_opt)

	return (diff_cup, diff_table)

def compute_weightDot(data, task):
	"""
	Given the weight data and the task that it was collected for, 
	computes the dot product between the weight at each time point
	and the ideal weight and averages the total.
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]

	# upsample the weights
	(aug_time, aug_cup, aug_table) = augment_weights(timestamp, weights)

	total = 0.0

	for t in range(len(aug_time)):
		# get weight at current time step
		w = np.array([aug_cup[t],aug_table[t]])
		# do dot product with ideal_w
		d = np.dot(w,ideal_w)
		total += d

	return total/len(aug_time)

def compute_weightAngleAvg(data, task):
	"""
	Computes angle with the true reward averaged over time
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]

	# upsample the weights
	(aug_time, aug_cup, aug_table) = augment_weights(timestamp, weights)

	total = 0.0

	for t in range(len(aug_time)):
		# get weight at current time step
		w = np.array([aug_cup[t],aug_table[t]])
		if np.linalg.norm(w) != 0.0:
			# do dot product with ideal_w
			num = np.dot(w,ideal_w)
			denom = np.linalg.norm(w)*np.linalg.norm(ideal_w)
			arg = num/denom
			theta = np.arccos(arg)
			total += theta

	return total/len(aug_time)

def compute_weightFinalAngle(data, task):
	"""
	Computes the angle between the final learned weight and the 
	ideal weight for this task.
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]

	# get final weight
	w_T = weights[-1]
	#print "w_T: " + str(w_T)

	d = np.dot(w_T,ideal_w)
	arg = d/(np.linalg.norm(w_T)*np.linalg.norm(ideal_w))
	theta = np.arccos(arg)
	return theta

def compute_weightFinalDot(data, task):
	"""
	Computes the dot product between the final learned weight and the 
	ideal weight for this task.
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]

	# get final weight
	w_T = weights[-1]

	d = np.dot(w_T,ideal_w)
	return d

def compute_weightL2FinalDiff(data, task):
	"""
	Computes the norm difference between the desired 
	and final learned theta.
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])
	
	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]

	# get final weight
	w_T = weights[-1]
	return np.linalg.norm(ideal_w - w_T)

def compute_awayScore(data, task):
	"""
	Computes the cumulative score for weight updates. 
	If the weight was updated in the wrong direction, then 
	then score 1 and 0 else. Want to minimize this metric.
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]

	# upsample the weights according to hz rate
	(aug_time, aug_cup, aug_table) = augment_weights(timestamp, weights)

	cup_prev = np.array([aug_time[0], aug_cup[0]])
	dcup_prev = np.linalg.norm(ideal_w[0] - cup_prev)	

	table_prev = np.array([aug_time[0], aug_table[0]])
	dtable_prev = np.linalg.norm(ideal_w[1] - table_prev)	

	cup_score = 0.0
	table_score = 0.0
	for t in range(1,len(aug_time)):

		ideal_cup = np.array([aug_time[t], ideal_w[0]])		
		ideal_table = np.array([aug_time[t], ideal_w[1]])

		cup_curr = np.array([aug_time[t], aug_cup[t]])
		dcup_curr = np.linalg.norm(ideal_cup - cup_curr)

		table_curr = np.array([aug_time[t], aug_table[t]])
		dtable_curr = np.linalg.norm(ideal_table - table_curr)

		if dcup_curr > dcup_prev:
		# if moved in the wrong direction, decrease score
			cup_score += 1.0

		if dtable_curr > dtable_prev:
		# if moved in the wrong direction, decrease score
			table_score += 1.0

		dcup_prev = dcup_curr
		dtable_prev = dtable_curr

	return (cup_score, table_score)

def compute_weightPathLength(data):
	"""
	Computes the path length in weight space for theta hat.
	Linearly interpolates between weight at time t and weight 
	at time t+1, and summing the length of the lines between each of those points.
	"""

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]
	total = 0.0

	# upsample the weights
	(aug_time, aug_cup, aug_table) = augment_weights(timestamp, weights)

	w_prev = np.array([aug_cup[0],aug_table[0]])

	pathLength = 0.0

	for t in range(1,len(aug_time)):
		w_curr = np.array([aug_cup[t],aug_table[t]])
		pathLength += np.linalg.norm(w_curr - w_prev)
		w_prev = w_curr

	return pathLength


def compute_weightFeatPathLength(data):
	"""
	Computes the path length in weight space.
	Linearly interpolates between (cup or table) weight at time t and weight 
	at time t+1, and summing the length of the lines between each of those points.
	"""

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]
	total = 0.0

	# upsample the weights
	(aug_time, aug_cup, aug_table) = augment_weights(timestamp, weights)

	cup_prev = np.array([aug_time[0],aug_cup[0]])
	table_prev = np.array([aug_time[0], aug_table[0]])

	cup_pathLength = 0.0
	table_pathLength = 0.0
	for t in range(1,len(aug_time)):
		cup_curr = np.array([aug_time[t],aug_cup[t]])
		table_curr = np.array([aug_time[t], aug_table[t]])

		cup_pathLength += np.linalg.norm(cup_curr - cup_prev)
		table_pathLength += np.linalg.norm(table_curr - table_prev)

		cup_prev = cup_curr
		table_prev = table_curr

	return (cup_pathLength, table_pathLength)

def augment_weights(time, weights):
	"""
	Augments the weight data with 0.1 sec timesteps
	"""
	cup_w = weights[:,0]
	table_w = weights[:,1]

	aug_time = [0.0]*200 # traj is 20 sec, sampling at 0.1 sec
	aug_cup = [0.0]*200
	aug_table = [0.0]*200

	aug_idx = 0
	idx = 0
	prev_cup = 0.0
	prev_table = 0.0
	times = np.arange(0.1, 20.0, 0.1)
	for t in times:
		aug_time[aug_idx] = t
		#clipped_t = round(time[idx][0],1)
		if idx < len(cup_w) and np.isclose(round(time[idx][0],1), t, rtol=1e-05, atol=1e-08, equal_nan=False):
			aug_cup[aug_idx] = cup_w[idx]
			aug_table[aug_idx] = table_w[idx]
			prev_cup = cup_w[idx]
			prev_table = table_w[idx]
			idx += 1
		else:
			aug_cup[aug_idx] = prev_cup
			aug_table[aug_idx] = prev_table
		aug_idx += 1

	aug_time[-1] = 20.0
	aug_cup[-1] = prev_cup
	aug_table[-1] = prev_table
	return (aug_time, aug_cup, aug_table)

if __name__ == '__main__':
	"""
	task = 1
	trial = 1
	method = "A"
	ID = 7
	effortData = data_io.parse_exp_data("force")
	data = effortData[ID][task][trial][method]
	effort = compute_effort(data,ID)
	print "effort 7: " + str(effort)
	
	ID = 9
	data = effortData[ID][task][trial][method]
	effort = compute_effort(data,ID)
	print "effort 9: " + str(effort)
 	"""

	filename = "metrics"
	save_parsed_data(filename, csvData=True, pickleData=True)
	
