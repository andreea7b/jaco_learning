import numpy as np
import math
import logging
import copy
import os
import pickle
import time

import trajopt_planner

import openrave_utils
from openrave_utils import *

# ----- Data parsing capabilities ---- #

def parse_exp_data(dataType):
	"""
	Creates a dictionary of all the data from the experiments 
	for the particular dataType

	Returns:
		ID 0 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		ID 1 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		...
		ID N --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
	-----
	dataType - "force", "tracked", "weights", "replanned"
	"""
	
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/study/"
	datapath = here + subdir + dataType

	validTypes = ["force", "weights", "replanned", "tracked"]
	if dataType not in validTypes:
		print dataType + " is not a valid data type!"
		return None

	data = {}

	for filename in os.listdir(datapath):
		# for each participant's data file, parse the participant's info
		info = filename.split(dataType)[1]
		if len(info) > 6:
			ID = int(info[0:2])
			task = int(info[2])
			method = info[3]
			trial = int(info[4])
		else:
			ID = int(info[0])
			task = int(info[1])
			method = info[2]
			trial = int(info[3])

		print "ID: " + str(ID) + ", task: " + str(task) + ", method: " + str(method) + ", trial: " + str(trial)

		# add participant info to dictionary
		if ID not in data:
			data[ID] = {}
		if task not in data[ID]:
			data[ID][task] = {}
		if trial not in data[ID][task]:
			data[ID][task][trial] = {}

		if dataType is "force":
			#(t_force, force) = utils.parse_force(filename)
			force = parse_force(filename)			
			data[ID][task][trial][method] = force
		elif dataType is "weights":
 			#(t_weights, weights) = utils.parse_weights(filename)
			weights = parse_weights(filename)		
			data[ID][task][trial][method] = weights
		elif dataType is "tracked":
			traj = parse_tracked_traj(filename)
			data[ID][task][trial][method] = traj
		elif dataType is "replanned":
			trajList = parse_replanned_trajList(filename)
			data[ID][task][trial][method] = trajList

	return data

# ----- De-pickling utilities ------- #

def parse_replanned_trajList(filename):
	"""
	Returns dictionary of trajectories, with timestamps as keys
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/replanned/"
	filepath = here + subdir + filename

	trajList = pickle.load( open( filepath, "rb" ) )
	return trajList

def parse_tracked_traj(filename):
	"""
	Returns trajectory
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/tracked/"
	filepath = here + subdir + filename

	traj = pickle.load( open( filepath, "rb" ) )
	return traj

	# returns only the waypoints, not timestamps!
	#return traj[:,1:len(traj)+1]

def parse_deformed_traj(filename):
	"""
	Returns trajectory
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/deformed/"
	filepath = here + subdir + filename

	traj = pickle.load( open( filepath, "rb" ) )

	return traj

def parse_weights(filename):
	"""
	Returns tuple: (timestamp list, weight list)
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/weights/"
	filepath = here + subdir + filename

	weights = pickle.load( open( filepath, "rb" ) )
	return weights
	#return (weights[:,0:1], weights[:,1:len(weights)+1])

def parse_force(filename):
	"""
	Returns tuple (timestamp list, force list)
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/force/"
	filepath = here + subdir + filename

	force = pickle.load( open( filepath, "rb" ) )
	return force
	#return (force[:,0:1], force[:,1:len(force)+1])

if __name__ == '__main__':
	"""
	data = parse_exp_data("weights")
	for ID in data.keys():
		for task in data[ID]:
			for trial in data[ID][task]:
				for method in data[ID][task][trial]:
					d = data[ID][task][trial][method]
					for i in range(len(d[:,1:len(d)+1])):
						print d[:,1:len(d)+1][i]
	#print data
	"""

	# ---- test tracked/deformed trajectory parsing ---- #
	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

	EXP_TASK = 2

	env, robot = openrave_utils.initialize(model_filename='jaco_dynamics')
	bodies = []

	plotTable(env)
	#plotCabinet(env)
	#plotTableMount(env,bodies)

	T = 20.0
	task = 1

	weights = [0.0,0.0]
	if task == 1:
		weights = [0.0,1.0]
	elif task == 2:
		weights = [1.0,1.0]

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
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, weights, 0.0, T, 0.1, seed=None)	
	plotCupTraj(env,robot,bodies,plan.waypts,color=[0,0,1],increment=5)


	filename = "tracked01A1.p"
	filenamew = "weights01A1.p"
	#filename = "tracked32B1.p"
	#traj = parse_tracked_traj(filename)
	#waypts = traj[:,1:len(traj)+1]

	timeweights = parse_weights(filenamew)
	w = timeweights[:,1:len(timeweights)+1]
	final_w = w[-1]

	plan.replan(start, goal, final_w, 0.0, T, 0.1, seed=None)	
	plotCupTraj(env,robot,bodies,plan.waypts,color=[0,1,0],increment=5)

	#filename = "deformed01A1.p"
	#waypts = exp.parse_deformed_traj(filename)	
	#plotCupTraj(env,robot,bodies,waypts,color=[0,1,0],increment=5)
	#plotCupTraj(env,robot,bodies,plan.waypts,color=[0,0,1],increment=5)
	print "final_w: " + str(final_w)	
	time.sleep(20)

