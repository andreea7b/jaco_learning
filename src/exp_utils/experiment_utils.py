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
import pickle

import openrave_utils
from openrave_utils import *

import data_io

class ExperimentUtils(object):
	
	def __init__(self):
		# stores dictionary of all the replanned trajectories
		self.replanned_trajList = {}

		# stores trajectory as it is deformed
		self.deformed_traj = None

		# stores the list of positions as the robot is moving 
		# in the form [timestamp, j1, j2, ... , j7]
		self.tracked_traj = None

		# stores weights over time 
		# always in the form [timestamp, weight1, weight2, weight3]
		self.weights = None

		# stores running list of forces applied by human
		# in the form [timestamp, j1, j2, ... , j7]
		self.tauH = None 

		# stores start and end time of the interaction
		self.startT = 0.0
		self.endT = 0.0

	# ----- Data saving/tracking utils ----- 

	def update_replanned_trajList(self, timestamp, traj):
		"""
		Updates the dictionary of replanned trajectories since the start.
		"""
		if timestamp not in self.replanned_trajList:
			self.replanned_trajList[timestamp] = traj

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
		if new_weight is None:
			print "in update_weights: new_weights are None..."
			return 
		new_w = np.array([timestamp] + new_weight)
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

	# ----- Saving (pickling) utilities ------- #

	def get_unique_filepath(self,subdir,filename):
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"+subdir+"/"
		filepath = here + subdir + filename + "1.p"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".p"
			i+=1

		return filepath

	def pickle_replanned_trajList(self, filename):
		"""
		Pickles the replanned_trajList data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("replanned",filename)
		pickle.dump(self.replanned_trajList, open( filepath, "wb" ) )


	def pickle_deformed_traj(self, filename):
		"""
		Pickles the deformed_traj data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("deformed",filename)
		pickle.dump(self.deformed_traj, open( filepath, "wb" ) )

	def pickle_tracked_traj(self, filename):
		"""
		Pickles the tracked_traj data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("tracked",filename)
		pickle.dump(self.tracked_traj, open( filepath, "wb" ) )

	def pickle_weights(self, filename):
		"""
		Pickles the weights data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("weights",filename)
		pickle.dump(self.weights, open( filepath, "wb" ) )

	def pickle_force(self, filename):
		"""
		Pickles the force data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("force",filename)
		pickle.dump(self.tauH, open( filepath, "wb" ) )


if __name__ == '__main__':
	env, robot = openrave_utils.initialize(model_filename='jaco_dynamics')
	bodies = []

	#exp = ExperimentUtils()
	
	# ---- test replanned trajectory saving and parsing ---- #
	"""	
	filename = "replanned01B1.p"
	trajList = exp.parse_replanned_trajList(filename)
	for t in trajList.keys():
		plotCupTraj(env,robot,bodies,trajList[t],color=[0,1,0],increment=1)
	"""

	# ---- test tracked/deformed trajectory saving and parsing ---- #

	"""
	filename = "tracked32B1.p"
	waypts = data_io.parse_tracked_traj(filename)
	#filename = "deformed01A1.p"
	#waypts = exp.parse_deformed_traj(filename)	
	plotCupTraj(env,robot,bodies,waypts,color=[0,1,0],increment=5)
	time.sleep(20)
	"""

	"""	
	# ---- test weights/force saving and parsing ---- #
	filename = "weights02B1.p"
	(time, weights) = exp.parse_weights(filename)
	plt.plot(time,weights.T[0],linewidth=4.0,label='Coffee')
	plt.plot(time,weights.T[1],linewidth=4.0,label='Table')
	plt.legend()
	plt.title("Weight (for features) changes over time")
	plt.show()		

	filename = "force02B1.p"
	(time, force) = exp.parse_force(filename)
	plt.plot(time,force,linewidth=4.0)
	plt.legend()
	plt.title("Force over time")
	plt.show()		
	"""
