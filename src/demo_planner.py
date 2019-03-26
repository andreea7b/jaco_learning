import numpy as np
from numpy import linalg
import time
import math
import json

from scipy.optimize import minimize, newton
from scipy.stats import chi2

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import openrave_utils
from openrave_utils import *

import copy
import os
import itertools
import pickle
import matplotlib.mlab as mlab

from trajopt_planner import Planner

# feature constacts (update gains and max weights)
UPDATE_GAINS = {'table':0.01, 'coffee':0.002, 'laptop':0.01, 'human':0.01, 'efficiency':0.1}

class demoPlanner(Planner):
	"""
	This class plans a trajectory from start to goal with TrajOpt.
	It supports learning capabilities from demonstrated human trajectories.
	"""

	def __init__(self, feat_list, task=None, traj_cache=None):

		# Call parent initialization
		super(demoPlanner, self).__init__(feat_list, task, traj_cache)

		# ---- important internal variables ---- #
		self.weights = [0.0]*self.num_features
		self.updates = [0.0]*self.num_features

	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, waypts_h):
		import pdb;pdb.set_trace()
		if waypts_h is not None:
			self.waypts_h = waypts_h
			new_features = self.featurize(self.waypts_h)
			old_features = self.featurize(self.waypts)

			if 'efficiency' in self.feat_list:
				Phi_H = np.array([new_features[0]] + [sum(x) for x in new_features[1:]])
				Phi_R = np.array([old_features[0]] + [sum(x) for x in old_features[1:]])
			else:
				Phi_H = np.array([sum(x) for x in new_features])
				Phi_R = np.array([sum(x) for x in old_features])

			# Determine alpha and max theta
			update_gains = [0.0] * self.num_features
			for feat in range(self.num_features):
				update_gains[feat] = UPDATE_GAINS[self.feat_list[feat]]
			update = Phi_H - Phi_R
			self.updates = update.tolist()

			curr_weight = self.weights - update_gains * update
			print "here is the update:", update
			print "here are the old weights:", self.weights
			print "here are the new weights:", curr_weight

			self.weights = curr_weight.tolist()

			return self.weights



