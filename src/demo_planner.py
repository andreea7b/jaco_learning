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
UPDATE_GAINS = {'table':0.1, 'coffee':0.02, 'laptop':0.3, 'human':0.5}
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':10.0, 'human':10.0}
FEAT_RANGE = {'table':0.6918574, 'coffee':1.87608702, 'laptop':1.00476554, 'human':3.2}

# table is relatively symmetric: [-1.0, 0.75]
# coffee: [-0.06, 1.0] OR [-0.03, 0.7]
# laptop: [0.0, 7.51]
# human: [0.0, 10.0]

# fit a chi-squared distribution to p(beta|r); numers are [deg_of_freedom, loc, scale]
P_beta = {"table0": [1.83701582842, 0.0, 0.150583961407], "table1": [2.8, 0.0, 0.4212940611], "coffee0": [1.67451171875, 0.0, 0.05], "coffee1": [2.8169921875, 0.0, 0.3], "human0": [2.14693459432, 0.0, 0.227738059531], "human1": [5.0458984375, 0.0, 0.25]}

# feature learning methods
ALL = "ALL"					# updates all features
MAX = "MAX"					# updates only feature that changed the most
BETA = "BETA"				# updates beta-adaptive features 

class demoPlanner(Planner):
	"""
	This class plans a trajectory from start to goal with TrajOpt.
	It supports learning capabilities from demonstrated human trajectories.
	"""

	def __init__(self, feat_method, feat_list, task=None, traj_cache=None):

		# Call parent initialization
		super(demoPlanner, self).__init__(feat_list, task, traj_cache)

		# ---- important internal variables ---- #
		self.feat_method = feat_method	# can be ALL, MAX, or BETA
		self.weights = [0.0]*self.num_features
		self.betas = [1.0]*self.num_features
		self.betas_u = [1.0]*self.num_features
		self.waypts_prev = None
		self.waypts_deform = None
		self.updates = [0.0]*self.num_features

	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, waypts_h):
		if waypts_h is not None:
			self.waypts_h = waypts_h
			new_features = self.featurize(self.waypts_h)
			old_features = self.featurize(self.waypts)

			Phi_H = np.array([new_features[0]] + [sum(x) for x in new_features[1:]])
			Phi_R = np.array([old_features[0]] + [sum(x) for x in old_features[1:]])

			# Determine alpha and max theta
			update_gains = [0.0] * self.num_features
			max_weights = [0.0] * self.num_features
			feat_range = [0.0] * self.num_features
			for feat in range(0, self.num_features):
				update_gains[feat] = UPDATE_GAINS[self.feat_list[feat]]
				max_weights[feat] = MAX_WEIGHTS[self.feat_list[feat]]
				feat_range[feat] = FEAT_RANGE[self.feat_list[feat]]
			update = Phi_H - Phi_R
			self.updates = update[1:].tolist()

			if self.feat_method == ALL or self.feat_method == BETA:
				# update all weights 
				curr_weight = self.weights - update_gains * update[1:]
			elif self.feat_method == MAX:
				print "updating max weight"
				change_in_features = np.divide(update[1:], feat_range)

				# get index of maximal change
				max_idx = np.argmax(np.fabs(change_in_features))

				# update only weight of feature with maximal change
				curr_weight = np.array([self.weights[i] for i in range(len(self.weights))])
				curr_weight[max_idx] = curr_weight[max_idx] - update_gains[max_idx]*update[max_idx+1]

			# clip values at max and min allowed weights
			for i in range(self.num_features):
				curr_weight[i] = np.clip(curr_weight[i], -max_weights[i], max_weights[i])

			if self.feat_method == BETA:
				l = 0
				beta = np.linalg.norm(curr_weight)
				print "here is beta1: ", beta
				beta = 1 / (np.linalg.norm(np.array(self.weights) - curr_weight) ** 2 / (2 * np.array(update_gains)) /
					- curr_weight * update[1:] + l * (np.linalg.norm(waypts_h) ** 2 - np.linalg.norm(self.waypts) ** 2))
				print "here is beta2: ", beta
				beta = 1 / np.abs(np.linalg.norm(waypts_h) ** 2 - np.linalg.norm(self.waypts) ** 2)
				print "here is beta3: ", beta
			print "here is the update:", update
			print "here are the old weights:", self.weights
			print "here are the new weights:", curr_weight

			self.weights = curr_weight.tolist()

			return self.weights



