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
UPDATE_GAIN = 0.004
#FEAT_RANGE = {'table':0.6918574, 'coffee':1.87608702, 'laptop':1.3706093, 'human':2.2249931, 'efficiency':0.20920897}
FEAT_RANGE = {'table':0.270624619494, 'coffee':0.974212104025, 'laptop':0.30402465675, 'human':0.687767885424, 'efficiency':0.18665647143383943}

class demoPlanner(Planner):
	"""
	This class plans a trajectory from start to goal with TrajOpt.
	It supports learning capabilities from demonstrated human trajectories.
	"""

	def __init__(self, feat_list, task=None, traj_cache=None):

		# Call parent initialization
		super(demoPlanner, self).__init__(feat_list, task, traj_cache)

		# ---- important internal variables ---- #
		self.updates = [0.0]*self.num_features

	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, waypts_h):
		import pdb;pdb.set_trace()
		if waypts_h is not None:
			self.waypts_h = waypts_h
			new_features = self.featurize(self.waypts_h)
			old_features = self.featurize(self.waypts)
			
			Phi_H = np.array([new_features[0]] + [sum(x) for x in new_features[1:]])
			Phi_R = np.array([old_features[0]] + [sum(x) for x in old_features[1:]])

			update = Phi_H - Phi_R
			i = 1 if 'efficiency' in self.feat_list else 0
			update[0] = update[0] / FEAT_RANGE['efficiency']

			for feat in range(i, self.num_features):
				update[feat-i+1] = update[feat-i+1] / FEAT_RANGE[self.feat_list[feat]]
			self.updates = update.tolist()

			if 'efficiency' not in self.feat_list:
				update = update[1:]

			curr_weight = self.weights - UPDATE_GAIN * update

			print "here is the update:", update
			print "here are the old weights:", self.weights
			print "here are the new weights:", curr_weight

			self.weights = curr_weight.tolist()

			return self.weights



