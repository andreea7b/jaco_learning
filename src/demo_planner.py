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
			self.updates = update.tolist()

			if 'efficiency' not in self.feat_list:
				update = update[1:]

			curr_weight = self.weights - alpha * update

			print "here is the update:", update
			print "here are the old weights:", self.weights
			print "here are the new weights:", curr_weight

			self.weights = curr_weight.tolist()

			return self.weights



