import numpy as np
import math

from scipy.optimize import minimize, newton
from scipy.stats import chi2

class TeleopLearner(object):
	"""
	This class performs correction inference given a trajectory and an input
	torque applied onto the trajectory.
	"""

	def __init__(self, feat_method, feat_list, environment, constants):

		# ---- Important internal variables ---- #
		self.feat_method = feat_method
		self.feat_list = feat_list
		self.num_features = len(self.feat_list)
		self.weights = [0.0] * self.num_features
		self.betas = [1.0] * self.num_features
		self.betas_u = [1.0] * self.num_features
		self.updates = [0.0] * self.num_features
		self.environment = environment

		self.alpha = constants["alpha"]
		self.n = constants["n"]
		UPDATE_GAINS = constants["UPDATE_GAINS"]
		MAX_WEIGHTS = constants["MAX_WEIGHTS"]
		FEAT_RANGE = constants["FEAT_RANGE"]

		self.update_gains = [UPDATE_GAINS[self.feat_list[feat]] for feat in range(self.num_features)]
		self.max_weights = [MAX_WEIGHTS[self.feat_list[feat]] for feat in range(self.num_features)]
		self.feat_range = [FEAT_RANGE[self.feat_list[feat]] for feat in range(self.num_features)]
		self.P_beta = constants["P_beta"]

	def learn_weights(self, traj, u_h, t):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between
		features of new trajectory and old trajectory.
		---

		Params:
			traj [Trajectory] -- Current trajectory that force was applied to.
			u_h [array] -- Human force applied onto the trajectory.
			t [float] -- Time where deformation was applied.

		Returns:
			weights [list] -- Learned weights.
		"""

		return self.weights

	def update_beliefs(self, beliefs):
		pass
