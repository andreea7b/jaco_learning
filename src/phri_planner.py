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
UPDATE_GAINS = {'table':2.0, 'coffee':2.0, 'laptop':100.0, 'human':20.0}
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':8.0, 'human':10.0}
FEAT_RANGE = {'table':0.6918574, 'coffee':1.87608702, 'laptop':1.00476554, 'human':3.2}

# fit a chi-squared distribution to p(beta|r); numers are [deg_of_freedom, loc, scale]
P_beta = {"table0": [1.83701582842, 0.0, 0.150583961407], "table1": [2.8, 0.0, 0.4212940611], "coffee0": [1.67451171875, 0.0, 0.05], "coffee1": [2.8169921875, 0.0, 0.3], "human0": [2.14693459432, 0.0, 0.227738059531], "human1": [5.0458984375, 0.0, 0.25]}

# feature learning methods
ALL = "ALL"					# updates all features
MAX = "MAX"					# updates only feature that changed the most
BETA = "BETA"				# updates beta-adaptive features 

class pHRIPlanner(Planner):
	"""
	This class plans a trajectory from start to goal with TrajOpt.
	It supports learning capabilities from physical human input.
	"""

	def __init__(self, feat_method, feat_list, task=None, traj_cache=None):

		# Call parent initialization
		super(pHRIPlanner, self).__init__(feat_list, task, traj_cache)

		# ---- important internal variables ---- #
		self.feat_method = feat_method	# can be ALL, MAX, or BETA
		self.betas = [1.0]*self.num_features
		self.betas_u = [1.0]*self.num_features
		self.waypts_prev = None
		self.waypts_deform = None
		self.updates = [0.0]*self.num_features

		# ---- DEFORMATION Initialization ---- #

		self.alpha = -0.01
		self.n = 5
		self.A = np.zeros((self.n+2, self.n))
		np.fill_diagonal(self.A, 1)
		for i in range(self.n):
			self.A[i+1][i] = -2
			self.A[i+2][i] = 1
		self.R = np.dot(self.A.T, self.A)
		Rinv = np.linalg.inv(self.R)
		Uh = np.zeros((self.n, 1))
		Uh[0] = 1
		self.H = np.dot(Rinv,Uh)*(np.sqrt(self.n)/np.linalg.norm(np.dot(Rinv,Uh)))

	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, u_h):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between 
		features of new trajectory and old trajectory
		---
		input is human force and returns updated weights 
		"""
		(waypts_deform, waypts_prev) = self.deform(u_h)	

		if waypts_deform is not None:
			self.waypts_deform = waypts_deform
			new_features = self.featurize(waypts_deform)
			old_features = self.featurize(waypts_prev)

			Phi_p = np.array([new_features[0]] + [sum(x) for x in new_features[1:]])
			Phi = np.array([old_features[0]] + [sum(x) for x in old_features[1:]])

			self.prev_features = Phi_p
			self.curr_features = Phi

			# Determine alpha and max theta
			update_gains = [0.0] * self.num_features
			max_weights = [0.0] * self.num_features
			feat_range = [0.0] * self.num_features
			for feat in range(0, self.num_features):
				update_gains[feat] = UPDATE_GAINS[self.feat_list[feat]]
				max_weights[feat] = MAX_WEIGHTS[self.feat_list[feat]]
				feat_range[feat] = FEAT_RANGE[self.feat_list[feat]]
			update = Phi_p - Phi
			self.updates = update[1:].tolist()

			if self.feat_method == ALL:
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
			elif self.feat_method == BETA:
				# beta-adaptive method
				update = update[1:]
				Phi_p = Phi_p[1:]
				Phi = Phi[1:]

				### First obtain the original beta rationality from the optimization problem ###
				# Set up the unconstrained optimization problem:
				def u_unconstrained(u):
					# Optimized manually; lambda_u can be changed according to user preferences
					if self.feat_list[i] == 'table':
						lambda_u = 20000
					elif self.feat_list[i] == 'human':
						lambda_u = 1500
					elif self.feat_list[i] == 'coffee':	
						lambda_u = 20000
					u_p = np.reshape(u, (7,1))
					(waypts_deform_p, waypts_prev) = self.deform(u_p)
					H_features = self.featurize_single(waypts_deform_p,i)
					Phi_H = sum(H_features)
					cost = (Phi_H - Phi_p[i])**2
					return cost

				# Constrained variant of the optimization problem
				def u_constrained(u):
					cost = np.linalg.norm(u)**2
					return cost

				# Set up the constraints:
				def u_constraint(u):
					u_p = np.reshape(u, (7,1))
					(waypts_deform_p, waypts_prev) = self.deform(u_p)
					H_features = self.featurize_single(waypts_deform_p,i)
					Phi_H = sum(H_features)
					cost = (Phi_H - Phi_p[i])**2
					return cost

				# Compute what the optimal action would have been wrt every feature
				for i in range(self.num_features):
					# Compute optimal action
					# Every feature requires a different optimizer because every feature is different in scale
					# Every feature also requires a different Newton-Rapson lambda
					if self.feat_list[i] == 'table':
						u_h_opt = minimize(u_constrained, np.zeros((7,1)), method='SLSQP', constraints=({'type': 'eq', 'fun': u_constraint}), options={'maxiter': 10, 'ftol': 1e-6, 'disp': True})
						l = math.pi
					elif self.feat_list[i] == 'human':
						u_h_opt = minimize(u_unconstrained, np.zeros((7,1)), options={'maxiter': 10, 'disp': True})
						l = 15.0
					elif self.feat_list[i] == 'coffee':
						u_h_opt = minimize(u_constrained, np.zeros((7,1)), method='SLSQP', constraints=({'type': 'eq', 'fun': u_constraint}), options={'maxiter': 10, 'ftol': 1e-6, 'disp': True})
						l = math.pi
					u_h_star = np.reshape(u_h_opt.x, (7, 1)) 

					# Compute beta based on deviation from optimal action
					beta_norm = 1.0/np.linalg.norm(u_h_star)**2
					self.betas[i] = self.num_features/(2*beta_norm*abs(np.linalg.norm(u_h)**2 - np.linalg.norm(u_h_star)**2))
					print "here is beta:", self.betas

					### Compute update using P(r|beta) for the beta estimate we just computed ###
					# Compute P(r|beta)
					mus1 = P_beta[self.feat_list[i]+"1"]
					mus0 = P_beta[self.feat_list[i]+"0"]
					p_r0 = chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) / (chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]))
					p_r1 = chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]) / (chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]))

					# Newton-Rapson setup; define function, derivative, and
					# call optimization method
					def f_theta(weights_p):
					    num = p_r1*np.exp(weights_p*update[i])
					    denom = p_r0*(l/math.pi)**(self.num_features/2.0)*np.exp(-l*update[i]**2) + num
					    return weights_p + update_gains[i]*num*update[i]/denom - self.weights[i]
					def df_theta(weights_p):
					    num = p_r0*(l/math.pi)**(self.num_features/2.0)*np.exp(-l*update[i]**2)
					    denom = p_r1*np.exp(weights_p*update[i])
					    return 1 + update_gains[i]*num/denom

					weight_p = newton(f_theta,self.weights[i],df_theta,tol=1e-04,maxiter=1000)
					
					num = p_r1*np.exp(weight_p*update[i])
					denom = p_r0*(l/math.pi)**(self.num_features/2.0)*np.exp(-l*update[i]**2) + num
					self.betas_u[i] = num/denom
					print "here is weighted beta:", self.betas_u
				# Compute new weights
				curr_weight = self.weights - np.array(self.betas_u)*update_gains*update

			# clip values at max and min allowed weights
			for i in range(self.num_features):
				curr_weight[i] = np.clip(curr_weight[i], 0.0, max_weights[i])
			print "here is the update:", update
			print "here are the old weights:", self.weights
			print "here are the new weights:", curr_weight

			self.weights = curr_weight.tolist()

			return self.weights

	def deform(self, u_h):
		"""
		Deforms the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		---
		input is human force, returns deformed and old waypts
		"""
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		deform_waypt_idx = self.curr_waypt_idx + 1

		if (deform_waypt_idx + self.n) > self.num_waypts:
			print "Deforming too close to end. Returning same trajectory"
			return (waypts_prev, waypts_prev)

		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		return (waypts_deform, waypts_prev)



