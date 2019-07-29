import numpy as np
import os
import itertools
import pickle
import matplotlib.pyplot as plt
import matplotlib
import ast

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

from utils.openrave_utils import *

class DemoLearner(object):
	"""
    This class performs demonstration inference given a human trajectory.
	"""

	def __init__(self, feat_method, feat_list, environment, constants):

		# ---- Important internal variables ---- #
		self.feat_method = feat_method
        self.feat_list = feat_list
        self.num_features = len(self.feat_list)
		self.weights = [0.0] * self.num_features
		self.beta = 1.0
        self.environment = environment

		# ---- Important discrete variables ---- #
        self.betas_list = constants["betas_list"].reverse()
        weight_vals = constants["weight_vals"]
		self.weights_list = list(iter.product(weight_vals, repeat=self.num_features))
        if (0.0,)*self.num_features in self.weights_list:
            self.weights_list.remove((0.0,)*self.num_features)
        self.weights_list = [w / np.linalg.norm(w) for w in self.weights_list]
		self.weights_list = set([tuple(i) for i in self.weights_list])
		self.weights_list = [list(i) for i in self.weights_list]
		self.num_betas = len(self.betas_list)
		self.num_weights = len(self.weights_list)

		# Construct uninformed prior
		P_bt = np.ones((self.num_betas, self.num_weights))
		self.P_bt = 1.0/self.num_betas * P_bt
		
        # Trajectory paths.
		here = os.path.dirname(os.path.realpath(__file__))
		self.traj_rand = pickle.load(open(here + constants["trajs_path"], "rb" ) )

	# ---- here's our algorithms for modifying the trajectory ---- #

	def learn_weights(self, trajs):
        new_features = [self.environment.featurize(traj.waypts) for traj in trajs]
			Phi_H = np.array([sum(x)/FEAT_RANGE[self.feat_list[i]] for i,x in enumerate(new_features)])
			print "Phi_H: ", Phi_H

			# Compute features for the normalizing trajectories.
			Phi_rands = []
			weight_rands = []
			num_trajs = len(self.traj_rand.keys())
			for rand_i, traj_str in enumerate(self.traj_rand.keys()):
				curr_traj = np.array(ast.literal_eval(traj_str))
				rand_features = self.featurize(curr_traj)
				Phi_rand = np.array([sum(x)/FEAT_RANGE[self.feat_list[i]] for i,x in enumerate(rand_features)])
				print "Phi_rand",rand_i, ": ",Phi_rand, "; weights: ", self.traj_rand[traj_str]
				Phi_rands.append(Phi_rand)
				weight_rands.append(self.traj_rand[traj_str])

			# Now compute probabilities for each beta and theta in the dictionary
			P_xi = np.zeros((self.num_betas, self.num_weights))
			for (weight_i, weight) in enumerate(self.weights_list):
				print "Initiating inference with the following weights: ", weight
				for (beta_i, beta) in enumerate(self.betas_list):
					# Compute -beta*(weight^T*Phi(xi_H))
					numerator = -beta * np.dot(weight, Phi_H)

					# Calculate the integral in log space
					logdenom = np.zeros((num_trajs+1,1))
					logdenom[-1] = -beta * np.dot(weight, Phi_H)

					# Compute costs for each of the random trajectories
					for rand_i in range(num_trajs):
						Phi_rand = Phi_rands[rand_i]

						# Compute each denominator log
						logdenom[rand_i] = -beta * np.dot(weight, Phi_rand)
					#if weight == [0.0,1.0,0.0]:
					#	import pdb;pdb.set_trace()
					# Compute the sum in log space
					A_max = max(logdenom)
					expdif = logdenom - A_max
					denom = A_max + np.log(sum(np.exp(expdif)))

					# Get P(xi_H | beta, weight) by dividing them
					P_xi[beta_i][weight_i] = np.exp(numerator - denom)

			P_obs = P_xi / sum(sum(P_xi))
			
			# Compute P(weight, beta | xi_H) via Bayes rule
			posterior = np.multiply(P_obs, self.P_bt)

			# Normalize posterior
			posterior = posterior / sum(sum(posterior))

			# Compute optimal expected weight
			P_weight = sum(posterior, 0)
			curr_weight = np.sum(np.transpose(self.weights_list)*P_weight, 1)

			P_beta = np.sum(posterior, axis=1)
			self.beta = np.dot(self.betas_list,P_beta)
			self.P_bt = posterior
			print("observation model: ", P_obs)
			print("posterior: ", self.P_bt)
			print("theta marginal: " + str(P_weight))
			print("beta marginal: " + str(P_beta))
			print("theta average: " + str(curr_weight))
			print("beta average: " + str(self.beta))

			self.weights = curr_weight
			self.visualize_posterior(self.P_bt)
			return self.weights
	
    
    def learnWeights_cont(self, waypts_h, alpha=0.002):
		if waypts_h is not None:
			new_features = self.featurize(self.waypts_h)
			old_features = self.featurize(self.waypts)
			
			Phi_H = np.array([sum(x) for x in new_features])
			Phi_R = np.array([sum(x) for x in old_features])

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

	def visualize_posterior(self, post):
		matplotlib.rcParams['font.sans-serif'] = "Arial"
		matplotlib.rcParams['font.family'] = "Times New Roman"
		matplotlib.rcParams.update({'font.size': 15})

		plt.imshow(post, cmap='Blues', interpolation='nearest')
		plt.colorbar(ticks=[0, 0.15, 0.3])
		plt.clim(0, 0.3)

		weights_rounded = [[round(i,2) for i in j] for j in self.weights_list]
		plt.xticks(range(len(self.weights_list)), weights_rounded, rotation = 'vertical')
		plt.yticks(range(len(self.betas_list)), list(self.betas_list))
		plt.xlabel(r'$\theta$', fontsize=15)
		plt.ylabel(r'$\beta$',fontsize=15)
		plt.title(r'Joint Posterior Belief b($\theta$, $\beta$)')
		plt.tick_params(length=0)
		plt.show()
		return
