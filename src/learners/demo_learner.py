import numpy as np
import os
import itertools
import pickle
import matplotlib.pyplot as plt
import matplotlib
import ast

class DemoLearner(object):
	"""
	This class performs demonstration inference given human trajectories.
	"""

	def __init__(self, feat_list, environment, constants):
		# ---- Important internal variables ---- #
		self.feat_list = feat_list
		self.num_features = len(self.feat_list)
		self.environment = environment

		FEAT_RANGE = constants["FEAT_RANGE"]
		self.feat_range = [FEAT_RANGE[self.feat_list[feat]] for feat in range(self.num_features)]

		# Set up discretization of theta and beta space.
		self.betas_list = constants["betas_list"]
		self.betas_list.reverse()
		weight_vals = constants["weight_vals"]
		self.weights_list = list(itertools.product(weight_vals, repeat=self.num_features))
		if (0.0,)*len(self.feat_list) in self.weights_list:
			self.weights_list.remove((0.0,) * self.num_features)
		self.weights_list = [w / np.linalg.norm(w) for w in self.weights_list]
		self.weights_list = set([tuple(i) for i in self.weights_list])
		self.weights_list = [list(i) for i in self.weights_list]
		self.num_betas = len(self.betas_list)
		self.num_weights = len(self.weights_list)

		# Construct uninformed prior
		self.P_bt = np.ones((self.num_betas, self.num_weights)) / (self.num_betas * self.num_weights)
	
		# Trajectory paths.
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
		self.traj_rand = pickle.load(open(here + constants["trajs_path"], "rb" ))
		
		# Compute features for the normalizing trajectories.
		self.Phi_rands = []
		for rand_i, traj_str in enumerate(self.traj_rand.keys()):
			curr_traj = np.array(ast.literal_eval(traj_str))
			rand_features = self.environment.featurize(curr_traj, self.feat_list)
			Phi_rand = np.array([sum(x)/self.feat_range[i] for i,x in enumerate(rand_features)])
			self.Phi_rands.append(Phi_rand)
		
	def learn_weights(self, trajs):
		# Project all trajectories into feature space.
		new_features = [np.sum(self.environment.featurize(traj.waypts, self.feat_list), axis=1) for traj in trajs]
		Phi_H = np.array(np.sum(np.matrix(new_features), axis=0) / self.feat_range).T
		print "Phi_H: ", Phi_H

		# Now compute probabilities for each beta and theta pair.
		num_trajs = len(self.traj_rand.keys())
		P_xi = np.zeros((self.num_betas, self.num_weights))
		for (weight_i, weight) in enumerate(self.weights_list):
			print "Initiating inference with the following weights: ", weight
			for (beta_i, beta) in enumerate(self.betas_list):
				# Compute -beta*(weight^T*Phi(xi_H))
				numerator = -beta * np.dot(weight, Phi_H)

				# Calculate the integral in log space
				logdenom = np.zeros((num_trajs + 1,1))
				logdenom[-1] = -beta * np.dot(weight, Phi_H)

				# Compute costs for each of the random trajectories
				for rand_i in range(num_trajs):
					Phi_rand = self.Phi_rands[rand_i]

					# Compute each denominator log
					logdenom[rand_i] = -beta * np.dot(weight, Phi_rand)

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
		self.P_bt = posterior

		# Compute optimal expected weight
		P_weight = sum(posterior, 0)
		weights = np.sum(np.transpose(self.weights_list)*P_weight, 1)
		P_beta = np.sum(posterior, axis=1)
		beta = np.dot(self.betas_list,P_beta)
		print("observation model: ", P_obs)
		print("posterior: ", self.P_bt)
		print("theta marginal: " + str(P_weight))
		print("beta marginal: " + str(P_beta))
		print("theta average: " + str(weights))
		print("beta average: " + str(beta))

		self.visualize_posterior(self.P_bt)
		return weights

	def visualize_posterior(self, post):
		matplotlib.rcParams['font.sans-serif'] = "Arial"
		matplotlib.rcParams['font.family'] = "Times New Roman"
		matplotlib.rcParams.update({'font.size': 15})

		plt.imshow(post, cmap='Blues', interpolation='nearest')
		#plt.colorbar(ticks=[0, 0.15, 0.3])
		#plt.clim(0, 0.3)

		weights_rounded = [[round(i,2) for i in j] for j in self.weights_list]
		plt.xticks(range(len(self.weights_list)), weights_rounded, rotation = 'vertical')
		plt.yticks(range(len(self.betas_list)), list(self.betas_list))
		plt.xlabel(r'$\theta$', fontsize=15)
		plt.ylabel(r'$\beta$',fontsize=15)
		plt.title(r'Joint Posterior Belief b($\theta$, $\beta$)')
		plt.tick_params(length=0)
		plt.show()
		return
