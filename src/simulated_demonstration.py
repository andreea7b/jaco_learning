#! /usr/bin/env python

import math
#import human_demonstrator#
import matplotlib.pyplot as plt
import numpy as np
import sys

from trajopt_planner import Planner

home_pos = [103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]
candlestick_pos = [180.0]*7

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

# feature constacts (update gains and max weights)
UPDATE_GAINS = {'table':2.0, 'coffee':2.0, 'laptop':100.0}
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':10.0, 'human':10.0}
FEAT_RANGE = {'table':0.6918574, 'coffee':1.87608702, 'laptop':1.00476554}

IMPEDANCE = 'A'
LEARNING = 'B'
DEMONSTRATION = 'C'

ALL = "ALL"						# updates all features
MAX = "MAX"						# updates only feature that changed the most
BETA = "BETA"					# updates beta-adaptive

class DemoJaco(object):

	def __init__(self, ID, method_type, record, feat_method, feat_list, feat_list_H, traj_cache=None, traj_rand=None):

		# method type - A=IMPEDANCE, B=LEARNING, C=DEMONSTRATION
		self.method_type = method_type

		# can be ALL, MAX, or BETA
		self.feat_method = feat_method

		# can be strings 'table', 'coffee', 'human', 'origin', 'laptop'
		self.feat_list = feat_list
		self.num_feats = len(self.feat_list)

		self.feat_list_H = feat_list_H
		self.num_feats_H = len(self.feat_list_H)

		# trajectory paths
		self.traj_cache = traj_cache
		self.traj_rand = traj_rand

		# record experimental data mode 
		if record == "F" or record == "f":
			self.record = False
		elif record == "T" or record == "t":
			self.record = True
		else:
			print("Oopse - it is unclear if you want to record data. Not recording data.")
			self.record = False

		self.weights = [0.0]*self.num_feats

		# ---- important discrete variables ---- #
		#self.weights_dict = [[-1.0, -1.0], [-1.0, 0.], [-1.0, 1.0], [0., -1.0], [0., 0.], [0., 1.0], [1.0, -1.0], [1.0, 0.], [1.0, 1.0]]
		self.weights_dict = [[-1.0], [0.], [1.0]]
		self.betas_dict = [0.01, 0.03, 0.1, 0.3, 1.0]

		self.num_betas = len(self.betas_dict)
		self.num_weights = len(self.weights_dict)

		# Construct uninformed prior
		P_bt = np.ones((self.num_betas, self.num_weights))
		self.P_bt = 1.0/self.num_betas * P_bt

		# initialize start/goal based on features
		# by default for table and laptop, these are the pick and place
		pick = pick_basic
		place = place_lower
		if 'human' in self.feat_list:
			pick = pick_shelf
			place = place_higher
		if 'coffee' in self.feat_list:
			pick = pick_basic_EEtilt

		start = np.array(pick)*(math.pi/180.0)
		goal = np.array(place)*(math.pi/180.0)
		self.start = start
		self.goal = goal

		self.T = 20.0

		# create the trajopt planner representing the human demonstrator
		self.planner = Planner(self.feat_list_H, None, self.traj_cache)


	def inferDemo(self, human_weights):
		# stores the current trajectory we are tracking, produced by planner
		print("\n\n----------- SIMULATED HUMAN NOW PLANNING -----------")
		self.traj = self.planner.replan(self.start, self.goal, human_weights, 0.0, self.T, 0.5, seed=None)
		print("\n\nTHIS IS THE HUMAN TRAJ: " + str(self.traj) + "\n\n")

		self.learnWeights(self.traj)
		print("\n\nTHESE ARE THE WEIGHTS: ")
		print(self.P_bt)
		print("\n\n")
		self.visualize_posterior(self.P_bt)
		print("DONE\n")
	

	def featurize(self, waypts):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory, output list of feature values
		"""
		features = [self.planner.velocity_features(waypts)]
		features += self.num_feats * [[0.0] * (len(waypts)-1)]
		for index in range(0,len(waypts)-1):
			for feat in range(1,self.num_feats+1):
				if self.feat_list[feat-1] == 'table':
					features[feat][index] = self.planner.table_features(waypts[index+1])
				elif self.feat_list[feat-1] == 'coffee':
					features[feat][index] = self.planner.coffee_features(waypts[index+1])
				elif self.feat_list[feat-1] == 'human':
					features[feat][index] = self.planner.human_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat-1] == 'laptop':
					features[feat][index] = self.planner.laptop_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat-1] == 'origin':
					features[feat][index] = self.planner.origin_features(waypts[index+1])
		return features

	
	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, traj):
	
		if traj is not None:
			old_features = self.featurize(self.traj)
			self.traj = traj
			new_features = self.featurize(self.traj)
			Phi_p = np.array([new_features[0]] + [sum(x) for x in new_features[1:]])
			Phi = np.array([old_features[0]] + [sum(x) for x in old_features[1:]])

			self.prev_features = Phi_p
			self.curr_features = Phi

			# Determine alpha and max theta
			update_gains = [0.0] * self.num_feats
			max_weights = [0.0] * self.num_feats
			feat_range = [0.0] * self.num_feats
			for feat in range(0, self.num_feats):
				update_gains[feat] = UPDATE_GAINS[self.feat_list[feat]]
				max_weights[feat] = MAX_WEIGHTS[self.feat_list[feat]]
				feat_range[feat] = FEAT_RANGE[self.feat_list[feat]]
			update = Phi_p - Phi

			if self.feat_method == ALL:
				# update all weights 
				curr_weight = self.weights - np.dot(update_gains, update[1:])
			elif self.feat_method == MAX:
				print("updating max weight")
				change_in_features = np.divide(update[1:], feat_range)

				# get index of maximal change
				max_idx = np.argmax(np.fabs(change_in_features))

				# update only weight of feature with maximal change
				curr_weight = [self.weights[i] for i in range(len(self.weights))]
				curr_weight[max_idx] = curr_weight[max_idx] - update_gains[max_idx]*update[max_idx+1]
			elif self.feat_method == BETA:
				# Now compute probabilities for each beta and theta in the dictionary
				P_xi = np.zeros((self.num_betas, self.num_weights))
				for (weight_i, weight) in enumerate(self.weights_dict):
					for (beta_i, beta) in enumerate(self.betas_dict):
						# Compute -beta*(weight^T*Phi(xi_H))
						numerator = -beta * np.dot([1] + weight, Phi_p)

						# Calculate the integral in log space
						num_trajs = self.traj_rand.shape[0]
						logdenom = np.zeros((num_trajs,1))

						# Compute costs for each of the random trajectories
						for rand_i in range(num_trajs):
							curr_traj = self.traj_rand[rand_i]
							rand_features = self.featurize(curr_traj)
							Phi_rand = np.array([rand_features[0]] + [sum(x) for x in rand_features[1:]])

							# Compute each denominator log
							logdenom[rand_i] = -beta * np.dot([1] + weight, Phi_rand)
						
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
				curr_weight = np.sum(np.transpose(self.weights_dict)*P_weight, 1)

				P_beta = np.sum(posterior, axis=1)
				self.beta = np.dot(self.betas_dict,P_beta)
				
				self.P_bt = posterior
				print("observation model:", P_obs)
				print("posterior", self.P_bt)
				print("theta marginal:", P_weight)
				print("beta average:", self.beta)
				print("update:", update[1:])
			print("curr_weight after = " + str(curr_weight))

			# clip values at max and min allowed weights
			for i in range(self.num_feats):
				curr_weight[i] = np.clip(curr_weight[i], -max_weights[i], max_weights[i])

			self.weights = curr_weight
			return self.weights

	
	def visualize_posterior(self, post):
		fig2, ax2 = plt.subplots()
		plt.imshow(post, cmap='RdBu', interpolation='nearest')
		plt.colorbar()
		plt.xticks(range(len(self.weights_dict)), list(self.weights_dict), rotation = 'vertical')
		plt.yticks(range(len(self.betas_dict)), list(self.betas_dict))
		plt.xlabel(r'$\theta$')
		plt.ylabel(r'$\beta$')
		plt.title("Joint posterior belief")
		plt.show()


if __name__ == '__main__':
	ID = 0 #ID = int(sys.argv[1])
	method_type = "A" #method_type = sys.argv[2]
	record = "F" #record = sys.argv[3]
	feat_method = "BETA" #feat_method = sys.argv[4]
	feat_list = ["table"] #feat_list = [x.strip() for x in sys.argv[5].split(',')]
	feat_list_H = ["table"] #feat_list_H = [x.strip() for x in sys.argv[6].split(',')]
	traj_cache = traj_rand = None
	traj_rand = np.load('./traj_dump/traj_cache_table.p')
	#if sys.argv[7] != 'None':
	#	traj_cache = sys.argv[6]
	#if sys.argv[8] != 'None':
	#	traj_rand = sys.argv[7]

	robot = DemoJaco(ID,method_type,record,feat_method,feat_list,feat_list_H,traj_cache,traj_rand)
	human_weights = [1.0]*robot.num_feats_H
	robot.inferDemo(human_weights)


