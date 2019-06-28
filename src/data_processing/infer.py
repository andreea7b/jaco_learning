import numpy as np
import math
import os
import itertools
import pickle
import glob
import sys
import ast
import matplotlib
import matplotlib.pyplot as plt

import trajoptpy
import openravepy
from openravepy import *

from utils.openrave_utils import *

# feature constants
MIN_WEIGHTS = {'table':0.0, 'coffee':-1.0, 'laptop':0.0, 'human':0.0, 'efficiency':0.0}
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':1.0, 'human':1.0, 'efficiency':1.0}
FEAT_RANGE = {'table':1.0, 'coffee':1.0, 'laptop':1.6, 'human':1.6, 'efficiency':0.01}

LAPTOP_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
CLOSE_LAPTOP_CENTER = [-1.3858/2.0, -0.1, 0.0]
HUMAN_CENTER = [-0.6, -0.55, 0.0]

class Infer(object):
	"""
	This class plans a trajectory from start to goal with TrajOpt. No learning involved.
	"""

	def __init__(self, feat_list, demo_id, task, demo_feat, traj_rand=None):

		# ---- important internal variables ---- #
		self.feat_list = feat_list		# 'table', 'human', 'coffee', 'origin', 'laptop'
		self.num_features = len(self.feat_list)

		# Can be a string number representing the participant ID or it can be "all", representing all demos.
		self.ID = demo_id
		self.task = task
		if self.task == "FAR":
			globals()['OBS_CENTER'] = LAPTOP_CENTER
		elif self.task == "CLOSE":
			globals()['OBS_CENTER'] = CLOSE_LAPTOP_CENTER
		else:
			print "ERROR: Task number not recognized."

		# ---- important discrete variables ---- #
		weights_span = [None]*self.num_features
		for feat in range(0,self.num_features):
			weights_span[feat] = list(np.linspace(MIN_WEIGHTS[feat_list[feat]], MAX_WEIGHTS[feat_list[feat]], num=3))
		self.weights_list = list(itertools.product(*weights_span))
		if (0.0,)*self.num_features in self.weights_list:
			self.weights_list.remove((0.0,)*self.num_features)
		self.weights_list = [w / np.linalg.norm(w) for w in self.weights_list]
		self.weights_list = set([tuple(i) for i in self.weights_list])	     # Make tuples out of these to find uniques.
		self.weights_list = [list(i) for i in self.weights_list]
		self.betas_list = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
		self.betas_list.reverse()
		self.num_betas = len(self.betas_list)
		self.num_weights = len(self.weights_list)

		# Construct uninformed prior
		P_bt = np.ones((self.num_betas, self.num_weights))
		self.P_bt = 1.0/self.num_betas * P_bt

		# ---- OpenRAVE Initialization ---- #

		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename)

		# insert any objects you want into environment
		self.bodies = []

		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotLaptop(self.env,self.bodies,OBS_CENTER)
		#plotCabinet(self.env)
		#plotSphere(self.env,self.bodies,OBS_CENTER,0.01)
		plotSphere(self.env,self.bodies,HUMAN_CENTER,0.01)

		# Load in random trajectories
		here = os.path.dirname(os.path.realpath(__file__))
		if traj_rand is None:
			traj_rand = "/traj_rand/traj_rand_merged_H.p"
		self.traj_rand = pickle.load( open( here + traj_rand, "rb" ) )
		import pdb;pdb.set_trace()
		# Load in requested demonstrations.
		self.demos = []
		self.demo_names = []
		if self.ID == "all":
			for demo_file in glob.glob(here+'/data/demonstrations/demos/*{}.p'.format(demo_feat)):
				demo = np.load(demo_file)
				self.demos.append(demo)
				self.demo_names.append(demo_file)
				plotTraj(self.env,self.robot,self.bodies, demo, size=0.015,color=[0, 0, 1])
		else:
			demo_file = here+'/data/demonstrations/demos/demo_{}_{}.p'.format(self.ID, demo_feat)
			demo = np.load(demo_file)
			self.demos.append(demo)
			plotTraj(self.env,self.robot,self.bodies, demo, size=0.015,color=[0, 0, 1])
		plotCupTraj(self.env,self.robot,self.bodies,[demo[-1]],color=[0,1,0])		

		# Create partition function values.
		self.Phi_rands = []
		num_trajs = len(self.traj_rand.keys())
		for rand_i, traj_str in enumerate(self.traj_rand.keys()):
			curr_traj = np.array(ast.literal_eval(traj_str))
			rand_features = self.featurize(curr_traj)
			Phi_rand = np.array([sum(x)/FEAT_RANGE[self.feat_list[i]] for i,x in enumerate(rand_features)])
			#print "Phi_rand",rand_i, ": ",Phi_rand, "; weights: ", self.traj_rand[traj_str]
			self.Phi_rands.append(Phi_rand)

	# ---- custom feature and cost functions ---- #

	def featurize(self, waypts):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory, output list of feature values
		"""
		features = [[0.0 for _ in range(len(waypts)-1)] for _ in range(0, self.num_features)]

		for index in range(0,len(waypts)-1):
			for feat in range(0, self.num_features):
				if self.feat_list[feat] == 'table':
					features[feat][index] = self.table_features(waypts[index+1])
				elif self.feat_list[feat] == 'coffee':
					features[feat][index] = self.coffee_features(waypts[index+1])
				elif self.feat_list[feat] == 'human':
					features[feat][index] = self.human_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat] == 'laptop':
					features[feat][index] = self.laptop_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat] == 'origin':
					features[feat][index] = self.origin_features(waypts[index+1])
				elif self.feat_list[feat] == 'efficiency':
					features[feat][index] = self.efficiency_features(waypts[index+1],waypts[index])
		return features

	# -- Velocity -- #

	def velocity_features(self, waypts):
		"""
		Computes total velocity cost over waypoints, confirmed to match trajopt.
		---
		input waypoint, output scalar feature
		"""
		vel = 0.0
		for i in range(1,len(waypts)):
			curr = waypts[i]
			prev = waypts[i-1]
			vel += np.linalg.norm(curr - prev)**2
		return vel 

	# -- Efficiency -- #

	def efficiency_features(self, waypt, prev_waypt):
		"""
		Computes efficiency cost for waypoint, confirmed to match trajopt.
		---
		input waypoint, output scalar feature
		"""
		return np.linalg.norm(waypt - prev_waypt)**2

	# -- Distance to Robot Base (origin of world) -- #

	def origin_features(self, waypt):
		"""
		Computes the total cost over waypoints based on 
		y-axis distance to table
		---
		input trajectory, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_y = coords[6][1]
		EEcoord_y = np.linalg.norm(coords[6])
		return EEcoord_y

	# -- Distance to Table -- #

	def table_features(self, waypt):
		"""
		Computes the total cost over waypoints based on 
		z-axis distance to table
		---
		input trajectory, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_z = coords[6][2]
		return EEcoord_z

	# -- Coffee (or z-orientation of end-effector) -- #

	def coffee_features(self, waypt):
		"""
		Computes the distance to table cost for waypoint
		by checking if the EE is oriented vertically according to pitch.
		Note: adding 1.5 to pitch to make it centered around 0
		---
		input trajectory, output scalar cost
		"""
		# get rotation transform, convert it to euler coordinates, and make sure the end effector is upright
		def mat2euler(mat):
			gamma = np.arctan2(mat[2,1], mat[2,2])
			beta = np.arctan2(-mat[2,0], np.sqrt(mat[2,1]**2 + mat[2,2]**2))
			alpha = np.arctan2(mat[1,0], mat[0,0])
			return np.array([gamma,beta,alpha])

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		R = EE_link.GetTransform()[:3,:3]
		[yaw, pitch, roll] = mat2euler(R)
		#return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))
		return (pitch + 1.5)

	# -- Distance to Laptop -- #

	def laptop_features(self, waypt, prev_waypt):
		"""
		Computes laptop cost over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions
		---
		input trajectory, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.laptop_dist(inter_waypt)
		return feature

	def laptop_dist(self, waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		laptop_xy = np.array(OBS_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	# -- Distance to Human -- #

	def human_features(self, waypt, prev_waypt):
		"""
		Computes laptop cost over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions
		---
		input trajectory, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.human_dist(inter_waypt)
		return feature

	def human_dist(self, waypt):
		"""
		Computes distance from end-effector to human in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from human
			+: EE is closer than 0.4 meters to human
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(HUMAN_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime

	def learnWeights(self, waypts_h):
		if waypts_h is not None:
			new_features = self.featurize(waypts_h)
			Phi_H = np.array([sum(x)/FEAT_RANGE[self.feat_list[i]] for i,x in enumerate(new_features)])
			print "Phi_H: ", Phi_H

			num_trajs = len(self.traj_rand.keys())
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

			# Compute optimal expected weight
			P_weight = sum(posterior, 0)
			curr_weight = np.sum(np.transpose(self.weights_list)*P_weight, 1)

			P_beta = np.sum(posterior, axis=1)
			self.beta = np.dot(self.betas_list,P_beta)
			print("observation model: ", P_obs)
			print("posterior: ", posterior)
			print("theta marginal: " + str(P_weight))
			print("beta marginal: " + str(P_beta))
			print("theta average: " + str(curr_weight))
			print("beta average: " + str(self.beta))

			self.weights = curr_weight
			self.visualize_posterior(posterior)
			print("\n------------ SIMULATED DEMONSTRATION DONE ------------\n")
			return posterior


	def learnWeightsAllDemos(self, waypts_h_array):
		if waypts_h_array is not None and len(waypts_h_array) > 1:
			new_features = np.matrix(self.featurize(waypts_h_array[0]))
			for i in range(len(waypts_h_array) - 1):
				next_features = np.matrix(self.featurize(waypts_h_array[i + 1]))
				new_features = new_features + next_features

			new_features = new_features.tolist()
			Phi_H = np.array([sum(x)/FEAT_RANGE[self.feat_list[i]] for i,x in enumerate(new_features)])
			print "Phi_H: ", Phi_H

			num_trajs = len(self.traj_rand.keys())
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
						Phi_rand = self.Phi_rands[rand_i]

						# Compute each denominator log
						logdenom[rand_i] = -beta * np.dot(weight, Phi_rand)

					# Compute the sum in log space
					A_max = max(logdenom)
					expdif = logdenom - A_max
					denom = A_max + np.log(sum(np.exp(expdif)))

					# Get P(xi_H | beta, weight) by dividing them
					P_xi[beta_i][weight_i] = np.exp(numerator - denom*len(waypts_h_array))

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
			print("observation model: ", P_obs)
			print("posterior: ", posterior)
			print("theta marginal: " + str(P_weight))
			print("beta marginal: " + str(P_beta))
			print("theta average: " + str(curr_weight))
			print("beta average: " + str(self.beta))

			self.weights = curr_weight
			self.visualize_posterior(posterior)
			print("\n------------ SIMULATED DEMONSTRATION DONE ------------\n")
			return posterior


	def visualize_posterior(self, post):
		matplotlib.rcParams['font.sans-serif'] = "Arial"
		matplotlib.rcParams['font.family'] = "Times New Roman"
		matplotlib.rcParams.update({'font.size': 15})

		plt.imshow(post, cmap='Blues', interpolation='nearest')
		plt.colorbar(ticks=[0, 0.03, 0.06])
		plt.clim(0, 0.06)

		weights_rounded = [[round(i,2) for i in j] for j in self.weights_list]
		plt.xticks(range(len(self.weights_list)), weights_rounded, rotation = 'vertical')
		plt.yticks(range(len(self.betas_list)), list(self.betas_list))
		plt.xlabel(r'$\theta$', fontsize=15)
		plt.ylabel(r'$\beta$',fontsize=15)
		plt.title(r'Joint Posterior Belief b($\theta$, $\beta$)')
		plt.tick_params(length=0)
		plt.show()
		return

### END CLASS


def infer_sequentially():
	inference = Infer(feat_list, demo_id, task, demo_feat)
	avg_post = np.zeros((inference.num_betas, inference.num_weights))
	for demo_i, demo in enumerate(inference.demos):
		posterior = inference.learnWeights(demo)
		avg_post += posterior
	avg_post = avg_post / len(inference.demos)
	inference.visualize_posterior(avg_post)

def infer_same_time():
	inference = Infer(feat_list, demo_id, task, demo_feat)
	posterior = inference.learnWeightsAllDemos(inference.demos)

if __name__ == '__main__':
	feat_list = [x.strip() for x in sys.argv[1].split(',')]
	demo_id = sys.argv[2]
	task = sys.argv[3]
	demo_feat = sys.argv[4]

	infer_sequentially()
	#infer_same_time()
	

