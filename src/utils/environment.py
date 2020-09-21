import numpy as np
import math
import copy
import torch
import os

import openravepy
from openravepy import *

from utils.openrave_utils import *
#from learned_feature import LearnedFeature


class Environment(object):
	"""
	This class creates an OpenRave environment and contains all the
	functionality needed for custom features and constraints.
	"""
	def __init__(self, model_filename, object_centers, feat_list, feat_range,
				 goals=[], LF_dict=None, use_viewer=True, plot_objects=True):
		# ---- Create environment ---- #
		self.env, self.robot = initialize(model_filename, use_viewer=use_viewer)

		# Insert any objects you want into environment.
		self.bodies = []
		self.object_centers = object_centers

		# Plot the table and table mount, and other desired objects.
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		if plot_objects:
			plotCabinet(self.env)
			plotLaptop(self.env,self.bodies,object_centers['LAPTOP_CENTER'])
			plotSphere(self.env,self.bodies,object_centers['HUMAN_CENTER'], 0.015)

		# Plot and add the goals
		i = 0
		self.goals = goals
		self.goal_locs = []
		for goal in goals:
			with self.robot:
				angles = goal if len(goal) == 10 else np.append(goal, np.array([0,0,0]))
				self.robot.SetDOFValues(angles)
				cartesian_coords = robotToCartesian(self.robot)
				goal_loc = cartesian_coords[6]
				self.goal_locs.append(goal_loc)
				if i == 0:
					color = [1, 0, 0]
				else:
					color = [0, 0, 1]
				i += 1
				plotSphere(self.env, self.bodies, goal_loc, 0.05, color=color) # TODO: change colors
		self.goal_locs = np.array(self.goal_locs)

		# true goal for presentation
		goal = np.array([260.0, 90.0, 180.0, 160.0, 270.0, 180.0, 280.0])*np.pi/180
		with self.robot:
			angles = goal if len(goal) == 10 else np.append(goal, np.array([0,0,0]))
			self.robot.SetDOFValues(angles)
			cartesian_coords = robotToCartesian(self.robot)
			goal_loc = cartesian_coords[6]
			plotSphere(self.env, self.bodies, goal_loc, 0.05, color=[0,1,0])

		# Create the initial feature function list.
		self.feat_func_list = []
		self.feat_list = feat_list
		self.num_feats = len(self.feat_list)
		self.feat_range = feat_range
		for feat in self.feat_list:
			if feat == 'table':
				self.feat_func_list.append(self.table_features)
			elif feat == 'coffee':
				self.feat_func_list.append(self.coffee_features)
			elif feat == 'human':
				self.feat_func_list.append(self.human_features)
			elif feat == 'laptop':
				self.feat_func_list.append(self.laptop_features)
			elif feat == 'origin':
				self.feat_func_list.append(self.origin_features)
			elif feat == 'efficiency':
				self.feat_func_list.append(self.efficiency_features)
			elif feat == 'proxemics':
				self.feat_func_list.append(self.proxemics_features)
			elif feat == 'betweenobjects':
				self.feat_func_list.append(self.betweenobjects_features)
		# TODO: add goal_dist function

		# Create a list of learned features.
		self.learned_feats = []

		# Initialize LF_dict optionally for learned features.
		self.LF_dict = LF_dict


	# ---- Compute features for all waypoints in trajectory ---- #
	def featurize(self, waypts, feat_idxs=None):
		"""
		Computes the features for a given trajectory.
		---
		Params:
			waypts -- trajectory waypoints
			feat_idx -- list of feature indices (optional)
		Returns:
			features -- list of feature values (T x num_features)
		"""
		# if no list of idx is provided use all of them
		if feat_idxs is None:
			feat_idxs = list(np.arange(self.num_feats))

		features = np.zeros((len(feat_idxs), len(waypts)-1))

		for index in range(len(waypts)-1):
			for feat in range(len(feat_idxs)):
				waypt = waypts[index+1]
				if self.feat_list[feat_idxs[feat]] == 'efficiency':
					waypt = np.concatenate((waypts[index+1],waypts[index]))
				features[feat][index] = self.featurize_single(waypt, feat_idxs[feat])

			# for feat in range(len(feat_idxs)):
			# 	feat_idx = feat_idxs[feat]
			# 	if self.feat_list[feat_idx] == 'table':
			# 		features[feat][index] = self.table_features(waypts[index+1])
			# 	elif self.feat_list[feat_idx] == 'coffee':
			# 		features[feat][index] = self.coffee_features(waypts[index+1])
			# 	elif self.feat_list[feat_idx] == 'human':
			# 		features[feat][index] = self.human_features(waypts[index+1],waypts[index])
			# 	elif self.feat_list[feat_idx] == 'laptop':
			# 		features[feat][index] = self.laptop_features(waypts[index+1],waypts[index])
			# 	elif self.feat_list[feat_idx] == 'origin':
			# 		features[feat][index] = self.origin_features(waypts[index+1])
			# 	elif self.feat_list[feat_idx] == 'efficiency':
			# 		features[feat][index] = self.efficiency_features(waypts[index+1],waypts[index])
			# 	# TODO: this is bad design because only 10 goals can be supported, but shouldn't matter
			# 	elif "goal" in self.feat_list[feat_idx]:
			# 		features[feat][index] = self.goal_dist_features(int(feat_list[feat][4]), waypts[index+1])
		return features

	# -- Compute single feature for single waypoint -- #
	def featurize_single(self, waypt, feat_idx):
		"""
		Computes given feature value for a given waypoint.
		---
		Params:
			waypt -- single waypoint
			feat_idx -- feature index
		Returns:
			featval -- feature value
		"""
		# If it's a learned feature, feed in raw_features to the NN.
		if 'learned_feature' in self.feat_list[feat_idx]:
			waypt = self.raw_features(waypt)
		# Compute feature value.
		featval = self.feat_func_list[feat_idx](waypt)
		if 'learned_feature': in self.feat_list[feat_idx]:
			featval = featval[0][0]
		else:
			if self.feat_range is not None:
				featval /= self.feat_range[self.feat_list[feat_idx]]
		return featval

	# -- Return raw features -- #
	def raw_features(self, waypt):
		"""
		Computes raw state space features for a given waypoint.
		---
		Params:
			waypt -- single waypoint
		Returns:
			raw_features -- list of raw feature values
		"""
		object_coords = np.array([self.object_centers[x] for x in self.object_centers.keys()])
		if torch.is_tensor(waypt):
			Tall = self.get_torch_transforms(waypt)
			coords = Tall[:,:3,3]
			orientations = Tall[:,:3,:3]
			object_coords = torch.from_numpy(object_coords)
			return torch.reshape(torch.cat((waypt.squeeze(), orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))
		else:
			if len(waypt) < 10:
				waypt_openrave = np.append(waypt.reshape(7), np.array([0, 0, 0]))
				waypt_openrave[2] += math.pi

			with self.robot:
				self.robot.SetDOFValues(waypt_openrave)
				coords = np.array(robotToCartesian(self.robot))
				orientations = np.array(robotToOrientation(self.robot))
			return np.reshape(np.concatenate((waypt.squeeze(), orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))

	def get_torch_transforms(self, waypt):
		"""
		Computes torch transforms for given waypoint.
		---
		Params:
			waypt -- single waypoint
		Returns:
			Tall -- Transform in torch for every joint (7D)
		"""
		# Manually compute a link transform given theta, alpha, and D (a is assumed to be 0).
		def transform(theta, alpha, D):
			T = torch.zeros([4, 4])
			T[0][0] = torch.cos(theta)
			T[0][1] = -torch.sin(theta)*torch.cos(alpha)
			T[0][2] = torch.sin(theta)*torch.sin(alpha)
			T[1][0] = torch.sin(theta)
			T[1][1] = torch.cos(theta)*torch.cos(alpha)
			T[1][2] = -torch.cos(theta)*torch.sin(alpha)
			T[2][1] = torch.sin(alpha)
			T[2][2] = torch.cos(alpha)
			T[2][3] = D
			T[3][3] = 1.0
			return T

		def swap_cols(T, i, j):
			return torch.cat((T[:, :i], T[:, j:j+1], T[:, i+1:j], T[:, i:i+1], T[:, j+1:]), dim=1)

		# These are robot measurements and DH parameters.
		# Ds are link distances. es are some minor joint displacement errors.
		# For each transform, we much be careful which D and/or e we pass in.
		# The manual is sort of correct about how to do this but not 100% right.
		# Contact abobu@berkeley.edu if you have questions about this code.
		e = torch.tensor(np.array([0.0016, 0.0098]), requires_grad=True)
		D = torch.tensor(np.array([0.15675, 0.11875, 0.205, 0.205, 0.2073, 0.10375, 0.10375]), requires_grad=True)
		alpha = torch.tensor(np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi]), requires_grad=True)
		sign1 = torch.tensor(np.array([[-1,1,-1,-1], [1,-1,1,1], [-1,1,-1,-1], [1,1,1,1]]), dtype=torch.float64, requires_grad=True)
		sign2 = torch.tensor(np.array([[1,-1,-1,-1], [-1,1,1,1], [1,-1,-1,-1], [1,1,1,1]]), dtype=torch.float64, requires_grad=True)
		sign3 = torch.tensor(np.array([[1,-1,1,-1], [-1,1,-1,1], [1,-1,1,-1], [1,1,1,1]]), dtype=torch.float64, requires_grad=True)

		# Now construct the list of transforms for all joints.

		### T01 transform from base to joint 1 ###
		T01 = transform(waypt[0], alpha[0], -D[0])
		Tall = (swap_cols(T01, 1, 2) * sign1).unsqueeze(0)

		### T02 transform from base to joint 2 ###
		T01 = transform(waypt[0], alpha[0], -(D[0]+D[1]))
		T12 = transform(waypt[1], alpha[1], -e[0])
		T02 = torch.matmul(T01, T12)
		Tall = torch.cat((Tall, (swap_cols(T02, 1, 2) * sign2).unsqueeze(0)))

		### T03 transform from base to joint 3 ###
		T23 = transform(waypt[2], alpha[2], -D[2])
		T03 = torch.matmul(T02, T23)
		Tall = torch.cat((Tall, (swap_cols(T03, 1, 2) * sign2).unsqueeze(0)))

		### T04 transform from base to joint 4 ###
		T23 = transform(waypt[2], alpha[2], -(D[2]+D[3]))
		T34 = transform(waypt[3], alpha[3], 0.0)
		T03 = torch.matmul(T02, T23)
		T04 = torch.matmul(T03, T34)
		Tall = torch.cat((Tall, (swap_cols(T04, 1, 2) * sign1).unsqueeze(0)))

		### T05 transform from base to joint 5 ###
		T34 = transform(waypt[3], alpha[3], -(e[0]+e[1]))
		T45 = transform(waypt[4], alpha[4], -D[4])
		T04 = torch.matmul(T03, T34)
		T05 = torch.matmul(T04, T45)
		Tall = torch.cat((Tall, (swap_cols(T05, 1, 2) * sign2).unsqueeze(0)))

		### T06 transform from base to joint 6 ###
		T45 = transform(waypt[4], alpha[4], -(D[4]+D[5]))
		T56 = transform(waypt[5], alpha[5], 0.0)
		T05 = torch.matmul(T04, T45)
		T06 = torch.matmul(T05, T56)
		Tall = torch.cat((Tall, (swap_cols(T06, 1, 2) * sign1).unsqueeze(0)))

		### T07 transform from base to joint 7 ###
		T67 = transform(waypt[6], alpha[6], -D[6])
		T07 = torch.matmul(T06, T67)
		Tall = torch.cat((Tall, (T07 * sign3).unsqueeze(0)))

		return Tall

	# -- Instantiate a new learned feature -- #

	def new_FERL_learned_feature(self, nb_layers, nb_units, checkpoint_name=None):
		"""
		Adds a new FERL learned feature to the environment.
		--
		Params:
			nb_layers -- number of NN layers
			nb_units -- number of NN units per layer
			checkpoint_name -- name of NN model to load (optional)
		"""
		self.learned_feats.append(LearnedFeature(nb_layers, nb_units, self.LF_dict))
		self.feat_list.append('learned_feature')
		self.num_feats += 1

		# make sure that existing goal weights get a zero added to them
		## initialize new feature weight with zero
		#self.weights = np.hstack((self.weights, np.zeros((1, ))))

		# If we can, load a model instead of a blank feature.
		if checkpoint_name is not None:
			here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
			self.learned_feats[-1] = torch.load(here+'/data/final_models/' + checkpoint_name)

		self.feat_func_list.append(self.learned_feats[-1].function)

	def load_meirl_learned_feature(self, planner, weight, save_dict_path):
		save_dict = torch.load(save_dict_path)
		self.new_meirl_learned_feature(planner,
									   weight,
									   save_dict['s_g_exp_trajs'],
									   save_dict['goal_poses'],
									   save_dict['known_feat_list'],
									   save_dict['NN_dict'],
									   save_dict['gen'])
		meirl_obj = self.learned_feats[-1]
		meirl_obj.cost_nn.load_state_dict(save_dict['cost_nn'])
		meirl_obj.max_label = save_dict['max_label']
		meirl_obj.min_label = save_dict['min_label']


	def new_meirl_learned_feature(self, planner, weight, s_g_exp_trajs, goal_poses, known_feat_list, NN_dict, gen, T=20., timestep=0.5):
		"""
		Adds a new maxent irl learned feature to the environment.
		--
		Params:
			nb_layers -- number of NN layers
			nb_units -- number of NN units per layer
			checkpoint_name -- name of NN model to load (optional)
		"""
		meirl_obj = DeepMaxEntIRL(self, planner, weight, s_g_exp_trajs, goal_poses, known_feat_list, NN_dict, gen, T, timestep)
		self.learned_feats.append(meirl_obj)
		self.feat_list.append('learned_feature')
		self.num_feats += 1
		self.feat_func_list.append(self.learned_feats[-1].function)

	# -- Goal Distance -- #

	def goal_dist_features(self, goal_num, waypt):
		with self.robot:
			self.robot.SetDOFValues(np.append(waypt.reshape(7), np.array([0,0,0])))
			coords = robotToCartesian(self.robot)[6]
		return np.linalg.norm(self.goal_locs[goal_num] - coords)**2

	# -- Efficiency -- #

	def efficiency_features(self, waypt):
		"""
		Computes efficiency feature for waypoint, confirmed to match trajopt.
		---
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar feature
		"""

		return np.linalg.norm(waypt[:7] - waypt[7:])**2

	# -- Distance to Robot Base (origin of world) -- #

	def origin_features(self, waypt):
		"""
		Computes the total feature value over waypoints based on
		y-axis distance to table.
		---
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EEcoord_y = np.linalg.norm(coords[6])
		return EEcoord_y

	# -- Distance to Table -- #

	def table_features(self, waypt, prev_waypt=None):
		"""
		Computes the total feature value over waypoints based on
		z-axis distance to table.
		---
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EEcoord_z = coords[6][2]
		return EEcoord_z

	# -- Coffee (or z-orientation of end-effector) -- #

	def coffee_features(self, waypt):
		"""
		Computes the coffee orientation feature value for waypoint
		by checking if the EE is oriented vertically.
		---
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi

		with self.robot:
			self.robot.SetDOFValues(waypt)
			EE_link = self.robot.GetLinks()[7]
			Rx = EE_link.GetTransform()[:3,0]
			return 1 - EE_link.GetTransform()[:3,0].dot([0,0,1])

	# -- Distance to Laptop -- #

	def laptop_features(self, waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar distance where
				0: EE is at more than 0.3 meters away from laptop
				+: EE is closer than 0.3 meters to laptop
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		laptop_xy = np.array(self.object_centers['LAPTOP_CENTER'][0:2])
		dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.3
		if dist > 0:
			return 0
		return -dist

	# -- Distance to Human -- #

	def human_features(self, waypt):
		"""
		Computes distance from end-effector to human in xy coords
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar distance where
				0: EE is at more than 0.4 meters away from human
				+: EE is closer than 0.4 meters to human
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(self.object_centers['HUMAN_CENTER'][0:2])
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	# -- Human Proxemics -- #

	def proxemics_features(self, waypt):
		"""
		Computes distance from end-effector to human proxemics in xy coords
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar distance where
				0: EE is at more than 0.3 meters away from human
				+: EE is closer than 0.3 meters to human
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(self.object_centers['HUMAN_CENTER'][0:2])
		# Modify ellipsis distance.
		EE_coord_xy[1] /= 3
		human_xy[1] /= 3
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.3
		if dist > 0:
			return 0
		return -dist

	# -- Between 2-objects -- #

	def betweenobjects_features(self, waypt):
		"""
		Computes distance from end-effector to 2 objects in xy coords.
		Params:
			waypt -- single waypoint
		Returns:
			dist -- scalar distance where
				0: EE is at more than 0.2 meters away from the objects and between
				+: EE is closer than 0.2 meters to the objects and between
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		object1_xy = np.array(self.object_centers['OBJECT1'][0:2])
		object2_xy = np.array(self.object_centers['OBJECT2'][0:2])

		# Determine where the point lies with respect to the segment between the two objects.
		o1EE = np.linalg.norm(object1_xy - EE_coord_xy)
		o2EE = np.linalg.norm(object2_xy - EE_coord_xy)
		o1o2 = np.linalg.norm(object1_xy - object2_xy)
		o1angle = np.arccos((o1EE**2 + o1o2**2 - o2EE**2) / (2*o1o2*o1EE))
		o2angle = np.arccos((o2EE**2 + o1o2**2 - o1EE**2) / (2*o1o2*o2EE))

		dist1 = 0
		if o1angle < np.pi/2 and o2angle < np.pi/2:
			dist1 = np.linalg.norm(np.cross(object2_xy - object1_xy, object1_xy - EE_coord_xy)) / o1o2 - 0.2
		dist1 = 0.8*dist1 # control how much less it is to go between the objects versus on top of them
		dist2 = min(np.linalg.norm(object1_xy - EE_coord_xy), np.linalg.norm(object2_xy - EE_coord_xy)) - 0.2

		if dist1 > 0 and dist2 > 0:
			return 0
		elif dist2 > 0:
			return -dist1
		elif dist1 > 0:
			return -dist2
		return -min(dist1, dist2)

	# ---- Custom environmental constraints --- #

	def table_constraint(self, waypt):
		"""
		Constrains z-axis of robot's end-effector to always be above the table.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			EE_link = self.robot.GetLinks()[10]
			EE_coord_z = EE_link.GetTransform()[2][3]
		if EE_coord_z > -0.1016:
			return 0
		return 10000

	def coffee_constraint(self, waypt):
		"""
		Constrains orientation of robot's end-effector to be holding coffee mug upright.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			EE_link = self.robot.GetLinks()[7]
			return EE_link.GetTransform()[:2,:3].dot([1,0,0])

	def coffee_constraint_derivative(self, waypt):
		"""
		Analytic derivative for coffee constraint.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
			return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()

	# ---- Helper functions ---- #

	def update_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos - 7x1 vector of current joint angles (radians)
		"""
		#pos = np.array([curr_pos[0][0],curr_pos[1][1],curr_pos[2][2]+math.pi,curr_pos[3][3],curr_pos[4][4],curr_pos[5][5],curr_pos[6][6],0,0,0])
		pos = np.array([curr_pos[0][0],curr_pos[1][1],curr_pos[2][2],curr_pos[3][3],curr_pos[4][4],curr_pos[5][5],curr_pos[6][6],0,0,0])
		with self.env:
			self.robot.SetDOFValues(pos)

	def update_vel(self, curr_vel):
		"""
		Updates DOF velocities in OpenRAVE simulation based on curr_vel.
		----
		curr_vel - 7x1 vector of current joint velocities (radians)
		"""
		vel = np.array([curr_vel[0][0],curr_vel[1][1],curr_vel[2][2],curr_vel[3][3],curr_vel[4][4],curr_vel[5][5],curr_vel[6][6],0,0,0])
		with self.env:
			self.robot.SetDOFVelocities(vel)

	def get_cartesian_coords(self, joint_angles):
		"""
		Note that joint_angles are assumed to be in radians
		"""
		if len(joint_angles) < 10:
			joint_angles = np.append(joint_angles.reshape(7), np.array([0,0,0]))
			joint_angles[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(joint_angles)
			return robotToCartesian(self.robot)[6]

	def kill_environment(self):
		"""
		Destroys openrave thread and environment for clean shutdown.
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime
