import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import math
import json

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import openrave_utils
from openrave_utils import *

import logging
import pid
import copy
import os
import itertools
import pickle

# feature constacts (update gains and max weights)
UPDATE_GAINS = {'table':2.0, 'coffee':2.0, 'laptop':100.0}
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':10.0}
FEAT_RANGE = {'table':0.6918574, 'coffee':1.87608702, 'laptop':1.00476554}

OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [0.0, 0.2, 0.0]

# feature learning methods
ALL = "ALL"					# updates all features
MAX = "MAX"					# updates only feature that changed the most
BETA = "BETA"				# updates beta-adaptive features 

class DiscretePlanner(object):
	"""
	This class plans a trajectory from start to goal
	with TrajOpt.
	"""

	def __init__(self, feat_method, feat_list, traj_cache=None, traj_rand=None, traj_optimal=None):

		# ---- important discrete variables ---- #
		#self.weights_dict = [[-1.0, -1.0], [-1.0, 0.], [-1.0, 1.0], [0., -1.0], [0., 0.], [0., 1.0], [1.0, -1.0], [1.0, 0.], [1.0, 1.0]]
		self.weights_dict = [[-0.5], [0.], [0.5]]
		self.betas_dict = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

		self.num_betas = len(self.betas_dict)
		self.num_weights = len(self.weights_dict)

		# Construct uninformed prior
		P_bt = np.ones((self.num_betas, self.num_weights))
		self.P_bt = 1.0/self.num_betas * P_bt

		# Decide on planning and deformation
		self.replan_weights = "WEIGHTED"    # can be ARGMAX, MEAN or WEIGHTED
		self.deform_method = "ALL"		# can be ONE or ALL

		# ---- important internal variables ---- #
		self.feat_method = feat_method	# can be ALL, MAX, or BETA
		self.feat_list = feat_list		# 'table', 'human', 'coffee', 'origin', 'laptop'
		self.num_features = len(self.feat_list)

		self.start_time = None
		self.final_time = None
		self.curr_waypt_idx = None

		# these variables are for trajopt
		self.waypts_plan = None
		self.num_waypts_plan = None
		self.step_time_plan = None
		self.MAX_ITER = 40

		# this is the cache of trajectories computed for all max/min weights
		self.traj_cache = self.traj_optimal = self.traj_rand = None	
		if traj_cache is not None:
			here = os.path.dirname(os.path.realpath(__file__))
			self.traj_cache = pickle.load( open( here + traj_cache, "rb" ) )
		if traj_rand is not None:
			here = os.path.dirname(os.path.realpath(__file__))
			self.traj_rand = pickle.load( open( here + traj_rand, "rb" ) )
		if traj_optimal is not None:
			here = os.path.dirname(os.path.realpath(__file__))
			self.traj_optimal = pickle.load( open( here + traj_optimal, "rb" ) )

		# these variables are for the upsampled trajectory
		self.waypts = None
		self.num_waypts = None
		self.step_time = None
		self.waypts_time = None

		self.weights = [0.0]*self.num_features
		self.waypts_prev = None

		# ---- Plotting weights & features over time ---- #
		self.weight_update = None
		self.update_time = None

		self.feature_update = None
		self.prev_features = None
		self.curr_features = None
		self.update_time2 = None

		# ---- OpenRAVE Initialization ---- #

		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename)

		# insert any objects you want into environment
		self.bodies = []

		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		#plotLaptop(self.env,self.bodies,OBS_CENTER)
		#plotCabinet(self.env)
		#plotSphere(self.env,self.bodies,OBS_CENTER,0.4)
		#plotSphere(self.env,self.bodies,HUMAN_CENTER,1)

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

	# ---- utilities/getter functions ---- #

	def get_waypts_plan(self):
		"""
		Returns reference to waypts_plan (used by trajopt)
		Used mostly for recording experimental data by pid_trajopt.py
		"""
		return self.waypts_plan

	# ---- custom feature and cost functions ---- #

	def featurize(self, waypts):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory, output list of feature values
		"""
		features = [self.velocity_features(waypts)]
		features += self.num_features * [[0.0] * (len(waypts)-1)]
		for index in range(0,len(waypts)-1):
			for feat in range(1,self.num_features+1):
				if self.feat_list[feat-1] == 'table':
					features[feat][index] = self.table_features(waypts[index+1])
				elif self.feat_list[feat-1] == 'coffee':
					features[feat][index] = self.coffee_features(waypts[index+1])
				elif self.feat_list[feat-1] == 'human':
					features[feat][index] = self.human_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat-1] == 'laptop':
					features[feat][index] = self.laptop_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat-1] == 'origin':
					features[feat][index] = self.origin_features(waypts[index+1])
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

	def velocity_cost(self, waypts):
		"""
		Computes the total velocity cost.
		---
		input trajectory, output scalar cost
		"""
		#mywaypts = np.reshape(waypts,(7,self.num_waypts_plan)).T
		return self.velocity_features(mywaypts)

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
		#plotSphere(self.env, self.bodies, [coords[6][0],0,0], size=20, color=[1,0,0])
		#plotSphere(self.env, self.bodies, [0,coords[6][1],0], size=20, color=[0,0,1])
		#plotSphere(self.env, self.bodies, [0,0,coords[6][2]], size=20, color=[0,1,0])
		#plotSphere(self.env, self.bodies, coords[6][0:3], size=20, color=[1,1,0])
		#print "EEcoord_y: " + str(EEcoord_y)
		return EEcoord_y

	def origin_cost(self, waypt):
		"""
		Computes the total distance from EE to base of robot cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.origin_features(waypt)
		feature_idx = self.feat_list.index('origin')
		return feature*self.weights[feature_idx]

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

	def table_cost(self, waypt):
		"""
		Computes the total distance to table cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.table_features(waypt)
		feature_idx = self.feat_list.index('table')
		return feature*self.weights[feature_idx]

	# -- Coffee (or z-orientation of end-effector) -- #

	def coffee_features(self, waypt):
		"""
		Computes the distance to table cost for waypoint
		by checking if the EE is oriented vertically.
		Note: [0,0,1] in the first *column* corresponds to the cup upright
		---
		input trajectory, output scalar cost
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))

	def coffee_cost(self, waypt):
		"""
		Computes the total coffee (EE orientation) cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.coffee_features(waypt)
		feature_idx = self.feat_list.index('coffee')
		return feature*self.weights[feature_idx]

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

	def laptop_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.laptop_features(curr_waypt,prev_waypt)
		feature_idx = self.feat_list.index('laptop')
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

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

	def human_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.human_features(curr_waypt,prev_waypt)
		feature_idx = self.feat_list.index('human')
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	# ---- custom constraints --- #

	def table_constraint(self, waypt):
		"""
		Constrains z-axis of robot's end-effector to always be 
		above the table.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_z = EE_link.GetTransform()[2][3]
		if EE_coord_z > 0:
			EE_coord_z = 0
		return -EE_coord_z

	def coffee_constraint(self, waypt):
		"""
		Constrains orientation of robot's end-effector to be 
		holding coffee mug upright.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
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
		self.robot.SetDOFValues(waypt)
		world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
		return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()


	# ---- here's trajOpt --- #

	def trajOptPose(self, start, goal, goal_pose):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Goal is a pose, not a configuration!
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input:
			start and goal pos, and a trajectory to seed trajopt with
		return:
			the waypts_plan trajectory
		"""

		print "I'm in trajopt_PLANNER trajopt pose!"

		# plot goal point
		#plotSphere(self.env, self.bodies, goal_pose, size=40)

		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]))
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		xyz_target = goal_pose
		quat_target = [1,0,0,0] # wxyz

		init_joint_target =  goal

		init_waypts = np.zeros((self.num_waypts_plan,7))

		for count in range(self.num_waypts_plan):
			init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)

		if self.traj_cache is not None:
			# choose seeding trajectory from cache if the weights match
			weights_span = [None]*self.num_features
			min_dist_w = [None]*self.num_features
			for feat in range(0,self.num_features):
				limit = MAX_WEIGHTS[self.feat_list[feat]]
				weights_span[feat] = list(np.arange(-limit, limit+.1, limit/2))
				min_dist_w[feat] = -limit

			weight_pairs = list(itertools.product(*weights_span))
			weight_pairs = [np.array(i) for i in weight_pairs]

			# current weights
			cur_w = np.array(self.weights)
			min_dist_idx = 0
			for (w_i, w) in enumerate(weight_pairs):
				dist = np.linalg.norm(cur_w - w)
				if dist < np.linalg.norm(cur_w - min_dist_w):
					min_dist_w = w
					min_dist_idx = w_i

			init_waypts = np.array(self.traj_cache[min_dist_idx])

		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"start_fixed" : True,
				"max_iter" : self.MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [1.0]}
			}
			],
			"constraints": [
			{
				"type": "pose",
				"params" : {"xyz" : xyz_target,
                            "wxyz" : quat_target,
                            "link": "j2s7s300_link_7",
							"rot_coeffs" : [0,0,0],
							"pos_coeffs" : [35,35,35],
                            }
			}
			],
			#"init_info": {
            #    "type": "straight_line",
            #    "endpoint": init_joint_target.tolist()
			#}
			"init_info": {
                "type": "given_traj",
                "data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env)

		for t in range(1,self.num_waypts_plan):
			if 'coffee' in self.feat_list:
				prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			if 'table' in self.feat_list:
				prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			if 'laptop' in self.feat_list:
				prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			if 'origin' in self.feat_list:
				prob.AddCost(self.origin_cost, [(t,j) for j in range(7)], "origin%i"%t)
			if 'human' in self.feat_list:
				prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "table%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		# plot resulting trajectory
		#plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, size=10,color=[0, 0, 1])
		#plotCupTraj(self.env,self.robot,self.bodies,self.waypts_plan, color=[0,1,0])		

		print "I'm done with trajopt pose!"

		return self.waypts_plan


	def trajOpt(self, start, goal, traj_seed=None):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""

		print "I'm in normal trajOpt!"

		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]))
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		# --- linear interpolation seed --- #
		if traj_seed is None:
			print "using straight line!"
			init_waypts = np.zeros((self.num_waypts_plan,7))
			for count in range(self.num_waypts_plan):
				init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		else:
			print "using traj seed!"
			init_waypts = traj_seed

		if self.traj_cache is not None:
			# choose seeding trajectory from cache if the weights match
			weights_span = [None]*self.num_features
			min_dist_w = [None]*self.num_features
			for feat in range(0,self.num_features):
				limit = MAX_WEIGHTS[self.feat_list[feat]]
				weights_span[feat] = list(np.arange(-limit, limit+.1, limit/2))
				min_dist_w[feat] = -limit

			weight_pairs = list(itertools.product(*weights_span))
			weight_pairs = [np.array(i) for i in weight_pairs]

			# current weights
			cur_w = np.array(self.weights)
			min_dist_idx = 0
			for (w_i, w) in enumerate(weight_pairs):
				dist = np.linalg.norm(cur_w - w)
				if dist < np.linalg.norm(cur_w - min_dist_w):
					min_dist_w = w
					min_dist_idx = w_i

			init_waypts = np.array(self.traj_cache[min_dist_idx])

		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter" : self.MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [1.0]}
			}
			],
			"constraints": [
			{
				"type": "joint",
				"params": {"vals": goal.tolist()}
			}
			],
			"init_info": {
                "type": "given_traj",
                "data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env)

		for t in range(1,self.num_waypts_plan):
			if 'coffee' in self.feat_list:
				prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			if 'table' in self.feat_list:
				prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			if 'laptop' in self.feat_list:
				prob.AddErrorCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "HINGE", "laptop%i"%t)
				prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			if 'origin' in self.feat_list:
				prob.AddCost(self.origin_cost, [(t,j) for j in range(7)], "origin%i"%t)
			if 'human' in self.feat_list:
				prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)


		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		#plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, size=10,color=[0, 0, 1])
		#plotCupTraj(self.env,self.robot,self.bodies,self.waypts_plan,color=[0,1,0])

		return self.waypts_plan


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

			if self.feat_method == ALL:
				# update all weights 
				curr_weight = self.weights - np.dot(update_gains, update[1:])
			elif self.feat_method == MAX:
				print "updating max weight"
				change_in_features = np.divide(update[1:], feat_range)

				# get index of maximal change
				max_idx = np.argmax(np.fabs(change_in_features))

				# update only weight of feature with maximal change
				curr_weight = [self.weights[i] for i in range(len(self.weights))]
				curr_weight[max_idx] = curr_weight[max_idx] - update_gains[max_idx]*update[max_idx+1]
			elif self.feat_method == BETA:
				# TODO: not working. If we deform all optimals, first precompute the deformations
				if self.deform_method == "ALL":
					traj_optimal_deform = [0] * self.num_weights
					for weight_i in range(self.num_weights):
						curr_timestep = int(self.waypts_time[self.curr_waypt_idx])
						u_h_p = np.reshape(waypts_deform[curr_timestep] - self.traj_optimal[weight_i][curr_timestep], (7,1))
						(waypts_partial, _) = self.deform_given_waypts(self.traj_optimal[weight_i], u_h_p)
						traj_optimal_deform[weight_i] = np.concatenate((waypts_prev[:curr_timestep+1], waypts_partial[curr_timestep+1:]))

				# Now compute probabilities for each beta and theta in the dictionary
				P_xi = np.zeros((self.num_betas, self.num_weights))
				for (weight_i, weight) in enumerate(self.weights_dict):
					for (beta_i, beta) in enumerate(self.betas_dict):
						# Compute -beta*(weight^T*Phi(xi_H))
						if self.deform_method == "ONE":
							numerator = -beta * np.dot([1] + weight, Phi_p)
						elif self.deform_method == "ALL":
							curr_features = self.featurize(traj_optimal_deform[weight_i])
							Phi_curr = np.array([curr_features[0]] + [sum(x) for x in curr_features[1:]])
							numerator = -beta * np.dot([1] + weight, Phi_curr)

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

				if self.replan_weights == "ARGMAX":
					# Get optimal weight and beta by doing argmax
					(best_beta_i, best_weight_i) = np.unravel_index(np.argmax(posterior), posterior.shape)
					curr_weight = self.weights_dict[best_weight_i]
				elif self.replan_weights == "MEAN":
					# Compute optimal expected weight
					P_weight = sum(posterior, 0)
					curr_weight = np.sum(np.transpose(self.weights_dict)*P_weight, 1)
				elif self.replan_weights == "WEIGHTED":
					# Another method uses beta-weighted expected weight
					P_weight = np.matmul(np.transpose(posterior), self.betas_dict)
					P_weight = P_weight / sum(P_weight,0)
					curr_weight = np.matmul(P_weight, self.weights_dict)

				self.P_bt = posterior
				print self.P_bt
				print(sum(self.P_bt, 0))

			print "curr_weight after = " + str(curr_weight)

			# clip values at max and min allowed weights
			for i in range(self.num_features):
				curr_weight[i] = np.clip(curr_weight[i], -max_weights[i], max_weights[i])

			self.weights = curr_weight
			return self.weights

	def deform(self, u_h):
		"""
		Deforms the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		---
		input is human force, returns deformed and old waypts
		"""
		deform_waypt_idx = self.curr_waypt_idx + 1
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))

		if (deform_waypt_idx + self.n) > self.num_waypts:
			print "Deforming too close to end. Returning same trajectory"
			return (waypts_prev, waypts_prev)

		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		return (waypts_deform, waypts_prev)

	def deform_given_waypts(self, waypts, u_h):
		"""
		Deforms the next n waypoints of the given upsampled trajectory
		updates the upsampled trajectory, stores old given trajectory
		---
		input is trajectory and human force, returns deformed and old waypts
		"""
		deform_waypt_idx = self.curr_waypt_idx + 1
		waypts_prev = copy.deepcopy(waypts)
		waypts_deform = copy.deepcopy(waypts)
		gamma = np.zeros((self.n,7))

		if (deform_waypt_idx + self.n) > self.num_waypts:
			print "Deforming too close to end. Returning same trajectory"
			return (waypts_prev, waypts_prev)

		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		return (waypts_deform, waypts_prev)

	# ---- replanning, upsampling, and interpolating ---- #

	def replan(self, start, goal, weights, start_time, final_time, step_time, seed=None):
		"""
		Replan the trajectory from start to goal given weights.
		---
		input trajectory parameters, update raw and upsampled trajectories
		"""
		if weights is None:
			return
		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.weights = weights
		print "weights in replan: " + str(weights)

		if 'coffee' in self.feat_list:
			place_pose = [-0.46513, 0.29041, 0.69497]
			self.trajOptPose(start, goal, place_pose)
		else:
			self.trajOpt(start, goal, traj_seed=seed)

		self.upsample(step_time)

		return self.waypts_plan

	def upsample(self, step_time):
		"""
		Put waypoints along trajectory at step_time increments.
		---
		input desired time increment, update upsampled trajectory
		"""
		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts

		t = self.start_time
		for i in range(num_waypts):
			if t >= self.final_time:
				waypts_time[i] = self.final_time
				waypts[i,:] = self.waypts_plan[self.num_waypts_plan - 1]
			else:
				deltaT = t - self.start_time
				prev_idx = int(deltaT/self.step_time_plan)
				prev = self.waypts_plan[prev_idx]
				next = self.waypts_plan[prev_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = prev+((t-prev_idx*self.step_time_plan)/self.step_time_plan)*(next-prev)
			t += step_time
		self.step_time = step_time
		self.num_waypts = num_waypts
		self.waypts = waypts
		self.waypts_time = waypts_time

	def downsample(self):
		"""
		Updates the trajopt trajectory from the upsampled trajectory.
		changes the trajopt waypoints between start and goal.
		"""
		for index in range(1,self.num_waypts_plan-1):
			t = self.start_time + index*self.step_time_plan
			target_pos = self.interpolate(t)
			self.waypts_plan[index,:] = target_pos.reshape((1,7))

	def interpolate(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_time:
			self.curr_waypt_idx = self.num_waypts - 1
			target_pos = self.waypts[self.curr_waypt_idx]
		else:
			deltaT = curr_time - self.start_time
			self.curr_waypt_idx = int(deltaT/self.step_time)
			prev = self.waypts[self.curr_waypt_idx]
			next = self.waypts[self.curr_waypt_idx + 1]
			ti = self.waypts_time[self.curr_waypt_idx]
			tf = self.waypts_time[self.curr_waypt_idx + 1]
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

	def update_curr_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos - 7x1 vector of current joint angles (degrees)
		"""
		pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0]+math.pi,curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])

		self.robot.SetDOFValues(pos)

	def plot_weight_update(self):
		"""
		Plots weight update over time.
		"""

		#plt.plot(self.update_time,self.weight_update.T[0],linewidth=4.0,label='Vel')
		plt.plot(self.update_time,self.weight_update.T[0],linewidth=4.0,label='Coffee')
		plt.plot(self.update_time,self.weight_update.T[1],linewidth=4.0,label='Table')
		#plt.plot(self.update_time,self.weight_update.T[2],linewidth=4.0,label='Laptop')
		plt.legend()
		plt.title("Weight (for features) changes over time")
		plt.show()		

	def plot_feature_update(self):
		"""
		Plots feature change over time.
		"""

		#plt.plot(self.update_time,self.weight_update.T[0],linewidth=4.0,label='Vel')
		plt.plot(self.update_time2,self.feature_update.T[1],linewidth=4.0,label='Coffee')
		plt.plot(self.update_time2,self.feature_update.T[2],linewidth=4.0,label='Table')
		#plt.plot(self.update_time2,self.feature_update.T[3],linewidth=4.0,label='Laptop')
		plt.legend()
		plt.title("Feature changes over time")
		plt.show()		

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime

if __name__ == '__main__':

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

	# for cup/human task:
	pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
	place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]	

	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0] 
	place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]

	#place_pose = [-0.58218719,  0.33018986,  0.10592141] # x, y, z for pick_lower_EEtilt

	# initialize start/goal based on task 
	pick = pick_basic_EEtilt #pick_basic
	place = place_lower #place_lower 

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights = [0.0, 0.0]
	T = 20.0

	feat_method = "ALL"
	numFeat = 2
	planner = Planner(2, False, featMethod, numFeat)

	"""
	if len(goal) < 10:
		waypt = np.append(goal.reshape(7), np.array([0,0,0]), 1)
		waypt[2] += math.pi
	planner.robot.SetDOFValues(waypt)
	coords = robotToCartesian(planner.robot)
	place_pose = [coords[6][0], coords[6][1], coords[6][2]]
	print place_pose
	plotSphere(planner.env,planner.bodies,place_pose,0.4)
	"""

	#place_pose = [-0.46513, 0.29041, 0.69497]

	traj = planner.replan(start, goal, weights, 0.0, T, 0.5)	

	weights = [1.0, 1.0]

	traj = planner.replan(start, goal, weights, 0.0, T, 0.5)	

	time.sleep(20)


