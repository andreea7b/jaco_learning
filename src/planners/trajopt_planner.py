import numpy as np
import math
import json
import copy
import torch

import trajoptpy

from utils.openrave_utils import *
from utils.trajectory import Trajectory

import pybullet as p
from utils.environment_utils import *


class TrajoptPlanner(object):
	"""
	This class plans a trajectory from start to goal with TrajOpt.
	"""
	def __init__(self, max_iter, num_waypts, environment, pb_environment, prefer_angles=True, use_constraint_learned=True):

		# ---- Important internal variables ---- #
		# These variables are trajopt parameters.
		self.MAX_ITER = max_iter
		self.num_waypts = num_waypts

		# Set OpenRAVE environment.
		self.environment = environment

		# Set pybullet environment.
		self.bullet_environment = pb_environment

		# whether to use goal angles over goal pose for planning
		self.prefer_angles = prefer_angles

		# whether to use constraints when there are learned features
		self.use_constraint_learned = use_constraint_learned

	# -- Interpolate feature value between neighboring waypoints to help planner optimization. -- #

	def interpolate_features(self, waypt, prev_waypt, feat_idx, NUM_STEPS=4):
		"""
		Computes feature value over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions.
		---
		input neighboring waypoints and feature function, output scalar feature
		"""
		feat_val = 0.0
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + ((1.0 + step)/NUM_STEPS)*(waypt - prev_waypt)
			feat_val += self.environment.featurize_single(inter_waypt, feat_idx)
		return feat_val / NUM_STEPS

	# ---- Costs ---- #

	def efficiency_clip_cost(self, waypt):
		"""
		Computes the total efficiency cost
		---
		input waypoint, output scalar cost
		"""
		feature_idx = self.environment.feat_list.index('efficiency_clip')
		#feature = self.interpolate_features(waypt, waypt, feature_idx, NUM_STEPS=1)
		feature = self.environment.featurize_single(waypt, feature_idx, planner_version=True)
		return feature*self.weights[feature_idx]

	def world_efficiency_cost(self, waypt):
		"""
		Computes the total world-space efficiency cost
		---
		input waypoint, output scalar cost
		"""
		feature_idx = self.environment.feat_list.index('world_efficiency')
		feature = self.interpolate_features(waypt, waypt, feature_idx, NUM_STEPS=1)
		return feature*self.weights[feature_idx]

	def efficiency_cost(self, waypt):
		"""
		Computes the total efficiency cost
		---
		input waypoint, output scalar cost
		"""
		feature_idx = self.environment.feat_list.index('efficiency')
		feature = self.interpolate_features(waypt, waypt, feature_idx, NUM_STEPS=1)
		return feature*self.weights[feature_idx]

	def origin_cost(self, waypt):
		"""
		Computes the total distance from EE to base of robot cost.
		---

		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feat_list.index('origin')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def table_cost(self, waypt):
		"""
		Computes the total distance to table cost.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feat_list.index('table')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def coffee_cost(self, waypt):
		"""
		Computes the total coffee (EE orientation) cost.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feat_list.index('coffee')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def laptop_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feat_list.index('laptop')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def human_cost(self, waypt):
		"""
		Computes the total distance to human cost.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feat_list.index('human')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def learned_feature_costs(self, waypt):
		"""
		Computes the cost for all the learned features.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		# get the number of learned features
		n_learned = np.count_nonzero(self.environment.is_learned_feat)

		feature_values = []
		for i, feature in enumerate(self.environment.learned_feats):
			# get the value of the feature
			feat_idx = self.environment.num_feats - n_learned + i
			if self.weights[feat_idx] == 0.0:
				feature_value = 0.0
			else:
				feature_value = self.interpolate_features(curr_waypt, prev_waypt, feat_idx, NUM_STEPS=1) # CHANGED
			feature_values.append(feature_value)
		# calculate the cost
		world_dist = np.linalg.norm(self.environment.get_cartesian_coords(curr_waypt) - self.environment.get_cartesian_coords(prev_waypt))
		return np.matmul(self.weights[-n_learned:], np.array(feature_values))*world_dist

	def learned_feature_cost_derivatives(self, waypt):
		"""
		Computes the cost derivatives for all the learned features.
		---
		input waypoint, output scalar cost
		"""
		# get the number of learned features
		n_learned = np.count_nonzero(self.environment.is_learned_feat)

		J = []
		sols = []
		for i, feature in enumerate(self.environment.learned_feats):
			feat_idx = self.environment.num_feats - n_learned + i
			if self.weights[feat_idx] == 0.0:
				J.append(np.zeros(len(waypt)))
				continue

			# Setup for computing Jacobian.
			x = torch.tensor(waypt, requires_grad=True)

			# Get the value of the feature
			feat_val = torch.tensor(0.0, requires_grad=True)
			NUM_STEPS = 1 # CHANGED
			for step in range(NUM_STEPS):
				delta = torch.tensor((1.0 + step)/NUM_STEPS, requires_grad=True)
				inter_waypt = x[:7] + delta * (x[7:] - x[:7])
				# Compute feature value.
				z = self.environment.feat_func_list[feat_idx](self.environment.raw_features(inter_waypt).float(), torchify=True)
				feat_val = feat_val + z
			y = feat_val / torch.tensor(float(NUM_STEPS), requires_grad=True)
			world_dist = torch.norm(self.environment.get_torch_transforms(x[7:])[6,0:3,3] - self.environment.get_torch_transforms(x[:7])[6,0:3,3])
			y = y * torch.tensor(self.weights[-n_learned+i], requires_grad=True) * world_dist
			y.backward()
			J.append(x.grad.data.numpy())
		return np.sum(np.array(J), axis = 0).reshape((1,-1))

	def _learned_feature_costs(self, waypt):
		"""
		Computes the cost for all the learned features.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		# get the number of learned features
		n_learned = np.count_nonzero(self.environment.is_learned_feat)

		feature_values = []
		for i, feature in enumerate(self.environment.learned_feats):
			# get the value of the feature
			feat_idx = self.environment.num_feats - n_learned + i
			if self.weights[feat_idx] == 0.0:
				feature_value = 0.0
			else:
				feature_value = self.interpolate_features(curr_waypt, prev_waypt, feat_idx, NUM_STEPS=1) # CHANGED
			feature_values.append(feature_value)
		# calculate the cost
		return np.matmul(self.weights[-n_learned:], np.array(feature_values))*np.linalg.norm(curr_waypt - prev_waypt)

	def _learned_feature_cost_derivatives(self, waypt):
		"""
		Computes the cost derivatives for all the learned features.
		---
		input waypoint, output scalar cost
		"""
		# get the number of learned features
		n_learned = np.count_nonzero(self.environment.is_learned_feat)

		J = []
		sols = []
		for i, feature in enumerate(self.environment.learned_feats):
			feat_idx = self.environment.num_feats - n_learned + i
			if self.weights[feat_idx] == 0.0:
				J.append(np.zeros(len(waypt)))
				continue

			# Setup for computing Jacobian.
			x = torch.tensor(waypt, requires_grad=True)

			# Get the value of the feature
			feat_val = torch.tensor(0.0, requires_grad=True)
			NUM_STEPS = 1 # CHANGED
			for step in range(NUM_STEPS):
				delta = torch.tensor((1.0 + step)/NUM_STEPS, requires_grad=True)
				inter_waypt = x[:7] + delta * (x[7:] - x[:7])
				# Compute feature value.
				z = self.environment.feat_func_list[feat_idx](self.environment.raw_features(inter_waypt).float(), torchify=True)
				feat_val = feat_val + z
			y = feat_val / torch.tensor(float(NUM_STEPS), requires_grad=True)
			y = y * torch.tensor(self.weights[-n_learned+i], requires_grad=True) * torch.norm(x[7:] - x[:7])
			y.backward()
			J.append(x.grad.data.numpy())
		return np.sum(np.array(J), axis = 0).reshape((1,-1))



	# TODO: add interpolation for goal_dist cost so it behaves like other costs
	def gen_goal_cost(self, goal_num):
		def goal_cost(waypt):
			prev_waypt = waypt[0:7]
			curr_waypt = waypt[7:14]
			feature_idx = self.feat_list.index('goal'+str(goal_num)+'_dist')
			feature = self.environment.goal_dist_features(goal_num, waypt)
			return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)
		return goal_cost

	# ---- Here's TrajOpt --- #

	def trajOpt(self, start, goal, goal_pose, traj_seed=None, EE_rot_angle=None):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		Paramters:
			start -- The start position.
			goal -- The goal position.
			goal_pose -- The goal pose (optional: can be None).
			traj_seed [optiona] -- An optional initial trajectory seed.

		Returns:
			waypts_plan -- A downsampled trajectory resulted from the TrajOpt
			optimization problem solution.
		"""
		support = np.arange(len(self.weights))[self.weights != 0.0]
		nonzero_feat_list = np.array(self.environment.feat_list)[support]
		contains_learned_feat = any(np.array(self.environment.is_learned_feat)[support])

		contains_world_eff = "world_efficiency" in nonzero_feat_list

		print "Planning with features:", nonzero_feat_list

		use_constraint = self.use_constraint_learned or not contains_learned_feat

		# --- Initialization --- #
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]))
		with self.environment.robot:
			self.environment.robot.SetDOFValues(aug_start)

			# --- Linear interpolation seed --- #
			if contains_world_eff:
				init_waypts = self.get_min_world_dist_waypts(start, goal_pose, self.num_waypts)
				print 'Using world_efficiency_cost initialization'
			elif traj_seed is None:
				#print("Using straight line initialization!")
				init_waypts = np.zeros((self.num_waypts,7))
				for count in range(self.num_waypts):
					init_waypts[count,:] = start + count/(self.num_waypts - 1.0)*(goal - start)
			else:
				#print("Using trajectory seed initialization!")
				init_waypts = traj_seed

			if EE_rot_angle is not None:
				rot_interp_weights = np.linspace(0, 1, self.num_waypts) ** 1.5
				init_waypts[:, 6] = init_waypts[:, 6] * (1 - rot_interp_weights) + EE_rot_angle * rot_interp_weights
			# --- Request construction --- #
			# If pose is given, must include pose constraint.
			if goal_pose is not None and not self.prefer_angles and use_constraint:
				#print("Using goal pose for trajopt computation.")
				xyz_target = goal_pose
				quat_target = [1,0,0,0] # wxyz
				constraint = [
					{
						"type": "pose",
						"params" : {"xyz" : xyz_target,
									"wxyz" : quat_target,
									"link": "j2s7s300_link_7",
									"rot_coeffs" : [0,0,0],
									"pos_coeffs" : [35,35,35],
									}
					}
				]
			elif use_constraint:
				#print("Using goal for trajopt computation.")
				constraint = [
					{
						"type": "joint",
						#"params": {"vals": goal.tolist()}
						"params": {"vals": list(goal)}
					}
				]
			else:
				constraint = []
			request = {
				"basic_info": {
					"n_steps": self.num_waypts,
					"manip" : "j2s7s300",
					"start_fixed" : True,
					"max_iter" : self.MAX_ITER
				},
				# this is implemented instead through the use of the efficiency feature
				"costs": [
				#{
				#	"type": "joint_vel",
				#	"params": {"coeffs": [0.22]}
				#}
				],
				"constraints": constraint,
				"init_info": {
					"type": "given_traj",
					"data": init_waypts.tolist()
				}
			}

			s = json.dumps(request)
			prob = trajoptpy.ConstructProblem(s, self.environment.env)
			for t in range(1, self.num_waypts):
				if 'coffee' in nonzero_feat_list:
					prob.AddCost(self.coffee_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "coffee%i"%t)
				if 'table' in nonzero_feat_list:
					prob.AddCost(self.table_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "table%i"%t)
				if 'laptop' in nonzero_feat_list:
					prob.AddCost(self.laptop_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "laptop%i"%t)
				if 'origin' in nonzero_feat_list:
					prob.AddCost(self.origin_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "origin%i"%t)
				if 'human' in nonzero_feat_list:
					prob.AddCost(self.human_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "human%i"%t)
				if 'efficiency' in nonzero_feat_list:
					prob.AddCost(self.efficiency_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "efficiency%i"%t)
				if 'world_efficiency' in nonzero_feat_list:
					prob.AddCost(self.world_efficiency_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "world_efficiency%i"%t)
				if 'efficiency_clip' in nonzero_feat_list:
					prob.AddCost(self.efficiency_clip_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "efficiency_clip%i"%t)
				if contains_learned_feat:
					prob.AddErrorCost(self.learned_feature_costs, self.learned_feature_cost_derivatives, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "ABS", "learned_features%i"%t)
			# give goal_dist cost 2 time points like the above costs
			goal_num = 0
			while 'goal'+str(goal_num)+'_dist' in nonzero_feat_list:
				for t in range(1, self.num_waypts):
					prob.AddCost(self.gen_goal_cost(goal_num), [(t,j) for j in range(7)], "goal%i_dist%i"%(goal_num, t))
				goal_num += 1
			for t in range(1,self.num_waypts - 1):
				prob.AddConstraint(self.environment.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)

			result = trajoptpy.OptimizeProblem(prob)
			return result.GetTraj()

	def replan(self, start, goal, goal_pose, weights, T, timestep, start_time=0.0,
			   seed=None, return_plan=False, return_both=False, EE_rot_angle=None):
		"""
		Replan the trajectory from start to goal given weights.
		---
		Parameters:
			start -- Start position
			goal -- List of goal angles
			goal_pose -- Goal pose (optional: can be None).
			weights -- Weights used for the planning objective.
			T [float] -- Time horizon for the desired trajectory.
			timestep [float] -- Frequency of waypoints in desired trajectory.
			belief -- Beliefs for each goal in goals
		Returns:
			traj [Trajectory] -- The optimal trajectory satisfying the arguments.
		"""
		assert weights is not None, "The weights vector is empty. Cannot plan without a cost preference."
		self.weights = weights

		num_traj_waypts = int((T-start_time)/timestep) + 1

		import time
		trajopt_start_time = time.time()

		# if we are only using world_efficiency cost, we can compute the optimal solution
		support = np.arange(len(self.weights))[self.weights != 0.0]
		nonzero_feat_list = np.array(self.environment.feat_list)[support]
		if len(nonzero_feat_list) == 1 and "world_efficiency" in nonzero_feat_list:
			traj = Trajectory(self.get_min_world_dist_waypts(start, goal_pose, num_traj_waypts, EE_rot_angle),
							  np.linspace(start_time, T, num_traj_waypts))
			traj_plan = Trajectory(self.get_min_world_dist_waypts(start, goal_pose, num_traj_waypts, EE_rot_angle),
								   np.linspace(start_time, T, num_traj_waypts))

			print "planning took:", time.time() - trajopt_start_time
			if return_both:
				return traj, traj_plan
			elif return_plan:
				return traj_plan
			else:
				return traj

		waypts = self.trajOpt(start, goal, goal_pose, traj_seed=seed, EE_rot_angle=EE_rot_angle)
		print "planning took:", time.time() - trajopt_start_time
		waypts_time = np.linspace(start_time, T, self.num_waypts)
		traj = Trajectory(waypts, waypts_time)
		if return_both:
			return traj.resample(num_traj_waypts), traj
		elif return_plan:
			return traj
		else:
			return traj.resample(num_traj_waypts)

	def get_min_world_dist_waypts(self, start, goal_pose, num_waypts, EE_rot_angle=None):
		goal_pose = np.array(goal_pose)

		move_robot(self.bullet_environment["robot"], np.append(start.reshape(7), np.array([0.0, 0.0, 0.0])))
		start_pose = robot_coords(self.bullet_environment["robot"])[-1]

		mix_coeffs = np.linspace(0, 1, num_waypts).reshape((num_waypts, 1))
		poses = start_pose.reshape((1,3)) * (1 - mix_coeffs) + goal_pose.reshape((1,3)) * mix_coeffs # shape (num_waypts, 3)

		waypts = np.empty((num_waypts, 7))
		waypts[0] = start
		for i in range(1, num_waypts):
			waypt = p.calculateInverseKinematics(self.bullet_environment["robot"], 7, poses[i])[:7]
			move_robot(self.bullet_environment["robot"], np.append(waypt, np.array([0.0, 0.0, 0.0])))
			waypts[i] = waypt

		if EE_rot_angle:
			rot_interp_weights = np.linspace(0, 1, num_waypts) ** 10
			waypts[:, 6] = waypts[:, 6] * (1 - rot_interp_weights) + EE_rot_angle * rot_interp_weights
		return waypts

def expected_goal(belief, goals):
	return np.sum([goal*prob for goal, prob in zip(goals, belief)], axis=0)
