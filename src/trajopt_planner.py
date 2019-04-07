import numpy as np
from numpy import linalg
import time
import math
import json

import trajoptpy
import openravepy
from openravepy import *

import openrave_utils
from openrave_utils import *

import copy
import os
import itertools
import pickle

# feature constants
#FEAT_RANGE = {'table':0.98, 'coffee':1.87608702, 'laptop':1.00476554, 'human':1.27253431, 'efficiency':0.0227859767}
FEAT_RANGE = {'table':1.0, 'coffee':1.0, 'laptop':1.0, 'human':1.0, 'efficiency':1.0}
OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [-0.5, -0.5, 0.0]

class Planner(object):
	"""
	This class plans a trajectory from start to goal with TrajOpt. No learning involved.
	"""

	def __init__(self, feat_list, task=None):

		# ---- important internal variables ---- #
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

		# this is the task
		self.task = task

		# these variables are for the upsampled trajectory
		self.waypts = None
		self.num_waypts = None
		self.step_time = None
		self.waypts_time = None
		self.weights = [0.0]*self.num_features

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
		plotCabinet(self.env)
		plotSphere(self.env,self.bodies,OBS_CENTER,0.4)
		plotSphere(self.env,self.bodies,HUMAN_CENTER,10.0)

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

	def velocity_cost(self, waypts):
		"""
		Computes the total velocity cost.
		---
		input trajectory, output scalar cost
		"""
		return self.velocity_features(waypts)

	# -- Efficiency -- #

	def efficiency_features(self, waypt, prev_waypt):
		"""
		Computes efficiency cost for waypoint, confirmed to match trajopt.
		---
		input waypoint, output scalar feature
		"""
		return np.linalg.norm(waypt - prev_waypt)**2 / FEAT_RANGE['efficiency']
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord1 = coords[6][:3]

		if len(prev_waypt) < 10:
			prev_waypt = np.append(prev_waypt.reshape(7), np.array([0,0,0]))
			prev_waypt[2] += math.pi
		self.robot.SetDOFValues(prev_waypt)
		coords = robotToCartesian(self.robot)
		EEcoord2 = coords[6][0:3]

		return np.linalg.norm(EEcoord1 - EEcoord2)**2
		"""

	def efficiency_cost(self, waypt):
		"""
		Computes the total efficiency cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.efficiency_features(curr_waypt,prev_waypt)
		feature_idx = self.feat_list.index('efficiency')
		return feature*self.weights[feature_idx]

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
		return EEcoord_z / FEAT_RANGE['table']

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
		return feature / FEAT_RANGE['laptop']

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
		return feature / FEAT_RANGE['human']

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

		print("I'm in trajopt_PLANNER trajopt pose!")

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

		# Check if efficiency is a feature; if not, use default weight.
		if "efficiency" in self.feat_list:
			coeff = self.weights[self.feat_list.index("efficiency")] / FEAT_RANGE['efficiency']
		else:
			coeff = 1.0

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
				"params": {"coeffs": [coeff]}
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

		print("I'm done with trajopt pose!")

		return self.waypts_plan


	def trajOpt(self, start, goal, traj_seed=None):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""

		print("I'm in normal trajOpt!")
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]))
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		# --- linear interpolation seed --- #
		if traj_seed is None:
			print("using straight line!")
			init_waypts = np.zeros((self.num_waypts_plan,7))
			for count in range(self.num_waypts_plan):
				init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		else:
			print("using traj seed!")
			init_waypts = traj_seed

		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter" : self.MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [0.0]}
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
			if 'efficiency' in self.feat_list:
				prob.AddCost(self.efficiency_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "efficiency%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		return self.waypts_plan

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
		print("weights in replan: " + str(weights))

		if 'coffee' in self.feat_list or self.task=="coffee":
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

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime


