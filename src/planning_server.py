#! /usr/bin/env python

import math
import sys, select, os
import time
from threading import Thread

import kinova_msgs.msg
from kinova_msgs.srv import *
from sensor_msgs.msg import Joy

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from utils.environment import Environment
from teleop_inference_base import TeleopInferenceBase

import numpy as np
import cPickle as pickle
import yaml
import socket

PORT_NUM = 10001

class PlanningServer(TeleopInferenceBase):
	def __init__(self):
		super(PlanningServer, self).__init__(True)

		# setup socket
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		sock.bind(('0.0.0.0', PORT_NUM))

		# wait for planning queries
		sock.listen(1)
		while True:
			connection, client_address = sock.accept()
			try:
				query_bytes = bytearray()
				while True:
					data = connection.recv(4096)
					if data:
						query_bytes.extend(data)
					else:
						break
				query = pickle.loads(str(query_bytes))
				#query = pickle.loads(str(query_bytes.decode('utf-8')))
				print 'query:'
				print query
				type, params = query
				out = [] # what to return
				if type == 0 or type == 2:
					# checking if indices were used instead of angles/poses
					if isinstance(params[1], int):
						params[1] = self.goals[params[1]]
					if isinstance(params[2], int):
						params[2] = self.goal_locs[params[2]]
					traj, plan = self.planner.replan(params[0],
													 params[1],
													 list(params[2]),
													 self.goal_weights[params[3]],
													 params[4],
													 params[5],
													 params[6],
													 params[7],
													 False,
													 True)
					# always plan and get both, then send the desired plan/traj/both back
					if not params[9]:
						if params[8]:
							p_out = plan
						else:
							p_out = traj
					else:
						p_out = [traj, plan]
					out.append(p_out)
				if type == 2:
					# change params to replicate a cost query
					params[0] = traj.waypts
					params[1] = params[3] # weight vector index
					params[2] = params[10] # add_pose_penalty
				if type == 1 or type == 2:
					support = np.arange(len(self.goal_weights[params[1]]))[self.goal_weights[params[1]] != 0.0]
					c_out = np.sum(self.goal_weights[params[1]][support] * np.sum(self.environment.featurize(params[0], support), axis=1))
					out.append(c_out)
				connection.sendall(pickle.dumps(out, 2))
				connection.shutdown(2)
			finally:
				connection.close()

class PlanningServer1():
	def __init__(self):
		self.setup_planner()

		# setup socket
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		sock.bind(('0.0.0.0', PORT_NUM))

		# wait for planning queries
		sock.listen(1)
		while True:
			connection, client_address = sock.accept()
			try:
				trajopt_query_bytes = bytearray()
				while True:
					data = connection.recv(4096)
					if data:
						trajopt_query_bytes.extend(data)
					else:
						break
				trajopt_query = pickle.loads(trajopt_query_bytes)
				print 'received planning query'
				print trajopt_query
				p_out = self.planner.replan(trajopt_query[0],
											trajopt_query[1],
											trajopt_query[2],
											trajopt_query[3],
											trajopt_query[4],
											trajopt_query[5],
											trajopt_query[6],
											trajopt_query[7],
											trajopt_query[8],
											trajopt_query[9])
				connection.sendall(pickle.dumps(p_out))
			finally:
				connection.close()


	def setup_planner(self):
		with open('../config/teleop_inference.yaml') as f:
			config = yaml.load(f)

		self.save_dir = config["setup"]["save_dir"]

		# ----- Goals and goal weights setup ----- #
		# TODO: remove one of these
		#self.goal_poses = np.array(config[""]["setup/goal_poses"))
		fixed_goals = [np.array(goal)*(math.pi/180.0) for goal in config["setup"]["goals"]]
		try:
			learned_goals = np.load('learned_goals.npy')
			self.goals = fixed_goals + learned_goals
		except IOError:
			self.goals = fixed_goals

		self.feat_list = config["setup"]["common_feat_list"]
		feat_range = {'table': 0.98,
					  'coffee': 1.0,
					  'laptop': 0.3,
					  'human': 0.3,
					  'efficiency': 0.22,
					  'proxemics': 0.3,
					  'betweenobjects': 0.2}
		common_weights = config["setup"]["common_feat_weights"]
		goals_weights = []
		goal_dist_feat_weight = config["setup"]["goal_dist_feat_weight"]
		if goal_dist_feat_weight != 0.0:
			# add features for distance from each of the goals
			common_weights = common_weights + ([0.] * len(self.goals))
			num_feats = len(self.feat_list)
			for goal_num in range(len(self.goals)):
				self.feat_list.append("goal"+str(goal_num)+"_dist")
				goal_weights = np.array(common_weights)
				goal_weights[num_feats + goal_num] = goal_dist_feat_weight
				goals_weights.append(goal_weights)
		else:
			# make copies of the common weights
			for goal_num in range(len(self.goals)):
				goals_weights.append(np.array(common_weights))
		self.goal_weights = goals_weights

		# Openrave parameters for the environment.
		model_filename = config["setup"]["model_filename"]
		object_centers = config["setup"]["object_centers"]
		self.environment = Environment(model_filename,
									   object_centers,
									   self.feat_list,
									   feat_range,
									   goals=self.goals,
									   use_viewer=False,
									   plot_objects=False)
		self.goal_locs = self.environment.goal_locs

		# ----- Planner Setup ----- #
		# Retrieve the planner specific parameters.
		planner_type = config["planner"]["type"]
		if planner_type == "trajopt":
			max_iter = config["planner"]["max_iter"]
			num_waypts = config["planner"]["num_waypts"]
			prefer_angles = config["planner"]["prefer_angles"]
			use_constraint_learned = config["planner"]["use_constraint_learned"]

			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment,
										  prefer_angles=prefer_angles, use_constraint_learned=use_constraint_learned)
		else:
			raise Exception('Planner {} not implemented.'.format(planner_type))
		# TODO: do something better than goals[0]?
		#self.traj, self.traj_plan = self.planner.replan(self.start, self.goals[0], None, self.goal_weights[0], self.T, self.timestep, return_both=True)

		# ----- Add in learned cost function goals -----
		for learned_goal_save_path in config["setup"]["learned_goals"]:
			# 1. create new weight vectors
			common_weights = common_weights + [0]
			for i in range(len(self.goal_weights)):
				self.goal_weights[i] = np.hstack((self.goal_weights[i], 0))
			learned_goal_weight = np.array(common_weights)
			learned_goal_weight[len(self.feat_list)] = 1.
			self.goal_weights.append(learned_goal_weight)

			# 2. add cost to environment
			#meirl_goal_save_path = "/root/catkin_ws/src/jaco_learning/data/pour_red_meirl.pt"
			# this reuses the first goal for the learned feature
			#self.environment.load_meirl_learned_feature(self.planner, learned_goal_weight, meirl_goal_save_path, goal=self.goals[0])
			# this uses the average demonstration final position
			self.environment.load_meirl_learned_feature(self.planner, learned_goal_weight, learned_goal_save_path)


if __name__ == "__main__":
	PlanningServer()
