#! /usr/bin/env python
"""


Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu), Matthew Zurek
"""

import math
import sys, select, os
import time
from threading import Thread


#from utils.environment_utils import *
from utils.trajectory import Trajectory

import numpy as np
import yaml
import cPickle as pickle


PORT_NUM = 10001

class TeleopInferenceBase(object):
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
	"""

	def __init__(self, is_server, config_file):
		self.server = is_server
		self.config_file = config_file
		# Load parameters and set up subscribers/publishers.
		self.load_parameters()

	def load_parameters(self):
		"""
		Loading parameters and setting up variables.
		"""
		with open(self.config_file) as f:
			config = yaml.load(f)
		self.config = config

		# ----- General Setup ----- #
		self.prefix = config["setup"]["prefix"]
		self.T = config["setup"]["T"]
		self.timestep = config["setup"]["timestep"]

		self.start = np.array(config["setup"]["start"])*(math.pi/180.0)
		#self.start += np.random.normal(0, 0.157, self.start.shape)

		# ----- Goals and goal weights setup ----- #
		# TODO: remove one of these
		#self.goal_poses = np.array(config["setup"]["goal_poses"))
		fixed_goals = [np.array(goal)*(math.pi/180.0) for goal in config["setup"]["goals"]]
		try:
			learned_goals = np.load('learned_goals.npy')
			self.goals = fixed_goals + learned_goals
		except IOError:
			self.goals = fixed_goals

		self.num_goals = len(self.goals) + len(config["setup"]["learned_goals"])
		self.visual_goals = [np.array(goal)*(math.pi/180.0) for goal in config["setup"]["visual_goals"]]

		if self.server:
			from utils.environment import Environment
			from planners.trajopt_planner import TrajoptPlanner
			from utils.environment_utils import *
			import pybullet as p

			self.feat_list = config["setup"]["common_feat_list"]
			feat_range = {'table': 0.98,
						  'coffee': 1.0,
						  'laptop': 0.3,
						  'human': 0.3,
						  'efficiency': 0.22,
						  'efficiency_clip': 0.22,
						  'world_efficiency': 0.22,
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

			# Setup pybullet environment

			# ------- setup pybullet -------
			physicsClient = p.connect(p.DIRECT)

			# Add path to data resources for the environment.
			p.setAdditionalSearchPath("../data/resources")

			# Setup the environment.
			self.bullet_environment = setup_environment([]) #setup with no goals


			# ----- Planner Setup ----- #
			# Retrieve the planner specific parameters.
			planner_type = config["planner"]["type"]
			if planner_type == "trajopt":
				max_iter = config["planner"]["max_iter"]
				num_waypts = config["planner"]["num_waypts"]
				prefer_angles = config["planner"]["prefer_angles"]
				use_constraint_learned = config["planner"]["use_constraint_learned"]

				# Initialize planner.
				self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment, self.bullet_environment,
											  prefer_angles=prefer_angles, use_constraint_learned=use_constraint_learned)
			else:
				raise Exception('Planner {} not implemented.'.format(planner_type))

			# ----- Add in learned cost function goals -----
			for learned_goal_save_path in config["setup"]["learned_goals"]:
				# 1. create new weight vectors
				common_weights = common_weights + [0]
				for i in range(len(self.goal_weights)):
					self.goal_weights[i] = np.hstack((self.goal_weights[i], 0))
				learned_goal_weight = np.array(common_weights)
				learned_goal_weight[len(self.feat_list)] = 50.
				self.goal_weights.append(learned_goal_weight)

				# 2. add cost to environment
				#meirl_goal_save_path = "/root/catkin_ws/src/jaco_learning/data/pour_red_meirl.pt"
				# this reuses the first goal for the learned feature
				#self.environment.load_meirl_learned_feature(self.planner, learned_goal_weight, meirl_goal_save_path, goal=self.goals[0])
				# this uses the average demonstration final position
				self.environment.load_meirl_learned_feature(self.planner, learned_goal_weight, learned_goal_save_path)

			self.common_weights = common_weights

		else: #client mode
			from controllers.pid_controller import PIDController
			from learners.teleop_learner import TeleopLearner
			from client import Client


			self.client = Client(PORT_NUM)
			self.environment = self.client


			# ----- Planner Setup ----- #
			self.planner = self.client

			# # Initialize planner and compute trajectory to track.
			# self.traj, self.traj_plan = self.planner.replan(self.start, 0, 0, 0, self.T, self.timestep, return_both=True)
			# Initialize with stationary trajectory
			traj = Trajectory([self.start, self.start], [0, self.T]).resample(int(self.T/self.timestep) + 1)
			self.traj, self.traj_plan = traj, traj.resample(config["planner"]["num_waypts"])

			# Track if you have reached the goal of the path and the episode start time
			self.start_T = None
			self.reached_goal = False

			# Save the current configuration.
			self.curr_pos = None

			# Save a history of waypts
			self.next_waypt_idx = 1
			self.traj_hist = np.zeros((int(self.T/self.timestep) + 1, 7))
			self.traj_hist[0] = self.start

			# ----- Controller Setup ----- #
			# Retrieve controller specific parameters.
			controller_type = config["controller"]["type"]
			if controller_type == "pid":
				# P, I, D gains.
				P = config["controller"]["p_gain"] * np.eye(7)
				I = config["controller"]["i_gain"] * np.eye(7)
				D = config["controller"]["d_gain"] * np.eye(7)

				# Stores proximity threshold.
				epsilon = config["controller"]["epsilon"]

				# Stores maximum COMMANDED joint torques.
				MAX_CMD = config["controller"]["max_cmd"] * np.eye(7)

				self.controller = PIDController(P, I, D, epsilon, MAX_CMD, self)
			else:
				raise Exception('Controller {} not implemented.'.format(controller_type))

			# Planner tells controller what plan to follow.
			self.controller.set_trajectory(self.traj)

			# Set the current goal for which we are assisting
			self.curr_goal = -1

			# Stores current COMMANDED joint velocities.
			self.cmd = np.zeros((7,7))

			self.assistance_method = config["learner"]["assistance_method"]
			self.alpha_method = config["learner"]["alpha_method"]
			self.alpha = 1. # in [0, 1]; higher numbers give more control to human
			self.zero_input_assist = config["learner"]["zero_input_assist"]
			self.joy_cmd = np.zeros((7,7))

			self.exp_data = {}
