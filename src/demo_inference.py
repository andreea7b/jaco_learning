#! /usr/bin/env python
"""
This class performs IRL inference on either pre-recorded human demonstrations,
or a simulated demonstration.
Author: Andreea Bobu (abobu@eecs.berkeley.edu)
"""
import math
import sys, os
import glob

from planners.trajopt_planner import TrajoptPlanner
from learners.demo_learner import DemoLearner
from utils.openrave_utils import *
from utils.environment import Environment

import numpy as np
import pickle, yaml

class DemoInference(object):
	"""
	This class performs IRL inference on human demonstrations or simulated demonstrations.
	"""
	def __init__(self, loadfile):
		with open(loadfile, 'r') as stream:
			params = yaml.load(stream)

		# ----- General Setup ----- #
		self.prefix = params["setup"]["prefix"]
		self.feat_list = params["setup"]["feat_list"]
		self.demo_spec = params["setup"]["demo_spec"]

		# Openrave parameters for the environment.
		model_filename = params["setup"]["model_filename"]
		object_centers = params["setup"]["object_centers"]
		self.environment = Environment(model_filename, object_centers)

		# Learner setup.
		constants = {}
		constants["trajs_path"] = params["learner"]["trajs_path"]
		constants["betas_list"] = params["learner"]["betas_list"]
		constants["weight_vals"] = params["learner"]["weight_vals"]
		constants["FEAT_RANGE"] = params["learner"]["FEAT_RANGE"]
		self.learner = DemoLearner(self.feat_list, self.environment, constants)

		if self.demo_spec == "simulate":
			# Task setup.
			pick = params["sim"]["task"]["start"]
			place = params["sim"]["task"]["goal"]
			self.start = np.array(pick)*(math.pi/180.0)
			self.goal = np.array(place)*(math.pi/180.0)
			self.goal_pose = None if params["sim"]["task"]["goal_pose"] == "None" else params["sim"]["task"]["goal_pose"]
			self.T = params["sim"]["task"]["T"]
			self.timestep = params["sim"]["task"]["timestep"]
			self.weights = params["sim"]["task"]["feat_weights"]
			
			# Planner Setup.
			planner_type = params["sim"]["planner"]["type"]
			if planner_type == "trajopt":
				max_iter = params["sim"]["planner"]["max_iter"]
				num_waypts = params["sim"]["planner"]["num_waypts"]
				
				# Initialize planner and compute trajectory simulation.
				self.planner = TrajoptPlanner(self.feat_list, max_iter, num_waypts, self.environment)
			else:
				raise Exception('Planner {} not implemented.'.format(planner_type))
			
			self.traj = [self.planner.replan(self.start, self.goal, self.goal_pose, self.weights, self.T, self.timestep)]
			plotTraj(self.environment.env, self.environment.robot, self.environment.bodies, self.traj[0].waypts, size=0.015,color=[0, 0, 1])
			plotCupTraj(self.environment.env, self.environment.robot, self.environment.bodies, [self.traj[0].waypts[-1]],color=[0,1,0])
		else:
			data_path = params["setup"]["demo_dir"]
			data_str = self.demo_spec.split("_")
			data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')) + data_path
			if data_str[0] == "all":
				file_str = data_dir + '/*task{}*.p'.format(data_str[1])
			else:
				file_str = data_dir + "/demo_ID{}_task{}*.p".format(data_str[0], data_str[1])

			self.traj = [pickle.load(open(demo_file, "rb")) for demo_file in glob.glob(file_str)]
		
		self.learner.learn_weights(self.traj)

if __name__ == '__main__':
	if len(sys.argv) < 1:
		print "ERROR: Need to provide parameters file (yaml) as input."
	else:
		DemoInference(sys.argv[1])


