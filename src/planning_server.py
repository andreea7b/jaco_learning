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
from learners.teleop_learner import TeleopLearner
from utils.environment import Environment
from utils.openrave_utils import robotToCartesian
from utils.environment_utils import *

import numpy as np
import pickle

import yaml
import json
import socket

PORT_NUM = 10000

class PlanningServer():
    def __init__():
        setup_planner()

        # setup socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', PORT_NUM))

        # wait for planning queries
        while True:
            connection, client_address = sock.accept()
            try:
                trajopt_query_bytes = bytearray()
                while True:
                    data = connection.recv()
                    if data:
                        trajopt_query_bytes.extend(data)
                    else:
                        break
                trajopt_query = json.loads(trajopt_query_bytes)
                print trajopt_query
                # plan here
                #connection.sendall(...plan...)
            finally:
                connection.close()


    def setup_planner():
        with open('../config/teleop_inference.yaml') as f:
            config = yaml.load(f)

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

if __name__ == "__main__":
    PlanningServer()
