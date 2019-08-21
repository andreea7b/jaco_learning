#! /usr/bin/env python
"""
This class performs IRL inference on either pre-recorded human demonstrations,
or a simulated demonstration.
Author: Andreea Bobu (abobu@eecs.berkeley.edu)
"""
import math
import sys, os

from planners.trajopt_planner import TrajoptPlanner
from learners.demo_learner import DemoLearner
from utils.openrave_utils import *

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
        constants["weights_vals"] = params["learner"]["weights_vals"]
        constants["FEAT_RANGE"] = params["learner"]["FEAT_RANGE"]
        self.learner = DemoLearner(self.feat_list, self.environment, constants)

        if self.demo_spec == "simulate":
            # Task setup.
            pick = params["sim"]["task"]["start"]
            place = params["sim"]["task"]["goal"]
            self.start = np.array(pick)*(math.pi/180.0)
            self.goal = np.array(place)*(math.pi/180.0)
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
            
            self.traj = [self.planner.replan(self.start, self.goal, self.weights, self.T, self.timestep)]
			plotTraj(self.environment.env, self.environment.robot, self.environment.bodies, self.traj.waypts, size=0.015,color=[0, 0, 1])
			plotCupTraj(self.environment.env, self.environment.robot, self.environment.bodies, [self.traj[0][-1]],color=[0,1,0])
        else:
            data_str = self.demo_spec.split("_")
            trajs_path = "data/demos/demo_{}_{}.p".format(data_str[0], data_str[1])
            here = os.path.dirname(os.path.realpath(__file__))
            self.traj = pickle.load(open(here + trajs_path, "rb" ))
        
        self.learner.learn_weights(self.traj)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print "ERROR: Need to provide parameters file (yaml) as input."
        return
	DemoInference(sys.argv[1])


