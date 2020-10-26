#! /usr/bin/env python

import math
import sys, select, os
import time
from threading import Thread

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from utils.environment import Environment
from teleop_inference_base import TeleopInferenceBase
from utils.trajectory import Trajectory

import numpy as np
import cPickle as pickle
import yaml

class Init(TeleopInferenceBase):
	def __init__(self):

		inference_config_file = "config/training_inference_config.yaml"
		super(Init, self).__init__(True, inference_config_file)

if __name__ == "__main__":
	Init()
