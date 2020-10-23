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

MEIRL_CONFIG_FILE_DICT = {
	2: "config/task2_meirl_config.yaml",
	3: "config/task3_meirl_config.yaml",
	4: "config/task4_meirl_config.yaml"
}

class Training(TeleopInferenceBase):
	def __init__(self, meirl_config_file):

		inference_config_file = "config/training_inference_config.yaml"
		super(Training, self).__init__(True, inference_config_file)

		# Setup meirl
		with open(meirl_config_file) as f:
			config = yaml.load(f)

		NN_dict = config['NN_dict']
		IRL_dict = config['IRL_dict']
		print 'IRL dict', IRL_dict

		# Load demonstrations
		npz_dict = np.load(config['demonstrations_path'], allow_pickle=True)

		s_g_exp_trajs = []
		for trajectory in npz_dict['demos']:
			waypts = trajectory
			waypts_time = np.linspace(0.0, self.T, waypts.shape[0])
			traj = Trajectory(waypts, waypts_time)

			# Downsample/Upsample trajectory to fit desired timestep and T.
			num_waypts = int(self.T / self.timestep) + 1
			if num_waypts < len(traj.waypts):
				demo = traj.downsample(int(self.T / self.timestep) + 1)
			else:
				demo = traj.upsample(int(self.T / self.timestep) + 1)
			s_g_exp_trajs.append([demo.waypts])

		# Train
		common_weights = self.common_weights + [0]
		for i in range(self.num_goals):
			self.goal_weights[i] = np.hstack((self.goal_weights[i], 0))
		learned_goal_weight = np.array(common_weights)
		learned_goal_weight[len(self.feat_list)] = 50.
		self.goal_weights.append(learned_goal_weight)

		self.environment.new_meirl_learned_feature(
			self.planner,
			learned_goal_weight,
			s_g_exp_trajs,
			npz_dict['FKs'],
			npz_dict['IK_goal'],
			npz_dict['FK_goal'],
			[],
			NN_dict,
			'waypt',
			name='_new'
		)
		meirl = self.environment.learned_feats[-1]
		meirl.deep_max_ent_irl(n_iters=IRL_dict['n_iters'],
							   n_cur_rew_traj=IRL_dict['n_cur_rew_traj'],
							   lr=IRL_dict['lr'],
							   weight_decay=IRL_dict['weight_decay'],
							   std=IRL_dict['std'])
		meirl.save(config['save_path'])
		print 'saved learned cost function in:', config['save_path']



if __name__ == "__main__":
	task = int(sys.argv[1])
	Training(MEIRL_CONFIG_FILE_DICT[task])
