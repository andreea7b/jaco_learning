#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu), Matthew Zurek
"""

import math
import sys, select, os
import time
from threading import Thread

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from learners.teleop_learner import TeleopLearner
#from utils.environment_utils import *

import numpy as np
import pickle

import pybullet as p


class TeleopInference():
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
	"""

	def __init__(self):
		# Load mode
		mode = config["setup"]["sim_mode"]

		# Load parameters and set up subscribers/publishers.
		self.load_parameters(mode)

		print "----------------------------------"
		if mode == "pybullet":
			print("Simulating robot, press ENTER to quit:")
			bullet_start = np.append(self.start.reshape(7), np.array([0.0, 0.0, 0.0]))
			move_robot(self.bullet_environment["robot"], bullet_start)
			# Start simulation.
			while True:
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break

				# Update position.
				self.keyboard_input_callback()

				# Update sim position with new velocity command.
				for i in range(len(self.cmd)):
					p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

				time.sleep(0.05)

			# Disconnect once the session is over.
			p.disconnect()

		print "----------------------------------"

	def load_parameters(self, mode):
		"""
		Loading parameters and setting up variables.
		"""
		# ----- General Setup ----- #
		self.prefix = config["setup"]["prefix"]
		self.T = config["setup"]["T"]
		self.timestep = config["setup"]["timestep"]
		self.save_dir = config["setup"]["save_dir"]

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

		if self.server:
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
			self.traj, self.traj_plan = self.planner.replan(self.start, self.goals[0], None, self.goal_weights[0], self.T, self.timestep, return_both=True)
		else:
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
			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment,
										  prefer_angles=prefer_angles, use_constraint_learned=use_constraint_learned)
			# TODO: do something better than goals[0]?
			self.traj, self.traj_plan = self.planner.replan(self.start, self.goals[0], None, self.goal_weights[0], self.T, self.timestep, return_both=True)

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

		# Stores current COMMANDED joint velocities.
		self.cmd = np.zeros((7,7))

		# ----- Learner Setup ----- #
		betas = np.array(config["learner"]["betas"])
		goal_beliefs = config["learner"]["goal_beliefs"]
		if goal_beliefs != "none":
			goal_beliefs = goal_beliefs / np.linalg.norm(goal_beliefs)
		else:
			goal_beliefs = np.ones(len(self.goals))/len(self.goals)
		assert(len(goal_beliefs) == len(self.goals))
		assert(len(goal_beliefs) == len(self.goal_weights))
		beta_priors = config["learner"]["beta_priors"]
		inference_method = config["learner"]["inference_method"]
		self.beta_method = config["learner"]["beta_method"]
		self.learner = TeleopLearner(self, goal_beliefs, beta_priors, betas, inference_method, self.beta_method)
		self.running_inference = False
		self.last_inf_idx = 0
		self.running_final_inference = False
		self.final_inference_done = False

		self.assistance_method = config["learner"]["assistance_method"]
		self.alpha = 1. # in [0, 1]; higher numbers give more control to human
		self.zero_input_assist = config["learner"]["zero_input_assist"]
		self.joy_cmd = np.zeros((7,7))

		if mode == "pybullet":
			# Connect to a physics simulator.
			physicsClient = p.connect(p.GUI, options="--opengl2")

			# Set camera angle.
			p.resetDebugVisualizerCamera(cameraDistance=2.50, cameraYaw=-85.6, cameraPitch=-17.6, cameraTargetPosition=[0.33,0.05,0.02])

			# Add path to data resources for the environment.
			p.setAdditionalSearchPath("../data/resources")

			# Setup the environment.
			self.bullet_environment = setup_environment(self.goals)

			# Get rid of gravity and make simulation happen in real time.
			p.setGravity(0, 0, 0)
			p.setRealTimeSimulation(1)

		self.exp_data = {
			'joint6_assist': []
		}

	def joint_angles_callback(self, curr_pos):
		"""
		Reads the latest position of the robot and publishes an
		appropriate velocity command to move the robot to the target.
		"""

		# Convert to radians.
		self.curr_pos = curr_pos*(math.pi/180.0)

		if self.start_T is not None and (time.time() - self.start_T >= self.timestep * self.next_waypt_idx):
			if not self.next_waypt_idx >= len(self.traj_hist):
				self.traj_hist[self.next_waypt_idx] = self.curr_pos.reshape(7)
				self.next_waypt_idx += 1
				#print "timestep:", self.next_waypt_idx
				if not self.running_inference:
					#print 'calling inference from', self.next_waypt_idx - 1
					self.running_inference = True
					self.inference_thread = Thread(target=self.learner.inference_step)
					self.inference_thread.start()
			elif not self.running_final_inference:
				self.running_final_inference = True
				self.inference_thread.join()
				self.inference_thread = Thread(target=self.learner.final_step)
				self.inference_thread.start()
			elif self.final_inference_done:
				pass

		ctl_cmd = self.controller.get_command(self.curr_pos)
		#print "joint 6 unblended assistance:", ctl_cmd[6,6]
		self.exp_data['joint6_assist'].append(ctl_cmd[6,6])
		if self.assistance_method == "blend":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
				if self.beta_method == "joint":
					goal, beta = self.learner.argmax_joint_beliefs
					#print 'goal:', goal, 'beta:', beta
					#print 'joint beliefs:', self.learner.joint_beliefs
				elif self.beta_method == "estimate":
					goal, beta = self.learner.argmax_estimate
					#print 'goal:', goal, 'beta:', beta
					#print 'beta estimates:', self.learner.beta_estimates
					#print 'goal beliefs:', self.learner.goal_beliefs
				self.alpha = beta_arbitration(beta)
				self.traj = self.learner.cache['goal_traj_by_idx'][self.last_inf_idx][goal]
				self.traj_plan = self.learner.cache['goal_traj_plan_by_idx'][self.last_inf_idx][goal]
				self.controller.set_trajectory(self.traj,
											   path_start_T=self.idx_to_time(self.last_inf_idx))
			if np.allclose(self.joy_cmd, np.zeros((7,7))) and not self.zero_input_assist:
				self.cmd = self.joy_cmd
			else:
				self.cmd = self.alpha * self.joy_cmd + (1. - self.alpha) * ctl_cmd
		elif self.assistance_method == "expected":
			raise NotImplementedError
		elif self.assistance_method == "none":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
			self.cmd = self.joy_cmd
		else:
			raise ValueError
		# Update cmd from PID based on current position.
		#self.cmd = self.controller.get_command(self.curr_pos)

	def keyboard_input_callback(self):
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.01, 0.01, 0.01]
		time_step = 0.05
		turn_step = 0.05
		EElink = 7

		# Get current EE position.
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		if p.B3G_LEFT_ARROW in keys:
			EEPos_new[1] -= dist_step[1]
		if p.B3G_RIGHT_ARROW in keys:
			EEPos_new[1] += dist_step[1]
		if p.B3G_UP_ARROW in keys:
			EEPos_new[0] -= dist_step[0]
		if p.B3G_DOWN_ARROW in keys:
			EEPos_new[0] += dist_step[0]
		if ord('i') in keys:
			EEPos_new[2] += dist_step[2]
		if ord('k') in keys:
			EEPos_new[2] -= dist_step[2]

		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			jointVelocities = (newPoses - jointPoses) / time_step
		if ord('j') in keys:
			jointVelocities[EElink] += turn_step / time_step
		if ord('l') in keys:
			jointVelocities[EElink] -= turn_step / time_step

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:8])

		# Move arm in openrave as well.
		joint_angles = np.diag(jointPoses[1:8] * (180/np.pi))
		self.joint_angles_callback(joint_angles)


	def idx_to_time(self, idx):
		return self.start_T + idx * self.timestep

def beta_arbitration(beta):
	#return 1 #all joystick
	return 0
	#return np.clip(1 / beta, 0, 1)
	#return np.clip(0.5 / beta, 0, 1)
	#return np.clip(np.exp(-beta + 0.1), 0, 1)

if __name__ == '__main__':
	TeleopInference()
