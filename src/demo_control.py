#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it maintains a fixed distance to a target. Additionally, it supports human-robot interaction in the form of offline demonstrations. 
Author: Andreea Bobu (abobu@eecs.berkeley.edu)
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math, copy
import sys, select, os
import time

from pid_trajopt import PIDControl
from planners import demo_planner
from utils import pid, openrave_utils, ros_utils
from data_processing import experiment_utils

import numpy as np
import pickle

class DemoControl(PIDControl):
	"""
	This class represents a node that moves the Jaco with PID control AND supports learning from human demonstrations.
	"""

	def __init__(self, ID, task, record, replay, simulate, feat_method, feat_list, feat_weight):
		
		# Load parameters
		self.load_parameters(ID, task, record, replay, simulate, feat_method, feat_list, feat_weight)
		
        # ---- ROS Setup ---- #

		rospy.init_node("demo_control")
		self.register_callbacks()

		# publish to ROS at 100hz
		r = rospy.Rate(100)

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"

		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()

		print "----------------------------------"

		if self.replay == False:
			# end admittance control mode
			self.stop_admittance_mode()

		# save experimental data for pHRI corrections (only if experiment started)
		if self.method_type == PHRI_LEARNING and self.record and self.reached_start:
			print "Saving experimental data to file..."
			if self.task == None:
				settings_string = str(ID) + "_" + self.method_type + "_" + self.feat_method + "_" + "_".join(feat_list) + "_correction_" + replay + "_"
			else:
				settings_string = str(ID) + "_" + self.method_type + "_" + self.feat_method + "_" + "_".join(feat_list) + "_correction_" + task + "_"
			weights_filename = "weights_" + settings_string
			betas_filename = "betas_" + settings_string
			betas_u_filename = "betas_u_" + settings_string
			force_filename = "force_" + settings_string
			interaction_pts_filename = "interaction_pts_" + settings_string
			tracked_filename = "tracked_" + settings_string
			deformed_filename = "deformed_" + settings_string
			deformed_waypts_filename = "deformed_waypts_" + settings_string
			replanned_filename = "replanned_" + settings_string
			replanned_waypts_filename = "replanned_waypts_" + settings_string
			updates_filename = "updates_" + settings_string

			self.expUtil.pickle_weights(weights_filename)
			self.expUtil.pickle_betas(betas_filename)
			self.expUtil.pickle_betas_u(betas_u_filename)
			self.expUtil.pickle_force(force_filename)
			self.expUtil.pickle_interaction_pts(interaction_pts_filename)
			self.expUtil.pickle_tracked_traj(tracked_filename)
			self.expUtil.pickle_deformed_trajList(deformed_filename)
			self.expUtil.pickle_deformed_wayptsList(deformed_waypts_filename)
			self.expUtil.pickle_replanned_trajList(replanned_filename)
			self.expUtil.pickle_replanned_wayptsList(replanned_waypts_filename)
			self.expUtil.pickle_updates(updates_filename)

		# If we are performing demonstration learning, we just finished receiving a demonstration.
		# Here we process the demonstration and perform inference on it.
		if (self.method_type == DEMONSTRATION_LEARNING or self.method_type == DISCRETE_DEMONSTRATION_LEARNING) and self.simulate == False:
			if self.replay is not False:
				loadfile = os.path.dirname(os.path.realpath(__file__)) + "/data/demonstrations/demos/demo" + "_" + str(ID) + "_" + replay + ".p"
				self.demo = np.load(loadfile)
				openrave_utils.plotTraj(self.planner.env,self.planner.robot,self.planner.bodies, self.demo, size=0.015,color=[0, 0, 1])
			else:
				# We tracked a real human trajectory. It is not a simulation.
				raw_demo = self.expUtil.tracked_traj[:,1:8]
				# 1. Trim ends of trajectory
				lo = 0
				hi = raw_demo.shape[0] - 1
				while np.linalg.norm(raw_demo[lo] - raw_demo[lo + 1]) < 0.01 and lo < hi:
					lo += 1
				while np.linalg.norm(raw_demo[hi] - raw_demo[hi - 1]) < 0.01 and hi > 0:
					hi -= 1
				raw_demo = raw_demo[lo:hi+1, :]

				# 2. Downsample to the same size as robot trajectory
				desired_length = self.planner.waypts.shape[0]
				step_size = float(raw_demo.shape[0]) / desired_length
				demo = []
				counter = 0
				while counter < raw_demo.shape[0]-1:
					demo.append(raw_demo[int(counter), :])
					counter += step_size
				self.demo = demo
				openrave_utils.plotTraj(self.planner.env,self.planner.robot,self.planner.bodies,self.demo, size=0.015,color=[0, 0, 1])

				# 3. Downsample to get trajopt plan
				desired_length = self.planner.num_waypts_plan
				step_size = float(raw_demo.shape[0]) / desired_length
				demo_plan = []
				counter = 0
				while counter < raw_demo.shape[0]-1:
					demo_plan.append(raw_demo[int(counter), :])
					counter += step_size
				demo_plan[0] = self.planner.waypts_plan[0]
				demo_plan[-1] = self.planner.waypts_plan[-1]
				self.demo_plan = np.array(demo_plan)

				print "Type [yes/y/Y] if you're happy with the demonstration."
				line = raw_input()
				if (line is not "yes") and (line is not "Y") and (line is not "y"):
					print "Not happy with demonstration. Terminating experiment."
					return

				if self.record == True:
					feat_string = "_".join(feat_list_H)
					filename = "demo" + "_" + str(ID) + "_" + feat_string
					savefile = self.expUtil.get_unique_filepath("demos",filename)
					pickle.dump(self.demo, open(savefile, "wb" ))

		if self.method_type == DEMONSTRATION_LEARNING:
			num_iter = 0
			while True:
				old_updates = np.array(self.planner.weights)
				self.weights = self.planner.learnWeights(np.array(self.demo),alpha=0.005)
				self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=self.demo_plan)
				new_updates = np.array(self.planner.weights)

				num_iter += 1
				print "error: ", np.linalg.norm(old_updates - new_updates)
				if np.linalg.norm(old_updates - new_updates) < 5e-4 or np.linalg.norm(new_updates) < 0.1:
					print "Finished in {} iterations".format(num_iter)
					break
			# Compute beta, the rationality coefficient.
			# Version 1 computes beta as a norm of pi:
			pi_new = copy.deepcopy(self.planner.weights)
			if 'efficiency' not in self.feat_list:
				pi_new = [1.0] + pi_new
			beta_new = np.linalg.norm(pi_new)
			theta_new = pi_new / beta_new

			# Version 2 computes beta by looking at the difference in cost:
			if 'efficiency' not in self.feat_list:
				self.planner.feat_list = ['efficiency'] + self.feat_list
				self.planner.num_features = len(self.feat_list) + 1

			self.planner.replan(self.start, self.goal, theta_new, 0.0, self.T, 0.5, seed=self.demo_plan)
			Phi_H = self.planner.featurize(self.demo)
			Phi_R = self.planner.featurize(self.planner.waypts)
			Phi_H = np.array([sum(x) for x in Phi_H])
			Phi_R = np.array([sum(x) for x in Phi_R])
			Phi_delta = Phi_H - Phi_R

			print "Phi_H - Phi_R: ", Phi_delta
			print "pi1, theta1, beta_norm: ", pi_new, theta_new, beta_new
			beta_new2 = 1.0 / np.abs(np.dot(theta_new, Phi_delta))
			print "beta_MLE: ", beta_new2
			beta_new3 = 1.0 / np.linalg.norm(Phi_delta)
			print "beta_phi: ", beta_new3
			beta_new4 = 1.0 / np.linalg.norm(self.demo - self.planner.waypts)
			print "beta_L2: ", beta_new4
		elif self.method_type == DISCRETE_DEMONSTRATION_LEARNING:
			self.weights = self.planner.learnWeights(np.array(self.demo))

			if self.record == True and self.replay is not False:
				feat_string = "_".join(self.feat_list)
				feat_H_string = "_".join(self.feat_list_H)
				filename_w = "weights" + "_" + str(ID) + "_" + feat_string + "_demo_" + feat_H_string
				savefile_w = self.expUtil.get_unique_filepath("weights",filename_w)
				pickle.dump(self.planner.P_bt, open(savefile_w, "wb" ))

	def load_parameters(self, ID, task, record, replay, simulate, feat_method, feat_list, feat_weight):
		"""
		Loading parameters.
		"""
        # Call super class parameters.

		# can be ALL, MAX, or BETA
		self.feat_method = feat_method

		# record experimental data mode 
		if record == "F" or record == "f":
			self.record = False
		elif record == "T" or record == "t" or record == "R":
			self.record = True
		else:
			print "Oopse - it is unclear if you want to record data. Not recording data."
			self.record = False

		# replay experimental data mode 
		if replay == "F" or replay == "f":
			self.replay = False
			# start admittance control mode
			self.start_admittance_mode()
		else:
			self.replay = replay

		# simulate data mode 
		# If true, we simulate data instead of getting it directly from the person.
		# Currently only applicable for demonstrations.
		if simulate == "F" or simulate == "f":
			self.simulate = False
		else:
			self.simulate = True
			self.weights_H = [0.0]*len(self.feat_list_H)

		# ---- Trajectory Setup ---- #

		# total time for trajectory
		self.T = 20.0   #TODO THIS IS EXPERIMENTAL - used to be 15.0

		# initialize trajectory weights and betas

		self.weights = [0.0]*self.num_feat
		if 'efficiency' in self.feat_list:
			self.weights[0] = 1.0
		self.betas = [1.0]*self.num_feat
		self.updates = [0.0]*self.num_feat

		# create the planner
		if self.method_type == DEMONSTRATION_LEARNING:
			# If demonstrations, use demo planner
			self.planner = demo_planner.demoPlanner(self.feat_list, self.task)
		elif self.method_type == DISCRETE_DEMONSTRATION_LEARNING:
			# If discrete demonstrations, use discrete demo planner
			self.planner = demo_planner_discrete.demoPlannerDiscrete(self.feat_list, self.task)

		if self.simulate is True:
			# We must simulate an ideal human trajectory according to the human's features.
			for feat in range(len(self.feat_list_H)):
				self.weights_H[feat] = MAX_WEIGHTS[self.feat_list_H[feat]]

			# Temporarily modify the planner in order to get simulated demonstration.
			self.planner.feat_list = self.feat_list_H
			self.planner.num_features = len(self.feat_list_H)

			self.planner.replan(self.start, self.goal, self.weights_H, 0.0, self.T, 0.5, seed=None)
			self.demo = self.planner.waypts
			self.demo_plan = self.planner.waypts_plan
			openrave_utils.plotTraj(self.planner.env,self.planner.robot,self.planner.bodies, self.demo, size=0.015,color=[0, 0, 1])
			openrave_utils.plotCupTraj(self.planner.env,self.planner.robot,self.planner.bodies,[self.demo[-1]],color=[0,1,0])

			# Reset the planner to the robot's original configuration.
			self.planner.feat_list = self.feat_list
			self.planner.num_features = len(self.feat_list)
			self.planner.weights = self.weights

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""

		# read the current joint angles from the robot
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		self.curr_pos = self.curr_pos*(math.pi/180.0)

		# update the OpenRAVE simulation 
		#self.planner.update_curr_pos(curr_pos)

		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(self.curr_pos)

		# update the experiment utils executed trajectory tracker
		if self.reached_start and not self.reached_goal:
			# update the experimental data with new position
			timestamp = time.time() - self.path_start_T
			self.expUtil.update_tracked_traj(timestamp, self.curr_pos)

		# update cmd from PID based on current position
		if self.reached_start:
			# Allow the person to move the end effector with no control resistance.
			self.cmd = np.zeros((7,7))
		else:
			self.cmd = self.PID_control(self.curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

if __name__ == '__main__':
	if len(sys.argv) < 9:
		print "ERROR: Not enough arguments. Specify ID, task, record, replay,
        simulate, feat_method, feat_list, feat_weight."
	else:
		ID = int(sys.argv[1])
		task = sys.argv[2]
		record = sys.argv[3]
		replay = sys.argv[4]
		simulate = sys.argv[5]
		feat_method = sys.argv[6]
		feat_list = [x.strip() for x in sys.argv[7].split(',')]
		feat_weight = [float(x.strip()) for x in sys.argv[8].split(',')]
	DemoControl(ID,task,record,replay,simulate,feat_method,feat_list,feat_weight)

