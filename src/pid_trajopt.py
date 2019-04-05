#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it maintains a fixed distance to a target. 
Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math, copy
import pid
import tf
import sys, select, os
import thread
import argparse
import actionlib
import time
import trajopt_planner, phri_planner, demo_planner, demo_planner_discrete
import ros_utils
import exp_utils.experiment_utils

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
from kinova_msgs.srv import *
from std_msgs.msg import Float32
from sympy import Point, Line

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pickle

# Jaco software name
prefix = 'j2s7s300_driver'

# Coordinates of 7DoF positions
home_pos = [103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]
candlestick_pos = [180.0]*7

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 225.0]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]

place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]
place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

# Constants for pHRI
epsilon = 0.10		# epsilon for when robot think it's at goal
INTERACTION_TORQUE_THRESHOLD = [0.88414821, 17.22751856, -0.40134936,  6.23537946, -0.90013662, 1.32379884,  0.10218059]
INTERACTION_TORQUE_EPSILON = [4.0, 5.0, 3.0, 4.0, 1.5, 1.5, 1.5]
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':5.0, 'human':6.0, 'efficiency':1.0}

# Method types based on which we determine the planner
IMPEDANCE = 'A'
PHRI_LEARNING = 'B'
DEMONSTRATION_LEARNING = 'C'
DISCRETE_DEMONSTRATION_LEARNING = 'D'
EXPERT = 'E'

class PIDVelJaco(object):
	"""
	This class represents a node that moves the Jaco with PID control.
	The joint velocities are computed as:

		V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)
	where:
		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change

	Subscribes to:
		/j2s7s300_driver/out/joint_angles	- Jaco sensed joint angles
		/j2s7s300_driver/out/joint_torques	- Jaco sensed joint torques

	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco commanded joint velocities

	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		sim_flag                  - flag for if in simulation or not
	"""

	def __init__(self, ID, task, method_type, record, replay, simulate, feat_method, feat_list):
		
		# Load parameters
		self.load_parameters(ID, task, method_type, record, replay, simulate, feat_method, feat_list)

		# ---- ROS Setup ---- #

		rospy.init_node("pid_trajopt")
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

		# save experimental data (only if experiment started)
		if self.record and self.reached_start:
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

		if self.replay == False:
			# end admittance control mode
			self.stop_admittance_mode()

		# If we are performind demonstration learning, we just finished receiving a demonstration.
		# Here we process the demonstration and perform inference on it.
		if self.simulate == False:
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

		if self.method_type == DEMONSTRATION_LEARNING:
			num_iter = 0
			while True:
				old_updates = np.array(self.planner.weights)
				self.weights = self.planner.learnWeights(np.array(self.demo),alpha=0.005)
				self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=self.demo_plan)
				new_updates = np.array(self.planner.weights)

				num_iter += 1
				print "error: ",np.linalg.norm(old_updates - new_updates)
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
			print "pi1, theta1, beta_norm: ", pi_new, theta_new, beta_new

			# Version 2 computes beta by looking at the difference in cost:
			if 'efficiency' not in self.feat_list:
				self.planner.feat_list = ['efficiency'] + self.feat_list
				self.planner.num_features = len(self.feat_list) + 1

			self.planner.replan(self.start, self.goal, theta_new, 0.0, self.T, 0.5, seed=self.demo_plan)
			Phi_H = self.planner.featurize(self.demo)
			Phi_R = self.planner.featurize(self.planner.waypts)
			Phi_H = np.array([Phi_H[0]] + [sum(x) for x in Phi_H[1:]])
			Phi_R = np.array([Phi_R[0]] + [sum(x) for x in Phi_R[1:]])
			Phi_delta = Phi_H - Phi_R

			print "Phi_H - Phi_R: ", Phi_delta
			beta_new2 = 1.0 / np.abs(np.dot(theta_new, Phi_delta))
			print "beta_MLE: ", beta_new2
			beta_new3 = 1.0 / np.linalg.norm(Phi_delta)
			print "beta_phi: ", beta_new3
			beta_new4 = 1.0 / np.linalg.norm(self.demo - self.planner.waypts)
			print "beta_L2: ", beta_new4
		elif self.method_type == DISCRETE_DEMONSTRATION_LEARNING:
			self.weights = self.planner.learnWeights(np.array(self.demo))

	def load_parameters(self, ID, task, method_type, record, replay, simulate, feat_method, feat_list):
		"""
		Loading parameters.
		"""
		# task, if any
		self.task = None if task == "None" else task

		# method type - A=IMPEDANCE, B=PHRI_LEARNING, C=DEMONSTRATION_LEARNING, D=DISCRETE_DEMONSTRATION_LEARNING ,E=EXPERT
		self.method_type = method_type

		# can be ALL, MAX, or BETA
		self.feat_method = feat_method

		# can be strings 'table', 'coffee', 'human', 'laptop'
		self.feat_list = feat_list
		self.num_feat = len(self.feat_list)

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
			assert ((self.method_type == DEMONSTRATION_LEARNING) or (self.method_type == DISCRETE_DEMONSTRATION_LEARNING)), "Cannot use simulated demonstrations for a non-demonstration method."
			self.simulate = simulate.split(",")
			self.weights_H = [0.0]*len(self.simulate)

		# ---- Trajectory Setup ---- #

		# total time for trajectory
		self.T = 20.0   #TODO THIS IS EXPERIMENTAL - used to be 15.0

		# initialize trajectory weights and betas

		self.weights = [0.0]*self.num_feat
		if 'efficiency' in self.feat_list:
			self.weights[0] = 1.0
		self.betas = [1.0]*self.num_feat
		self.updates = [0.0]*self.num_feat

		# initialize start/goal based on features
		# by default for table and laptop, these are the pick and place
		# depending on the task for study, pick and place will be different
		pick = pick_basic
		if self.task is None:
			if "coffee" in self.feat_list:
				pick = pick_basic_EEtilt
		elif self.task == "coffee":
			pick = pick_basic_EEtilt
		place = place_lower

		start = np.array(pick)*(math.pi/180.0)
		goal = np.array(place)*(math.pi/180.0)
		self.start = start
		self.goal = goal

		self.curr_pos = None

		# if in demo mode, then set the weights to be optimal
		if self.method_type == EXPERT:
			if self.task == "table":
				feat_list = ["table"]
			elif self.task == "coffee":
				feat_list = ["coffee"]
			for feat in range(0,self.num_feat):
				self.weights[feat] = MAX_WEIGHTS[feat_list[feat]]

		# create the planner
		if self.method_type == PHRI_LEARNING:
			# If physical interactions, use pHRI planner
			self.planner = phri_planner.pHRIPlanner(self.feat_method, self.feat_list, self.task)
		elif self.method_type == DEMONSTRATION_LEARNING:
			# If demonstrations, use demo planner
			self.planner = demo_planner.demoPlanner(self.feat_list, self.task)
		elif self.method_type == DISCRETE_DEMONSTRATION_LEARNING:
			# If discrete demonstrations, use discrete demo planner
			self.planner = demo_planner_discrete.demoPlannerDiscrete(self.feat_list, self.task)
		else:
			# Otherwise, use simple trajopt planner
			self.planner = trajopt_planner.Planner(self.feat_list, self.task)

		if self.simulate is not False:
			# We must simulate an ideal human trajectory according to the human's features.
			for feat in range(len(self.simulate)):
				self.weights_H[feat] = MAX_WEIGHTS[self.simulate[feat]]

			# Temporarily modify the planner in order to get simulated demonstration.
			self.planner.feat_list = self.simulate
			self.planner.num_features = len(self.simulate)

			self.planner.replan(self.start, self.goal, self.weights_H, 0.0, self.T, 0.5, seed=None)
			self.demo = self.planner.waypts
			self.demo_plan = self.planner.waypts_plan

			# Reset the planner to the robot's original configuration.
			self.planner.feat_list = self.feat_list
			self.planner.num_features = len(self.feat_list)
			self.planner.weights = self.weights

		# stores the current trajectory we are tracking, produced by planner
		# If in DEMONSTRATION_LEARNING, this trajectory won't be used.
		self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=None)
		print "original traj: " + str(self.traj)

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		# keeps running time since beginning of program execution
		self.process_start_T = time.time()
		# keeps running time since beginning of path
		self.path_start_T = None

		# ----- Controller Setup ----- #

		# stores maximum COMMANDED joint torques
		max_cmd_torque = 40.0		
		self.max_cmd = max_cmd_torque*np.eye(7)
		# stores current COMMANDED joint torques
		self.cmd = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# P, I, D gains 
		p_gain = 50.0
		i_gain = 0.0
		d_gain = 20.0
		self.P = p_gain*np.eye(7)
		self.I = i_gain*np.eye(7)
		self.D = d_gain*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

		# ---- Experimental Utils ---- #
		self.expUtil = exp_utils.experiment_utils.ExperimentUtils()
		# update the list of replanned trajectories with new trajectory
		self.expUtil.update_replanned_trajList(0.0, self.traj)
		# update the list of replanned waypoints with new waypoints
		self.expUtil.update_replanned_wayptsList(0.0, self.planner.waypts)

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# create joint-velocity publisher
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)
	
	def start_admittance_mode(self):
		"""
		Switches Jaco to admittance-control mode using ROS services
		"""
		service_address = prefix+'/in/start_force_control'
		rospy.wait_for_service(service_address)
		try:
			startForceControl = rospy.ServiceProxy(service_address, Start)
			startForceControl()
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None

	def stop_admittance_mode(self):
		"""
		Switches Jaco to position-control mode using ROS services
		"""
		service_address = prefix+'/in/stop_force_control'
		rospy.wait_for_service(service_address)
		try:
			stopForceControl = rospy.ServiceProxy(service_address, Stop)
			stopForceControl()
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		return -self.controller.update_PID(error)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for
		plotting & analysis
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))
		interaction = False
		for i in range(7):
			#center torques around zero
			torque_curr[i][0] -= INTERACTION_TORQUE_THRESHOLD[i]
			if np.fabs(torque_curr[i][0]) > INTERACTION_TORQUE_EPSILON[i] and self.reached_start:
				interaction = True
		# if experienced large enough interaction force, then deform traj
		if interaction:
			#print "--- INTERACTION ---"
			if self.reached_start and not self.reached_goal:
				timestamp = time.time() - self.path_start_T
				self.expUtil.update_tauH(timestamp, torque_curr)
				self.expUtil.update_interaction_point(timestamp, self.curr_pos)

				if self.method_type == PHRI_LEARNING:
					self.weights = self.planner.learnWeights(torque_curr)
					self.betas = self.planner.betas
					self.betas_u = self.planner.betas_u
					self.updates = self.planner.updates

					print "in joint torques callback: going to plan..."
					self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=self.traj)
					print "in joint torques callback: finished planning -- self.traj = " + str(self.traj)

					# update the experimental data with new weights and new betas
					timestamp = time.time() - self.path_start_T
					self.expUtil.update_weights(timestamp, self.weights)
					self.expUtil.update_betas(timestamp, self.betas)
					self.expUtil.update_betas_u(timestamp, self.betas_u)
					self.expUtil.update_updates(timestamp, self.updates)

					# update the list of replanned trajectories with new trajectory
					self.expUtil.update_replanned_trajList(timestamp, self.traj)

					# update the list of replanned trajectory waypts with new trajectory
					self.expUtil.update_replanned_wayptsList(timestamp, self.planner.waypts)

					# store deformed trajectory
					deformed_traj = self.planner.get_waypts_plan()
					self.expUtil.update_deformed_trajList(timestamp, deformed_traj)

					# store deformed trajectory waypoints
					deformed_traj_waypts = self.planner.waypts_deform
					self.expUtil.update_deformed_wayptsList(timestamp, deformed_traj_waypts)

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
		if self.reached_start and (self.method_type == DEMONSTRATION_LEARNING or self.method_type == DISCRETE_DEMONSTRATION_LEARNING):
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

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:
			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)
			dist_from_start = np.fabs(dist_from_start)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()

				# set start time and the original weights as experimental data
				self.expUtil.set_startT(self.path_start_T)
				timestamp = time.time() - self.path_start_T
			else:
				#print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:			
			t = time.time() - self.path_start_T

			# get next target position from position along trajectory

			self.target_pos = self.planner.interpolate(t + 0.1)

			# check if the arm reached the goal, and restart path
			if not self.reached_goal:
				#print "REACHED START --> EXECUTING PATH"
				dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)
				dist_from_goal = np.fabs(dist_from_goal)

				# check if every joint is close enough to goal configuration
				close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]

				# if all joints are close enough, robot is at goal
				is_at_goal = all(close_to_goal)

				if is_at_goal:
					self.reached_goal = True
			else:
				#print "REACHED GOAL! Holding position at goal."
				self.target_pos = self.goal_pos

				# TODO: this should only set it once!
				self.expUtil.set_endT(time.time())

if __name__ == '__main__':
	if len(sys.argv) < 8:
		print "ERROR: Not enough arguments. Specify ID, method_type, record, replay, feat_method, feat_list"
	else:
		ID = int(sys.argv[1])
		task = sys.argv[2]
		method_type = sys.argv[3]
		record = sys.argv[4]
		replay = sys.argv[5]
		simulate = sys.argv[6]
		feat_method = sys.argv[7]
		feat_list = [x.strip() for x in sys.argv[8].split(',')]
	PIDVelJaco(ID,task,method_type,record,replay,simulate,feat_method,feat_list)