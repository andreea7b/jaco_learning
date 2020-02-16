#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target. Additionally, it supports human-robot
interaction in the form of online physical corrections.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import sys, select, os
import time

import kinova_msgs.msg
from kinova_msgs.srv import *

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from learners.teleop_learner import TeleopLearner
from utils import ros_utils
from utils.environment import Environment

import numpy as np
import pickle

class TeleopInference():
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.

	Subscribes to:
		/$prefix$/out/joint_angles	- Jaco sensed joint angles
		/$prefix$/out/joint_torques - Jaco sensed joint torques

	Publishes to:
		/$prefix$/in/joint_velocity	- Jaco commanded joint velocities
	"""

	def __init__(self):
		# Create ROS node.
		rospy.init_node("teleop_inference")

		# Load parameters and set up subscribers/publishers.
		self.load_parameters()
		self.register_callbacks()

		# Start admittance control mode.
		ros_utils.start_admittance_mode(self.prefix)

		# Publish to ROS at 100hz.
		r = rospy.Rate(100)

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"

		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break
			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg((180/np.pi)*self.cmd))
			r.sleep()

		print "----------------------------------"

		ros_utils.stop_admittance_mode(self.prefix)

	def load_parameters(self):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""
		# ----- General Setup ----- #
		self.prefix = rospy.get_param("setup/prefix")
		self.start = np.array(rospy.get_param("setup/start"))*(math.pi/180.0)
		self.goals = np.array(rospy.get_param("setup/goals"))*(math.pi/180.0)
		self.goal_pose = None if rospy.get_param("setup/goal_pose") == "None" else rospy.get_param("setup/goal_pose")
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.save_dir = rospy.get_param("setup/save_dir")
		self.feat_list = rospy.get_param("setup/feat_list")
		self.weights = rospy.get_param("setup/feat_weights")
		self.INTERACTION_VELOCITY_EPSILON = np.array(rospy.get_param("setup/INTERACTION_VELOCITY_EPSILON"))

		# Openrave parameters for the environment.
		model_filename = rospy.get_param("setup/model_filename")
		object_centers = rospy.get_param("setup/object_centers")
		for goal_num in range(len(self.goals)):
			# assumes the goal either contains the finger angles (10DOF) or does not (7DOF)
			if len(self.goals[goal_num]) == 7:
				object_centers["GOAL"+str(goal_num)+" ANGLES"] = np.pad(self.goals[goal_num], (0,3), mode='constant')
			else:
				object_centers["GOAL"+str(goal_num)+" ANGLES"] = self.goals[goal_num]
		# object centers holds xyz coords of objects and radian joint coords of goals
		self.environment = Environment(model_filename, object_centers)

		# ----- Planner Setup ----- #
		# Retrieve the planner specific parameters.
		planner_type = rospy.get_param("planner/type")
		prior_belief = rospy.get_param("planner/belief") # this is also used to initialize the learner
		if planner_type == "trajopt":
			max_iter = rospy.get_param("planner/max_iter")
			num_waypts = rospy.get_param("planner/num_waypts")

			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(self.feat_list, max_iter, num_waypts, self.environment)
		else:
			raise Exception('Planner {} not implemented.'.format(planner_type))

		self.traj = self.planner.replan(self.start, self.goals, self.goal_pose, self.weights, self.T, self.timestep, belief=prior_belief)
		self.traj_plan = self.traj.downsample(self.planner.num_waypts)

		# Track if you have reached the start/goal of the path.
		self.reached_start = False
		self.reached_goal = False

		# Save the intermediate target configuration.
		self.curr_pos = None
		self.prev_pos = None
		self.curr_time = None
		self.prev_time = None

		# ----- Controller Setup ----- #
		# Retrieve controller specific parameters.
		controller_type = rospy.get_param("controller/type")
		if controller_type == "pid":
			# P, I, D gains.
			P = rospy.get_param("controller/p_gain") * np.eye(7)
			I = rospy.get_param("controller/i_gain") * np.eye(7)
			D = rospy.get_param("controller/d_gain") * np.eye(7)

			# Stores proximity threshold.
			epsilon = rospy.get_param("controller/epsilon")

			# Stores maximum COMMANDED joint torques.
			MAX_CMD = rospy.get_param("controller/max_cmd") * np.eye(7)

			self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
		else:
			raise Exception('Controller {} not implemented.'.format(controller_type))

		# Planner tells controller what plan to follow.
		self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint torques.
		self.cmd = np.eye(7)

		# ----- Learner Setup ----- #
		constants = {}
		constants["UPDATE_GAINS"] = rospy.get_param("learner/UPDATE_GAINS")
		constants["MAX_WEIGHTS"] = rospy.get_param("learner/MAX_WEIGHTS")
		constants["FEAT_RANGE"] = rospy.get_param("learner/FEAT_RANGE")
		constants["P_beta"] = rospy.get_param("learner/P_beta")
		constants["alpha"] = rospy.get_param("learner/alpha")
		constants["n"] = rospy.get_param("learner/n")
		self.feat_method = rospy.get_param("learner/type")
		self.learner = TeleopLearner(self.feat_method, self.feat_list, self.environment, constants, self.goals, prior_belief)

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# Create joint-velocity publisher.
		self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# Create subscriber to joint_angles.
		rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# Create subscriber to joint torques
		rospy.Subscriber(self.prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target.
		"""
		self.prev_pos = self.curr_pos
		self.prev_time = self.curr_time
		self.curr_time = time.time()

		# Read the current joint angles from the robot.
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = self.curr_pos*(math.pi/180.0)

		interaction = False
		if self.prev_pos is not None:
			dt = self.curr_time - self.prev_time
			obs_vel = ((self.curr_pos - self.prev_pos) / dt).reshape(7)
			if any(np.fabs(obs_vel - self.cmd.diagonal()) > self.INTERACTION_VELOCITY_EPSILON) and self.reached_start:
				print "interaction detected"
				interaction = True

		# Update cmd from PID based on current position.
		self.cmd = self.controller.get_command(self.curr_pos)

		# Check is start/goal has been reached.
		if self.controller.path_start_T is not None:
			self.reached_start = True
		if self.controller.path_end_T is not None:
			self.reached_goal = True
		

		if interaction:
			self.learner.update_beliefs(self.prev_pos.reshape(7), self.curr_pos.reshape(7))
			self.traj = self.planner.replan(self.start, self.goals, self.goal_pose, self.weights, self.T, self.timestep, seed=self.traj_plan.waypts, belief=self.learner.beliefs)
			self.traj_plan = self.traj.downsample(self.planner.num_waypts)
			self.controller.set_trajectory(self.traj)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for
		plotting & analysis
		"""
		# Read the current joint torques from the robot.
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))
		interaction = False
		return
		for i in range(7):
			# Center torques around zero.
			torque_curr[i][0] -= self.INTERACTION_TORQUE_THRESHOLD[i]
			#torque_curr[i][0] -= self.cmd[i][i]
			# Check if interaction was not noise.
			if np.fabs(torque_curr[i][0]) > self.INTERACTION_TORQUE_EPSILON[i] and self.reached_start:
				interaction = True

		# If we experienced large enough interaction force, then learn.
		if interaction:
			# for testing the thresholds/epsilons
			#print "interaction detected"
			#print torque_curr.reshape((1,7))
			#print np.fabs(torque_curr[:, 0]) > self.INTERACTION_TORQUE_EPSILON
			#print self.cmd
			return
			if self.reached_start and not self.reached_goal:
				timestamp = time.time() - self.controller.path_start_T

				self.weights = self.learner.learn_weights(self.traj, torque_curr, timestamp)
				# learn weights here
				betas = self.learner.betas
				betas_u = self.learner.betas_u
				updates = self.learner.updates

				self.traj = self.planner.replan(self.start, self.goals, self.goal_pose, self.weights, self.T, self.timestep, seed=self.traj_plan.waypts, belief=belief)
				self.traj_plan = self.traj.downsample(self.planner.num_waypts)
				self.controller.set_trajectory(self.traj)

if __name__ == '__main__':
	TeleopInference()
