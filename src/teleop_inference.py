#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu), Matthew Zurek
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
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
from utils import ros_utils
from utils.environment import Environment
from utils.openrave_utils import robotToCartesian

import numpy as np
import pickle

from openravepy import RaveCreatePhysicsEngine, RaveCreateModule


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

		# Load mode
		mode = rospy.get_param("setup/sim_mode")

		# Load parameters and set up subscribers/publishers.
		self.load_parameters(mode)
		self.register_callbacks(mode)

		# For presentation recording (TODO: delete later)
		#recorder = RaveCreateModule(self.sim_environment.env, 'viewerrecorder')
		#self.sim_environment.env.AddModule(recorder, '')
		#codecs = recorder.SendCommand('GetCodecs')
		#filename = 'sim.mpg'
		#codec = 13

		# Start admittance control mode.
		if mode == "real":
			ros_utils.start_admittance_mode(self.prefix)

		# Publish to ROS at 100hz.
		r = rospy.Rate(100)

		print "----------------------------------"
		if mode == "real":
			print "Moving robot, press ENTER to quit:"

			while not rospy.is_shutdown():
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break
				self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg((180/np.pi)*self.cmd))
				r.sleep()

		elif mode == "sim":
			print "Simulating robot, press ENTER to quit:"
			physics_engine = RaveCreatePhysicsEngine(self.sim_environment.env, 'ode')
			self.sim_environment.env.SetPhysicsEngine(physics_engine)
			self.sim_environment.env.StartSimulation(1e-1, True)
			self.sim_environment.robot.SetDOFValues( np.hstack((self.start, np.array([0,0,0]))) )

			# TODO: remove
			#self.recorder = recorder
			#recorder.SendCommand('Start 640 480 30 codec %d timing realtime filename %s\nviewer %s'%(codec, filename, self.sim_environment.env.GetViewer().GetName()))

			loop_iter = 0
			while not rospy.is_shutdown():
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break
				## TODO: this should be done in another thread like a normal subscriber
				if loop_iter % 10 == 0:
					joint_angles = np.diag(self.sim_environment.robot.GetDOFValues() * (180/np.pi))
					self.joint_angles_callback(ros_utils.cmd_to_JointAnglesMsg(joint_angles))
					loop_iter = 0
					self.sim_environment.update_vel(self.cmd) ## TODO: make sure that update_vel is supposed to take radians
				loop_iter += 1
				r.sleep()

		print "----------------------------------"

		if mode == "real":
			ros_utils.stop_admittance_mode(self.prefix)

	def load_parameters(self, mode):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""
		# ----- General Setup ----- #
		self.prefix = rospy.get_param("setup/prefix")
		self.start = np.array(rospy.get_param("setup/start"))*(math.pi/180.0)
		# TODO: remove one of these
		#self.goal_poses = np.array(rospy.get_param("setup/goal_poses"))
		fixed_goals = np.array(rospy.get_param("setup/goals"))*(math.pi/180.0)
		try:
			learned_goals = np.load('learned_goals.npy')
			self.goals = np.vstack((fixed_goals, learned_goals))
		except IOError:
			self.goals = fixed_goals
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.save_dir = rospy.get_param("setup/save_dir")
		self.feat_list = rospy.get_param("setup/feat_list")
		self.weights = rospy.get_param("setup/feat_weights")
		self.weights = self.weights + ([0.] * len(self.goals))
		self.goal_weights = []
		num_feats = len(self.feat_list)
		for goal_num in range(len(self.goals)):
			self.feat_list.append("goal"+str(goal_num)+"_dist")
			goal_weights = list(self.weights)
			#goal_weights[num_feats + goal_num] = 1.
			self.goal_weights.append(goal_weights)

		# Openrave parameters for the environment.
		model_filename = rospy.get_param("setup/model_filename")
		object_centers = rospy.get_param("setup/object_centers")
		self.environment = Environment(model_filename, object_centers,
									   goals=self.goals,
		                               use_viewer=(mode == "real"),
									   plot_objects=False)
		# turns off the viewer for the calculations environment when in sim mode
		self.goal_locs = self.environment.goal_locs

		# ----- Planner Setup ----- #
		# Retrieve the planner specific parameters.
		planner_type = rospy.get_param("planner/type")
		if planner_type == "trajopt":
			max_iter = rospy.get_param("planner/max_iter")
			num_waypts = rospy.get_param("planner/num_waypts")

			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(self.feat_list, max_iter, num_waypts, self.environment)
		else:
			raise Exception('Planner {} not implemented.'.format(planner_type))
		# TODO: do something better than goals[0]?
		self.traj = self.planner.replan(self.start, self.goals[0], None, self.goal_weights[0], self.T, self.timestep)
		self.traj_plan = self.traj.downsample(self.planner.num_waypts)

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

			self.controller = PIDController(P, I, D, epsilon, MAX_CMD, self)
		else:
			raise Exception('Controller {} not implemented.'.format(controller_type))

		# Planner tells controller what plan to follow.
		self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint velocities.
		self.cmd = np.zeros((7,7))

		# ----- Learner Setup ----- #
		betas = np.array(rospy.get_param("learner/betas"))
		if len(fixed_goals) == len(self.goals):
			goal_beliefs = np.ones(len(self.goals))/len(self.goals)
		else:
			goal_beliefs = rospy.get_param("learner/goal_beliefs")
		beta_priors = rospy.get_param("learner/beta_priors")
		inference_method = rospy.get_param("learner/inference_method")
		self.beta_method = rospy.get_param("learner/beta_method")
		self.learner = TeleopLearner(self, goal_beliefs, beta_priors, betas, inference_method, self.beta_method)
		self.running_inference = False
		self.last_inf_idx = -1

		# ----- Input Device Setup ----- #
		self.joy_environment = Environment(model_filename, object_centers,
										   goals=self.goals,
										   use_viewer=False,
										   plot_objects=False)
		self.joy_cmd = np.zeros((7,7))
		self.assistance_method = rospy.get_param("learner/assistance_method")
		self.alpha = 1. # in [0, 1]; higher numbers give more control to human

		# ----- Simulation Setup ----- #
		if mode == "sim":
			self.sim_environment = Environment(model_filename, object_centers,
											   goals=self.goals,
			                                   use_viewer=True,
											   plot_objects=False)

	def register_callbacks(self, mode):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		if mode == "real":
			# Create joint-velocity publisher.
			self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)
			# Create subscriber to joint_angles.
			rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		elif mode == "sim":
			pass
		# Create subscriber to input joystick.
		rospy.Subscriber('joy', Joy, self.joystick_input_callback, queue_size=1)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate velocity command to move the robot to the target.
		"""
		# TODO: remove
		#if self.next_waypt_idx >= 40:
		#	self.recorder.SendCommand('Stop')

		# Read the current joint angles from the robot.
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = curr_pos*(math.pi/180.0)

		if self.start_T is not None and \
		   (time.time() - self.start_T >= self.timestep * self.next_waypt_idx) \
		    and not self.next_waypt_idx >= len(self.traj_hist):
			self.traj_hist[self.next_waypt_idx] = self.curr_pos.reshape(7)
			self.next_waypt_idx += 1
			print "timestep:", self.next_waypt_idx
			if not self.running_inference:
				self.running_inference = True
				inference_thread = Thread(target=self.learner.inference_step)
				inference_thread.start()

		if self.assistance_method == "blend":
			ctl_cmd = self.controller.get_command(self.curr_pos)
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
				if self.beta_method == "joint":
					goal, beta = self.learner.argmax_joint_beliefs
					print 'goal:', goal, 'beta:', beta
					print 'joint beliefs:', self.learner.joint_beliefs
				elif self.beta_method == "estimate":
					goal, beta = self.learner.argmax_estimate
					print 'goal:', goal, 'beta:', beta
					print 'beta estimates:', self.learner.beta_estimates
					print 'goal beliefs:', self.learner.goal_beliefs
				self.alpha = beta_arbitration(beta)
				self.traj = self.learner.cache['goal_traj_by_idx'][self.last_inf_idx][goal]
				self.traj_plan = self.learner.cache['goal_traj_plan_by_idx'][self.last_inf_idx][goal]
				self.controller.set_trajectory(self.traj,
											   path_start_T=self.idx_to_time(self.last_inf_idx))
			if np.allclose(self.joy_cmd, np.zeros((7,7))):
				self.cmd = self.joy_cmd
			else:
				self.cmd = self.alpha * self.joy_cmd + (1. - self.alpha) * ctl_cmd
		elif self.assistance_method == "expected":
			raise NotImplementedError
		else:
			raise ValueError
		# Update cmd from PID based on current position.
		#self.cmd = self.controller.get_command(self.curr_pos)

	def joystick_input_callback(self, msg):
		"""
		Reads joystick commands
		"""
		#start = time.time()
		FREQ = 10
		#joy_input = (msg.axes[1], msg.axes[0], msg.axes[2]) # corrects orientation
		joy_input = (msg.axes[0], -msg.axes[1], msg.axes[4])

		#pos = self.curr_pos.reshape(7) + np.array([0,0,np.pi,0,0,0,0])
		curr_angles = np.append(self.curr_pos.reshape(7), np.array([0,0,0]))

		# not using EE orientation
		dis = np.array(joy_input)
		# preserving EE orientation
		#dis = np.array(joy_input + (0,0,0))

		# clamp/scale dis
		dis = dis * 0.5 / FREQ
		#dis = np.clip(dis, -0.5, 0.5)

		err = dis
		angles = np.copy(curr_angles)
		if np.linalg.norm(err) < 0.1 / FREQ:
			self.joy_cmd = np.zeros((7, 7))
			return
		with self.joy_environment.robot:
			self.joy_environment.robot.SetDOFValues(angles)
			xyz = robotToCartesian(self.joy_environment.robot)[6]
			for k in range(3):
				Jt = self.joy_environment.robot.ComputeJacobianTranslation(7, xyz)
				#Jo = self.joy_environment.robot.ComputeJacobianAxisAngle(7)
				J = Jt # J = np.vstack((Jt, Jo))
				angle_step_dir = np.dot(J.T, err)
				pred_xyz_step = np.dot(J, angle_step_dir)
				step_size = np.dot(err, pred_xyz_step)/(np.linalg.norm(pred_xyz_step) ** 2)
				angles += step_size * angle_step_dir
				self.joy_environment.robot.SetDOFValues(angles)
				new_xyz = robotToCartesian(self.joy_environment.robot)[6]
				err -= (new_xyz - xyz)
				xyz = new_xyz
		cmd = (angles - curr_angles) * FREQ
		# clamp large joints if you want here
		self.joy_cmd = np.diag(cmd[:7])
		#end = time.time()
		#print 'command time', end - start

	def _joystick_input_callback(self, msg):
		"""
		Reads joystick commands
		"""
		#joy_cmd = (msg.axes[1], msg.axes[0], msg.axes[2]) # corrects orientation
		joy_cmd = (msg.axes[1], msg.axes[0], msg.axes[3])
		#pos = self.curr_pos.reshape(7) + np.array([0,0,np.pi,0,0,0,0])
		pos = self.curr_pos.reshape(7)
		with self.joy_environment.robot:
			self.joy_environment.robot.SetDOFValues(np.append(pos, np.array([0,0,0])))
			xyz = robotToCartesian(self.joy_environment.robot)[6]
			Jt = self.joy_environment.robot.ComputeJacobianTranslation(7, xyz)
			Jo = self.joy_environment.robot.ComputeJacobianAxisAngle(7)

		# not using EE orientation
		#J, dis = Jt, np.array(joy_cmd)
		# preserving EE orientation
		J, dis = np.vstack((Jt, Jo)), np.array(joy_cmd + (0,0,0))

		# clamp/scale dis
		#dis = 0.1 * dis
		# dis = np.clip(dis, -limit, limit)

		# using pseudoinverse
		#J_inv = np.linalg.pinv(J)
		#cmd = np.dot(J_inv, dis)

		# using transpose
		#cmd = np.dot(J.T, dis)

		# using damped least squares
		lamb = 0.5
		A = np.dot(J, J.T) + lamb * np.eye(J.shape[0])
		cmd = np.dot(J.T, np.linalg.solve(A, dis))

		# clamp large joints if you want here
		self.cmd = np.diag(cmd)

	def idx_to_time(self, idx):
		return self.start_T + idx * self.timestep

def beta_arbitration(beta):
	#return 1
	return np.clip(1 / beta, 0, 1)
	#return np.clip(0.5 / beta, 0, 1)
	#return np.clip(np.exp(-beta + 0.1), 0, 1)

if __name__ == '__main__':
	TeleopInference()
