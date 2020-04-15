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

from openravepy import RaveCreatePhysicsEngine


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
			self.sim_environment.env.StartSimulation(1e-2, True)
			self.sim_environment.robot.SetDOFValues( np.hstack((self.start, np.array([0,0,0]))) )

			loop_iter = 0
			while not rospy.is_shutdown():
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break
				## TODO: this should be done in another thread like a normal subscriber
				if loop_iter % 10 == 0:
					joint_angles = self.sim_environment.robot.GetDOFValues() * (180/np.pi)
					self.joint_angles_callback(ros_utils.cmd_to_JointAnglesMsg(np.diag(joint_angles)))
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
		self.goals = np.array(rospy.get_param("setup/goals"))*(math.pi/180.0)
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.save_dir = rospy.get_param("setup/save_dir")
		self.feat_list = rospy.get_param("setup/feat_list")
		self.weights = rospy.get_param("setup/feat_weights")

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
		self.traj = self.planner.replan(self.start, self.goals[0], None, self.weights, self.T, self.timestep)
		self.traj_plan = self.traj.downsample(self.planner.num_waypts)
		print self.traj.waypts
		print self.traj.waypts_time
		print self.traj_plan

		# Track if you have reached the start/goal of the path.
		self.reached_start = False
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

			self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
		else:
			raise Exception('Controller {} not implemented.'.format(controller_type))

		# Planner tells controller what plan to follow.
		self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint velocities.
		self.cmd = np.zeros((7,7))

		# ----- Learner Setup ----- #
		betas = np.array(rospy.get_param("learner/betas"))
		prior_belief = rospy.get_param("learner/belief")
		inference_method = rospy.get_param("learner/inference_method")
		self.learner = TeleopLearner(self, prior_belief, betas, inference_method)
		self.running_inference = False

		# ----- Input Device Setup ----- #
		self.joy_environment = Environment(model_filename, object_centers,
										   goals=self.goals,
										   use_viewer=False,
										   plot_objects=False)

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
		# Read the current joint angles from the robot.
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = curr_pos*(math.pi/180.0)

		if self.reached_start and \
		   (time.time() - self.controller.path_start_T >= self.timestep * self.next_waypt_idx) \
		   and not self.reached_goal and not self.next_waypt_idx >= len(self.traj_hist):
			self.traj_hist[self.next_waypt_idx] = self.curr_pos.reshape(7)
			self.next_waypt_idx += 1
			print "next timestep"
			print self.next_waypt_idx
			if not self.running_inference:
				self.running_inference = True
				inference_thread = Thread(target=self.learner.inference_step)
				inference_thread.start()

		# Update cmd from PID based on current position.
		#self.cmd = self.controller.get_command(self.curr_pos)
		self.controller.get_command(self.curr_pos)

		# Check is start/goal has been reached.
		if self.controller.path_start_T is not None:
			self.reached_start = True
		if self.controller.path_end_T is not None:
			self.reached_goal = True

	def joystick_input_callback(self, msg):
		"""
		Reads joystick commands
		"""
		#joy_cmd = (msg.axes[1], msg.axes[0], msg.axes[2]) # corrects orientation
		joy_cmd = (msg.axes[1], msg.axes[0], msg.axes[3])

		#pos = self.curr_pos.reshape(7) + np.array([0,0,np.pi,0,0,0,0])
		curr_angles = np.append(self.curr_pos.reshape(7), np.array([0,0,0]))

		# not using EE orientation
		dis = np.array(joy_cmd)
		# preserving EE orientation
		#dis = np.array(joy_cmd + (0,0,0))

		# clamp/scale dis
		dis = 0.01 * dis
		#dis = np.clip(dis, -0.5, 0.5)

		curr_err = np.zeros(len(dis))
		err = curr_err + dis
		angles = curr_angles
		print 'current angles:', curr_angles # TODO: remove after testing
		print 'current error:', curr_err # TODO: remove after testing
		start = time.time() # TODO: remove after testing
		with self.joy_environment.robot:
			self.joy_environment.robot.SetDOFValues(angles)
			xyz = robotToCartesian(self.joy_environment.robot)[6]
			for k in range(10):
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
		end = time.time() # TODO: remove after testing
		print 'time:', end - start # TODO: remove after testing
		cmd = (angles - curr_angles) / 0.01

		# clamp large joints if you want here
		self.cmd = np.diag(cmd)

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


if __name__ == '__main__':
	TeleopInference()
