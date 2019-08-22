#! /usr/bin/env python
"""
This node records human demonstrations and saves them to files.

Given a start, a goal, and specific controller parameters, the controller moves
the Jaco manipulator at the start, after which the node awaits for human input
tracking a demonstrated trajectory. Once the user hits Enter, the recording
ends and the resulting waypoints are processed into a trajectory and saved.

Author: Andreea Bobu (abobu@eecs.berkeley.edu)
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math, copy
import sys, select, os
import time

import kinova_msgs.msg
from kinova_msgs.srv import *

from controllers.pid_controller import PIDController
from utils import openrave_utils, ros_utils, experiment_utils
from utils.environment import Environment
from utils.trajectory import Trajectory

import numpy as np
import pickle

class DemoRecorder(object):
	"""
	This class represents a node that moves the Jaco to a start position and
	allows the human to give a demonstration for recording.

	Subscribes to:
		/$prefix$/out/joint_angles	- Jaco sensed joint angles

	Publishes to:
		/$prefix$/in/joint_velocity	- Jaco commanded joint velocities
	"""

	def __init__(self):
		# Create ROS node.
		rospy.init_node("demo_recorder")

		# Load parameters and set up subscribers/publishers.
		self.load_parameters()
		self.register_callbacks()
		ros_utils.start_admittance_mode(self.prefix)

		# Publish to ROS at 100hz
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

		# Process and save the recording.
		raw_demo = self.expUtil.tracked_traj[:,1:8]
				
		# Trim ends of waypoints and create Trajectory.
		lo = 0
		hi = raw_demo.shape[0] - 1
		while np.linalg.norm(raw_demo[lo] - raw_demo[lo + 1]) < 0.01 and lo < hi:
			lo += 1
		while np.linalg.norm(raw_demo[hi] - raw_demo[hi - 1]) < 0.01 and hi > 0:
			hi -= 1
		waypts = raw_demo[lo:hi+1, :]
		waypts_time = np.linspace(0.0, self.T, waypts.shape[0])
		traj = Trajectory(waypts, waypts_time)

		# Downsample/Upsample trajectory to fit desired timestep and T.
		num_waypts = int(self.T / self.timestep) + 1
		if num_waypts < len(traj.waypts):
			demo = traj.downsample(int(self.T / self.timestep) + 1)
		else:
			demo = traj.upsample(int(self.T / self.timestep) + 1)

		# Decide whether to save trajectory
		openrave_utils.plotTraj(self.environment.env, self.environment.robot,
								self.environment.bodies, demo.waypts, size=0.015, color=[0, 0, 1])

		print "Type [yes/y/Y] if you're happy with the demonstration."
		line = raw_input()
		if (line is not "yes") and (line is not "Y") and (line is not "y"):
			print "Not happy with demonstration. Terminating experiment."
		else:
			print "Please type in the ID number (e.g. [0/1/2/...])."
			ID = raw_input()
			print "Please type in the task number (e.g. [0/1/2/...])."
			task = raw_input()
			filename = "demo" + "_ID" + ID + "_task" + task
			savefile = self.expUtil.get_unique_filepath("demos",filename)
			pickle.dump(demo, open(savefile, "wb" ))
			print "Saved demonstration in {}.".format(savefile)
		
		ros_utils.stop_admittance_mode(self.prefix)

	def load_parameters(self):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""

		# ----- General Setup ----- #
		self.prefix = rospy.get_param("setup/prefix")
		self.start = np.array(rospy.get_param("setup/start"))*(math.pi/180.0)
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.save_dir = rospy.get_param("setup/save_dir")

		# Openrave parameters for the environment.
		model_filename = rospy.get_param("setup/model_filename")
		object_centers = rospy.get_param("setup/object_centers")
		self.environment = Environment(model_filename, object_centers)

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

		# Tell controller to move to start.
		self.controller.set_trajectory(Trajectory([self.start], [0.0]))

		# Stores current COMMANDED joint torques.
		self.cmd = np.eye(7)

		# Utilities for recording data.
		self.expUtil = experiment_utils.ExperimentUtils(self.save_dir)

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# Create joint-velocity publisher.
		self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# Create subscriber to joint_angles.
		rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
	
	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target.
		"""

		# Read the current joint angles from the robot.
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		curr_pos = curr_pos*(math.pi/180.0)

		# Update cmd from PID based on current position.
		if self.controller.path_start_T is not None:
			# Allow the person to move the end effector with no control resistance.
			self.cmd = np.zeros((7,7))

			# Update the experiment utils executed trajectory tracker.
			timestamp = time.time() - self.controller.path_start_T
			self.expUtil.update_tracked_traj(timestamp, curr_pos)
		else:
			self.cmd = self.controller.get_command(curr_pos)

if __name__ == '__main__':
	DemoRecorder()


