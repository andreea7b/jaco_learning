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
from utils import ros_utils, experiment_utils

import numpy as np
import pickle

class pHRIInference():
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
        rospy.init_node("phri_inference")

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

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()

		print "----------------------------------"

		# Ask whether to save experimental data for pHRI corrections.
		print "Type [yes/y/Y] if you'd like to save experimental data."
		line = raw_input()
		if (line is not "yes") and (line is not "Y") and (line is not "y"):
			print "Not happy with recording. Terminating experiment."
        else:
			print "Please type in the ID number (e.g. [0/1/2/...])."
            ID = raw_input()
            print "Please type in the task number."
            task = raw_input()
            print "Saving experimental data to file..."
			settings_string = ID + "_" + self.feat_method + "_" + "_".join(feat_list) + "_correction_" + task + "_"
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

        ros_utils.stop_admittance_mode(self.prefix)

	def load_parameters(self):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""
		# ----- General Setup ----- #
        self.prefix = rospy.get_param("setup/prefix")
        pick = rospy.get_param("setup/start")
        place = rospy.get_param("setup/goal")
		self.start = np.array(pick)*(math.pi/180.0)
		self.goal = np.array(place)*(math.pi/180.0)
		self.goal_pose = None if rospy.get_param("setup/goal_pose") == "None" else rospy.get_param("setup/goal_pose")
        self.T = rospy.get_param("setup/T")
        self.timestep = rospy.get_param("setup/timestep")
        self.save_dir = rospy.get_param("setup/save_dir")
        self.feat_list = rospy.get_param("setup/feat_list")
        self.weights = rospy.get_param("setup/feat_weights")
        self.INTERACTION_TORQUE_THRESHOLD = rospy.get_param("setup/INTERACTION_TORQUE_THRESHOLD")
        self.INTERACTION_TORQUE_EPSILON = rospy.get_param("INTERACTION_TORQUE_EPSILON")

        # Openrave parameters for the environment.
        model_filename = rospy.get_param("setup/model_filename")
        object_centers = rospy.get_param("setup/object_centers")
        self.environment = Environment(model_filename, object_centers)
        
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
		
        self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.weights, self.T, self.timestep)
        self.traj_plan = self.traj.downsample(self.planner.num_waypts)

        # Track if you have reached the start/goal of the path.
		self.reached_start = False
		self.reached_goal = False
		
        # Save the intermediate target configuration. 
		self.curr_pos = None

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
        self.feat_method = rospy.get_param("learner/type")
        self.learner = PHRILearner(self.feat_method, self.feat_list, self.environment, constants)
	
         # ---- Experimental Utils ---- #
		self.expUtil = experiment_utils.ExperimentUtils(self.save_dir)
		# Update the list of replanned plans with new trajectory plan.
		self.expUtil.update_replanned_trajList(0.0, self.traj_plan.waypts)
		# Update the list of replanned waypoints with new waypoints.
		self.expUtil.update_replanned_wayptsList(0.0, self.traj.waypts)
    
    def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# Create joint-velocity publisher.
		self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# Create subscriber to joint_angles.
		rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
        # Create subscriber to joint torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)
	
	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target.
		"""
		# Read the current joint angles from the robot.
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = self.curr_pos*(math.pi/180.0)

		# Update cmd from PID based on current position.
		self.cmd = self.controller.get_command(self.curr_pos)
		
        # Check is start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
            self.expUtil.set_startT(self.controller.path_start_T)
        if self.controller.path_end_T is not None:
            self.reached_goal = True
            self.expUtil.set_endT(self.controller.path_end_T)

        # Update the experiment utils executed trajectory tracker.
		if self.reached_start and not self.reached_goal:
			timestamp = time.time() - self.controller.path_start_T
			self.expUtil.update_tracked_traj(timestamp, self.curr_pos)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for
		plotting & analysis
		"""
		# Read the current joint torques from the robot.
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))
		interaction = False
		for i in range(7):
			# Center torques around zero.
			torque_curr[i][0] -= self.INTERACTION_TORQUE_THRESHOLD[i]
            # Check if interaction was not noise.
			if np.fabs(torque_curr[i][0]) > self.INTERACTION_TORQUE_EPSILON[i] and self.reached_start:
				interaction = True
		
        # If we experienced large enough interaction force, then learn.
		if interaction:
            if self.reached_start and not self.reached_goal:
                timestamp = time.time() - self.controller.path_start_T
                self.expUtil.update_tauH(timestamp, torque_curr)
                self.expUtil.update_interaction_point(timestamp, self.curr_pos)

                self.weights = self.learner.learn_weights(self.traj, torque_curr)
                betas = self.learner.betas
                betas_u = self.learner.betas_u
                updates = self.learner.updates

                self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.weights, 
												self.T, self.timestep, seed=self.traj_plan.waypts)
                self.traj_plan = self.traj.downsample(self.planner.num_waypts)
				self.controller.set_trajectory(self.traj)

                # Update the experimental data with new weights and new betas.
                timestamp = time.time() - self.controller.path_start_T
                self.expUtil.update_weights(timestamp, self.weights)
                self.expUtil.update_betas(timestamp, betas)
                self.expUtil.update_betas_u(timestamp, betas_u)
                self.expUtil.update_updates(timestamp, updates)

                # Update the list of replanned plans with new trajectory plan.
                self.expUtil.update_replanned_trajList(timestamp, self.traj_plan.waypts)

                # Update the list of replanned trajectory waypts with new trajectory.
                self.expUtil.update_replanned_wayptsList(timestamp, self.traj.waypts)

                # Store deformed trajectory plan.
                deformed_traj = self.learner.traj_deform.downsample(self.planner.num_waypts)
                self.expUtil.update_deformed_trajList(timestamp, deformed_traj.waypts)

                # Store deformed trajectory waypoints.
                self.expUtil.update_deformed_wayptsList(timestamp, self.learner.traj_deform.waypts)

if __name__ == '__main__':
	pHRIControl()



