#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target. Additionally, it supports human-robot
interaction in the form of online physical corrections. 
Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
"""
import rospy
import math
import sys, select, os
import time

from pid_trajopt import PIDControl
from planners import phri_planner
from utils import pid, openrave_utils, ros_utils
from data_processing import experiment_utils

import numpy as np
import pickle

# Constants for pHRI
epsilon = 0.10		# epsilon for when robot think it's at goal
INTERACTION_TORQUE_THRESHOLD = [0.88414821, 17.22751856, -0.40134936,  6.23537946, -0.90013662, 1.32379884,  0.10218059]
INTERACTION_TORQUE_EPSILON = [4.0, 5.0, 3.0, 4.0, 1.5, 1.5, 1.5]
MAX_WEIGHTS = {'table':25.0, 'coffee':1.0, 'laptop':40.0, 'human':12.0, 'efficiency':0.5}

class pHRIControl(PIDControl):
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
	"""

	def __init__(self, ID, task, feat_method, feat_list, feat_weight, record, replay):
		
		# Load parameters
		self.load_parameters(ID, task, feat_method, feat_list, feat_weight, record, replay)
        
        # ---- ROS Setup ---- #

		rospy.init_node("phri_control")
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

	def load_parameters(self, ID, task, feat_method, feat_list, feat_weight, record, replay):
		"""
		Loading parameters.
		"""
        super(PIDControl, self).load_parameters(task, feat_list, feat_weight)

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

		# initialize trajectory weights and betas
		self.betas = [1.0]*self.num_feat
		self.updates = [0.0]*self.num_feat

		# create the planner
		self.planner = phri_planner.pHRIPlanner(self.feat_method, self.feat_list, self.task)

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

if __name__ == '__main__':
	if len(sys.argv) < 8:
		print "ERROR: Not enough arguments. Specify ID, task, record, replay, feat_method, feat_list, feat_weight"
	else:
		ID = int(sys.argv[1])
		task = sys.argv[2]
		record = sys.argv[3]
		replay = sys.argv[4]
		feat_method = sys.argv[5]
		feat_list = [x.strip() for x in sys.argv[6].split(',')]
		feat_weight = [float(x.strip()) for x in sys.argv[7].split(',')]
	pHRIControl(ID,task,record,replay,feat_method,feat_list,feat_weight)


