#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco
so that it maintains a fixed distance to a target. 
Author: Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import pid
import tf
import sys, select, os
import thread
import argparse
import actionlib
import time
import trajopt_planner
#import planner
import traj
import ros_utils
import experiment_utils

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

import openrave_utils
from openrave_utils import *

prefix = 'j2s7s300_driver'

home_pos = [103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]
candlestick_pos = [180.0]*7

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]


epsilon = 0.10							# epislon for when robot think it's at goal
MAX_CMD_TORQUE = 40.0					# max command robot can send
INTERACTION_TORQUE_THRESHOLD = 8.0		# threshold when interaction is measured 

HUMAN_TASK = 0
COFFEE_TASK = 1
TABLE_TASK = 2
LAPTOP_TASK = 3

ZERO_FEEDBACK = 'A'
HAPTIC_FEEDBACK = 'B'

ALL = 0 						# updates all features
MAX = 1							# updates only feature that changed the most
LIKELY = 2						# updates the most likely feature 

class PIDVelJaco(object): 
	"""
	This class calbirates max and min values for features.
	"""

	def __init__(self, task, sim):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		# task type - table, laptop, or coffee task
		self.task = task
	
		# stores if you want to compute min/max in sim or from real robot
		if sim == "F" or sim == "f":
			self.sim = False
		elif sim == "T" or sim == "t":
			self.sim = True

		print self.sim
		# start admittance control mode
		if not self.sim:
			self.start_admittance_mode()

		# ---- Trajectory Setup ---- #

		# total time for trajectory
		self.T = 15.0

		# initialize start/goal based on task 
		if self.task == COFFEE_TASK or self.task == HUMAN_TASK:
			pick = pick_shelf
		else:
			pick = pick_basic

		if self.task == LAPTOP_TASK or self.task == HUMAN_TASK:
			place = place_higher
		else:
			place = place_lower
		
		# initialize trajectory weights for the min and max feature count trajectory
		self.max_weights = [0.0, 0.0, 0.0]
		self.min_weights = [0.0, 0.0, 0.0]
		# set the desired task's weight super high to simulate max
		self.min_weights[self.task-1] = 100.0

		print "min_weights: " + str(self.min_weights)
		print "max_weights: " + str(self.max_weights)

		start = np.array(pick)*(math.pi/180.0)
		goal = np.array(place)*(math.pi/180.0)
		self.start = start
		self.goal = goal
		self.curr_pos = None

		# create the trajopt planner and plan from start to goal
		self.planner = trajopt_planner.Planner(self.task, False, ALL)

		# stores the min feature count trajectory
		#self.min_traj = self.planner.replan(self.start, self.goal, self.min_weights, 0.0, self.T, 0.5)
		self.min_traj = self.planner.replan(self.start, self.goal, self.min_weights, 0.0, self.T, 0.5)		

		#self.planner.plotWaypts(self.min_traj.waypts, color=[1,0,0])
		plotCupTraj(self.planner.env,self.planner.robot,self.planner.bodies,self.min_traj,color=[1,0,0])	
		new_features = self.planner.featurize(self.min_traj)
		min_Phi = np.array([new_features[0], sum(new_features[1]), sum(new_features[2]), sum(new_features[3])])

		print "min summed features: " + str(min_Phi)


		# stores the max feature count trajectory
		self.max_traj = self.planner.replan(self.start, self.goal, self.max_weights, 0.0, self.T, 0.5)
		#self.planner.plotWaypts(self.max_traj.waypts, color=[0,1,0])
		plotCupTraj(self.planner.env,self.planner.robot,self.planner.bodies,self.max_traj,color=[0,1,0])		
		new_features2 = self.planner.featurize(self.max_traj)
		max_Phi = np.array([new_features2[0], sum(new_features2[1]), sum(new_features2[2]), sum(new_features2[3])])

		print "max summed features: " + str(max_Phi)

		print "feature range: " + str(max_Phi - min_Phi)

		time.sleep(20)
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# ---- ROS Setup ---- #

		if not self.sim:
			rospy.init_node("max_min_calib")

			# create joint-velocity publisher
			self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

			# create subscriber to joint_angles
			rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
			# create subscriber to joint_torques
			rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

			# publish to ROS at 100hz
			r = rospy.Rate(100) 

			print "----------------------------------"
			print "Press ENTER to quit:"
		
			while not rospy.is_shutdown():

				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break

				r.sleep()
		
			print "----------------------------------"


			# end admittance control mode
			self.stop_admittance_mode()

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

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for 
		plotting & analysis
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))


	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot 
		"""
	
		# read the current joint angles from the robot
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		self.curr_pos = self.curr_pos*(math.pi/180.0)	

		if self.task == COFFEE_TASK:
			featurized = self.planner.coffee_features(self.curr_pos)
			print "curr pos: " + str(self.curr_pos)
			print "COFFEE feature for curr pos: " + str(featurized)
		elif self.task == TABLE_TASK:
			featurized = self.planner.table_features(self.curr_pos)
			print "curr pos: " + str(self.curr_pos)
			print "TABLE feature for curr pos: " + str(featurized)
		elif self.task == LAPTOP_TASK:
			featurized = self.planner.laptop_dist(self.curr_pos)
			print "curr pos: " + str(self.curr_pos)
			print "LAPTOP feature for curr pos: " + str(featurized)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "ERROR: Not enough arguments. Specify task number and if you want sim or not."
	else:	

		task = int(sys.argv[1])
		sim = sys.argv[2]
		PIDVelJaco(task, sim)

