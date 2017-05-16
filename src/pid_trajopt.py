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
import plot
import trajopt_planner
import sim_robot

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

prefix = 'j2s7s300_driver'

home_pos = [103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]
candlestick_pos = [180.0]*7

pos1 = [14.30,162.95,190.75,124.03,176.10,188.25,167.94]
pos2 = [121.89,159.32,213.20,109.06,153.09,185.10,170.77]

waypt1 = [136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861]
waypt2 = [271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644]
waypt3 = [338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655]

epsilon = 0.10
interaction_thresh = 5.0

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
		/j2s7s300_driver/out/joint_angles	- Jaco joint angles
	
	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco joint velocities 
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		sim_flag 				  - flag for if in simulation or not
	"""

	def __init__(self, p_gain, i_gain, d_gain, sim_flag):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		# ---- ROS Setup ---- #
		rospy.init_node("pid_vel_trajopt")

		# switch robot to torque-control mode if not in simulation
		#if not sim_flag:
			#self.init_torque_mode()

		# ---- Trajectory Setup ---- #

		# get trajectory planner
		T = 6

		p1 = home_pos #pos1 #candlestick_pos
		p2 = candlestick_pos #pos2 #home_pos
		start = np.array(p1)*(math.pi/180.0)
		goal = np.array(p2)*(math.pi/180.0)

		# create the trajopt planner from start to goal
		self.planner = trajopt_planner.Planner(start, goal, T)

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		print "HAS REACHED START? " + str(self.reached_start)

		# ------------------------- #

		self.max_cmd = 40*np.eye(7)
		# stores current COMMANDED joint torques
		self.cmd = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# ----- Controller Setup ----- #

		print "PID Gains: " + str(p_gain) + ", " + str(i_gain) + "," + str(d_gain)

		self.p_gain = p_gain
		self.i_gain = i_gain
		self.d_gain = d_gain

		# P, I, D gains 
		self.P = self.p_gain*np.eye(7)
		self.I = self.i_gain*np.eye(7)
		self.D = self.d_gain*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

		# ---------------------------- #

		# stuff for plotting
		self.plotter = plot.Plotter(self.p_gain,self.i_gain,self.d_gain)

		# keeps running time since beginning of program execution
		self.process_start_T = time.time() 
		# keeps running time since beginning of path
		self.path_start_T = None 

		# create joint-velocity publisher
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)
		# create publisher of the cartesian waypoints
		self.waypt_pub = rospy.Publisher('/cartesian_waypts', geometry_msgs.msg.PoseArray, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_state --> get joint state in radians
		rospy.Subscriber(prefix + '/out/joint_state', sensor_msgs.msg.JointState, self.joint_state_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)


		# publish to ROS at 1000hz
		r = rospy.Rate(100) 

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"
		
		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.vel_pub.publish(self.cmd_to_JointVelocityMsg()) 
			self.waypt_pub.publish(self.waypts_to_PoseArrayMsg())
			r.sleep()

		# plot the error over time after finished
		tot_path_time = time.time() - self.path_start_T

		#self.plotter.plot_PID(tot_path_time)
		self.plotter.plot_tau_PID(tot_path_time)


	def cmd_to_JointTorqueMsg(self):
		"""
		Returns a JointTorque Kinova msg from an array of torques
		"""
		jointCmd = kinova_msgs.msg.JointTorque()
		jointCmd.joint1 = self.cmd[0][0];
		jointCmd.joint2 = self.cmd[1][1];
		jointCmd.joint3 = self.cmd[2][2];
		jointCmd.joint4 = self.cmd[3][3];
		jointCmd.joint5 = self.cmd[4][4];
		jointCmd.joint6 = self.cmd[5][5];
		jointCmd.joint7 = self.cmd[6][6];
		
		return jointCmd

	def cmd_to_JointVelocityMsg(self):
		"""
		Returns a JointVelocity Kinova msg from an array of velocities
		"""
		jointCmd = kinova_msgs.msg.JointVelocity()
		jointCmd.joint1 = self.cmd[0][0];
		jointCmd.joint2 = self.cmd[1][1];
		jointCmd.joint3 = self.cmd[2][2];
		jointCmd.joint4 = self.cmd[3][3];
		jointCmd.joint5 = self.cmd[4][4];
		jointCmd.joint6 = self.cmd[5][5];
		jointCmd.joint7 = self.cmd[6][6];

		return jointCmd

	def waypts_to_PoseArrayMsg(self):
		"""
		Returns a PoseArray msg from an array of 3D carteian waypoints
		"""
		poseArray = geometry_msgs.msg.PoseArray()
		poseArray.header.stamp = rospy.Time.now()
		poseArray.header.frame_id = "/root"

		cart_waypts = self.planner.cartesian_waypts
		for i in range(len(cart_waypts)):
			somePose = geometry_msgs.msg.Pose()
			somePose.position.x = cart_waypts[i][0]
			somePose.position.y = cart_waypts[i][1]
			somePose.position.z = cart_waypts[i][2]

			somePose.orientation.x = 0.0
			somePose.orientation.y = 0.0
			somePose.orientation.z = 0.0
			somePose.orientation.w = 1.0
			poseArray.poses.append(somePose)

		return poseArray

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		#print "error: " + str(error)
		return -self.controller.update_PID(error)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for 
		plotting & analysis
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# save running list of joint torques
		self.joint_torques = np.column_stack((self.joint_torques,torque_curr))

		# update the plot of joint torques over time
		#t = time.time() - self.process_start_T
		#self.plotter.update_joint_torque(torque_curr, force_theta, force_mag, t)

	def joint_state_callback(self, msg):
		"""		
		Reads the joint state in radians
		"""
		curr_pos = msg.position
		print "curr pos (rad): " + str(curr_pos)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
		# read the current joint angles from the robot
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		curr_pos = curr_pos*(math.pi/180.0)	

		
		self.planner.update_curr_pos(curr_pos)

		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(curr_pos)

		# update cmd from PID based on current position
		self.cmd = self.PID_control(curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

		# update plotter with new error measurement, torque command, and path time
		curr_time = time.time() - self.process_start_T
		cmd_tau = np.diag(self.controller.cmd).reshape((7,1))

		print "target_pos: " + str(self.target_pos)
		print "curr_pos: " + str(curr_pos)
		#print "cmd: " + str(self.cmd)
		#dist_to_target = -((self.target_pos - curr_pos + math.pi)%(2*math.pi) - math.pi)
		dist_to_target = -((self.target_pos - curr_pos + math.pi)%(2*math.pi) - math.pi)
		print "dist to target: " + str(dist_to_target)

		self.plotter.update_PID_plot(self.controller.p_error, self.controller.i_error, self.controller.d_error, cmd_tau, curr_time)

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:
			#dist_from_start = np.fabs(curr_pos - self.start_pos)
			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_start = np.fabs(dist_from_start)
			print "dist from start: " + str(dist_from_start)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()
				# for plotting, save time when path execution started
				self.plotter.set_path_start_time(time.time() - self.process_start_T)
			else:
				print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:
			print "REACHED START --> EXECUTING PATH"

			t = time.time() - self.path_start_T
			# get next target position from position along trajectory
			self.target_pos = self.planner.interpolate(t)
			print "t: " + str(t)
			print "new target pos from planner: " + str(self.target_pos)
		# check if the arm reached the goal, and restart path
		if not self.reached_goal:
			#dist_from_goal = np.fabs(curr_pos - self.goal_pos)
			
			dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_goal = np.fabs(dist_from_goal)
			##print "dist from goal: " + str(dist_from_goal)

			# check if every joint is close enough to goal configuration
			close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]
			
			# if all joints are close enough, robot is at goal
			is_at_goal = all(close_to_goal)
			
			if is_at_goal:
				self.reached_goal = True
		else:
			print "REACHED GOAL! Holding position at goal."
			self.target_pos = self.goal_pos

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print "ERROR: Not enough arguments. Specify p_gains, i_gains, d_gains, sim_flag."
	else:	
		p_gains = float(sys.argv[1])
		i_gains = float(sys.argv[2])
		d_gains = float(sys.argv[3])
		sim_flag = int(sys.argv[4])

		PIDVelJaco(p_gains,i_gains,d_gains,sim_flag)
	
