import numpy as np
from numpy import linalg
from numpy import linspace
import time
import math

import interactpy
from interactpy import *

import logging
import pid
import copy

class Trajectory(object):
	"""
	This class represents a robot trajectory from start to goal.
	"""

	def __init__(self):

		self.start_time = None				# trajectory start time
		self.final_time = None				# trajectory end time
		self.curr_waypt_idx = None			# current waypoint controller is moving to

		# trajopt variables
		self.waypts_plan = None				# trajectory computed by trajopt
		self.num_waypts_plan = None			# number of waypoints trajopt uses
		self.step_time_plan = None			# time difference between waypoints in trajopt

		# upsampled traj variables
		self.waypts = None					# trajectory upsampled from trajopt trajectory
		self.num_waypts = None				# number of waypoints in upsampled traj
		self.step_time = None				# time difference between upsampled waypoints
		self.waypts_time = None				# array of lenth num_waypts, with time stamp for each point

		# ---- DEFORMATION Initialization ---- #

		self.alpha = -0.01
		self.n = 5
		self.A = np.zeros((self.n+2, self.n)) 
		np.fill_diagonal(self.A, 1)
		for i in range(self.n):
			self.A[i+1][i] = -2
			self.A[i+2][i] = 1
		self.R = np.dot(self.A.T, self.A)
		Rinv = np.linalg.inv(self.R)
		Uh = np.zeros((self.n, 1))
		Uh[0] = 1
		self.H = np.dot(Rinv,Uh)*(np.sqrt(self.n)/np.linalg.norm(np.dot(Rinv,Uh)))

	# ---- utilities/getter functions ---- #

	def get_waypts_plan(self):
		"""
		Returns reference to waypts_plan (used by trajopt)
		Used mostly for recording experimental data by pid_trajopt.py
		"""
		return self.waypts_plan

	def deform(self, u_h):
		"""
		Deforms the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		---
		input is human force, returns deformed and old waypts
		"""
		deform_waypt_idx = self.curr_waypt_idx + 1
		if (deform_waypt_idx + self.n) > self.num_waypts:
			return (None, None)
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		#plotTraj(self.env, self.robot, self.bodies, self.waypts_plan, [1, 0, 0])
		return (waypts_deform, waypts_prev)

	def upsample(self, step_time):
		"""
		Put waypoints along trajectory at step_time increments.
		---
		input desired time increment, update upsampled trajectory
		"""
		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts
		
		t = self.start_time
		for i in range(num_waypts):
			if t >= self.final_time:
				waypts_time[i] = self.final_time
				waypts[i,:] = self.waypts_plan[self.num_waypts_plan - 1]
			else:
				deltaT = t - self.start_time
				prev_idx = int(deltaT/self.step_time_plan)
				prev = self.waypts_plan[prev_idx]
				next = self.waypts_plan[prev_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = prev+((t-prev_idx*self.step_time_plan)/self.step_time_plan)*(next-prev)
			t += step_time
		self.step_time = step_time
		self.num_waypts = num_waypts
		self.waypts = waypts
		self.waypts_time = waypts_time

	def downsample(self):
		"""
		Updates the trajopt trajectory from the upsampled trajectory.
		changes the trajopt waypoints between start and goal.
		"""
		for index in range(1,self.num_waypts_plan-1):
			t = self.start_time + index*self.step_time_plan
			target_pos = self.interpolate(t)
			self.waypts_plan[index,:] = target_pos.reshape((1,7))
			
	def interpolate(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_time:
			self.curr_waypt_idx = self.num_waypts - 1
			target_pos = self.waypts[self.curr_waypt_idx]
		else:
			deltaT = curr_time - self.start_time
			self.curr_waypt_idx = int(deltaT/self.step_time)
			prev = self.waypts[self.curr_waypt_idx]
			next = self.waypts[self.curr_waypt_idx + 1]
			ti = self.waypts_time[self.curr_waypt_idx]
			tf = self.waypts_time[self.curr_waypt_idx + 1]
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev		
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

