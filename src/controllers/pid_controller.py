import math
import time

import pid

import numpy as np

class PIDController(object):
	"""
	This class represents a PID controller for the Jaco Kinova arm.
	The joint velocities are computed as:

		V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)
	where:
		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change

	Required parameters:
		P, I, D    - gain terms for the PID controller
		epsilon    - proximity threshold
		max_cmd    - maximum allowed torque command
	"""

	def __init__(self, P, I, D, epsilon, max_cmd):
		# ----- PID Parameter Setup ----- #

		# Basic PID controller initialization.
		self.pid = pid.PID(P,I,D,0,0)

		# Stores proximity threshold.
		self.epsilon = epsilon

		# Stores maximum COMMANDED joint torques.
		self.max_cmd = max_cmd 

	def set_trajectory(self, trajectory):
		"""
		Setter method that sets the trajectory and relevant parameters.
		"""
		# Set the trajectory, which may be updated.
		self.traj = trajectory

		# Save the intermediate target configuration. 
		self.target_pos = self.traj.waypts[0]

		# Tracks running time since beginning and end of the path.
		self.path_start_T = None
		self.path_end_T = None

	def get_command(self, current_pos):
		"""
		Reads the latest position of the robot and returns an
		appropriate torque command to move the robot to the target.

		Parameters:
			current_pos - A waypoint determining the robot's position.

		Returns:
		    cmd - The next control command to get to the updated target.
		"""

		# First update the target position if needed.
		# Check if the arm is at the start of the path to execute.
		if self.path_start_T is None:
			dist_from_start = -((current_pos - self.traj.waypts[0] + math.pi)%(2*math.pi) - math.pi)
			dist_from_start = np.fabs(dist_from_start)

			# Check if every joint is close enough to start configuration.
			is_at_start = all([dist_from_start[i] < self.epsilon for i in range(7)])
			if is_at_start:
				self.path_start_T = time.time()
		else:			
			t = time.time() - self.path_start_T

			# Get next target position from position along trajectory.
			self.target_pos = self.traj.interpolate(t + 0.1)

			# Check if the arm reached the goal.
			if self.path_end_T is None:
				dist_from_goal = -((current_pos - self.traj.waypts[-1] + math.pi)%(2*math.pi) - math.pi)
				dist_from_goal = np.fabs(dist_from_goal)

				# Check if every joint is close enough to goal configuration.
				is_at_goal = all([dist_from_goal[i] < self.epsilon for i in range(7)])
				if is_at_goal:
					self.path_end_T = time.time()

		# Update cmd from PID based on current position.
		error = -((self.target_pos - current_pos + math.pi)%(2*math.pi) - math.pi)
		cmd = -self.pid.update_PID(error)

		# Check if each angular torque is within set limits.
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

		return cmd
