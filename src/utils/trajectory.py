import numpy as np
import copy

class Trajectory(object):
	"""
	This class represents a trajectory object, supporting operations such as
	interpolating, downsampling waypoints, upsampling waypoints, etc.
	"""
	def __init__(self, waypts, waypts_time):
		self.waypts = waypts
		self.waypts_time = waypts_time
		self.num_waypts = len(waypts)
		if self.num_waypts > 1:
			self.timestep = (self.waypts_time[-1] - self.waypts_time[0]) / (self.num_waypts - 1)

	def upsample(self, num_waypts):
		"""
		Upsample the trajectory waypoints to num_waypts length.
		Error if num_waypts is smaller than current waypts length.

		Params:
			num_waypts [int] -- Number of waypoints to downsample to.

		Returns:
			upsampled_waypts [Trajectory] -- Downsampled trajectory.
		"""
		assert num_waypts >= len(self.waypts), "Upsampling requires a larger number of waypoints to upsample to. Your number is smaller."
		assert len(self.waypts) > 1, "Cannot upsample a one-waypoint trajectory."

		timestep = (self.waypts_time[-1] - self.waypts_time[0]) / (num_waypts - 1)
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts

		t = self.waypts_time[0]
		for i in range(num_waypts):
			if t >= self.waypts_time[-1]:
				waypts_time[i] = self.waypts_time[-1]
				waypts[i,:] = self.waypts[-1]
			else:
				curr_waypt_idx = int((t - self.waypts_time[0]) / self.timestep)
				curr_waypt = self.waypts[curr_waypt_idx]
				next_waypt = self.waypts[curr_waypt_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = curr_waypt + ((t - curr_waypt_idx * self.timestep) / self.timestep) * (next_waypt - curr_waypt)
			t += timestep
		return Trajectory(waypts, waypts_time)

	def downsample(self, num_waypts):
		"""
		Downsample the trajectory waypoints to num_waypts length.
		Error if num_waypts is larger than current waypts length.

		Params:
			num_waypts [int] -- Number of waypoints to downsample to.

		Returns:
			downsampled_waypts [Trajectory] -- Downsampled trajectory.
		"""
		assert num_waypts <= len(self.waypts), "Downsampling requires a smaller number of waypoints to downsample to. Your number is larger."
		assert len(self.waypts) > 1, "Cannot downsample a one-waypoint trajectory."

		timestep = (self.waypts_time[-1] - self.waypts_time[0]) / (num_waypts - 1)
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts

		for index in range(num_waypts):
			t = self.waypts_time[0] + index * timestep
			waypts_time[index] = t
			waypts[index,:] = self.interpolate(t).reshape((1,7))

		return Trajectory(waypts, waypts_time)

	def interpolate(self, t):
		"""
		Gets the desired position along trajectory at time t by interpolating between waypoints.

		Params:
			t [float] -- The time of desired interpolation along path.

		Returns:
			waypt [array] -- Interpolated waypoint at time t.
		"""
		assert len(self.waypts) > 1, "Cannot interpolate a one-waypoint trajectory."

		if t >= self.waypts_time[-1]:
			# If interpolating after end of trajectory, return last waypoint.
			waypt = self.waypts[-1]
		else:
			curr_waypt_idx = int((t - self.waypts_time[0]) / self.timestep)
			curr_waypt = self.waypts[curr_waypt_idx]
			next_waypt = self.waypts[curr_waypt_idx + 1]
			curr_t = self.waypts_time[curr_waypt_idx]
			next_t = self.waypts_time[curr_waypt_idx + 1]
			waypt = curr_waypt + (next_waypt - curr_waypt) * ((t - curr_t) / (next_t - curr_t))
		waypt = np.array(waypt).reshape((7,1))
		return waypt

	def deform(self, u_h, t, alpha, n):
		"""
		Deforms the next n waypoints of the trajectory.
		
		Params:
			u_h -- Deformation torque.
			t [float] -- The time of deformation.
			alpha -- Alpha deformation parameter.
			n -- Width of deformation parameter.

		Returns:
			trajectory -- Deformed trajectory.
		"""

		# ---- DEFORMATION Initialization ---- #
		A = np.zeros((n+2, n))
		np.fill_diagonal(A, 1)
		for i in range(n):
			A[i+1][i] = -2
			A[i+2][i] = 1
		R = np.dot(A.T, A)
		Rinv = np.linalg.inv(R)
		Uh = np.zeros((n, 1))
		Uh[0] = 1
		H = np.dot(Rinv,Uh)*(np.sqrt(n)/np.linalg.norm(np.dot(Rinv,Uh)))

		# ---- Deformation process ---- #
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((n,7))
		deform_waypt_idx = int((t - self.waypts_time[0]) / self.timestep) + 1

		if (deform_waypt_idx + n) > self.num_waypts:
			print "Deforming too close to end. Returning same trajectory"
			return Trajectory(waypts_deform, self.waypts_time)

		for joint in range(7):
			gamma[:,joint] = alpha*np.dot(H, u_h[joint])
		waypts_deform[deform_waypt_idx : n + deform_waypt_idx, :] += gamma
		return Trajectory(waypts_deform, self.waypts_time)

