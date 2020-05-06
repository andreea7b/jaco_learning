import numpy as np
import math
import copy

import openravepy
from openravepy import *

from utils.openrave_utils import *

class Environment(object):
	"""
	This class creates an OpenRave environment and contains all the
	functionality needed for custom features and constraints.
	"""
	def __init__(self, model_filename, object_centers, goals=None, use_viewer=True, plot_objects=True):
		# ---- Create environment ---- #
		self.env, self.robot = initialize(model_filename, use_viewer=use_viewer)

		# Insert any objects you want into environment.
		self.bodies = []
		self.object_centers = object_centers

		# Plot the table and table mount, and other desired objects.
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		if plot_objects:
			plotCabinet(self.env)
			plotLaptop(self.env,self.bodies,object_centers['LAPTOP_CENTER'])
			plotSphere(self.env,self.bodies,object_centers['HUMAN_CENTER'], 0.015)

		# Plot and add the goals
		if goals is not None:
			self.goal_locs = []
			for goal in goals:
				with self.robot:
					angles = goal if len(goal) == 10 else np.append(goal, np.array([0,0,0]))
					self.robot.SetDOFValues(angles)
					cartesian_coords = robotToCartesian(self.robot)
					goal_loc = cartesian_coords[6]
					self.goal_locs.append(goal_loc)
					plotSphere(self.env, self.bodies, goal_loc, 0.05, color=[1,0,0]) # may need to change colors
			self.goal_locs = np.array(self.goal_locs)

		# true goal for presentation
		goal = np.array([260.0, 90.0, 180.0, 160.0, 270.0, 180.0, 280.0])*np.pi/180
		with self.robot:
			angles = goal if len(goal) == 10 else np.append(goal, np.array([0,0,0]))
			self.robot.SetDOFValues(angles)
			cartesian_coords = robotToCartesian(self.robot)
			goal_loc = cartesian_coords[6]
			plotSphere(self.env, self.bodies, goal_loc, 0.05, color=[0,1,0])

	# ---- Custom environmental features ---- #
	def featurize(self, waypts, feat_list):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory waypoints, output list of feature values
		"""
		num_features = len(feat_list)
		features = [[0.0 for _ in range(len(waypts)-1)] for _ in range(0, num_features)]

		for index in range(len(waypts)-1):
			for feat in range(num_features):
				if feat_list[feat] == 'table':
					features[feat][index] = self.table_features(waypts[index+1])
				elif feat_list[feat] == 'coffee':
					features[feat][index] = self.coffee_features(waypts[index+1])
				elif feat_list[feat] == 'human':
					features[feat][index] = self.human_features(waypts[index+1],waypts[index])
				elif feat_list[feat] == 'laptop':
					features[feat][index] = self.laptop_features(waypts[index+1],waypts[index])
				elif feat_list[feat] == 'origin':
					features[feat][index] = self.origin_features(waypts[index+1])
				elif feat_list[feat] == 'efficiency':
					features[feat][index] = self.efficiency_features(waypts[index+1],waypts[index])
				# TODO: this is bad design because only 10 goals can be supported, but shouldn't matter
				elif "goal" in feat_list[feat]:
					features[feat][index] = self.goal_dist_features(int(feat_list[feat][4]), waypts[index+1])
		return features

	# -- Goal Distance -- #
	def goal_dist_features(self, goal_num, waypt):
		with self.robot:
			self.robot.SetDOFValues(np.append(waypt.reshape(7), np.array([0,0,0])))
			coords = robotToCartesian(self.robot)[6]
		return np.linalg.norm(self.goal_locs[goal_num] - coords)**2

	# -- Efficiency -- #
	def efficiency_features(self, waypt, prev_waypt):
		"""
		Computes efficiency feature for waypoint, confirmed to match trajopt.
		---
		input waypoint, output scalar feature
		"""
		return np.linalg.norm(waypt - prev_waypt)**2

	# -- Distance to Robot Base (origin of world) -- #
	def origin_features(self, waypt):
		"""
		Computes the total feature value over waypoints based on
		y-axis distance to table.
		---
		input waypoint, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EEcoord_y = coords[6][1]
		EEcoord_y = np.linalg.norm(coords[6])
		## QUESTION: why is the above line here (either it or the one above it is unnecessary)
		return EEcoord_y

	# -- Distance to Table -- #
	def table_features(self, waypt):
		"""
		Computes the total feature value over waypoints based on
		z-axis distance to table.
		---
		input waypoint, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EEcoord_z = coords[6][2]
		return EEcoord_z

	# -- Coffee (or z-orientation of end-effector) -- #
	def coffee_features(self, waypt):
		"""
		Computes the distance to table feature value for waypoint
		by checking if the EE is oriented vertically according to pitch.
		Note: adding 1.5 to pitch to make it centered around 0
		---
		input waypoint, output scalar feature
		"""
		# get rotation transform, convert it to euler coordinates, and make sure the end effector is upright
		def mat2euler(mat):
			gamma = np.arctan2(mat[2,1], mat[2,2])
			beta = np.arctan2(-mat[2,0], np.sqrt(mat[2,1]**2 + mat[2,2]**2))
			alpha = np.arctan2(mat[1,0], mat[0,0])
			return np.array([gamma,beta,alpha])

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			EE_link = self.robot.GetLinks()[7]
			R = EE_link.GetTransform()[:3,:3]
		[yaw, pitch, roll] = mat2euler(R)
		#return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))
		## QUESTION: why 1.5? should this be pi/2?
		return (pitch + 1.5)

	# -- Distance to Laptop -- #
	def laptop_features(self, waypt, prev_waypt):
		"""
		Computes laptop feature value over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions.
		---
		input neighboring waypoints, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.laptop_dist(inter_waypt)
		return feature

	def laptop_dist(self, waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		laptop_xy = np.array(self.object_centers['LAPTOP_CENTER'][0:2])
		dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	# -- Distance to Human -- #
	def human_features(self, waypt, prev_waypt):
		"""
		Computes human feature value over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions.
		---
		input neighboring waypoints, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.human_dist(inter_waypt)
		return feature

	def human_dist(self, waypt):
		"""
		Computes distance from end-effector to human in xy coords
		input trajectory, output scalar distance where
			0: EE is at more than 0.4 meters away from human
			+: EE is closer than 0.4 meters to human
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(self.object_centers['HUMAN_CENTER'][0:2])
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	# ---- Custom environmental constraints --- #
	def table_constraint(self, waypt):
		"""
		Constrains z-axis of robot's end-effector to always be
		above the table.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			EE_link = self.robot.GetLinks()[10]
			EE_coord_z = EE_link.GetTransform()[2][3]
		if EE_coord_z > 0:
			EE_coord_z = 0
		return -EE_coord_z

	def coffee_constraint(self, waypt):
		"""
		Constrains orientation of robot's end-effector to be
		holding coffee mug upright.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			EE_link = self.robot.GetLinks()[7]
			return EE_link.GetTransform()[:2,:3].dot([1,0,0])

	def coffee_constraint_derivative(self, waypt):
		"""
		Analytic derivative for coffee constraint.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(waypt)
			world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
			return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()

	# ---- Helper functions ---- #
	def update_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos - 7x1 vector of current joint angles (radians)
		"""
		#pos = np.array([curr_pos[0][0],curr_pos[1][1],curr_pos[2][2]+math.pi,curr_pos[3][3],curr_pos[4][4],curr_pos[5][5],curr_pos[6][6],0,0,0])
		pos = np.array([curr_pos[0][0],curr_pos[1][1],curr_pos[2][2],curr_pos[3][3],curr_pos[4][4],curr_pos[5][5],curr_pos[6][6],0,0,0])
		with self.env:
			self.robot.SetDOFValues(pos)

	def update_vel(self, curr_vel):
		"""
		Updates DOF velocities in OpenRAVE simulation based on curr_vel.
		----
		curr_vel - 7x1 vector of current joint velocities (radians)
		"""
		vel = np.array([curr_vel[0][0],curr_vel[1][1],curr_vel[2][2],curr_vel[3][3],curr_vel[4][4],curr_vel[5][5],curr_vel[6][6],0,0,0])
		with self.env:
			self.robot.SetDOFVelocities(vel)

	def get_cartesian_coords(self, joint_angles):
		"""
		Note that joint_angles are assumed to be in radians
		"""
		if len(joint_angles) < 10:
			joint_angles = np.append(joint_angles.reshape(7), np.array([0,0,0]))
			joint_angles[2] += math.pi
		with self.robot:
			self.robot.SetDOFValues(joint_angles)
			return robotToCartesian(self.robot)[6]

	def kill_environment(self):
		"""
		Destroys openrave thread and environment for clean shutdown.
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime
