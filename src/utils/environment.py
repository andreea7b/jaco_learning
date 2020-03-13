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
	def __init__(self, model_filename, object_centers):
		# ---- Create environment ---- #
		self.env, self.robot = initialize(model_filename)

		# Insert any objects you want into environment.
		self.bodies = []
		self.object_centers = object_centers

		# Plot the table and table mount, and other desired objects.
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotLaptop(self.env,self.bodies,object_centers['LAPTOP_CENTER'])
		plotCabinet(self.env)
		plotSphere(self.env,self.bodies,object_centers['HUMAN_CENTER'], 0.015)

		# Plot the goals
		for key in object_centers.keys():
			if "GOAL" in key:
				if "ANGLES" in key:
					with self.robot:
						angles = object_centers[key] if len(object_centers[key]) == 10 else object_centers[key]+[0,0,0]
						self.robot.SetDOFValues(object_centers[key])
						cartesian_coords = robotToCartesian(self.robot)
						obj_center = cartesian_coords[6]
				else:
					obj_center = object_centers[key]
				plotSphere(self.env, self.bodies, obj, 0.015) # may need to change colors

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
		return features

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
		Computes laptop feature value over waypoints, interpolating and
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

	def update_curr_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos - 7x1 vector of current joint angles (degrees)
		"""
		pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0]+math.pi,curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])
		self.robot.SetDOFValues(pos)

	def get_cartesian_coords(self, joint_angles):
		"""
		Note that joint_angles are assumed to be in degrees
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
