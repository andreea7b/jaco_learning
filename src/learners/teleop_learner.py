import numpy as np
import math

class TeleopLearner(object):
	"""
	This class performs goal and confidence inference given user inputs.
	"""

	def __init__(self, planner, planner_vars, environment, goals, feat_list, weights, beliefs, betas):
		# ---- Important internal variables ---- #
		# TODO: instead of copying all of these variables over, just store the TeleopInference object
		self.planner = planner
		self.T, self.timestep, self.start = planner_vars
		self.environment = environment
		self.weights = weights
		self.feat_list = feat_list

		self.goals = goals # goals should be cartesian coordinates
		self.num_goals = len(goals)
		self.goals_beliefs = beliefs
		assert(len(goals) == len(beliefs))

		self.betas = betas
		# beliefs is len(goals) x len(betas)
		self.beliefs = np.outer(beliefs, np.ones(len(betas),)/len(betas))

		# compute the costs of optimal trajectories to all goals for later
		self.optimal_costs = np.zeros(self.num_goals)
		for i in range(self.num_goals):
			traj = self.planner.replan(self.start, None, self.goals[i], self.weights, self.T, self.timestep)
			# TODO: remove these print statements once this is tested
			print environment.featurize(traj)
			print np.sum(environment.featurize(traj), axis=1)
			print weights
			traj_cost = np.sum(self.weights * np.sum(self.environment.featurize(traj), axis=1))
			self.optimal_costs[i] = traj_cost

	def update_beliefs(self, traj):
		"""
		Updates the beliefs given a trajectory traj
		"""
		curr_waypt = TODO
		curr_waypt_idx = TODO
		curr_time = TODO
		traj_cost = np.sum(self.weights * np.sum(self.environment.featurize(traj), axis=1))
		goal_traj_costs = np.zeros(self.num_goals)
		for i in range(self.num_goals):
			goal_traj = self.planner.replan(curr_waypt, None, self.goals[i], self.weights,
											self.T - curr_time, self.timestep)
			goal_traj_costs[i] = np.sum(self.weights * np.sum(self.environment.featurize(goal_traj), axis=1))\
		#prob_u_given_g = np.exp(np.outer(-(traj_cost + goal_traj_costs), self.betas)) \
		#               / np.exp(np.outer(-self.optimal_costs, self.betas))




		# to_xyz = self.environment.get_cartesian_coords
		# pos_xyz = to_xyz(old_angles)
		# cartesian_displacement = to_xyz(new_angles) - pos_xyz
		# goal_displacements = self.goals_xyz - pos_xyz
		# goal_costs = angle_costs(cartesian_displacement, goal_displacements)
		#
		# prob_u_given_g = np.exp(np.outer(goal_costs, -self.betas))
		# updated_beliefs = prob_u_given_g * self.beliefs
		# updated_beliefs = updated_beliefs / np.sum(updated_beliefs)
		# print updated_beliefs
		# self.beliefs = updated_beliefs
		#
		# self.goal_beliefs = self.beliefs.sum(axis=1)
		# print self.goal_beliefs



# def angle_costs(obs_dir, goal_dirs):
# 	u_obs_dir = obs_dir/np.linalg.norm(obs_dir)
# 	u_goal_dirs = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, goal_dirs)
# 	def angle_from_obs(u_goal_dir):
# 		return np.arccos(np.clip(np.dot(u_goal_dir, u_obs_dir), -1.0, 1.0))
# 	return np.fabs(np.apply_along_axis(angle_from_obs, 1, u_goal_dirs))
