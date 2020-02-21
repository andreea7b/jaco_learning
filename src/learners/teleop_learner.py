import numpy as np
import math

from scipy.optimize import minimize, newton
from scipy.stats import chi2

class TeleopLearner(object):
	"""
	This class performs correction inference given a trajectory and an input
	torque applied onto the trajectory.
	"""

	def __init__(self, environment, goals, beliefs, betas):

		# ---- Important internal variables ---- #
		self.environment = environment

		self.goals = goals
		self.goals_xyz = np.array([self.environment.get_cartesian_coords(goal) for goal in goals])
		self.num_goals = len(goals)

		self.goals_beliefs = beliefs
		assert(len(goals) == len(beliefs))
		self.betas = betas
		self.beliefs = np.outer(beliefs, np.ones(len(betas),)/len(betas))
		# beliefs is a len(goals) x len(betas)

	def learn_weights(self, traj, u_h, t):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between
		features of new trajectory and old trajectory.
		---

		Params:
			traj [Trajectory] -- Current trajectory that force was applied to.
			u_h [array] -- Human force applied onto the trajectory.
			t [float] -- Time where deformation was applied.

		Returns:
			weights [list] -- Learned weights.
		"""

		return self.weights

	def update_beliefs(self, old_angles, new_angles):
		"""
		Updates the beliefs given an interaction that moved the robot from old_angles to new_angles
		"""
		to_xyz = self.environment.get_cartesian_coords
		pos_xyz = to_xyz(old_angles)
		cartesian_displacement = to_xyz(new_angles) - pos_xyz
		goal_displacements = self.goals_xyz - pos_xyz
		goal_costs = angle_costs(cartesian_displacement, goal_displacements)
		#goal_costs = angle_costs(new_angles - old_angles, self.goals - old_angles)
		

		#beta = 100
		#prob_fn = lambda beta, costs: np.exp(-beta * costs)
		# likelihood of the action given each goal
		#prob_u_given_g = prob_fn(goal_costs)
		prob_u_given_g = np.exp(np.outer(goal_costs, -self.betas))
		updated_beliefs = prob_u_given_g * self.beliefs
		updated_beliefs = updated_beliefs / np.sum(updated_beliefs)
		print updated_beliefs
		self.beliefs = updated_beliefs

		self.goal_beliefs = self.beliefs.sum(axis=1)
		print self.goal_beliefs



def angle_costs(obs_dir, goal_dirs):
	u_obs_dir = obs_dir/np.linalg.norm(obs_dir)
	u_goal_dirs = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, goal_dirs)
	def angle_from_obs(u_goal_dir):
		return np.arccos(np.clip(np.dot(u_goal_dir, u_obs_dir), -1.0, 1.0))
	return np.fabs(np.apply_along_axis(angle_from_obs, 1, u_goal_dirs))
