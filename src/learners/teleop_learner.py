import numpy as np
import math

class TeleopLearner(object):
	"""
	This class performs goal and confidence inference given user inputs.
	"""

	def __init__(self, main, goal_beliefs, betas, inference_method="dragan"):
		self.main = main # store a reference to the TeleopInference object
		self.goals_beliefs = goal_beliefs
		assert(len(main.goals) == len(beliefs))
		self.betas = betas
		# joint_beliefs is len(goals) x len(betas)
		# assumes uniform prior over betas
		self.joint_beliefs_prior = np.outer(beliefs, np.ones(len(betas),)/len(betas))
		self.joint_beliefs = self.joint_beliefs_prior

		if inference_method == "dragan":
			# precompute the costs of optimal trajectories to all goals for later
			self.optimal_costs = np.zeros(len(goals_beliefs))
			for i in range(len(goals_beliefs)):
				traj = main.planner.replan(main.start, main.goals[i], main.goal_locs[i], main.weights, main.T, main.timestep)
				# TODO: remove these print statements once this is tested
				print main.environment.featurize(traj)
				print np.sum(main.environment.featurize(traj), axis=1)
				print main.weights
				traj_cost = np.sum(main.weights * np.sum(main.environment.featurize(traj), axis=1))
				self.optimal_costs[i] = traj_cost
			self.inference_step = self._dragan_belief_update
		elif inference_method == "javdani":
			raise NotImplementedError
		else:
			raise ValueError

	def _dragan_belief_update(self):
		main = self.main
		curr_traj = main.traj_hist[:main.next_waypt_idx]
		curr_traj_cost = np.sum(main.weights * np.sum(main.environment.featurize(curr_traj), axis=1))
		goal_traj_costs = np.zeros(len(self.goals_beliefs))
		curr_time = main.next_waypt_idx * main.timestep
		for i in range(len(self.goals_beliefs)):
			# TODO: redo in a way that is not time-invariant (need to modify featurize and planner)
			goal_traj = main.planner.replan(curr_pos, main.goals[i], main.goal_locs[i], main.weights,
											main.T - curr_time, main.timestep)
			goal_traj_costs[i] = np.sum(main.weights * np.sum(main.environment.featurize(goal_traj), axis=1))
		cond_prob_traj = np.exp(np.outer(curr_traj_cost + goal_traj_costs - self.optimal_costs, -self.betas)) * \
		                 (self.betas / (2*np.pi)) ** (main.next_waypt_idx / 2)
		prob_traj_joint = cond_prob_traj * self.joint_beliefs_prior
		self.joint_beliefs = prob_traj_joint / np.sum(prob_traj_joint)
		print self.joint_beliefs #TODO: remove once this is tested
		main.running_inference = False
