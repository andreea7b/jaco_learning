import numpy as np
import math

class TeleopLearner(object):
	"""
	This class performs goal and confidence inference given user inputs.
	"""

	def __init__(self, main, goal_beliefs, beta_beliefs, betas, inference_method="dragan"):
		self.main = main # store a reference to the TeleopInference object
		self.goal_beliefs = goal_beliefs
		self.beta_beliefs = beta_beliefs
		assert(len(main.goals) == len(goal_beliefs))
		self.betas = betas
		# joint_beliefs is shape (len(goals), len(betas))
		self.joint_beliefs_prior = np.outer(goal_beliefs, beta_beliefs)
		self.joint_beliefs = self.joint_beliefs_prior
		self.last_inf_idx = -1 # holds the index of the last time from which inference was run
		# so other parts of the program don't need to recompute trajectories
		self.cache = {}

		if inference_method == "dragan":
			self._update_argmax_joint()
			# precompute the costs of optimal trajectories to all goals for later
			self.optimal_costs = np.zeros(len(goal_beliefs))
			self.cache['goal_traj_by_idx'] = {0: []} # these can be reused elsewhere
			self.cache['goal_traj_plan_by_idx'] = {0: []}
			for i in range(len(goal_beliefs)):
				traj, traj_plan = main.planner.replan(main.start, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
				                                      main.T, main.timestep, return_both=True)
				traj_cost = np.sum(main.goal_weights[i] * np.sum(main.environment.featurize(traj.waypts, main.feat_list), axis=1))
				self.optimal_costs[i] = traj_cost
				self.cache['goal_traj_by_idx'][0].append(traj)
				self.cache['goal_traj_plan_by_idx'][0].append(traj_plan)
			self.inference_step = self._dragan_belief_update
		elif inference_method == "javdani":
			raise NotImplementedError
		else:
			raise ValueError

	def _dragan_belief_update(self):
		main = self.main
		this_idx = main.next_waypt_idx - 1
		curr_traj = main.traj_hist[:this_idx + 1]
		curr_pos = curr_traj[-1]
		curr_traj_features = np.sum(main.environment.featurize(curr_traj, main.feat_list), axis=1)
		curr_traj_costs = np.array([np.sum(main.goal_weights[i] * curr_traj_features) for i in range(len(self.goal_beliefs))])
		goal_traj_costs = np.zeros(len(self.goal_beliefs))
		curr_time = this_idx * main.timestep
		self.cache['goal_traj_by_idx'][this_idx] = []
		self.cache['goal_traj_plan_by_idx'][this_idx] = []
		for i in range(len(self.goal_beliefs)):
			# TODO: redo in a way that is not time-invariant (modify planner.trajOpt to take waypoint times and pass them into costs)
			goal_traj, goal_traj_plan = main.planner.replan(curr_pos, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
											                main.T - curr_time, main.timestep, return_both=True)
			goal_traj_costs[i] = np.sum(main.goal_weights[i] * np.sum(main.environment.featurize(goal_traj.waypts, main.feat_list), axis=1))
			self.cache['goal_traj_by_idx'][this_idx].append(goal_traj)
			self.cache['goal_traj_plan_by_idx'][this_idx].append(goal_traj_plan)
		cond_prob_traj = np.exp(np.outer(curr_traj_costs + goal_traj_costs - self.optimal_costs, -self.betas)) * \
		                 (self.betas / (2*np.pi)) ** (this_idx / 2)
		prob_traj_joint = cond_prob_traj * self.joint_beliefs_prior
		self.joint_beliefs = prob_traj_joint / np.sum(prob_traj_joint)
		self._update_argmax_joint()
		self.last_inf_idx = this_idx
		main.running_inference = False

	def _update_argmax_joint(self):
		goal, beta_idx = np.unravel_index(np.argmax(self.joint_beliefs), self.joint_beliefs.shape)
		beta = self.betas[beta_idx]
		self.argmax_joint_beliefs = (goal, beta)
