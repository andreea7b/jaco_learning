import numpy as np
import math
import time

class TeleopLearner(object):
	"""
	This class performs goal and confidence inference given user inputs.
	"""

	def __init__(self, main, goal_priors, beta_priors, betas, inference_method, beta_method):
		self.main = main # store a reference to the TeleopInference object
		self.goal_priors = goal_priors
		self.goal_beliefs = goal_priors
		self.beta_priors = beta_priors
		assert(len(main.goals) == len(goal_priors))
		self.betas = betas
		self.cache = {}
		# precompute the costs of optimal trajectories to all goals for later
		self.optimal_costs = np.zeros(len(goal_priors))
		self.cache['goal_traj_by_idx'] = {0: []} # these can be reused elsewhere
		self.cache['goal_traj_plan_by_idx'] = {0: []}
		for i in range(len(goal_priors)):
			traj, traj_plan = main.planner.replan(main.start, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
												  main.T, main.timestep, return_both=True)
			traj_cost = np.sum(main.goal_weights[i] * np.sum(main.environment.featurize(traj.waypts, main.feat_list), axis=1))
			self.optimal_costs[i] = traj_cost
			self.cache['goal_traj_by_idx'][0].append(traj)
			self.cache['goal_traj_plan_by_idx'][0].append(traj_plan)
		self.last_inf_idx = 0 # holds the index of the last time from which inference was run
		if beta_method == "joint":
			# joint_beliefs is shape (len(goals), len(betas))
			self.joint_beliefs_prior = np.outer(goal_priors, beta_priors)
			self.joint_beliefs = self.joint_beliefs_prior
			if inference_method == "dragan":
				self._update_argmax_joint()
				self.inference_step = lambda: self._dragan_update(True)
				self.final_step = lambda: self._dragan_update_final(True)
			elif inference_method == "javdani":
				raise NotImplementedError
			else:
				raise ValueError
		elif beta_method == "estimate":
			self.beta_estimates = np.full(len(goal_priors), 1e-6)
			if inference_method == "dragan":
				self._update_argmax_estimate()
				self.inference_step = lambda: self._dragan_update(False)
				self.final_step = lambda: self._dragan_update_final(False)
			elif inference_method == "javdani":
				raise NotImplementedError
			else:
				raise ValueError
		else:
			raise ValueError

	def _dragan_update(self, is_joint):
		main = self.main
		start = time.time() # TODO: remove
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
			# Using default (straight line) initialization
			#goal_traj, goal_traj_plan = main.planner.replan(curr_pos, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
			#								                main.T - curr_time, main.timestep, return_both=True)
			# Using last plan initialization
			#seed = self.cache['goal_traj_plan_by_idx'][self.last_inf_idx][i].waypts
			#seed[0] = curr_pos
			#goal_traj, goal_traj_plan = main.planner.replan(curr_pos, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
			#								                main.T - curr_time, main.timestep, return_both=True,
			#								                seed=seed)
			# Using last final waypoint and straight line initialization
			goal_waypt = self.cache['goal_traj_plan_by_idx'][self.last_inf_idx][i].waypts[-1]
			goal_traj, goal_traj_plan = main.planner.replan(curr_pos, goal_waypt, list(main.goal_locs[i]), main.goal_weights[i],
											                main.T - curr_time, main.timestep, return_both=True)
			goal_traj_costs[i] = np.sum(main.goal_weights[i] * np.sum(main.environment.featurize(goal_traj.waypts, main.feat_list), axis=1))
			self.cache['goal_traj_by_idx'][this_idx].append(goal_traj)
			self.cache['goal_traj_plan_by_idx'][this_idx].append(goal_traj_plan)
		suboptimality = curr_traj_costs + goal_traj_costs - self.optimal_costs
		suboptimality *= (3.5 / (0.01 * 1.))
		print 'suboptimality:', suboptimality
		print 'suboptimality/time:', suboptimality / this_idx
		if is_joint: # joint inference over beta and goals
			cond_prob_traj = np.exp(np.outer(suboptimality, -self.betas)) * (((self.betas/(2*np.pi))**this_idx)**(7/2))
			prob_traj_joint = cond_prob_traj * self.joint_beliefs_prior
			self.joint_beliefs = prob_traj_joint / np.sum(prob_traj_joint)
			self._update_argmax_joint()
		else: # MAP estimation of beta
			self.beta_estimates = (this_idx * 7 / 2) / (suboptimality + self.beta_priors)
			cond_prob_traj = np.exp(suboptimality * -self.beta_estimates) * (((self.beta_estimates/(2*np.pi))**this_idx)**(7/2))
			prob_traj_joint = cond_prob_traj * self.goal_priors
			self.goal_beliefs = prob_traj_joint / np.sum(prob_traj_joint)
			self._update_argmax_estimate()
		self.last_inf_idx = this_idx
		end = time.time() #TODO: remove
		print 'inference time:', end - start #TODO: remove
		main.running_inference = False

	def _dragan_update_final(self, is_joint):
		main = self.main
		traj_features = np.sum(main.environment.featurize(main.traj_hist, main.feat_list), axis=1)
		traj_costs = np.array([np.sum(main.goal_weights[i] * traj_features) for i in range(len(self.goal_beliefs))])
		constraint_costs = np.zeros(len(self.goal_beliefs))
		#curr_time = len(main.traj_hist) * main.timestep
		for i in range(len(self.goal_beliefs)):
			# calculate constraint violation costs
			if "efficiency" in main.feat_list:
				constraint_costs[i] = main.environment.goal_dist_features(i, main.traj_hist[-1])
				constraint_costs[i] *= main.goal_weights[i][main.feat_list.index("efficiency")]
				constraint_costs[i] *= 1 # TODO: tune
		suboptimality = curr_traj_costs + constraint_costs - self.optimal_costs
		suboptimality *= (3.5 / (0.01 * 1.))
		print 'final suboptimality:', suboptimality
		print 'final suboptimality/time:', suboptimality / this_idx
		if is_joint: # joint inference over beta and goals
			cond_prob_traj = np.exp(np.outer(suboptimality, -self.betas)) * (((self.betas/(2*np.pi))**this_idx)**(7/2))
			prob_traj_joint = cond_prob_traj * self.joint_beliefs_prior
			self.joint_beliefs = prob_traj_joint / np.sum(prob_traj_joint)
			self._update_argmax_joint()
			print 'final beta: ', self.argmax_joint_beliefs[1]
		else: # MAP estimation of beta
			self.beta_estimates = (this_idx * 7 / 2) / (suboptimality + self.beta_priors)
			cond_prob_traj = np.exp(suboptimality * -self.beta_estimates) * (((self.beta_estimates/(2*np.pi))**this_idx)**(7/2))
			prob_traj_joint = cond_prob_traj * self.goal_priors
			self.goal_beliefs = prob_traj_joint / np.sum(prob_traj_joint)
			self._update_argmax_estimate()
			print 'final beta: ', self.argmax_estimate[1]
		if (is_joint and self.argmax_joint_beliefs[1] < 0.3) or (not is_joint and self.argmax_estimate[1] < 0.3):
			# learn new goal here
			print 'detected new goal:', main.traj_hist[-1]

	def _update_argmax_joint(self):
		goal, beta_idx = np.unravel_index(np.argmax(self.joint_beliefs), self.joint_beliefs.shape)
		beta = self.betas[beta_idx]
		self.argmax_joint_beliefs = (goal, beta)

	def _update_argmax_estimate(self):
		goal = np.argmax(self.goal_beliefs)
		beta = self.beta_estimates[goal]
		self.argmax_estimate = (goal, beta)
