import numpy as np
import math
import time
import pybullet as p

class TeleopLearner(object):
	"""
	This class performs goal and confidence inference given user inputs.
	"""

	def __init__(self, main, goal_priors, beta_priors, betas, inference_method, beta_method):
		# TODO: remove (for final project)
		self.beta_hist = []
		self.inf_hist = []
		self.sub_hist = []
		self.belief_hist = []

		self.main = main # store a reference to the TeleopInference object
		self.goal_priors = goal_priors
		self.goal_beliefs = goal_priors
		self.beta_priors = beta_priors
		self.betas = betas

		self.last_inf_idx = 0 # holds the index of the last time from which inference was run

		self.inference_method = inference_method
		if inference_method == "none" or inference_method == "collect":
			self.inference_step = self._no_inference
			self.final_step = self._no_inference_final
			return

		self.cache = {}
		# precompute the costs of optimal trajectories to all goals for later
		self.optimal_costs = np.zeros(len(goal_priors))
		self.cache['goal_traj_by_idx'] = {0: []} # these can be reused elsewhere
		self.cache['goal_traj_plan_by_idx'] = {0: []}


		for i in range(main.num_goals):
			#traj, traj_plan = main.planner.replan(main.start, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
			#									  main.T, main.timestep, return_both=True)
			#support = np.arange(len(main.goal_weights[i]))[main.goal_weights[i] != 0.0]
			#traj_cost = np.sum(main.goal_weights[i][support] * np.sum(main.environment.featurize(traj.waypts, support), axis=1))

			# Server versions:
			# Using IK from goal loc as seed:
			#goal_waypt = p.calculateInverseKinematics(main.bullet_environment["robot"], 7, main.goal_locs[i])[:7]
			(traj, traj_plan), traj_cost = main.planner.replan_and_get_cost(main.start, main.IK_goals[i], i, i,
																			main.T, main.timestep, return_both=True)
			# # Using server's goal position seed:
			# (traj, traj_plan), traj_cost = main.planner.replan_and_get_cost(main.start, i, i, i,
			# 																main.T, main.timestep, return_both=True)

			self.optimal_costs[i] = traj_cost
			self.cache['goal_traj_by_idx'][0].append(traj)
			self.cache['goal_traj_plan_by_idx'][0].append(traj_plan)
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
		print "optimal costs", self.optimal_costs

	def _dragan_update(self, is_joint):
		main = self.main
		start = time.time() # TODO: remove
		this_idx = main.next_waypt_idx - 1
		#print 'doing inference from', this_idx
		curr_traj = main.traj_hist[:this_idx + 1]
		curr_pos = curr_traj[-1]
		#curr_traj_features = np.sum(main.environment.featurize(curr_traj), axis=1)
		#curr_traj_costs = np.array([np.sum(main.goal_weights[i] * curr_traj_features) for i in range(main.num_goals)])
		curr_traj_costs = np.array([main.planner.get_cost(curr_traj, i) for i in range(main.num_goals)])
		print 'curr_traj costs:', curr_traj_costs

		goal_traj_costs = np.zeros(main.num_goals)
		curr_time = this_idx * main.timestep
		self.cache['goal_traj_by_idx'][this_idx] = []
		self.cache['goal_traj_plan_by_idx'][this_idx] = []
		for i in range(main.num_goals):
			# # TODO: redo in a way that is not time-invariant (modify planner.trajOpt to take waypoint times and pass them into costs)
			# # Using default (straight line) initialization
			# #goal_traj, goal_traj_plan = main.planner.replan(curr_pos, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
			# #								                main.T - curr_time, main.timestep, return_both=True)
			# # Using last plan initialization
			# #seed = self.cache['goal_traj_plan_by_idx'][self.last_inf_idx][i].waypts
			# #seed[0] = curr_pos
			# #goal_traj, goal_traj_plan = main.planner.replan(curr_pos, main.goals[i], list(main.goal_locs[i]), main.goal_weights[i],
			# #								                main.T - curr_time, main.timestep, return_both=True,
			# #								                seed=seed)
			# # Using last final waypoint and straight line initialization
			# goal_waypt = self.cache['goal_traj_plan_by_idx'][self.last_inf_idx][i].waypts[-1]
			# # with cartesian goal constraint (as long as planner has prefer_angles=False)
			# goal_traj, goal_traj_plan = main.planner.replan(curr_pos, goal_waypt, list(main.goal_locs[i]), main.goal_weights[i],
			# 								                main.T - curr_time, main.timestep, return_both=True)
			# # without cartesian goal constraint
			# #goal_traj, goal_traj_plan = main.planner.replan(curr_pos, goal_waypt, None, main.goal_weights[i],
			# #								                main.T - curr_time, main.timestep, return_both=True)
			# support = np.arange(len(main.goal_weights[i]))[main.goal_weights[i] != 0.0]
			# goal_traj_costs[i] = np.sum(main.goal_weights[i][support] * np.sum(main.environment.featurize(goal_traj.waypts, support), axis=1))
			# print 'planned goal traj len', len(goal_traj.waypts)

			# ----- Server versions: -----

			# TODO: this segfaults, since TeleopLearner is in different thread than main (which is using pybullet)
			# # Using IK from goal loc as seed:
			# goal_waypt = p.calculateInverseKinematics(main.bullet_environment["robot"], 7, main.goal_locs[i])[:7]
			# (goal_traj, goal_traj_plan), goal_traj_costs[i] = main.planner.replan_and_get_cost(curr_pos, goal_waypt, i, i,
			# 																				   main.T - curr_time,
			# 																				   main.timestep,
			# 																				   return_both=True)
			# print 'cost', goal_traj_costs[i]
			# Using IK_goal
			(goal_traj, goal_traj_plan), goal_traj_costs[i] = main.planner.replan_and_get_cost(curr_pos, main.IK_goals[i], i, i,
																							   main.T - curr_time,
																							   main.timestep,
																							   return_both=True)
			print 'cost', goal_traj_costs[i]

			# # Using server's goal position seed:
			#(goal_traj, goal_traj_plan), goal_traj_costs[i] = main.planner.replan_and_get_cost(curr_pos, i, i, i,
			#																				   main.T - curr_time,
			#																				   main.timestep,
			#																				   return_both=True)
			#print 'cost', goal_traj_costs[i]

			self.cache['goal_traj_by_idx'][this_idx].append(goal_traj)
			self.cache['goal_traj_plan_by_idx'][this_idx].append(goal_traj_plan)
		print 'goal traj costs', goal_traj_costs
		suboptimality = curr_traj_costs + goal_traj_costs - self.optimal_costs
		#suboptimality *= (1. / self.optimal_costs)
		#print 'suboptimality:', suboptimality
		#print 'suboptimality/time:', suboptimality / this_idx
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
			# TODO: remove (for final project)
			self.beta_hist.append(self.beta_estimates)
			self.inf_hist.append(this_idx)
			self.sub_hist.append(suboptimality)
			self.belief_hist.append(self.goal_beliefs)

		self.last_inf_idx = this_idx
		end = time.time() #TODO: remove
		print 'inference time:', end - start #TODO: remove
		main.running_inference = False

	def _dragan_update_final(self, is_joint):
		main = self.main
		# traj_features = np.sum(main.environment.featurize(main.traj_hist), axis=1)
		# traj_costs = np.array([np.sum(main.goal_weights[i] * traj_features) for i in range(main.num_goals)])
		# constraint_costs = np.zeros(main.num_goals)
		# this_idx = len(main.traj_hist)
		# curr_time = this_idx * main.timestep
		# for i in range(main.num_goals):
		# 	# calculate constraint violation costs
		# 	if "efficiency" in main.feat_list:
		# 		constraint_costs[i] = main.environment.goal_dist_features(i, main.traj_hist[-1])
		# 		constraint_costs[i] *= main.goal_weights[i][main.feat_list.index("efficiency")]
		# 		constraint_costs[i] *= 10 # TODO: tune
		# print 'constraint_costs', constraint_costs
		# suboptimality = traj_costs + constraint_costs - self.optimal_costs
		this_idx = len(main.traj_hist)
		curr_time = this_idx * main.timestep
		traj_costs = np.array([main.planner.get_cost(main.traj_hist, i, i) for i in range(main.num_goals)])
		suboptimality = traj_costs - self.optimal_costs
		#suboptimality *= (1. / self.optimal_costs)
		print 'curr traj costs', traj_costs
		print 'optimal_costs', self.optimal_costs
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

			# TODO: remove all of the below (for final project)
			self.beta_hist.append(self.beta_estimates)
			self.inf_hist.append(this_idx)
			self.sub_hist.append(suboptimality)
			self.belief_hist.append(self.goal_beliefs)
			#np.save('/sub_hist.npy', np.array(self.sub_hist))
			#np.save('/inf_hist.npy', np.array(self.inf_hist))
			print 'beta:', np.array(self.beta_hist)
			print 'suboptimality:', np.array(self.sub_hist)
			print 'inference times:', np.array(self.inf_hist)
			main.exp_data['beta_hist'] = np.array(self.beta_hist)
			main.exp_data['sub_hist'] = np.array(self.sub_hist)
			main.exp_data['belief_hist'] = np.array(self.belief_hist)

			print 'final beta:', self.argmax_estimate[1]
		if (is_joint and self.argmax_joint_beliefs[1] < 0.3) or (not is_joint and self.argmax_estimate[1] < 0.3):
			# learn new goal here
			print 'detected new goal:', main.traj_hist[-1]
		#print 'joint6_assist', main.exp_data['joint6_assist']
		#import pdb; pdb.set_trace()

		#np.save('/home/matthew/traj_hist_nobj10.npy', np.array(main.traj_hist))
		#print 'saved trajectory'
		main.final_inference_done=True

	def _no_inference(self):
		main = self.main
		this_idx = main.next_waypt_idx - 1
		self.last_inf_idx = this_idx
		print 'ran dummy inference'
		main.running_inference = False

	def _no_inference_final(self):
		main = self.main
		if self.inference_method == "collect":
			np.save('placeholder', np.array(main.traj_hist))
			print 'saved trajectory'
		main.final_inference_done = True

	def _update_argmax_joint(self):
		goal, beta_idx = np.unravel_index(np.argmax(self.joint_beliefs), self.joint_beliefs.shape)
		beta = self.betas[beta_idx]
		self.argmax_joint_beliefs = (goal, beta)

	def _update_argmax_estimate(self):
		goal = np.argmax(self.goal_beliefs)
		beta = self.beta_estimates[goal]
		self.argmax_estimate = (goal, beta)
