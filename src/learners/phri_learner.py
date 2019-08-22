import numpy as np
import math

from scipy.optimize import minimize, newton
from scipy.stats import chi2

class PHRILearner(object):
	"""
	This class performs correction inference given a trajectory and an input
	torque applied onto the trajectory.
	"""

	def __init__(self, feat_method, feat_list, environment, constants):

		# ---- Important internal variables ---- #
		self.feat_method = feat_method
		self.feat_list = feat_list
		self.num_features = len(self.feat_list)
		self.weights = [0.0] * self.num_features
		self.betas = [1.0] * self.num_features
		self.betas_u = [1.0] * self.num_features
		self.updates = [0.0] * self.num_features
		self.environment = environment

		self.alpha = constants["alpha"]
		self.n = constants["n"]
		UPDATE_GAINS = constants["UPDATE_GAINS"]
		MAX_WEIGHTS = constants["MAX_WEIGHTS"]
		FEAT_RANGE = constants["FEAT_RANGE"]

		self.update_gains = [UPDATE_GAINS[self.feat_list[feat]] for feat in range(self.num_features)]
		self.max_weights = [MAX_WEIGHTS[self.feat_list[feat]] for feat in range(self.num_features)]
		self.feat_range = [FEAT_RANGE[self.feat_list[feat]] for feat in range(self.num_features)]
		self.P_beta = constants["P_beta"]

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
		self.traj = traj
		self.traj_deform = traj.deform(u_h, t, self.alpha, self.n)

		new_features = self.environment.featurize(self.traj_deform.waypts, self.feat_list)
		old_features = self.environment.featurize(traj.waypts, self.feat_list)

		Phi_p = np.array([sum(x) for x in new_features])
		Phi = np.array([sum(x) for x in old_features])

		update = Phi_p - Phi
		self.updates = update.tolist()

		if self.feat_method == "all":
			# Update all weights. 
			curr_weight = self.all_update(update)
		elif self.feat_method == "max":
			# Update only weight of maximal change.
			curr_weight = self.max_update(update)
		elif self.feat_method == "beta":
			# Set up the optimization problem.
			def u_constrained(u):
				cost = np.linalg.norm(u)**2
				return cost

			def u_constraint(u, idx):
				u_p = np.reshape(u, (7,1))
				waypts_deform_p = self.traj.deform(u_p, t, self.alpha, self.n).waypts
				H_features = self.environment.featurize(waypts_deform_p, [self.feat_list[idx]])[0]
				Phi_H = sum(H_features)
				cost = (Phi_H - Phi_p[idx])**2
				return cost

			for i in range(self.num_features):
				# Compute optimal action.
				u_h_opt = minimize(u_constrained, np.zeros((7,1)), method='SLSQP', 
									constraints=({'type': 'eq', 'fun': u_constraint, 'args': (i,)}), 
									options={'maxiter': 10, 'ftol': 1e-6, 'disp': True})
				l = math.pi
				u_h_star = np.reshape(u_h_opt.x, (7, 1)) 

				# Compute beta based on deviation from optimal action.
				beta_norm = 1.0 / np.linalg.norm(u_h_star) ** 2
				self.betas[i] = self.num_features / (2 * beta_norm * abs(np.linalg.norm(u_h)**2 - np.linalg.norm(u_h_star)**2))
				print "Here is beta:", self.betas

				### Compute update using P(r|beta) for the beta estimate we just computed ###
				# Compute P(r|beta)
				mus1 = self.P_beta[self.feat_list[i]+"1"]
				mus0 = self.P_beta[self.feat_list[i]+"0"]
				p_r0 = chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) / (chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]))
				p_r1 = chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]) / (chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]))

				# Newton-Rapson setup; define function, derivative, and call optimization method.
				def f_theta(weights_p):
					num = p_r1 * np.exp(weights_p * update[i])
					denom = p_r0 * (l/math.pi) ** (self.num_features/2.0) * np.exp(-l*update[i]**2) + num
					return weights_p + self.update_gains[i] * num * update[i]/denom - self.weights[i]
				def df_theta(weights_p):
					num = p_r0 * (l/math.pi) ** (self.num_features/2.0) * np.exp(-l*update[i]**2)
					denom = p_r1 * np.exp(weights_p*update[i])
					return 1 + self.update_gains[i] * num / denom

				weight_p = newton(f_theta,self.weights[i],df_theta,tol=1e-04,maxiter=1000)

				num = p_r1 * np.exp(weight_p * update[i])
				denom = p_r0 * (l/math.pi) ** (self.num_features/2.0) * np.exp(-l*update[i]**2) + num
				self.betas_u[i] = num/denom
				print "Here is weighted beta:", self.betas_u

			curr_weight = self.beta_update(update)
		else:
			raise Exception('Learning method {} not implemented.'.format(self.feat_method))

		# Clip values at max and min allowed weights.
		for i in range(self.num_features):
			curr_weight[i] = np.clip(curr_weight[i], 0.0, self.max_weights[i])
		print "Here is the update:", update
		print "Here are the old weights:", self.weights
		print "Here are the new weights:", curr_weight

		self.weights = curr_weight.tolist()
		return self.weights

	def all_update(self, update):
		return self.weights - self.update_gains * update
	
	def max_update(self, update):
		change_in_features = np.divide(update, self.feat_range)

		# Get index of maximal change.
		max_idx = np.argmax(np.fabs(change_in_features))

		# Update only weight of feature with maximal change.
		curr_weight = np.array([self.weights[i] for i in range(len(self.weights))])
		curr_weight[max_idx] = curr_weight[max_idx] - self.update_gains[max_idx]*update[max_idx]
		return curr_weight

	def beta_update(self, update):
		return self.weights - np.array(self.betas_u) * self.update_gains * update
