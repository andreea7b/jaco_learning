from maxent_irl_utils import *
from utils.transform_input import transform_input, get_subranges
import random
import torch.optim as optim
from tqdm import trange


class DeepMaxEntIRL:
	"""
	This class contains the demonstrations, function, and training procedure to apply Deep Maximum Entropy IRL to
	the 7DoF Robot arm, learning a reward/cost from demonstrations.

	Initi:
	s_g_exp_trajs	A list of lists, each sublist containing demonstrations with the same start & goal pair
	goal_poses		List of goal poses for each start & goal pair
	known_feat_list	List of strings with the known features
	gen				The mode with which induced near optimal trajectories with the current cost are calculated
					can be 'waypt' (get TrajOpt optimal trajectory and perturb waypoints)
					or 'cost' (perturb the cost function slightly and get TrajOpt optimal trajectories for all of them)
	T, timestep		Settings for TrajOpt
	obj_center_dict	Dict of human & laptop positions used when calculating the known feature values
	feat_range_dict	Dict of factors for the different features to scale them to 0-1. -> used to calculate known features


	NN_dict			Settings for the cost function NN see example belo
					NN_dict = {'n_layers': 2, 'n_units':128, 'sin':False, 'cos':False, 'noangles':True, 'norot':True, 'rpy':False, 'lowdim':False,
						   '6D_laptop':False, '6D_human':False, '9D_coffee':False}

	sin, cos, noangles	: noangles means no angles in the input space, sin/cos and are transforms of the default 7 angles
	norot, EErot, rpy	: if norot True no rotation matrices are in the input space, if EErot only the rotation matrix
						of the endeffector and not of the other joints, rpy: the Roll/Pitch/Yaw representation is used
	lowdim				: only the endeffector xyz, and rotation matrix plus the angles & object xyz
	9D_coffee			: only the 9 entries of the Endeffector rotation matrix as input
	6D_laptop, 6D_human	: only the endeffector xyz plus the xyz of the laptop or the human is used as input space

	"""

	def __init__(self, env, planner, weight, s_g_exp_trajs, goal_poses, known_feat_list, NN_dict, gen, T=20., timestep=0.5):
		self.planner = planner
		self.weight = weight # weight to use in planner for this feature
		# planner settings
		self.T = T
		self.timestep = timestep

		self.env = env
		self.gen = gen

		# care about known features
		self.known_feat_list = known_feat_list
		self.known_feat_transformer = TorchFeatureTransform(env.object_centers, known_feat_list, env.feat_range)

		# get some derivative data from the s_g_exp_trajs
		self.init_s_g_exp_trajs = s_g_exp_trajs
		self.s_g_exp_trajs = []
		self.starts = []
		self.goals = []
		self.goal_poses = goal_poses
		# full data for normalization
		self.full_exp_data = np.empty((0, 97), float)

		for s_g_trajs in s_g_exp_trajs:
			self.starts.append(s_g_trajs[0][0, :7])
			self.goals.append(s_g_trajs[0][-1, :7])
			full_dim_trajs = map_to_raw_dim(self.env, s_g_trajs)
			self.s_g_exp_trajs.append(full_dim_trajs)
			for traj in full_dim_trajs:
				self.full_exp_data = np.vstack((self.full_exp_data, traj))
		self.NN_dict = NN_dict
		self.max_label = 1.
		self.min_label = 0.

		# get the input dim & instantiate cost NN
		self.raw_input_dim = transform_input(torch.ones(97), NN_dict).shape[1]
		self.cost_nn = ReLuNet(NN_dict['n_layers'], NN_dict['n_units'], self.raw_input_dim, input_residuals=len(known_feat_list))

	def function(self, x, torchify=False, norm=False):
		"""
			Cost Function used for Training & in Trajopt to calculate induced trajectories
			----
			Input:
			x			Nx97 tensor raw state input
			torchify	if the output should be a torch tensor
			norm		if the output should be normalized

			Output:
			y 			Nx1 output tensor
		"""
		# used for Trajopt that optimizes it
		x_raw = self.torchify(x)

		# transform 97D input
		x_trans = transform_input(x_raw, self.NN_dict)

		# add known feature values
		if len(self.known_feat_list) > 0:
			known_features = self.known_feat_transformer.featurize(x_raw)
			# add inputs together
			x = torch.cat((x_trans, known_features), dim=1)
		else:
			x = x_trans

		y = self.cost_nn(x)
		if not norm:
			if torchify:
				return y
			else:
				return y.detach().numpy()
		else:
			y = (y - self.min_label) / (self.max_label - self.min_label)
			if torchify:
				return y
			else:
				return y.detach().numpy()

	def torchify(self, x):
		"""
		Transforms numpy input to torch tensors.
		"""
		x = torch.Tensor(x)
		if len(x.shape) == 1:
			x = x.unsqueeze(0)
		return x

	def update_normalizer(self):
		"""
		Update the max & min labels used for normalization based on all states in the demonstrations.
		"""
		# Note: if there are few expert demo, it might lead to too low max_label (feature values get high)
		all_logits = self.function(self.full_exp_data, norm=False)
		self.max_label = np.amax(all_logits)
		self.min_label = np.amin(all_logits)

	def get_trajs_with_cur_reward(self, n_traj, std, start, goal, pose):
		"""
			Generate a set of induced trajectories for the current reward/cost.
			----
			Input:
			n_traj		desired number of trajectories
			start		start position for TrajOpt
			end			end position for TrajOpt
			pose		pose for TrajOpt

			Output:  	List of n_traj induced trajectories in 97D
		"""
		if self.gen == 'waypt':
			cur_rew_traj = generate_Gaus_MaxEnt_trajs(self.planner, self.weight, std,
													  n_traj, start, goal, pose, self.T, self.timestep)
		# note/TODO: 'cost' doesn't work
		elif self.gen == 'cost':
			cur_rew_traj = generate_cost_perturb_trajs(self.planner, self.env, std,
													   n_traj, start, goal, pose, self.T, self.timestep)
		else:
			print("gen has to be either waypt or cost")
			assert False
		return map_to_raw_dim(self.env, cur_rew_traj)

	def get_total_cost(self, waypt_array):
		"""
			Calculate the total cost over a nx97D input tensor using the unnormalized function.
			----
			Input:
			waypt_array		a nx97D torch tensor of waypoints

			Output:  a scalar of the total cost under the current cost function
		"""
		waypt_rewards = self.function(waypt_array, torchify=True, norm=False)
		return torch.sum(waypt_rewards, 0)

	def deep_max_ent_irl(self, n_iters, n_cur_rew_traj=1, lr=1e-3, weight_decay=0.01, std=0.01):
		"""
			The training function for deep maximum entropy IRL.
			----
			Input:
			n_iters				number of iterations (one iteration goes through all demonstrations once)
			n_cur_rew_traj		number of induced trajectories calculated for the current reward
			std					standard deviation used when calculating near-optimal trajectories
			lr, weight_decay	settings for the optimizer

			Output:  a scalar of the total cost under the current cost function
		"""
		loss_log = []
		optimizer = optim.Adam(self.cost_nn.parameters(), lr=lr, weight_decay=weight_decay)
		with trange(n_iters) as T:
			for it in T:
				T.set_description('Iteration %i' % it)
				# Step 0: generate traj under current reward
				s_g_specific_trajs = []
				if self.goal_poses is None:
					g_poses = [None for _ in range(len(self.starts))]
				else:
					g_poses = self.goal_poses

				for start, goal, goal_pose in zip(self.starts, self.goals, g_poses):
					s_g_specific_trajs.append(self.get_trajs_with_cur_reward(n_cur_rew_traj, std, start, goal, goal_pose))

				# make dataset random
				s_g_indices = np.arange(len(self.starts)).tolist()
				random.shuffle(s_g_indices)

				loss_tracker = []
				# iterate over start_goal configurations
				for j in s_g_indices:
					indices = np.arange(min(len(s_g_specific_trajs[j]), len(self.s_g_exp_trajs[j]))).tolist()
					random.shuffle(indices)

					# Subloop for the multiple trajectories within one S-G Configuration
					# Note: one batch is exactly two trajectories, one demo and one induced one with the same starrt
					# and goal as the demonstration.
					for i in indices:
						exp_traj = self.s_g_exp_trajs[j][i % len(self.s_g_exp_trajs[j])]
						cur_rew_traj = s_g_specific_trajs[j][i % n_cur_rew_traj]

						# Step 1: calculate the cost for expert & current optimal
						exp_rew = self.get_total_cost(exp_traj)
						cur_opt_rew = self.get_total_cost(cur_rew_traj)

						# Step 2: calculate loss & backpropagate
						loss = (exp_rew - cur_opt_rew)
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
						loss_tracker.append(loss.item())

				loss_log.append(sum(loss_tracker) / len(loss_tracker))

				T.set_postfix(avg_loss=loss_log[-1])

		# update normalizer once in the end
		self.update_normalizer()

	def save(self, path):
	    torch.save({
	        "known_feat_list": self.known_feat_list,
	        "s_g_exp_trajs": self.init_s_g_exp_trajs,
	        "goal_poses": self.goal_poses,
	        "NN_dict": self.NN_dict,
	        "gen": self.gen,
	        "cost_nn_state_dict": self.cost_nn.state_dict(),
	        "max_label": self.max_label,
	        "min_label": self.min_label
	    }, path)

	def load_cost_nn_state_dict(self, state_dict):
		self.cost_nn.load_state_dict(state_dict)
