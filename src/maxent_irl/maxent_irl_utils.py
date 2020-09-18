import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# NN stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# stuff for TarjOpt
from src.planners.trajopt_planner import TrajoptPlanner
from src.utils.plot_utils import *
from src.utils.environment import Environment


def map_to_raw_dim(env, expert_demos):
	"""
		Map all trajectories in a list from 7D configuration space to 97D raw input space
		----
		Input:
		env				an environment object
		expert_demos	list of demonstrations
	"""
	out_demos = []
	for j, traj in enumerate(expert_demos):
		raw_feature_traj = []
		temp_traj = traj.copy()
		for i in range(temp_traj.shape[0]):
			out = env.raw_features(temp_traj[i, :])
			raw_feature_traj.append(out)
		out_demos.append(np.array(raw_feature_traj))

	return out_demos


def generate_cost_perturb_trajs(planner, env, std, n_traj, start, goal, goal_pose, T, timestep):
	"""
		Generate a set of n_traj near optimal trajectories under the reward in the env.
		Idea: let's perturb the cost and start-goal positions slightly to get a set of near-optimal trajectories
		----
		Input:
		env						an environment object containing the current cost function as learned_feature
		planner					TrajOpt planner object
		std						standard deviation used for perturbation
		n_traj					number of induced trajectories to produce
		start, goal, goal_pose	settings for TrajOpt
		T, timestep				settings for TrajOpt
	"""

	gt_weights = env.weights
	# Step 1: generate nominal optimal plan
	opt_traj = planner.replan(start, goal, goal_pose, T, timestep, seed=None)

	# Step 2: generate n_demos-1 soft-optimal trajectories
	expert_demos = [opt_traj.waypts]

	for _ in range(n_traj - 1):
		# perturb weights in env
		env.weights = gt_weights + np.random.normal(loc=0, scale=std, size=env.weights.shape[0])
		# perturb start & goal slightly
		cur_start = start + np.random.normal(loc=0, scale=std, size=7)
		cur_goal = goal + np.random.normal(loc=0, scale=std, size=7)
		# plan with perturbed weights
		traj = planner.replan(cur_start, cur_goal, goal_pose, T, timestep, seed=None).waypts
		# append
		expert_demos.append(np.array(traj))

	# reset env weights to gt
	env.weights = gt_weights
	return expert_demos


def generate_Gaus_MaxEnt_trajs(planner, scale, n_traj, start, goal, goal_pose, T, timestep):
	"""
		Generate a set of n_traj near optimal trajectories under the reward in the env.
		Idea: let's perturb the 7D angle waypoints with a normal distribution to get soft-optimal trajectories
		----
		Input:
		env						an environment object containing the current cost function as learned_feature
		planner					TrajOpt planner object
		std						standard deviation used for perturbation
		n_traj					number of induced trajectories to produce
		start, goal, goal_pose	settings for TrajOpt
		T, timestep				settings for TrajOpt
	"""
	opt_traj = planner.replan(start, goal, goal_pose, T, timestep, seed=None)

	# Step 2: generate n_demos-1 soft-optimal trajectories
	expert_demos = [opt_traj.waypts]

	for _ in range(n_traj - 1):
		cur_traj = []
		for i in range(opt_traj.waypts.shape[0]):
			cur_traj.append(opt_traj.waypts[i, :] + np.random.normal(loc=0, scale=scale, size=7))
		expert_demos.append(np.array(cur_traj))

	return expert_demos


def init_env(feat_list, weights, env_only=False,
			 object_centers={'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]},
			 feat_range = {'table': 0.98, 'coffee': 1.0, 'laptop': 0.3, 'human': 0.3, 'efficiency': 0.22, 'proxemics': 0.3, 'betweenobjects': 0.2}
			 ):
	"""
		initialize an openrave environment and TrajOpt planner and return them.
		----
		Input:
		feat_list		List of strings with the active features
		weights			List of weights for the feat_list features
		env_only		if True, only an env object gets created & returned
		obj_center_dict	Dict of human & laptop positions
		feat_range_dict	Dict of factors for the different features to scale them to 0-1.
	"""
	model_filename = "jaco_dynamics"
	feat_range = [feat_range[feat_list[feat]] for feat in range(len(feat_list))]

	# Planner
	max_iter = 50
	num_waypts = 5

	environment = Environment(model_filename, object_centers, feat_list, feat_range, np.array(weights), viewer=False)
	if env_only:
		return environment
	else:
		planner = TrajoptPlanner(max_iter, num_waypts, environment)
		return environment, planner


class ReLuNet(nn.Module):
	"""
		Neural Network with leaky ReLu and last layer softplus non-linearity used to approximate the cost/reward func
		----
		Init:
		nb_layers		number of hidden layers for the NN
		nb_units		number of hidden units per layer for the NN
		raw_input_dim	dimensionality of input
		input_residuals	How many of the last entries of the input are known features -> appended in the last layer
	"""
	def __init__(self, nb_layers, nb_units, raw_input_dim, input_residuals=0):
		super(ReLuNet, self).__init__()

		self.nb_layers = nb_layers
		self.input_residuals = input_residuals

		layers = []
		dim_list = [raw_input_dim] + [nb_units] * nb_layers + [1]

		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))

		self.fc = nn.ModuleList(layers)

		# initialize weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				torch.nn.init.zeros_(m.bias)
		self.apply(weights_init)

		# last layer combining from DNN & known features
		self.weighting = nn.Linear(1 + input_residuals, 1, bias=True)

	def forward(self, x):
		if self.input_residuals > 0:
			x_residuals = x[:, -self.input_residuals:]
			# calculate normal path
			x = F.leaky_relu(self.fc[0](x[:, :-self.input_residuals]))
			for layer in self.fc[1:]:
				x = F.leaky_relu(layer(x))
			# combine normal path & residuals
			x = F.softplus(self.weighting(torch.cat((x, x_residuals), dim=1)))
		else:
			for layer in self.fc[:-1]:
				x = F.leaky_relu(layer(x))
			x = F.softplus(self.fc[-1](x))
		return x

class DNN(nn.Module):
	"""
	Creates a NN with leaky ReLu non-linearity.
	---
	input nb_layers, nb_units, input_dim
	output scalar
	"""
	def __init__(self, nb_layers, nb_units, input_dim):
		super(DNN, self).__init__()
		self.nb_layers = nb_layers

		layers = []
		dim_list = [input_dim] + [nb_units] * nb_layers + [1]

		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i+1]))

		self.fc = nn.ModuleList(layers)

		# initialize weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				torch.nn.init.zeros_(m.bias)

		self.apply(weights_init)

	def forward(self, x):
		for layer in self.fc[:-1]:
			x = F.leaky_relu(layer(x))
		x = F.softplus(self.fc[-1](x))
		return x

def plot_IRL_comparison(IRL, viz_idx=range(88, 91), frame_idx=None):
	"""
		Plot the set of demonstrations & the current set of induced trajectories with same start-goal position.
		Color denotes the value of the current cost/reward function.
		----
		Input:
		IRL		a DeepMaxEntIRL object
	"""
	plot_trajs([s_g_trajs[0] for s_g_trajs in IRL.s_g_exp_trajs],
			   IRL.env.object_centers,
			   'Expert Trajectory with learned reward',
			   IRL.function,
			   viz_idx,
			   frame_idx)

	if IRL.goal_poses is None:
		g_poses = [None for _ in range(len(IRL.starts))]
	else:
		g_poses = IRL.goal_poses
	trajs = []
	for start, goal, goal_pose in zip(IRL.starts, IRL.goals, g_poses):
		trajs.append(IRL.get_trajs_with_cur_reward(1, 0.01, start, goal, goal_pose)[0])
	plot_trajs(trajs,
			   IRL.env.object_centers,
			   'Current Trajectory with learned reward',
			   IRL.function,
			   viz_idx,
			   frame_idx)

def plot_trajs(demos, object_centers, title='some_title', func=None, viz_idx=range(88, 91), frame_idx=None):
	"""
		Plot a set of demonstrations in 3D
		----
		Input:
		demos			list of 97D demonstrations
		object_centers	dict of centers for laptop and human
		title			title for plot
		func			function for labeling the points, if None the z-coordinate is used
	"""
	# get laptop position
	laptop = object_centers['LAPTOP_CENTER']
	human = object_centers['HUMAN_CENTER']
	# Experts
	points = np.empty((0, 4))
	for traj in demos:
		if func is not None:
			labels = func(traj)
		else:
			labels = traj[:, 90].reshape((-1, 1))
		euclidean = traj[:, viz_idx]
		points = np.vstack((points, np.hstack((euclidean, labels))))
	df = pd.DataFrame(points)
	fig = go.Figure(data=go.Scatter3d(x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers',
									  marker=dict(color=df.iloc[:, 3], showscale=True), showlegend=False))
	#fig.data[0]['text'] = ['color: ' + str(round(i,2)) for i in fig.data[0]['marker']['color']]

	if frame_idx is not None:
		axes_points = np.empty((0, 12))
		for traj in demos:
			euclidean = traj[:, viz_idx]
			axis_vectors = [euclidean]
			for axis_num in range(3):
				start_idx = 7 + 9 * frame_idx + axis_num
				axis_vectors.append(euclidean + 0.1*traj[:, start_idx: start_idx+9: 3])
			axes_points = np.vstack((axes_points, np.hstack(axis_vectors)))
		for axis_num in range(3):
			color = [0, 0, 0]
			color[axis_num] = 255
			color = tuple(color)
			add_fig_vectors(fig,
							axes_points[:, 0], axes_points[:, 1], axes_points[:, 2],
							axes_points[:, 3+axis_num*3], axes_points[:, 4+axis_num*3], axes_points[:, 5+axis_num*3],
							color=color)



	fig.add_scatter3d(x=[laptop[0]], y=[laptop[1]], z=[laptop[2]], mode='markers',
					  marker=dict(size=10, color='black'), showlegend=False, hovertext=['Laptop'])
	fig.add_scatter3d(x=[human[0]], y=[human[1]], z=[human[2]], mode='markers',
					  marker=dict(size=10, color='red'), showlegend=False, hovertext=['Human'])
	fig.update_layout(title=title)
	fig.show()

def add_fig_vectors(fig, base_x, base_y, base_z, head_x, head_y, head_z, color=(255, 0, 0)):
	for x, y, z, u, v, w in zip(base_x, base_y, base_z, head_x, head_y, head_z):
		fig.add_scatter3d(x=[x, u], y=[y, v], z=[z, w],
						  line = dict(color = "rgb"+str(color),
									  width = 6),
						  marker = dict(size=0),
						  showlegend=False)

class TorchFeatureTransform(object):
	"""
		Torch version of all features that are in environment. Needed because to calculate the gradient for TrajOpt
		the path from 7D input to cost output has to be fully differentiable (and with known features we need this).
		----
		Init:
		object_centers	Dict of human & laptop positions used when calculating the feature values
		feat_range_dict	Dict of factors for the different features to scale them to 0-1. -> used to calculate feature values
		feat_list		List of strings of all active features
	"""
	def __init__(self, object_centers, feat_list, feat_range_dict):
		self.object_centers = object_centers
		self.feature_list = feat_list
		self.num_features = len(self.feature_list)
		feat_range = [feat_range_dict[feat_list[feat]] for feat in range(len(feat_list))]
		self.feat_range = feat_range

		self.feature_func_list = []

		for feat in self.feature_list:
			if feat == 'table':
				self.feature_func_list.append(self.table_features)
			elif feat == 'coffee':
				self.feature_func_list.append(self.coffee_features)
			elif feat == 'human':
				self.feature_func_list.append(self.human_features)
			elif feat == 'laptop':
				self.feature_func_list.append(self.laptop_features)

	def featurize(self, high_dim_waypt):
		"""
		Input: 97D raw torch Tensor
		Output: #known_features torch Tensor
		"""
		features = torch.empty((high_dim_waypt.shape[0], 0), requires_grad=True)
		for feat_func in self.feature_func_list:
			features = torch.cat((features, feat_func(high_dim_waypt).unsqueeze(1)), dim=1)

		return features

	def table_features(self, high_dim_waypt):
		"""
		Input: 97D raw torch Tensor
		Output: torch scalar
		"""
		return high_dim_waypt[:, 90] / self.feat_range[self.feature_list.index("table")]

	def coffee_features(self, high_dim_waypt):
		"""
		Computes the coffee orientation feature value for waypoint
		by checking if the EE is oriented vertically.
		---
		input waypoint, output scalar feature
		"""
		featval = 1 - high_dim_waypt[:, 67]

		return featval / self.feat_range[self.feature_list.index("coffee")]

	def laptop_features(self, high_dim_waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		EE_coord_xy = high_dim_waypt[:, 88:90]
		laptop_xy = torch.Tensor(self.object_centers['LAPTOP_CENTER'][0:2])
		dist = torch.norm(EE_coord_xy - laptop_xy, dim=1) - 0.3

		return -((dist < 0) * dist) / self.feat_range[self.feature_list.index('laptop')]

	def human_features(self, high_dim_waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		EE_coord_xy = high_dim_waypt[:, 88:90]
		human_xy = torch.Tensor(self.object_centers['HUMAN_CENTER'][0:2])
		dist = torch.norm(EE_coord_xy - human_xy, dim=1) - 0.3
		return -((dist < 0) * dist) / self.feat_range[self.feature_list.index('human')]
