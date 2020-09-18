from maxent_irl_utils import *

class GT_Reward_Expert:
	"""
	This class contains an environment with the GT reward and is used to simulate near-optimal demonstrations.

	Initi:
	gt_weights		List of weights for the feat_list features
	feat_list		List of strings with the active features
	gen				The mode with which induced near optimal trajectories with the GT cost are calculated
					can be 'waypt' (get TrajOpt optimal trajectory and perturb waypoints)
					or 'cost' (perturb the cost function slightly and get TrajOpt optimal trajectories for all of them)

	starts			list of start positions
	goals			list of goal positions
	goal_poses		List of goal poses OR default None
	combi			If True: take all combinations of start & goal pairs
	T, timestep		Settings for TrajOpt

	obj_center_dict	Dict of human & laptop positions used when calculating the feature values
	feat_range_dict	Dict of factors for the different features to scale them to 0-1. -> used to calculate feature values

	"""

	def __init__(self, feat_list, gt_weights, gen, starts, goals, goal_poses=None, combi=False, T=20., timestep=0.5,
				 obj_center_dict = {'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]},
				 feat_range_dict = {'table': 0.98, 'coffee': 1.0, 'laptop': 0.3, 'human': 0.3, 'efficiency': 0.22, 'proxemics': 0.3, 'betweenobjects': 0.2}):

		# instantiate an environment & trajOpt planner
		env, planner = init_env(feat_list, gt_weights, object_centers=obj_center_dict, feat_range=feat_range_dict)
		self.env = env
		self.planner = planner
		self.s_g_exp_trajs = []
		self.gen = gen

		if goal_poses is not None and len(goals) != len(goal_poses):
			print("Goal pose needs to be either None or same length as len(goals)")
			assert False

		if combi:
			combis = [(x, y) for x in range(len(starts)) for y in range(len(goals))]
			self.starts = [starts[tup[0]] for tup in combis]
			self.goals = [goals[tup[1]] for tup in combis]
			if goal_poses is not None:
				self.goal_poses = [goal_poses[tup[1]] for tup in combis]
		else:
			self.starts = starts[:min(len(starts), len(goals))]
			self.goals = goals[:min(len(starts), len(goals))]
			if goal_poses is not None:
				self.goal_poses = goal_poses

		if goal_poses is None:
			self.goal_poses = [None for _ in range(len(self.starts))]

		self.T = T
		self.timestep = timestep

	def generate_expert_demos(self, n_per_s_g, std=0.01):
		"""
			Use trajopt and some perturbation method to generate near-optimal demonstrations under the GT reward
			----
			Input:
			n_per_s_g	how many demonstrations per start-goal pair
			std			standard deviation used to induce near-optimality
		"""
		for start, goal, goal_pose in zip(self.starts, self.goals, self.goal_poses):
			if self.gen == 'waypt':
				expert_demos = generate_Gaus_MaxEnt_trajs(self.planner, self.env, std,
														  n_per_s_g, start, goal, goal_pose, self.T, self.timestep)
			elif self.gen == 'cost':
				expert_demos = generate_cost_perturb_trajs(self.planner, self.env, std,
														   n_per_s_g, start, goal, goal_pose, self.T, self.timestep)
			# add for that s_g configuration
			self.s_g_exp_trajs.append(expert_demos)

	def generate_rand_start_goal(self, n_trajs, min_dist=0.7):
		"""
			Generates and adds a random set of start-goal pairs that are at least min_dist apart
			----
			Input:
			n_trajs		how many start-goal pairs
			min_dist	minimum distance the start-goal pairs should be apart
		"""
		trajs = []
		starts = []
		goals = []
		while len(trajs) < n_trajs:
			# sample
			start_sample = np.random.uniform(low=0, high=2 * math.pi, size=7)
			goal_sample = np.random.uniform(low=0, high=2 * math.pi, size=7)
			# plan
			opt_traj = self.planner.replan(start_sample, goal_sample, None, self.T, self.timestep, seed=None)
			# get raw and x,y,z of start and end of the trajectory
			raw = map_to_raw_dim(self.env, [opt_traj.waypts])
			distance = np.linalg.norm(raw[0][0][88:91] - raw[0][-1][88:91])
			if distance > min_dist:
				trajs.append(raw[0])
				starts.append(start_sample)
				goals.append(goal_sample)
		self.starts = self.starts + starts
		self.goals = self.goals + goals
		self.goal_poses = self.goal_poses + [None]*len(starts)

	def return_trajs(self):
		"""
			Returns the list of lists of expert demonstrations
		"""
		return self.s_g_exp_trajs

	def load_trajs(self, trajectories):
		"""
			Loads in a list of lists of expert demonstrations
		"""
		self.s_g_exp_trajs = trajectories

	def plot_trajs(self):
		"""
			Plot the current set of expert demonstrations in 3D space, color is the z coordinate.
		"""
		all_trajs = []
		for s_g_demos in self.s_g_exp_trajs:
			high_dim_demos = []
			for angle_traj in s_g_demos:
				high_dim_demos.append(map_to_raw_dim(self.env, [angle_traj])[0])
			all_trajs = all_trajs + high_dim_demos
		plot_trajs(all_trajs, object_centers=self.env.object_centers, title='Expert Trajectories')
