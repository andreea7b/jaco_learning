from trajopt_planner import *
import pickle

if __name__ == '__main__':

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0]
	pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

	place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
	place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

	T = 20.0

	feat_method = "ALL"
	feat_list = "coffee"
	feat_list = [x.strip() for x in feat_list.split(',')]
	num_features = len(feat_list)
	planner = Planner(feat_method, feat_list)
	MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':10.0, 'human':10.0}

	# initialize start/goal based on features
	# by default for table and laptop, these are the pick and place
	pick = pick_basic
	place = place_lower
	if 'human' in feat_list:
		pick = pick_shelf
		place = place_higher
	if 'coffee' in feat_list:
		pick = pick_basic_EEtilt

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights_span = [None]*num_features
	for feat in range(0,num_features):
		limit = MAX_WEIGHTS[feat_list[feat]]
		weights_span[feat] = list(np.arange(-limit, limit+.1, limit/2))

	weight_pairs = list(itertools.product(*weights_span))
	num_trajs = len(weight_pairs)
	traj_cache = [0] * num_trajs
	
	for (w_i, w) in enumerate(weight_pairs):
		traj = planner.replan(start, goal, w, 0.0, T, 0.5)	
		traj_cache[w_i] = traj

	traj_cache = np.array(traj_cache)
	print traj_cache

	savestr = "_".join(feat_list)
	savefile = "traj_cache_"+savestr+".p"
	pickle.dump(traj_cache, open( savefile, "wb" ) )
	print "Saved in: ", savefile
