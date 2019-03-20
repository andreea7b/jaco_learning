import itertools
import math
import numpy as np
import pickle
import sys

from trajopt_planner import Planner


def generate_rand_trajs(features):
	# Before calling this function, you need to decide what features you care
    # about, from a choice of table, coffee, human, origin, and laptop

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0]
	pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

	place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
	place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

	MIN_WEIGHTS = {'table':-1.0, 'coffee':0.0, 'laptop':0.0, 'human':0.0}
	MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':8.0, 'human':10.0}

	T = 20.0

	feat_method = "ALL"
	traj_rand = []
	all_feats = features
	feat_list = all_feats

	
	planner = Planner(feat_list)
	num_features = len(feat_list)

	# initialize start/goal based on features
	# by default for table and laptop, these are the pick and place
	pick = pick_basic_EEtilt
	place = place_lower

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights_span = [None]*num_features
	for feat in range(0,num_features):
		weights_span[feat] = list(np.linspace(MIN_WEIGHTS[feat_list[feat]], MAX_WEIGHTS[feat_list[feat]], num=5))

	weight_pairs = list(itertools.product(*weights_span))
	weight_pairs = [np.array(i) for i in weight_pairs]
	import pdb;pdb.set_trace()
	for (w_i, weights) in enumerate(weight_pairs):
		print(feat_list)
		print(weights)
		planner.replan(start, goal, list(weights), 0.0, T, 0.5)
		traj = planner.waypts
		traj_rand.append(traj)


	print(traj_rand)
	traj_rand = np.array(traj_rand)
	savestr = "_".join(all_feats) + "_"
	savefile = "lala_" + savestr + ".p"
	pickle.dump(traj_rand, open( savefile, "wb" ) )

	#return traj_rand


if __name__ == '__main__':
	features = sys.argv[1:]
	generate_rand_trajs(features)



