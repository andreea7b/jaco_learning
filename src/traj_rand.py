from discrete_trajopt_planner import *
import pickle
import numpy as np
import itertools

if __name__ == '__main__':
    # Before calling this function, you need to decide what features you care
    # about, from a choice of table, coffee, human, origin, and laptop

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0]
	pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

	place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
	place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

	T = 20.0

	feat_method = "ALL"
	feat_list = "human"
	feat_list = [x.strip() for x in feat_list.split(',')]
	num_features = len(feat_list)
	planner = DiscretePlanner(feat_method, feat_list)

	# initialize start/goal based on features
	# by default for table and laptop, these are the pick and place
	pick = pick_basic_EEtilt
	place = place_lower
	if 'coffee' in feat_list:
		pick = pick_basic_EEtilt

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights_pairs = planner.weights_dict
	num_trajs = len(weights_pairs)
	traj_rand = [0] * num_trajs

	for (w_i, weights) in enumerate(weights_pairs):
		planner.replan(start, goal, weights, 0.0, T, 0.5)
		traj = planner.waypts
 		traj_rand[w_i] = traj

	traj_rand = np.array(traj_rand)
	print traj_rand
	#print "-------"

	savestr = "_".join(feat_list)
	savefile = "traj_rand_"+savestr+".p"
	pickle.dump(traj_rand, open( savefile, "wb" ) )
