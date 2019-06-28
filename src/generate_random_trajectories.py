import itertools
import math
import numpy as np
import pickle
import sys

from planners.trajopt_planner import Planner

def generate_rand_trajs(feat_list):
	# Before calling this function, you need to decide what features you care
    # about, from a choice of table, coffee, human, origin, and laptop

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0]
	pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

	place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]
	place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

	MIN_WEIGHTS = {'table':0.0, 'coffee':0.0, 'laptop':0.0, 'human':0.0, 'efficiency':1.0}
	MAX_WEIGHTS = {'table':20.0, 'coffee':1.0, 'laptop':18.0, 'human':14.0, 'efficiency':1.0}
	NUM_WEIGHTS = {'table':6, 'coffee':0, 'laptop':10, 'human':8, 'efficiency':1}

	T = 20.0

	traj_rand = {}
	feat_list = [x.strip() for x in feat_list.split(',')]
	planner = Planner(feat_list)
	num_features = len(feat_list)

	# initialize start/goal based on features
	# by default for table and laptop, these are the pick and place
	pick = pick_basic
	place = place_lower

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights_span = [None]*num_features
	for feat in range(0,num_features):
		if feat_list[feat] == "table":
			weights_span[feat] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 7.0, 8.0]
		elif feat_list[feat] == "laptop":
			weights_span[feat] = [0.0, 20.0, 21.0, 22.0, 24.0, 26.0, 30.0, 40.0]
		else:
			weights_span[feat] = list(np.linspace(MIN_WEIGHTS[feat_list[feat]], MAX_WEIGHTS[feat_list[feat]], num=NUM_WEIGHTS[feat_list[feat]]))

	weight_pairs = list(itertools.product(*weights_span)) 		 # Create all weight products
	weight_pairs = [list(i) for i in weight_pairs]

	for (w_i, weights) in enumerate(weight_pairs):
		planner.replan(start, goal, weights, 0.0, T, 0.5)
		Phi = planner.featurize(planner.waypts)
		# Getting rid of bad, out-of-bounds trajectories
		if sum(Phi[1]) < 0.0:
			continue
		traj = planner.waypts.tolist()
		if repr(traj) not in traj_rand:
			traj_rand[repr(traj)] = weights

	savefile = "traj_rand_close.p"
	pickle.dump(traj_rand, open( savefile, "wb" ))
	print "Saved in: ", savefile
	print "Used the following weight-combos: ", weight_pairs

if __name__ == '__main__':
	feat_list = sys.argv[1]
	generate_rand_trajs(feat_list)



