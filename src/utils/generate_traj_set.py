import itertools
import math
import numpy as np
import pickle
import sys

from utils.environment import Environment
from planners.trajopt_planner import TrajoptPlanner

def generate_traj_set(feat_list):
	# Before calling this function, you need to decide what features you care
    # about, from a choice of table, coffee, human, origin, and laptop.
	pick = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	place = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)
    goal_pose = None
	T = 20.0
	timestep = 0.5
    WEIGHTS_DICT = {"table": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 7.0, 8.0], 
                    "coffee": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    "laptop": [0.0, 20.0, 21.0, 22.0, 24.0, 26.0, 30.0, 40.0], 
                    "human": [0.0, 20.0, 21.0, 22.0, 24.0, 26.0, 30.0, 40.0],
                    "efficiency": [1.0]}
	    
	# Openrave parameters for the environment.
	model_filename = "jaco_dynamics"
	object_centers = {'HUMAN_CENTER': [-0.6,-0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.1,0.0]}
	environment = Environment(model_filename, object_centers)

	# Planner Setup
	max_iter = 50
	num_waypts = 5
	feat_list = [x.strip() for x in feat_list.split(',')]
	num_features = len(feat_list)
	planner = TrajoptPlanner(feat_list, max_iter, num_waypts, environment)

    # Construct set of weights of interest.
	weights_span = [None]*num_features
	for feat in range(0,num_features):
        weights_span[feat] = WEIGHTS_DIST[feat_list[feat]]
	weight_pairs = list(itertools.product(*weights_span))
	weight_pairs = [list(i) for i in weight_pairs]

    traj_rand = {}
	for (w_i, weights) in enumerate(weight_pairs):
        traj = self.planner.replan(start, goal, goal_pose, weights, T, timestep)
		Phi = environment.featurize(traj.waypts)
		# Getting rid of bad, out-of-bounds trajectories
		if any(phi < 0.0 for phi in Phi):
			continue
		traj = traj.waypts.tolist()
		if repr(traj) not in traj_rand:
			traj_rand[repr(traj)] = weights

	savefile = "traj_rand_close.p"
	pickle.dump(traj_rand, open( savefile, "wb" ))
	print "Saved in: ", savefile
	print "Used the following weight-combos: ", weight_pairs

if __name__ == '__main__':
	feat_list = sys.argv[1]
	generate_traj_set(feat_list)



