import numpy as np
import plotly.express as px
import pandas as pd
import os
import math
from openrave_utils import robotToCartesian
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Plot stuff
def angles_to_coords(data, feat, env):
    """
    	Transforms an array of 7D Angle coordinates to xyz coordinates expect for coffee (to x vector of EE rotation vec.)
    """
    coords_list = np.empty((0, 3), float)
    for i in range(data.shape[0]):
        waypt = data[i]
        if len(waypt) < 10:
            waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
            waypt[2] += math.pi
        env.robot.SetDOFValues(waypt)
        if feat == "coffee":
            EE_link = env.robot.GetLinks()[7]
            coords_list = np.vstack((coords_list, EE_link.GetTransform()[:3,0]))
        else:
            coords = robotToCartesian(env.robot)
            coords_list = np.vstack((coords_list, coords[6]))
    return coords_list


def plot_gt3D_one_feat(parent_dir, feat, env):
    """
        Plot the ground truth 3D Half-Sphere for a specific feature.
    """
    data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
    npzfile = np.load(data_file)
    train = npzfile['x'][:,:7]
    labels = npzfile['y']
    labels = labels.reshape(len(labels), 1)
    euclidean = angles_to_coords(train, feat, env)
    df = pd.DataFrame(np.hstack((euclidean, labels)))
    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.show()


def plot_gt3D(parent_dir, env, title='GT Cost Value over 3D Reachable Set'):
	"""
		Plot the ground truth 3D Half-Sphere for the environment.
	"""
	raw_waypts, gt_cost = get_coords_gt_cost(env, parent_dir)
	fig = px.scatter_3d(x=raw_waypts[:,88], y=raw_waypts[:,89], z=raw_waypts[:,90], color=gt_cost)
	fig.update_layout(title=title)
	fig.show()


def get_coords_gt_cost(expert_env, parent_dir, gen=False, n_waypoints=10000):
	"""
	Function to get the gt cost over a set of random reachable points for the expert_env.
	---
	Input:
		expert_env	environment object used to generate gt cost
		parent_dir	needed if gen=False and we load a random set of reachable points
		gen			False -> loads random set of reachable points, if True: generates one
		n_waypoints	If gen=True, this is the size of the set generated
	"""
	# Step 1: Generate ground truth data, sampling uniformly from 7D angle space
	if gen == True:
		waypts = np.random.uniform(size=(n_waypoints, 7), low=0, high=np.pi*2)
		# Transform to 97D
		raw_waypts = []
		for waypt in waypts:
			raw_waypts.append(expert_env.raw_features(waypt))
		raw_waypts = np.array(raw_waypts)

	else:
		# load coordinates above the table
		data_file = parent_dir + '/data/gtdata/data_table.npz'
		npzfile = np.load(data_file)
		raw_waypts = npzfile['x']

	# generate gt_labels
	feat_idx = list(np.arange(expert_env.num_features))
	features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(expert_env.feature_list))]
	for index in range(len(raw_waypts)):
		for feat in range(len(feat_idx)):
			features[feat][index] = expert_env.featurize_single(raw_waypts[index,:7], feat_idx[feat])
	features = np.array(features).T
	gt_cost = np.matmul(features, np.array(expert_env.weights).reshape(-1,1))

	return raw_waypts, gt_cost


def plot_learned_traj(feature_function, train_data, env, feat='table'):
	"""
		Plot the traces labled with the function values of feature_function.
	"""
	output = feature_function(train_data)
	euclidean = angles_to_coords(train_data[:, :7], feat, env)
	fig = px.scatter_3d(x=euclidean[:,0], y=euclidean[:,1], z=euclidean[:,2], color=output)
	fig.update_layout(title='Traces with learned function values')
	fig.show()


def plot_learned3D(parent_dir, feature_function, env, viz_idx=range(88, 91), feat='table', title='Learned function over 3D Reachable Set'):
	"""
		Plot the learned 3D ball over the 10.000 points Test Set in the gt_data
	"""
	data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
	npzfile = np.load(data_file)
	train = npzfile['x'][:,:7]
	train_raw = np.empty((0, 97), float)
	for dp in train:
		train_raw = np.vstack((train_raw, env.raw_features(dp)))
	labels = feature_function(train_raw)
	euclidean = train_raw[:, viz_idx]
	#euclidean = angles_to_coords(train, feat, env)
	fig = px.scatter_3d(x=euclidean[:, 0], y=euclidean[:, 1], z=euclidean[:, 2], color=labels)
	fig.update_layout(title=title)
	fig.show()
