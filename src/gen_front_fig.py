import openravepy
from openravepy import *
from catkin.find_in_workspaces import find_in_workspaces
import numpy as np
import math
import pickle
import time
import trajopt_planner

import openrave_utils
from openrave_utils import *

ALL = "ALL"						# updates all features
MAX = "MAX"						# updates only feature that changed the most
BETA = "BETA"					# updates beta-adaptive

pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 225.0]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

path = "/home/anca/catkin_ws/src/beta_adaptive_pHRI/src/data/experimental/"

def gen_default():
	# start and end for ROBOT_TASK = Table
	pick = pick_basic_EEtilt
	place = place_lower
	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	feat_list = ["table"]
	feat_method = ALL
	task = "table"
	T = 20.0

	weights = [0.0]

	# create the trajopt planner and plan from start to goal
	planner = trajopt_planner.Planner(feat_method, feat_list, task, None)
	
	# stores the current trajectory we are tracking, produced by planner
	traj = planner.replan(start, goal, weights, 0.0, T, 0.5, seed=None)

	# save out the upsampled waypoints (for better plotting)
	pickle.dump(planner.waypts, open(path+"default_traj.p", "wb"))

def gen_corl():
	# start and end for ROBOT_TASK = Table
	pick = pick_basic_EEtilt
	place = place_lower
	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	feat_list = ["table"]
	feat_method = ALL
	task = "table"
	T = 20.0

	weights = [0.85485018] # final table weight from weights_2_B_ALL_table_correction_human_1.p

	# create the trajopt planner and plan from start to goal
	planner = trajopt_planner.Planner(feat_method, feat_list, task, None)

	# stores the current trajectory we are tracking, produced by planner
	traj = planner.replan(start, goal, weights, 0.0, T, 0.5, seed=None)

	# save out the upsampled waypoints (for better plotting)
	pickle.dump(planner.waypts, open(path+"corl_traj.p", "wb"))

def gen_beta():
	# start and end for ROBOT_TASK = Table
	pick = pick_basic_EEtilt
	place = place_lower
	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	feat_list = ["table"]
	feat_method = BETA
	task = "table"
	T = 20.0

	weights = [0.0070389] # final table weight from weights_2_B_BETA_table_correction_human_1.p

	# create the trajopt planner and plan from start to goal
	planner = trajopt_planner.Planner(feat_method, feat_list, task, None)

	# stores the current trajectory we are tracking, produced by planner
	traj = planner.replan(start, goal, weights, 0.0, T, 0.5, seed=None)

	# save out the upsampled waypoints (for better plotting)
	pickle.dump(planner.waypts, open(path+"beta_traj.p", "wb"))

def gen_front_fig():
	beta_path = path + "beta_traj.p" #path + "tracked/tracked_2_B_BETA_table_correction_human_1.p"
	default_path = path + "tracked/tracked_2_B_ALL_table_correction_human_1.p"
#"default_traj.p"
	corl_path = path + "corl_traj.p" #path + "tracked/tracked_2_B_ALL_table_correction_human_1.p"

	beta_traj = pickle.load(open(beta_path, "rb"))
	corl_traj = pickle.load(open(corl_path, "rb"))
	default_traj = pickle.load(open(default_path, "rb"))
	
	env, robot = openrave_utils.initialize(model_filename='jaco_dynamics')
	bodies = []

	orange = [0.96862745, 0.58823529, 0.2745098]
	blue = [0. , 0.50196078, 0.50196078]
	grey = [0.8, 0.8, 0.8]
	plotTable(env)
	plotTableMount(env, bodies)
	plotTraj(env,robot,bodies,beta_traj, size=10, color=orange)
	#plotTraj(env,robot,bodies,default_traj[:,1:], size=10, color=orange)
	plotTraj(env,robot,bodies,corl_traj, size=10, color=grey)

	time.sleep(35)

if __name__ == '__main__':
	#gen_default()
	#gen_beta()
	#gen_corl()
	gen_front_fig()