import numpy as np
import trajoptpy
import or_trajopt
import openravepy
from openravepy import *
import sys, select, os

import openrave_utils
from openrave_utils import *

if __name__ == '__main__':
	if len(sys.argv) < 1:
		print "ERROR: Not enough arguments. Specify trajectory pathfile"
	elif len(sys.argv) > 2:
		iact_pts = sys.argv[2]
		here = os.path.dirname(os.path.realpath(__file__))
		iact_pts = pickle.load( open( here + iact_pts, "rb" ) )
	traj_path = sys.argv[1]
    
    # initialize robot and empty environment
	model_filename = 'jaco_dynamics'
	env, robot = initialize(model_filename)

	# insert any objects you want into environment
	bodies = []

	# plot the table and table mount
	plotTable(env)
	plotTableMount(env,bodies)

	here = os.path.dirname(os.path.realpath(__file__))
	trajs = pickle.load( open( here + "/" + traj_path, "rb" ) )
	if isinstance(trajs,dict):
		for (timestep, waypts_plan) in trajs.items():
			print "timestep: ", timestep
			plotTraj(env,robot,bodies,waypts_plan, size=8,color=[0, 0, 1])
			raw_input("Press Enter to continue...")
			bodies = []
	elif trajs.ndim == 3:
		for waypts_plan in trajs:
			plotTraj(env,robot,bodies,waypts_plan, size=8,color=[0, 0, 1])
			raw_input("Press Enter to continue...")
			bodies = []
	else:
		plotTraj(env,robot,bodies,trajs[:,1:], size=8,color=[0, 0, 1])
		if 'iact_pts' in locals():
			plotTraj(env,robot,bodies,iact_pts[:,1:], size=10,color=[1, 0, 0])
		raw_input("Press Enter to continue...")
		

