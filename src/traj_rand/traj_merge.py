import numpy as np
import ast
import glob

if __name__ == '__main__':
	first = np.load("traj_rand_close.p")
	second = np.load("traj_rand.p")
	third = np.load("traj_rand_small.p")
	merged = first.copy()
	merged.update(second)
	merged.update(third)
	"""
	for traji, traj1 in enumerate(merged.keys()):
		traj_1 = np.array(ast.literal_eval(traj1))
		for trajj, traj2 in enumerate(merged.keys()):
			traj_2 = np.array(ast.literal_eval(traj2))
			if trajj > traji and np.max(np.abs(traj_1-traj_2)) < 0.03:
				merged.pop(traj2)
		print "Done with trajecotry {}".format(traji)
	"""
	merged = np.load("traj_rand_merged.p")
	for demo_file in glob.glob('../data/demonstrations/demos/demo*'):
		traj_demo = np.array(np.load(demo_file))
		merged[repr(traj_demo.tolist())] = [100.0]
	import pdb;pdb.set_trace()
