from trajopt_planner import *
import pickle

if __name__ == '__main__':

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0]

	# initialize start/goal based on task 
	pick = pick_basic # pick_basic_EEtilt  
	place = place_lower #place_lower 

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	T = 20.0

	featMethod = "ALL"
	numFeat = 1
	planner = Planner(2, False, featMethod, numFeat)

	traj_cache = {}

	cup_weights = np.arange(-1.0, 1.1, 0.5)
	table_weights = np.arange(-1.0, 1.1, 0.5)
	#laptop_weights = np.arange(-10.0, 11.0, 5.0)

	for cup in cup_weights:
		for table in table_weights:
			weights = [cup, table]
			traj = planner.replan(start, goal, weights, 0.0, T, 0.5)	

			if cup not in traj_cache:
				traj_cache[cup] = {}
			if table not in traj_cache[cup]:
				traj_cache[cup][table] = traj

			#if laptop not in traj_cache[cup][table]:
			#	traj_cache[cup][table][laptop] = traj


	print traj_cache
	#print "-------"

	pickle.dump(traj_cache, open( "traj_cache_1feat.p", "wb" ) )
	#pickle.dump(traj_cache, open( "traj_cache_2feat.p", "wb" ) )
