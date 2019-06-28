import os
import pickle

# ----- Data parsing capabilities ---- #

def parse_exp_data(dataType):
	"""
	Creates a dictionary of all the data from the experiments 
	for the particular dataType

	Returns:
		ID 0 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		ID 1 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		...
		ID N --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
	-----
	dataType - "force", "tracked", "weights", "replanned"
	"""
	
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/study/"
	datapath = here + subdir + dataType

	validTypes = ["force", "weights", "replanned", "tracked"]
	if dataType not in validTypes:
		print dataType + " is not a valid data type!"
		return None

	data = {}

	for filename in os.listdir(datapath):
		# for each participant's data file, parse the participant's info
		info = filename.split(dataType)[1]
		if len(info) > 6:
			ID = int(info[0:2])
			task = int(info[2])
			method = info[3]
			trial = int(info[4])
		else:
			ID = int(info[0])
			task = int(info[1])
			method = info[2]
			trial = int(info[3])

		print "ID: " + str(ID) + ", task: " + str(task) + ", method: " + str(method) + ", trial: " + str(trial)

		# add participant info to dictionary
		if ID not in data:
			data[ID] = {}
		if task not in data[ID]:
			data[ID][task] = {}
		if trial not in data[ID][task]:
			data[ID][task][trial] = {}

		if dataType is "force":
			force = parse_force(filename)			
			data[ID][task][trial][method] = force
		elif dataType is "weights":
			weights = parse_weights(filename)		
			data[ID][task][trial][method] = weights
		elif dataType is "tracked":
			traj = parse_tracked_traj(filename)
			data[ID][task][trial][method] = traj
		elif dataType is "replanned":
			trajList = parse_replanned_trajList(filename)
			data[ID][task][trial][method] = trajList

	return data

# ----- De-pickling utilities ------- #

def parse_replanned_trajList(filename):
	"""
	Returns dictionary of trajectories, with timestamps as keys
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/replanned/"
	filepath = here + subdir + filename

	trajList = pickle.load( open( filepath, "rb" ) )
	return trajList

def parse_tracked_traj(filename):
	"""
	Returns trajectory
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/tracked/"
	filepath = here + subdir + filename

	traj = pickle.load( open( filepath, "rb" ) )
	return traj

def parse_deformed_traj(filename):
	"""
	Returns trajectory
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/deformed/"
	filepath = here + subdir + filename

	traj = pickle.load( open( filepath, "rb" ) )

	return traj

def parse_weights(filename):
	"""
	Returns tuple: (timestamp list, weight list)
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/weights/"
	filepath = here + subdir + filename

	weights = pickle.load( open( filepath, "rb" ) )
	return weights

def parse_force(filename):
	"""
	Returns tuple (timestamp list, force list)
	"""
	# get the current script path
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/force/"
	filepath = here + subdir + filename

	force = pickle.load( open( filepath, "rb" ) )
	return force

