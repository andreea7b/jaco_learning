import numpy as np
import trajoptpy
import or_trajopt
import openravepy
from openravepy import *
import sys, select, os
import matplotlib.pyplot as plt

import openrave_utils
from openrave_utils import *

if __name__ == '__main__':
	if len(sys.argv) < 1:
		print "ERROR: Not enough arguments. Specify pathfile"
	else:
		beta_path = sys.argv[1]

	here = os.path.dirname(os.path.realpath(__file__))
	betas = pickle.load( open( here + beta_path, "rb" ) )

	plt.plot(betas)
	plt.show()

