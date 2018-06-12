import openravepy
from openravepy import *
from prpy.bind import bind_subclass
from archierobot import ArchieRobot
from catkin.find_in_workspaces import find_in_workspaces
import numpy as np
import logging
import math

# Silence the planning logger to prevent spam.
logging.getLogger('prpy.planning.base').addHandler(logging.NullHandler())

robot_starting_dofs = np.array([-1, 2, 0, 2, 0, 4, 0, 1.11022302e-16,  -1.11022302e-16, 3.33066907e-16])

def initialize(model_filename='jaco', envXML=None):
	'''
	Load and configure the JACO robot. If envXML is not None, loads environment.
	Returns robot and environment.
	-----
	NOTE: 
	IF YOU JUST WANT TO DO COMPUTATIONS THROUGH OPENRAVE
	AND WANT MULTPILE INSTANCES TO OPEN, THEN HAVE TO TURN OFF
	QTCOIN AND ALL VIEWER FUNCTIONALITY OR IT WILL CRASH. 
	'''
	env = openravepy.Environment()
	if envXML is not None:
		env.LoadURI(envXML)
	#env.SetViewer('qtcoin')

	# Assumes the robot files are located in the data folder of the
	# kinova_description package in the catkin workspace.
	urdf_uri = 'package://iact_control/src/data/'+model_filename+'.urdf'
	srdf_uri = 'package://iact_control/src/data/'+model_filename+'.srdf'
	or_urdf = openravepy.RaveCreateModule(env, 'urdf')
	robot_name = or_urdf.SendCommand('load {:s} {:s}'.format(urdf_uri, srdf_uri))
	robot = env.GetRobot(robot_name)
	bind_subclass(robot, ArchieRobot)

	robot.SetActiveDOFs(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
	robot.SetDOFValues(robot_starting_dofs)
	
	# = env.GetViewer()
	#viewer.SetSize(700,500)
	#cam_params = np.array([[-0.99885711, -0.01248719, -0.0461361 , -0.18887213],
	#	   [ 0.02495645,  0.68697757, -0.72624996,  2.04733515],
	#	   [ 0.04076329, -0.72657133, -0.68588079,  1.67818344],
	#	   [ 0.        ,  0.        ,  0.        ,  1.        ]])
	#viewer.SetCamera(cam_params)
	
	return env, robot

# ------- Plotting & Conversion Utils ------- #

def robotToCartesian(robot):
	"""
	Converts robot configuration into a list of cartesian 
	(x,y,z) coordinates for each of the robot's links.
	------
	Returns: 7-dimensional list of 3 xyz values
	"""
	links = robot.GetLinks()
	cartesian = [None]*7
	i = 0
	for i in range(1,8):
		link = links[i] 
		tf = link.GetTransform()
		cartesian[i-1] = tf[0:3,3]

	return cartesian

def manipToCartesian(robot, offset_z):
	"""
	Gets center of robot's manipulator in cartesian space
	------
	Params: robot object
			offset in m from the base of the manip to the center
	Returns: xyz of center of robot manipulator
	"""
	links = robot.GetLinks()
	manipTf = links[7].GetTransform() 
	rot = manipTf[0:3,0:3]
	xyz = manipTf[0:3,3]
	offset = np.array([0,0,offset_z]).T
	return xyz

def plotCupTraj(env,robot,bodies,waypts,color=[0,1,0], increment=1):
	"""
	Plots trajectory of the cup
	"""

	for i in range(0,len(waypts),increment):
		waypoint = waypts[i]
		print "waypt: " +str(waypoint)
		dof = np.append(waypoint, np.array([1, 1, 1]))
		dof[2] += math.pi
		robot.SetDOFValues(dof)

		links = robot.GetLinks()
		manipTf = links[9].GetTransform() 

		# load mug into environment
		objects_path = find_in_workspaces(
				project='iact_control',
				path='src/data',
				first_match_only=True)[0]
		env.Load('{:s}/mug1.dae'.format(objects_path))
		mug = env.GetKinBody('mug')
		mug.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array(color))
		angle = -np.pi/2

		rot_x = np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]]) 

		rot_y = np.array([[np.cos(angle),0,np.sin(angle),0],[0,1,0,0],[-np.sin(angle),0,np.cos(angle),0],[0,0,0,1]]) 

		rot_z = np.array([[np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),0,np.cos(angle),0],[0,0,1,0],[0,0,0,1]]) 

		trans = np.array([[0,0,0,-0.02],[0,0,0,0.02],[0,0,0,-0.02],[0,0,0,1]]) 

		rotated = np.dot(manipTf+trans,rot_x)
		mug.SetTransform(rotated)

		body = mug
		body.SetName("pt"+str(len(bodies)))
		env.Add(body, True)
		bodies.append(body)

def plotMug(env, bodies, transform, color=[1,0,0]):
	"""
	Plots mug at specific transform
	"""
	objects_path = find_in_workspaces(
				project='iact_control',
				path='src/data',
				first_match_only=True)[0]
	env.Load('{:s}/mug1.dae'.format(objects_path))
	mug = env.GetKinBody('mug')
	mug.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array(color))
	mug.SetTransform(transform)
	body = mug
	body.SetName("pt"+str(len(bodies)))
	env.Add(body, True)
	bodies.append(body)
	return mug

def plotTraj(env,robot,bodies,waypts, size=10, color=[0, 1, 0]):
	"""
	Plots the best trajectory found or planned
	"""
	for i in range(0,len(waypts),1):
		waypoint = waypts[i]
		dof = np.append(waypoint, np.array([1, 1, 1]))
		dof[2] += math.pi
		robot.SetDOFValues(dof)
		coord = robotToCartesian(robot)
		# sz=0.015
		# 0.009
		plotSphere(env, bodies, coord[6], size, color)

def plotSphere(env, bodies, coords, size=10, color=[0, 0, 1]):
	"""
	Plots a single sphere in OpenRAVE center at coords(x,y,z) location
	"""
	bodies.append(env.plot3(points=np.array((coords[0],coords[1],coords[2])), pointsize=size, colors=np.array(((color[0],color[1],color[2])))))

def plotTable(env):
	"""
	Plots the robot table in OpenRAVE.
	"""
	# load table into environment
	objects_path = find_in_workspaces(
			project='iact_control',
			path='src/data',
			first_match_only=True)[0]
	env.Load('{:s}/table.xml'.format(objects_path))
	table = env.GetKinBody('table')
	table.SetTransform(np.array([[0.0, 1.0,  0.0, -0.8128/2], #should be negative 1?
  								 [1.0, 0.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1143], #-0.7874
			                     [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.9, 0.75, 0.75])
	table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

def plotCabinet(env):
	"""
	Plots the cabinet in OpenRAVE.
	"""
	# load table into environment
	objects_path = find_in_workspaces(
			project='iact_control',
			path='src/data',
			first_match_only=True)[0]
	env.Load('{:s}/cabinet.xml'.format(objects_path))
	cabinet = env.GetKinBody('cabinet')
	cabinet.SetTransform(np.array([[0.0, -1.0,  0.0, 0.6],
  								 [1.0, 0.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, 0], 
			                     [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.05,0.6,0.3])
	cabinet.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

"""
def plotMug(env):
	# load table into environment
	objects_path = find_in_workspaces(
			project='iact_control',
			path='src/data',
			first_match_only=True)[0]
	env.Load('{:s}/mug.xml'.format(objects_path))
"""

def plotMan(env):
	"""
	Plots a human in OpenRAVE.
	"""
	# load table into environment
	objects_path = find_in_workspaces(
			project='iact_control',
			path='src/data',
			first_match_only=True)[0]
	env.Load('{:s}/manifest.xml'.format(objects_path))


def plotTableMount(env,bodies):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
	# create robot base attachment
	body = RaveCreateKinBody(env, '')
	body.InitFromBoxes(np.array([[0,0,0, 0.3048/2,0.8128/2,0.1016/2]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
			                     [0.0, 1.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1016/2],
			                     [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("robot_mount")
	env.Add(body, True)

	color = np.array([0.9, 0.58, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

def plotLaptop(env,bodies,pos):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
	# create robot base attachment
	body = RaveCreateKinBody(env, '')
	#12 x 9 x 1 in, 0.3048 x 0.2286 x 0.0254 m
	# divide by 2: 0.1524 x 0.1143 x 0.0127
	#20 in from robot base
	body.InitFromBoxes(np.array([[0,0,0,0.1143,0.1524,0.0127]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, pos[0]],
			                     [0.0, 1.0,  0.0, pos[1]+0.1],
			                     [0.0, 0.0,  1.0, pos[2]-0.1016],
			                     [0.0, 0.0,  0.0, 1.0]]))
#	body.SetTransform(np.array([[1.0, 0.0,  0.0, (-1.3858/2 - 0.1)],
#			                     [0.0, 1.0,  0.0, 0],
#			                     [0.0, 0.0,  1.0, -0.1016],
#			                     [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("laptop")
	env.Add(body, True)
	color = np.array([0, 0, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

def executePathSim(env,robot,waypts):
	"""
	Executes in the planned trajectory in simulation
	"""
	traj = RaveCreateTrajectory(env,'')
	traj.Init(robot.GetActiveConfigurationSpecification())
	for i in range(len(waypts)):
		traj.Insert(i, waypts[i])
	robot.ExecutePath(traj)
	
