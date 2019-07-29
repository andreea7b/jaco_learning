"""
This node demonstrates velocity-based PID control by moving the Jaco to track a given trajectory.

The initial trajectory also serves as an indicator of the start and goal
configurations. The controller first goes to the start of the trajectory, after
which it tracks it until a) it reaches the end of the trajectory (the goal) OR 
b) another trajectory is passed in.

If a new trajectory is demanded, the controller will track it starting
from the interpolated waypoint at the current time.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import sys, select, os
import time

from utils import pid, ros_utils

import kinova_msgs.msg
from kinova_msgs.srv import *

import numpy as np

# Jaco software name
prefix = 'j2s7s300_driver'

# Coordinates of 7DoF positions
pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_tilted = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 225.0]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_pose = [-0.46513, 0.29041, 0.69497] # x, y, z for pick_lower_EEtilt

class PIDController(object):
	"""
	This class represents a node that moves the Jaco with PID control.
	The joint velocities are computed as:

		V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)
	where:
		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change

	Subscribes to:
		/j2s7s300_driver/out/joint_angles	- Jaco sensed joint angles

	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco commanded joint velocities

	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		sim_flag                  - flag for if in simulation or not
	"""

	def __init__(self, trajectory):
		
		# Load parameters
		self.load_parameters(trajectory)

		# ---- ROS Setup ---- #

		rospy.init_node("pid_controller")
		self.register_callbacks()

		# publish to ROS at 100hz
		r = rospy.Rate(100)

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"

		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()

		print "----------------------------------"

	def load_parameters(self, task, trajectory):
		"""
		Loading parameters and setting up variables.
        Parameters:
            task [string] -- Task name that determines start and goal positions.
            feat_list [list] -- List of features the robot is considering. 
                                Features can be: 'table', 'coffee', 'human', 'laptop', 'efficiency'.
            feat_weight [list] -- Initial weight vector. Must have the same size as feat_list.
		"""

        # Set task, if any.
        self.task = None if task == "None" else task

		# ---- Trajectory Setup ---- #
        self.traj = trajectory

		# Set total time for trajectory

		# Initialize start/goal based on task
        pick = pick_tilted if self.task == "tilted" else pick_basic
        place = place_lower # By default. If desired, can implement other goals.

		self.start = np.array(pick)*(math.pi/180.0)
		self.goal = np.array(place)*(math.pi/180.0)
		self.curr_pos = None

        # Initialize planner and compute trajectory to track.
		self.planner = trajopt_planner.Planner(self.feat_list, self.task)
		self.traj = self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5, seed=None)
		print "original traj: " + str(self.traj)

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		# keeps running time since beginning of path
		self.path_start_T = None

		# ----- Controller Setup ----- #

		# stores maximum COMMANDED joint torques
		max_cmd_torque = 40.0		
		self.max_cmd = max_cmd_torque*np.eye(7)
		# stores current COMMANDED joint torques
		self.cmd = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# P, I, D gains 
		p_gain = 50.0
		i_gain = 0.0
		d_gain = 20.0
		self.P = p_gain*np.eye(7)
		self.I = i_gain*np.eye(7)
		self.D = d_gain*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# create joint-velocity publisher
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
	
	def start_admittance_mode(self):
		"""
		Switches Jaco to admittance-control mode using ROS services.
		"""
		service_address = prefix+'/in/start_force_control'
		rospy.wait_for_service(service_address)
		try:
			startForceControl = rospy.ServiceProxy(service_address, Start)
			startForceControl()
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None

	def stop_admittance_mode(self):
		"""
		Switches Jaco to position-control mode using ROS services.
		"""
		service_address = prefix+'/in/stop_force_control'
		rospy.wait_for_service(service_address)
		try:
			stopForceControl = rospy.ServiceProxy(service_address, Stop)
			stopForceControl()
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		return -self.controller.update_PID(error)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""

		# read the current joint angles from the robot
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		self.curr_pos = self.curr_pos*(math.pi/180.0)

		# update the OpenRAVE simulation 
		#self.planner.update_curr_pos(curr_pos)

		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(self.curr_pos)

		# update the experiment utils executed trajectory tracker
		if self.reached_start and not self.reached_goal:
			# update the experimental data with new position
			timestamp = time.time() - self.path_start_T
			self.expUtil.update_tracked_traj(timestamp, self.curr_pos)

		# update cmd from PID based on current position
		self.cmd = self.PID_control(self.curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:
			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)
			dist_from_start = np.fabs(dist_from_start)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()

				# set start time and the original weights as experimental data
				self.expUtil.set_startT(self.path_start_T)
				timestamp = time.time() - self.path_start_T
			else:
				#print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:			
			t = time.time() - self.path_start_T

			# get next target position from position along trajectory

			self.target_pos = self.planner.interpolate(t + 0.1)

			# check if the arm reached the goal, and restart path
			if not self.reached_goal:
				#print "REACHED START --> EXECUTING PATH"
				dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)
				dist_from_goal = np.fabs(dist_from_goal)

				# check if every joint is close enough to goal configuration
				close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]

				# if all joints are close enough, robot is at goal
				is_at_goal = all(close_to_goal)

				if is_at_goal:
					self.reached_goal = True
			else:
				#print "REACHED GOAL! Holding position at goal."
				self.target_pos = self.goal_pos

                # TODO: this should only set it once!
				self.expUtil.set_endT(time.time())

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print "ERROR: Not enough arguments. Specify task, feat_list, feat_weight."
	else:
		task = sys.argv[1]
		feat_list = [x.strip() for x in sys.argv[2].split(',')]
		feat_weight = [float(x.strip()) for x in sys.argv[3].split(',')]
	PIDControl(task,feat_list,feat_weight)
