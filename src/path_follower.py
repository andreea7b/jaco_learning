#! /usr/bin/env python
"""
This node moves the Jaco using a specified controller, tracking a trajectory
given by a specified planner.

Given a start, a goal, and specific planner and controller parameters, the
planner plans a path from start to goal, while the controller moves the Jaco
manipulator along the path.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""

import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import sys, select, os
import time

from utils import ros_utils
from controllers import pid_controller
from planners import trajopt_planner

import kinova_msgs.msg
from kinova_msgs.srv import *

import numpy as np

class PathFollower(object):
	"""
	This class represents a node that computes an optimal path and moves the Jaco along.

	Subscribes to:
		/$prefix$/out/joint_angles	- Jaco sensed joint angles

	Publishes to:
		/$prefix$/in/joint_velocity	- Jaco commanded joint velocities
	"""

	def __init__(self):
		
		# Create ROS node.
        rospy.init_node("path_follower")

        # Load parameters and set up subscribers/publishers.
		self.load_parameters()
		self.register_callbacks()

		# Rublish to ROS at 100hz.
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

	def load_parameters(self):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""

		# ----- General Setup ----- #
        self.prefix = rospy.get_param("general/prefix")
        self.planner_type = rospy.get_param("general/planner_type")
        self.controller_type = rospy.get_param("general/controller_type")

        pick = rospy.get_param("general/start")
        place = rospy.get_param("general/goal")
		self.start = np.array(pick)*(math.pi/180.0)
		self.goal = np.array(place)*(math.pi/180.0)
        self.T = rospy.get_param("general/T")
        self.timestep = rospy.get_param("general/timestep")
        
        # Openrave parameters for the environment.
        model_filename = rospy.get_param("planner/openrave/model_filename")
        object_centers = rospy.get_param("planner/openrave/object_centers")
        self.environment = Environment(self.feat_list, model_filename, object_centers)
        
		# ----- Planner Setup ----- #
        # Retrieve the planner specific parameters.
        if self.planner_type == "trajopt":
            max_iter = rospy.get_param("planner/max_iter")
            num_waypts = rospy.get_param("planner/num_waypts")
            self.feat_list = rospy.get_param("planner/feat_list")
            self.feat_weights = rospy.get_param("planner/feat_weights")
            
            # Initialize planner and compute trajectory to track.
		    self.planner = TrajoptPlanner(self.feat_list, max_iter, num_waypts, environment)
        else:
            raise Exception('Planner {} not implemented.'.format(self.planner_type))
		
        self.traj = self.planner.replan(self.start, self.goal, self.weights,
                                        self.T, self.timestep seed=None)
        
        # Track if you have reached the start/goal of the path.
		self.reached_start = False
		self.reached_goal = False
		
        # Save the intermediate target configuration. 
		self.current_pos = None

		# ----- Controller Setup ----- #
        # Retrieve controller specific parameters.
        if self.controller_type == "pid":
            # P, I, D gains.
            P = rospy.get_param("controller/p_gain") * np.eye(7)
            I = rospy.get_param("controller/i_gain") * np.eye(7)
            D = rospy.get_param("controller/d_gain") * np.eye(7)

            # Stores proximity threshold.
            epsilon = rospy.get_param("controller/epsilon")
		    
            # Stores maximum COMMANDED joint torques.
            MAX_CMD = rospy.get_param("controller/max_cmd") * np.eye(7)
            
            self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
        else:
            raise Exception('Controller {} not implemented.'.format(self.controller_type))
            
        # Planner tells controller what plan to follow.
        self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint torques.
		self.cmd = np.eye(7)

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# Create joint-velocity publisher.
		self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# Create subscriber to joint_angles.
		rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
	
	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target.
		"""

		# Read the current joint angles from the robot.
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = self.curr_pos*(math.pi/180.0)

		# Update cmd from PID based on current position.
		self.cmd = -self.controller.get_command(self.curr_pos)
		
        # Check is start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
            self.expUtil.set_startT(self.controller.path_start_T)
        if self.controller.path_end_T is not None:
            self.reached_goal = True
            self.expUtil.set_endT(self.controller.path_end_T)

        # update the experiment utils executed trajectory tracker
		if self.reached_start and not self.reached_goal:
			# update the experimental data with new position
			timestamp = time.time() - self.path_start_T
			self.expUtil.update_tracked_traj(timestamp, self.curr_pos)

if __name__ == '__main__':
    path_follower = PathFollower()
