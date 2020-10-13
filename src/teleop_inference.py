#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu), Matthew Zurek
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import sys, select, os
import time
from threading import Thread

import kinova_msgs.msg
from kinova_msgs.srv import *
from sensor_msgs.msg import Joy

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from learners.teleop_learner import TeleopLearner
from utils import ros_utils
from utils.environment import Environment
from utils.openrave_utils import robotToCartesian
from utils.environment_utils import *

import numpy as np
import pickle

from openravepy import RaveCreatePhysicsEngine, RaveCreateModule
import pybullet as p


class TeleopInference():
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.

	Subscribes to:
		/$prefix$/out/joint_angles	- Jaco sensed joint angles
		/$prefix$/out/joint_torques - Jaco sensed joint torques

	Publishes to:
		/$prefix$/in/joint_velocity	- Jaco commanded joint velocities
	"""

	def __init__(self):
		# Create ROS node.
		rospy.init_node("teleop_inference")

		# Load mode
		mode = rospy.get_param("setup/sim_mode")

		# Load parameters and set up subscribers/publishers.
		self.load_parameters(mode)
		self.register_callbacks(mode)

		# Start admittance control mode.
		if mode == "real":
			ros_utils.start_admittance_mode(self.prefix)

		# Publish to ROS at 100hz.
		r = rospy.Rate(100)

		print "----------------------------------"
		if mode == "real":
			print "Moving robot, press ENTER to quit:"

			while not rospy.is_shutdown():
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break
				self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg((180/np.pi)*self.cmd))
				r.sleep()
			ros_utils.stop_admittance_mode(self.prefix)

		elif mode == "sim":
			print "Simulating robot, press ENTER to quit:"
			physics_engine = RaveCreatePhysicsEngine(self.sim_environment.env, 'ode')
			self.sim_environment.env.SetPhysicsEngine(physics_engine)
			self.sim_environment.env.StartSimulation(1e-1, True)
			self.sim_environment.robot.SetDOFValues( np.hstack((self.start, np.array([0,0,0]))) )

			loop_iter = 0
			while not rospy.is_shutdown():
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break
				## TODO: this should be done in another thread like a normal subscriber
				if loop_iter % 10 == 0:
					joint_angles = np.diag(self.sim_environment.robot.GetDOFValues() * (180/np.pi))
					self.joint_angles_callback(ros_utils.cmd_to_JointAnglesMsg(joint_angles))
					loop_iter = 0
					self.sim_environment.update_vel(self.cmd) ## TODO: make sure that update_vel is supposed to take radians
				loop_iter += 1
				r.sleep()

        elif mode == "pybullet":
            # Start simulation.
            cameraDistance = 1
            cameraYaw = 35
            cameraPitch = -35
            forward = 0
            turn = 0
            linkIdx = 7
            p.changeDynamics(self.bullet_environment["robot"], linkIdx, linearDamping=0.9)
            for i in range(11):
                p.setJointMotorControl2(self.bullet_environment["robot"], i, p.VELOCITY_CONTROL, force=0)
			while not rospy.is_shutdown():
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break
                EEPos = robot_coords(self.bullet_environment["robot"])[linkIdx-1]
                p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, EEPos)
                camInfo = p.getDebugVisualizerCamera()
                camForward = camInfo[5]
                forward, turn = self.keyboard_input_callback()
                force = [forward * camForward[0], forward * camForward[1], 0]
                cameraYaw = cameraYaw + turn
                if (forward):
                    p.applyExternalForce(self.bullet_environment["robot"], linkIdx, force, EEPos, flags=p.WORLD_FRAME)
                p.stepSimulation()
				r.sleep()

            # Disconnect once the session is over.
            p.disconnect()

        print "----------------------------------"
        self.joy_subscriber.unregister()
		del self.joy_environment
		del self.environment

	def load_parameters(self, mode):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""
		# ----- General Setup ----- #
		self.prefix = rospy.get_param("setup/prefix")
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.save_dir = rospy.get_param("setup/save_dir")

		self.start = np.array(rospy.get_param("setup/start"))*(math.pi/180.0)
		self.start += np.random.normal(0, 0.157, self.start.shape)

		# ----- Goals and goal weights setup ----- #
		# TODO: remove one of these
		#self.goal_poses = np.array(rospy.get_param("setup/goal_poses"))
		fixed_goals = [np.array(goal)*(math.pi/180.0) for goal in rospy.get_param("setup/goals")]
		try:
			learned_goals = np.load('learned_goals.npy')
			self.goals = fixed_goals + learned_goals
		except IOError:
			self.goals = fixed_goals

		self.feat_list = rospy.get_param("setup/common_feat_list")
		feat_range = {'table': 0.98,
					  'coffee': 1.0,
					  'laptop': 0.3,
					  'human': 0.3,
					  'efficiency': 0.22,
					  'proxemics': 0.3,
					  'betweenobjects': 0.2}
		common_weights = rospy.get_param("setup/common_feat_weights")
		goals_weights = []
		goal_dist_feat_weight = rospy.get_param("setup/goal_dist_feat_weight")
		if goal_dist_feat_weight != 0.0:
			# add features for distance from each of the goals
			common_weights = common_weights + ([0.] * len(self.goals))
			num_feats = len(self.feat_list)
			for goal_num in range(len(self.goals)):
				self.feat_list.append("goal"+str(goal_num)+"_dist")
				goal_weights = np.array(common_weights)
				goal_weights[num_feats + goal_num] = goal_dist_feat_weight
				goals_weights.append(goal_weights)
		else:
			# make copies of the common weights
			for goal_num in range(len(self.goals)):
				goals_weights.append(np.array(common_weights))
		self.goal_weights = goals_weights

		# Openrave parameters for the environment.
		model_filename = rospy.get_param("setup/model_filename")
		object_centers = rospy.get_param("setup/object_centers")
		self.environment = Environment(model_filename,
									   object_centers,
									   self.feat_list,
									   feat_range,
									   goals=self.goals,
		                               use_viewer=False,
									   plot_objects=False)
		self.goal_locs = self.environment.goal_locs

		# ----- Planner Setup ----- #
		# Retrieve the planner specific parameters.
		planner_type = rospy.get_param("planner/type")
		if planner_type == "trajopt":
			max_iter = rospy.get_param("planner/max_iter")
			num_waypts = rospy.get_param("planner/num_waypts")

			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment)
		else:
			raise Exception('Planner {} not implemented.'.format(planner_type))
		# TODO: do something better than goals[0]?
		self.traj, self.traj_plan = self.planner.replan(self.start, self.goals[0], None, self.goal_weights[0], self.T, self.timestep, return_both=True)

		# Track if you have reached the goal of the path and the episode start time
		self.start_T = None
		self.reached_goal = False

		# Save the current configuration.
		self.curr_pos = None

		# Save a history of waypts
		self.next_waypt_idx = 1
		self.traj_hist = np.zeros((int(self.T/self.timestep) + 1, 7))
		self.traj_hist[0] = self.start

		# ----- Add in learned cost function goals -----
		for learned_goal_save_path in rospy.get_param('setup/learned_goals'):
			# 1. create new weight vectors
			common_weights = common_weights + [0]
			for i in range(len(self.goal_weights)):
				self.goal_weights[i] = np.hstack((self.goal_weights[i], 0))
			learned_goal_weight = np.array(common_weights)
			learned_goal_weight[len(self.feat_list)] = 1.
			self.goal_weights.append(learned_goal_weight)

			# 2. add cost to environment
			#meirl_goal_save_path = "/root/catkin_ws/src/jaco_learning/data/pour_red_meirl.pt"
			# this reuses the first goal for the learned feature
			#self.environment.load_meirl_learned_feature(self.planner, learned_goal_weight, meirl_goal_save_path, goal=self.goals[0])
			# this uses the average demonstration final position
			self.environment.load_meirl_learned_feature(self.planner, learned_goal_weight, learned_goal_save_path)

		# ----- Controller Setup ----- #
		# Retrieve controller specific parameters.
		controller_type = rospy.get_param("controller/type")
		if controller_type == "pid":
			# P, I, D gains.
			P = rospy.get_param("controller/p_gain") * np.eye(7)
			I = rospy.get_param("controller/i_gain") * np.eye(7)
			D = rospy.get_param("controller/d_gain") * np.eye(7)

			# Stores proximity threshold.
			epsilon = rospy.get_param("controller/epsilon")

			# Stores maximum COMMANDED joint torques.
			MAX_CMD = rospy.get_param("controller/max_cmd") * np.eye(7)

			self.controller = PIDController(P, I, D, epsilon, MAX_CMD, self)
		else:
			raise Exception('Controller {} not implemented.'.format(controller_type))

		# Planner tells controller what plan to follow.
		self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint velocities.
		self.cmd = np.zeros((7,7))

		# ----- Learner Setup ----- #
		betas = np.array(rospy.get_param("learner/betas"))
		goal_beliefs = rospy.get_param("learner/goal_beliefs")
		if goal_beliefs != "none":
			goal_beliefs = goal_beliefs / np.linalg.norm(goal_beliefs)
		else:
			goal_beliefs = np.ones(len(self.goals))/len(self.goals)
		assert(len(goal_beliefs) == len(self.goals))
		assert(len(goal_beliefs) == len(self.goal_weights))
		beta_priors = rospy.get_param("learner/beta_priors")
		inference_method = rospy.get_param("learner/inference_method")
		self.beta_method = rospy.get_param("learner/beta_method")
		self.learner = TeleopLearner(self, goal_beliefs, beta_priors, betas, inference_method, self.beta_method)
		self.running_inference = False
		self.last_inf_idx = 0
		self.running_final_inference = False
		self.final_inference_done = False

        self.assistance_method = rospy.get_param("learner/assistance_method")
        self.alpha = 1. # in [0, 1]; higher numbers give more control to human
        self.zero_input_assist = rospy.get_param("learner/zero_input_assist")
        self.joy_cmd = np.zeros((7,7))

        if mode == "pybullet":
            # Connect to a physics simulator.
            physicsClient = p.connect(p.GUI)

            # Add path to data resources for the environment.
            p.setAdditionalSearchPath("../data/resources")

            # Setup the environment.
            self.bullet_environment = setup_environment()

            # Get rid of gravity and make simulation happen in real time.
            p.setGravity(0, 0, 0)
            p.setRealTimeSimulation(0)
        else:
            # ----- Input Device Setup ----- #
            self.joy_environment = Environment(model_filename,
                                               object_centers,
                                               list(), # doesn't need to know about features
                                               dict(),
                                               #goals=self.goals,
                                               use_viewer=False,
                                               plot_objects=False)

            # ----- Simulation Setup ----- #
            if mode == "sim":
                self.sim_environment = Environment(model_filename,
                                                   object_centers,
                                                   list(),
                                                   dict(),
                                                   goals=self.goals,
                                                   use_viewer=True,
                                                   plot_objects=False)

            self.exp_data = {
                'joint6_assist': []
            }

	def register_callbacks(self, mode):
		"""
		Sets up all the publishers/subscribers needed.
		"""
        if mode == "pybullet":
            return
		elif mode == "real":
			# Create joint-velocity publisher.
			self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)
			# Create subscriber to joint_angles.
			self.joint_subscriber = rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)

        # Create subscriber to input joystick.
		self.joy_subscriber = rospy.Subscriber('joy', Joy, self.joystick_input_callback, queue_size=1)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate velocity command to move the robot to the target.
		"""

		# Read the current joint angles from the robot.
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = curr_pos*(math.pi/180.0)

		if self.start_T is not None and (time.time() - self.start_T >= self.timestep * self.next_waypt_idx):
			if not self.next_waypt_idx >= len(self.traj_hist):
				self.traj_hist[self.next_waypt_idx] = self.curr_pos.reshape(7)
				self.next_waypt_idx += 1
				#print "timestep:", self.next_waypt_idx
				if not self.running_inference:
					#print 'calling inference from', self.next_waypt_idx - 1
					self.running_inference = True
					self.inference_thread = Thread(target=self.learner.inference_step)
					self.inference_thread.start()
			elif not self.running_final_inference:
				self.running_final_inference = True
				self.inference_thread.join()
				self.inference_thread = Thread(target=self.learner.final_step)
				self.inference_thread.start()
			elif self.final_inference_done:
				pass

		ctl_cmd = self.controller.get_command(self.curr_pos)
		#print "joint 6 unblended assistance:", ctl_cmd[6,6]
		self.exp_data['joint6_assist'].append(ctl_cmd[6,6])
		if self.assistance_method == "blend":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
				if self.beta_method == "joint":
					goal, beta = self.learner.argmax_joint_beliefs
					#print 'goal:', goal, 'beta:', beta
					#print 'joint beliefs:', self.learner.joint_beliefs
				elif self.beta_method == "estimate":
					goal, beta = self.learner.argmax_estimate
					#print 'goal:', goal, 'beta:', beta
					#print 'beta estimates:', self.learner.beta_estimates
					#print 'goal beliefs:', self.learner.goal_beliefs
				self.alpha = beta_arbitration(beta)
				self.traj = self.learner.cache['goal_traj_by_idx'][self.last_inf_idx][goal]
				self.traj_plan = self.learner.cache['goal_traj_plan_by_idx'][self.last_inf_idx][goal]
				self.controller.set_trajectory(self.traj,
											   path_start_T=self.idx_to_time(self.last_inf_idx))
			if np.allclose(self.joy_cmd, np.zeros((7,7))) and not self.zero_input_assist:
				self.cmd = self.joy_cmd
			else:
				self.cmd = self.alpha * self.joy_cmd + (1. - self.alpha) * ctl_cmd
		elif self.assistance_method == "expected":
			raise NotImplementedError
		elif self.assistance_method == "none":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
			self.cmd = self.joy_cmd
		else:
			raise ValueError
		# Update cmd from PID based on current position.
		#self.cmd = self.controller.get_command(self.curr_pos)

    def keyboard_input_callback(self):
        keys = p.getKeyboardEvents()
        for k, v in keys.items():

            if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
              turn = -0.5
            if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
              turn = 0
            if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
              turn = 0.5
            if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
              turn = 0

            if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED)):
              forward = 1
            if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
              forward = 0
            if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED)):
              forward = -1
            if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
              forward = 0

        return forward, turn

	def joystick_input_callback(self, msg):
		"""
		Reads joystick commands
		"""
		#start = time.time()
		FREQ = 10
		#joy_input = (msg.axes[1], msg.axes[0], msg.axes[2]) # corrects orientation
		joy_input = (msg.axes[0], -msg.axes[1], msg.axes[4])

		#pos = self.curr_pos.reshape(7) + np.array([0,0,np.pi,0,0,0,0])
		curr_angles = np.append(self.curr_pos.reshape(7), np.array([0,0,0]))

		# not using EE orientation
		dis = np.array(joy_input)
		# preserving EE orientation
		#dis = np.array(joy_input + (0,0,0))
		#dis = self.get_target_displacement(joy_input) #TODO

		# clamp/scale dis
		dis = dis * 0.25 / FREQ
		#dis = np.clip(dis, -0.5, 0.5)

		err = dis
		angles = np.copy(curr_angles)
		if np.linalg.norm(err) < 0.1 / FREQ:
			cmd = np.zeros(7)
		else:
			with self.joy_environment.robot:
				self.joy_environment.robot.SetDOFValues(angles)
				xyz = robotToCartesian(self.joy_environment.robot)[6]
				#xyz = np.append(xyz, np.zeros(3))
				for k in range(3):
					Jt = self.joy_environment.robot.ComputeJacobianTranslation(7, xyz)
					#Jo = self.joy_environment.robot.ComputeJacobianAxisAngle(7)
					J = Jt
					#J = np.vstack((Jt, Jo))

					#J = self.get_jacobian(self.joy_environment.robot, xyz) #TODO

					angle_step_dir = np.dot(J.T, err)
					pred_xyz_step = np.dot(J, angle_step_dir)
					step_size = np.dot(err, pred_xyz_step)/(np.linalg.norm(pred_xyz_step) ** 2)
					angles += step_size * angle_step_dir
					self.joy_environment.robot.SetDOFValues(angles)
					new_xyz = robotToCartesian(self.joy_environment.robot)[6]
					err -= (new_xyz - xyz)
					xyz = new_xyz
			cmd = (angles - curr_angles) * FREQ
			# clamp large joints if you want here



		# can modify joint 6 separately since the above always gives it velocity 0
		msg.axes[2] - msg.axes[5]
		cmd[6] = (msg.axes[2] - msg.axes[5])

		self.joy_cmd = np.diag(cmd[:7])
		#end = time.time()
		#print 'command time', end - start

	def _joystick_input_callback(self, msg):
		"""
		Reads joystick commands
		"""
		#joy_cmd = (msg.axes[1], msg.axes[0], msg.axes[2]) # corrects orientation
		joy_cmd = (msg.axes[1], msg.axes[0], msg.axes[3])
		#pos = self.curr_pos.reshape(7) + np.array([0,0,np.pi,0,0,0,0])
		pos = self.curr_pos.reshape(7)
		with self.joy_environment.robot:
			self.joy_environment.robot.SetDOFValues(np.append(pos, np.array([0,0,0])))
			xyz = robotToCartesian(self.joy_environment.robot)[6]
			Jt = self.joy_environment.robot.ComputeJacobianTranslation(7, xyz)
			Jo = self.joy_environment.robot.ComputeJacobianAxisAngle(7)

		# not using EE orientation
		#J, dis = Jt, np.array(joy_cmd)
		# preserving EE orientation
		J, dis = np.vstack((Jt, Jo)), np.array(joy_cmd + (0,0,0))

		# clamp/scale dis
		#dis = 0.1 * dis
		# dis = np.clip(dis, -limit, limit)

		# using pseudoinverse
		#J_inv = np.linalg.pinv(J)
		#cmd = np.dot(J_inv, dis)

		# using transpose
		#cmd = np.dot(J.T, dis)

		# using damped least squares
		lamb = 0.5
		A = np.dot(J, J.T) + lamb * np.eye(J.shape[0])
		cmd = np.dot(J.T, np.linalg.solve(A, dis))

		# clamp large joints if you want here
		self.cmd = np.diag(cmd)

	def idx_to_time(self, idx):
		return self.start_T + idx * self.timestep

def beta_arbitration(beta):
	#return 1 #all joystick
	return np.clip(1 / beta, 0, 1)
	#return np.clip(0.5 / beta, 0, 1)
	#return np.clip(np.exp(-beta + 0.1), 0, 1)

if __name__ == '__main__':
	TeleopInference()
