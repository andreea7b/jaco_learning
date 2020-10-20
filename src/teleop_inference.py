#! /usr/bin/env python
"""


Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu), Matthew Zurek
"""

import math
import sys, select, os
import time
from threading import Thread

#from utils.environment_utils import *

import pybullet as p
import numpy as np
from controllers.pid_controller import PIDController
from learners.teleop_learner import TeleopLearner
from utils.environment_utils import *


from teleop_inference_base import TeleopInferenceBase


class TeleopInference(TeleopInferenceBase):
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
	"""

	def __init__(self):
		super(TeleopInference, self).__init__(False)

		# ------- setup pybullet -------
		#physicsClient = p.connect(p.GUI)
		physicsClient = p.connect(p.GUI, options="--opengl2")

		# Set camera angle.
		p.resetDebugVisualizerCamera(cameraDistance=2.50, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[-0.8,0.05,0.02])

		# Add path to data resources for the environment.
		p.setAdditionalSearchPath("../data/resources")

		# Setup the environment.
		self.bullet_environment = setup_environment(self.goals)

		# Get rid of gravity and make simulation happen in real time.
		p.setGravity(0, 0, 0)
		p.setRealTimeSimulation(1)

		# Start simulation.
		if self.inference_method == "collect":
			N = 5

			# Add demonstration recording buttons.
			self.buttons = [p.addUserDebugParameter("Stop Recording", 1, 0, 0),
							p.addUserDebugParameter("Next Demo", 1, 0, 0),
							p.addUserDebugParameter("Save Demo", 1, 0, 0)]
			self.numPush = [0, 0, 0]
		else:
			N = 1
		self.queries = 0

		while self.queries < N:
			print "Attempting round {}.".format(self.queries+1)
			bullet_start = np.append(self.start.reshape(7), np.array([0.0, 0.0, 0.0]))
			move_robot(self.bullet_environment["robot"], bullet_start)
			self.demo = [np.append(np.array([0.0]), bullet_start)]
			self.running = True
			self.record = True

			# Start simulation.
			while self.running:
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break

				# Update position.
				self.keyboard_input_callback()

				# Look after button presses.
				if self.inference_method == "collect":
					self.update_buttons()
				else:
					self.queries = 1

				# Update sim position with new velocity command.
				for i in range(len(self.cmd)):
					p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

				time.sleep(0.05)

		# Disconnect once the session is over.
		p.disconnect()

		print "----------------------------------"

	def joint_angles_callback(self, curr_pos):
		"""
		Reads the latest position of the robot and publishes an
		appropriate velocity command to move the robot to the target.
		"""
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

		ctl_cmd = self.controller.get_command(self.curr_pos.reshape(7,1))
		#print "joint 6 unblended assistance:", ctl_cmd[6,6]
		#TODO: decide what to log
		#self.exp_data['joint6_assist'].append(ctl_cmd[6,6])
		if self.assistance_method == "blend":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
				if self.beta_method == "joint":
					goal, beta = self.learner.argmax_joint_beliefs
					#print 'goal:', goal, 'beta:', beta
					print 'joint beliefs:', self.learner.joint_beliefs
					belief = np.max(self.learner.joint_beliefs[goal])
				elif self.beta_method == "estimate":
					goal, beta = self.learner.argmax_estimate
					belief = self.learner.goal_beliefs[goal]
					#print 'goal:', goal, 'beta:', beta
					#print 'beta estimates:', self.learner.beta_estimates
					#print 'goal beliefs:', self.learner.goal_beliefs
				self.alpha = self.beta_arbitration(beta, belief)
				if goal != self.curr_goal:
					print 'new assistance trajectory, goal:', goal
					self.curr_goal = goal
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

	def replay_trace(self, trace):
		for waypt in trace:
			for jointIndex in range(p.getNumJoints(self.bullet_environment["robot"])):
				p.resetJointState(self.bullet_environment["robot"], jointIndex, waypt[jointIndex])
			time.sleep(0.01)

	def update_buttons(self):
		stopPushes = p.readUserDebugParameter(self.buttons[0])
		nextPushes = p.readUserDebugParameter(self.buttons[1])
		savePushes = p.readUserDebugParameter(self.buttons[2])

		if stopPushes > self.numPush[0]:
			print "Stopping recording. If happy with the recording, press Save Demo; otherwise press Next Demo."
			self.numPush[0] = stopPushes
			if self.record == True:
				self.record = False
				# Pre-process the recorded data.
				trace = np.squeeze(np.array(self.demo))
				lo = 0
				hi = trace.shape[0] - 1
				while lo < hi and np.linalg.norm(trace[lo] - trace[lo + 1]) < 0.01:
					lo += 1
				while hi > 0 and np.linalg.norm(trace[hi] - trace[hi - 1]) < 0.01:
					hi -= 1
				trace = trace[lo:hi+1, :]
				self.replay_trace(trace)
				self.trace = trace
			else:
				print "Can't stop a recording that hasn't started yet."

		if nextPushes > self.numPush[1]:
			self.numPush[1] = nextPushes
			self.running = False

		if savePushes > self.numPush[2]:
			self.numPush[2] = savePushes
			if self.record == False:
				traj = np.array(self.trace)[:,1:8]
				np.save('data/demo{}.npy'.format(self.queries), traj)
				print 'Saved trajectory {}.'.format(self.queries+1)
				self.queries += 1
				self.running = False
			else:
				print "Can't save while recording! Please stop the recording first."

		if self.record:
			state = p.getJointStates(self.bullet_environment["robot"],
									 range(p.getNumJoints(self.bullet_environment["robot"])))
			waypt = [s[0] for s in state]
			self.demo.append(np.array(waypt))


	def keyboard_input_callback(self):
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.01, 0.01, 0.01]
		time_step = 0.05
		turn_step = 0.05
		EElink = 7

		# Get current EE position.
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		if p.B3G_LEFT_ARROW in keys:
			EEPos_new[1] -= dist_step[1]
		if p.B3G_RIGHT_ARROW in keys:
			EEPos_new[1] += dist_step[1]
		if p.B3G_UP_ARROW in keys:
			EEPos_new[0] -= dist_step[0]
		if p.B3G_DOWN_ARROW in keys:
			EEPos_new[0] += dist_step[0]
		if ord('i') in keys:
			EEPos_new[2] += dist_step[2]
		if ord('k') in keys:
			EEPos_new[2] -= dist_step[2]

		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			jointVelocities = (newPoses - jointPoses) / time_step
		if ord('j') in keys:
			jointVelocities[EElink] += turn_step / time_step
		if ord('l') in keys:
			jointVelocities[EElink] -= turn_step / time_step

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:8])

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			#joint_angles = np.diag(jointPoses[1:8] * (180/np.pi))
			self.joint_angles_callback(joint_angles)
		else:
			self.cmd = self.joy_cmd


	def idx_to_time(self, idx):
		return self.start_T + idx * self.timestep

	def beta_arbitration(self, beta, belief):
		if self.alpha_method == 'prob':
			return belief
		elif self.alpha_method == 'beta':
			return 1 #all joystick
			return 0
			#return np.clip(1 / beta, 0, 1)
			#return np.clip(0.5 / beta, 0, 1)
			#return np.clip(np.exp(-beta + 0.1), 0, 1)

if __name__ == '__main__':
	TeleopInference()
