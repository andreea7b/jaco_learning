#! /usr/bin/env python
"""
Runs gravity compensation mode on the JACO2 7DOF. 
"""

import roslib; roslib.load_manifest('kinova_demo')
import rospy
import actionlib
import sys, select, os
import kinova_msgs.msg
from kinova_msgs.srv import *

def init_torque_mode():
	"""
	Switches Jaco to torque-control mode using ROS services
	"""
	# use service to set torque control parameters	
	service_address = '/j2s7s300_driver/in/set_torque_control_parameters'	
	rospy.wait_for_service(service_address)
	try:
		setTorqueParameters = rospy.ServiceProxy(service_address, SetTorqueControlParameters)
		setTorqueParameters()           
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

	# use service to switch to torque control	
	service_address = '/j2s7s300_driver/in/set_torque_control_mode'	
	rospy.wait_for_service(service_address)
	try:
		switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
		switchTorquemode(1)           
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e
		return None	

def end_torque_mode():
	"""
	Ends torque-control mode using ROS service
	"""
	# switch back to position control after finished 
	service_address = '/j2s7s300_driver/in/set_torque_control_mode'	
	rospy.wait_for_service(service_address)
	try:
		switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
		switchTorquemode(0)
		return None
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e
		return None

def zeroJointTorqueMsg():
	"""
	Returns a zero JointTorque Kinova msg
	"""
	jointCmd = kinova_msgs.msg.JointTorque()
	jointCmd.joint1 = 0.0;
	jointCmd.joint2 = 0.0;
	jointCmd.joint3 = 0.0;
	jointCmd.joint4 = 0.0;
	jointCmd.joint5 = 0.0;
	jointCmd.joint6 = 0.0;
	jointCmd.joint7 = 0.0;
	
	return jointCmd

def joint_position_client(angle_set, prefix):
	"""
	Moves robot to desired position
	"""
	action_address = '/' + prefix + 'driver/joints_action/joint_angles'    
	client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmJointAnglesAction)       
	client.wait_for_server()    

	goal = kinova_msgs.msg.ArmJointAnglesGoal()
	goal.angles.joint1 = angle_set[0]
	goal.angles.joint2 = angle_set[1]
	goal.angles.joint3 = angle_set[2]
	goal.angles.joint4 = angle_set[3]
	goal.angles.joint5 = angle_set[4]
	goal.angles.joint6 = angle_set[5]
	goal.angles.joint7 = angle_set[6]    	

	client.send_goal(goal)

	client.wait_for_result(rospy.Duration(100.0))

	# Prints out the result of executing the action
	return client.get_result()  

if __name__ == '__main__':
	rospy.init_node("grav_comp")

	# start torque mode
	init_torque_mode()

	# create joint-torque publisher
	torque_pub = rospy.Publisher('/j2s7s300_driver/in/joint_torque', kinova_msgs.msg.JointTorque, queue_size=1)

	r = rospy.Rate(100) 

	#print "----------------------------------"
	#print "Moving robot to candle like position..."
	#result = joint_position_client([180]*7, 'j2s7s300_')
	#print "Done."

	print "----------------------------------"
	print "Grav comp mode starting, press ENTER to quit:"
	
	while not rospy.is_shutdown():

		if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
			line = raw_input()
			break

		torque_pub.publish(zeroJointTorqueMsg()) 
		r.sleep()

	# cleanly end torque mode
	end_torque_mode()
	
