# beta_adaptive_pHRI: Adaptive Physical HRI

Control, planning, and learning system for physical human-robot interaction (pHRI) with a JACO2 7DOF robotic arm. Learning is adaptive based on how relevant the human's interaction is. 

## Dependencies
* Ubuntu 14.04, ROS Indigo, OpenRAVE, Python 2.7
* or_trajopt, or_urdf, or_rviz, prpy, pr_ordata
* kinova-ros
* fcl

## Running the Adaptive Physical HRI System
### Setting up the JACO2 Robot
Turn the robot on and put it in home position by pressing and holding the center (yellow) button on the joystick.
 
In a new terminal, turn on the Kinova API by typing:
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2s7s300 use_urdf:=true
```
### Starting the controller, planner, and learning system
In another terminal window, run:
```
roslaunch beta_adaptive_pHRI trajoptPID.launch ID:=0 task:=None method_type:=A record:=F replay:=F feat_method:=BETA feat_list:="table,coffee"
```
Command-line options include:
* `ID`: Participant/user identification number (for experiments and data saving)
* `task`: Task string {Bring close to table = table, Correct coffee cup orientation = coffee, Free roam = None}
* `method_type`: Sets the pHRI control method {impedance control = A, impedance + learning from pHRI = B, demonstration = C}
* `replay`: Replays a recorded bag of joint torques and angles
* `record`: Records the interaction forces, measured trajectories, and cost function weights for a task {record data = T, don't record = F}
* `feat_method`: Learning method used. 
  * ALL = update all features at once, according to A. Bajcsy* , D.P. Losey*, M.K. O'Malley, and A.D. Dragan. [Learning Robot Objectives from Physical Human Robot Interaction.](http://proceedings.mlr.press/v78/bajcsy17a/bajcsy17a.pdf) Conference on Robot Learning (CoRL), 2017.
  * MAX = update one feature at a time, according to A. Bajcsy , D.P. Losey, M.K. O'Malley, and A.D. Dragan. [Learning from Physical Human Corrections, One Feature at a Time.](https://dl.acm.org/citation.cfm?id=3171267) International Confernece on Human-Robot Interaction (HRI), 2018.
  * BETA = relevance adaptive method (ours) soon to be published.
* `feat_list`: List of features the robot's internal representation contains. Options: "table" (distance to table), "coffee" (coffee cup orientation), "human" (distance to human), and any combination of these.

### References
* https://github.com/abajcsy/iact_control
* TrajOpt Planner: http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/index.html
* PID Control Reference: https://w3.cs.jmu.edu/spragunr/CS354/handouts/pid.pdf
