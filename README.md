# beta_adaptive_pHRI: Adaptive Physical HRI

Implementation of traditional, interactive, and adaptive control schemes for a Kinova 7DOF Jaco robotic arm. 
ROS, Python, and Kinova API. Testing interaction integrated into planning. 

## Running the Controllers
### PID Controller 
In the terminal, run:
```
roslaunch iact_control jaco_pid_demo.launch
```
You can change the P,I, and D gains simply through the terminal by specifying:
```
roslaunch iact_control jaco_pid_demo.launch p_gain:=100 i_gain:=0 d_gain:=15
```
You can also change the target goal configuration though the terminal by specifying:
```
roslaunch iact_control jaco_pid_demo.launch j0:=180 j1:=180 j2:=180 j3:=180 j4:=180 j5:=180 j5:=180 j6:=180    
```
### References
* PID Control Reference: https://w3.cs.jmu.edu/spragunr/CS354/handouts/pid.pdf
