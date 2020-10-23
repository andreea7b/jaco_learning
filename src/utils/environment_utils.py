import pybullet as p
import numpy as np
import random


def setup_environment(goals):
    objectID = {}

    # Add a table.
    pos = [-0.675, 0, -0.675]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["table"] = p.loadURDF("table/table.urdf", pos, orientation, useFixedBase=True)

    # Add a robot support.
    pos = [0, 0, -0.025]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["stand"] = p.loadURDF("support.urdf", pos, orientation, useFixedBase=True)

    # Add the laptop.
    pos = [-0.4, -0.25, -0.05]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["laptop"] = p.loadURDF("laptop.urdf", pos, orientation, useFixedBase=True)

    # Add the Jaco robot and initialize it.
    pos = [0, 0, 0]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["robot"] = p.loadURDF("jaco.urdf", pos, orientation, useFixedBase=True)

    # Add goal visuals.
    for i,goal in enumerate(goals):
        print "plotting goal"
        move_robot(objectID["robot"], np.append(goal.reshape(7), np.array([0, 0, 0])))
        pos = robot_coords(objectID["robot"])[-1]
        p.loadURDF("sphere_transparent.urdf", pos, useFixedBase=True, globalScaling=0.02)
        #p.addUserDebugLine(pos, pos+0.01, [0, 0, 1], 10)
        #p.addUserDebugText("Goal {}".format(i), pos, [0, 0, 0], 10)

    # Add a mug in Jaco's hand.
    targetPos = [-0.7, 0.0, 0.6]
    targetOr = p.getQuaternionFromEuler([0, -1.5, 0])
    jointPoses = p.calculateInverseKinematics(objectID["robot"], 7, targetPos, targetOr)
    move_robot(objectID["robot"], jointPoses)
    objectID["mug"] = p.loadURDF("dinnerware/cup/cup_small.urdf", globalScaling=1.0)
    cid = p.createConstraint(objectID["robot"], 7, objectID["mug"], -1, p.JOINT_FIXED, [0, 0, 0], [0.0, 0.0, -0.13],
                             [0, 0, 0], childFrameOrientation=p.getQuaternionFromEuler([0, -np.pi/2, 0]))

    return objectID


def move_robot(robotID, jointPoses):
    """
    Move the robot to a legal position.
    """
    for jointIndex in range(p.getNumJoints(robotID)-1):
        p.resetJointState(robotID, jointIndex+1, jointPoses[jointIndex])


def robot_coords(robotID):
    states = p.getLinkStates(robotID, range(11))
    coords = np.array([s[0] for s in states])
    return coords[1:8]


def robot_orientations(robotID):
    states = p.getLinkStates(robotID, range(11))
    orientations = np.array([p.getMatrixFromQuaternion(s[1]) for s in states])
    return orientations[1:8]


# -- Return raw features -- #
def raw_features(objectID, waypt):
    """
    Computes raw state space features for a given waypoint.
    ---
    Params:
        waypt -- single waypoint
    Returns:
        raw_features -- list of raw feature values
    """
    if len(waypt) < 11:
        waypt = np.append(np.append(np.array([0]), waypt.reshape(7)), np.array([0,0,0]))
    # Get relevant objects in the environment.
    posH, _ = p.getBasePositionAndOrientation(objectID["human"])
    posL, _ = p.getBasePositionAndOrientation(objectID["laptop"])
    object_coords = np.array([posH, posL])

    # Get xyz coords and orientations.
    for jointIndex in range(p.getNumJoints(objectID["robot"])):
        p.resetJointState(objectID["robot"], jointIndex, waypt[jointIndex])

    coords = robot_coords(objectID["robot"])
    orientations = robot_orientations(objectID["robot"])
    return np.reshape(np.concatenate((waypt[1:8], orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))


def upsample(trace, num_waypts, objectID):
    if trace.shape[0] >= num_waypts:
        return trace
    trace = trace[:,:7]
    timestep = 1.0 / (trace.shape[0] - 1)
    timestep_up = 1.0 / (num_waypts - 1)
    t = 0
    trace_up = np.zeros((num_waypts,7))
    for i in range(num_waypts):
        if t >= 1:
            trace_up[i] = trace[-1]
        else:
            curr_idx = int(t / timestep)
            curr_waypt = trace[curr_idx]
            next_waypt = trace[curr_idx + 1]
            trace_up[i] = curr_waypt + ((t - curr_idx * timestep) / timestep) * (next_waypt - curr_waypt)
        t += timestep_up
    raw_trace = []
    for waypt in trace_up:
        raw_trace.append(raw_features(objectID, waypt))
    return np.array(raw_trace)
