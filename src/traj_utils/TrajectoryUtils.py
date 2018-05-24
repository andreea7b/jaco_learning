import copy
import numpy as np
import openravepy as orp

"""
Utilities that deal with a trajectory.

Perhaps in the future these functions will be methods in a class we make to extend
the OpenRAVE Trajectory class.
"""

def SampleSpecificGroup(trajectory, time, 
                        group_name_prefix="joint_values"):
    """
    Sample the trajectory at a given time, and return the values of a specific
        group at that time (by default, returns the joint values at that time).
    @param trajectory the trajectory to sample
    @param time the time at which to sample at
    @param group_name_prefix a prefix of the group you want to sample. By default
        the prefix of the joint values group.
    """
    assert time < trajectory.GetDuration(),\
        "Trajectory must be sampled at a time before it ends"
    spec = trajectory.GetConfigurationSpecification()
    specific_group = spec.GetGroupFromName(group_name_prefix)
    return trajectory.Sample(time, specific_group)

def GetGroupValues(waypoint, group_name_prefix, configuration_specification):
    """
    In a waypoint of a trajectory with configuration_specification, get the
    values of the group with prefix group_name_prefix. So, for example, if
    group_name_prefix was "joint_velocities", GetGroupValues would return
    the joint velocities of that waypoint.
    @param waypoint the waypoint which we want to get the group values from.
    @param configuration_specification the configuration specification of the
        waypoint.
    @parm group_name_prefix the prefix for the group name that we are changing.
    """
    group_offset = GroupOffset(configuration_specification,
                               group_name_prefix=group_name_prefix)
    group_length = GroupLength(configuration_specification,
                               group_name_prefix=group_name_prefix)
    if group_length == 1:
        return waypoint[group_offset]
    return waypoint[group_offset:group_offset + group_length]

def GetGroupValuesAtWaypointIndex(
    trajectory, waypoint_index, group_name_prefix):
    waypoint = trajectory.GetWaypoint(waypoint_index)
    return\
        GetGroupValues(waypoint, group_name_prefix,
                       trajectory.GetConfigurationSpecification())

def SetGroupValues(waypoint, group_values, group_name_prefix,
             configuration_specification):
    """
    In a waypoint of a trajectory with configuration_specification, set the
    group with prefix group_name_prefix to the group_values.
    @param waypoint the waypoint which we want to alter
    @param group_values the values that we want to set that group to.
    @param configuration_specification the configuration_specification of the
        waypoint.
    @parm group_name_prefix the prefix for the group name that we are changing.
    """
    new_waypoint = waypoint[:]
    group_offset = GroupOffset(configuration_specification,
                               group_name_prefix=group_name_prefix)
    group_length = GroupLength(configuration_specification,
                               group_name_prefix=group_name_prefix)
    new_waypoint[group_offset:group_offset + group_length] = group_values
    return new_waypoint

def SetGroupValuesAtWaypointIndex(
    trajectory, env, index, group_values, group_name_prefix):
    waypoint = trajectory.GetWaypoint(waypoint_index)
    new_waypoint = SetGroupValues(waypoint, group_values, group_name_prefix,
                                  trajectory.GetConfigurationSpecification())
    new_trajectory = CopyTrajectory(trajectory, env)
    new_trajectory.Insert(index, new_waypoint)
    return new_trajectory

    
def GroupOffset(configuration_specification, group_name_prefix="joint_values"):
    group = configuration_specification.GetGroupFromName(group_name_prefix)
    if group is not None:
        return group.offset
    else:
        return None

def GroupLength(configuration_specification, group_name_prefix="joint_values"):
    group = configuration_specification.GetGroupFromName(group_name_prefix)
    if group is not None:
        return group.dof
    else:
        return None

def CopyTrajectory(trajectory, env):
    new_trajectory = orp.RaveCreateTrajectory(env, "")
    new_trajectory.Init(trajectory.GetConfigurationSpecification())
    for i in range(trajectory.GetNumWaypoints()):
        waypoint = trajectory.GetWaypoint(i)
        waypoint_copy = waypoint[:]
        new_trajectory.Insert(i, waypoint_copy)
    return new_trajectory

def MakeDofValuesIntoTrajectory(env, dof_values, time_spacing,
                                configuration_specification=None):
    """
    Given a set of DOF Values, create a trajectory that has those configurations
    as waypoints that are spaced in time according to the time_spacing array.
    @param dof_values a 2D array where each row is a DOF values configuration
    @param time_spacing a list of values that state which second of the
        output trajectory each waypoint should fall on.
    @param configuration_specification the configuration specification of the
        output trajectory.
    """
    assert len(dof_values) == len(time_spacing),\
        "Each entry in the time_spacing array should correspond to a row\
        in the dof_values array"
    assert time_spacing[0] == 0,\
        "The trajectory must start at its zero-th second"
    num_dofs = dof_values.shape[1]
    if configuration_specification is None:
        configuration_specification = DefaultArchieConfigurationSpec()
    trajectory = orp.RaveCreateTrajectory(env, "")
    trajectory.Init(configuration_specification)
    joint_values_offset =\
        GroupOffset(configuration_specification, "joint_values")
    joint_velocities_offset =\
        GroupOffset(configuration_specification, "joint_velocities")
    delta_time_offset =\
        GroupOffset(configuration_specification, "deltatime")
    iswaypoint_offset =\
        GroupOffset(configuration_specification, "iswaypoint")
    for i in range(len(dof_values)):
        joint_values = dof_values[i,:]
        if i == 0:
            delta_time = 0
            joint_velocities = np.zeros(num_dofs)
        else:
            delta_time = time_spacing[i] - time_spacing[i-1]
            joint_velocities = (dof_values[i,:] - dof_values[i-1,:])\
                               / delta_time
        waypoint = np.zeros(configuration_specification.GetDOF())
        waypoint[joint_values_offset:joint_values_offset + len(joint_values)]\
            = joint_values
        waypoint[joint_velocities_offset:joint_velocities_offset +
             len(joint_velocities)] = joint_velocities
        waypoint[delta_time_offset] = delta_time
        waypoint[iswaypoint_offset] = 1
        trajectory.Insert(trajectory.GetNumWaypoints(), waypoint)
    return trajectory

def MakeDofValuesIntoEvenlySpacedTrajectory(
        env, dof_values, total_time, configuration_specification=None):
    """
    Given a set of DOF Values, create a trajectory that has those configurations
    as waypoints evenly-spaced in time.  The output trajectory should take
    total_time seconds.
    @param dof_values a 2D array where each row is a DOF values configuration
    @param total_time how long the output trajectory should take
    """
    if configuration_specification == None:
        configuration_specification = DefaultArchieConfigurationSpec()
    time_step = total_time*1.0 / len(dof_values)
    time_spacing = np.arange(0, total_time, time_step)
    return MakeDofValuesIntoTrajectory(
        env, dof_values, time_spacing, 
        configuration_specification=configuration_specification)

def ChangeWaypointsToMatchTimeSpacing(env, trajectory, time_spacing):
    """
    Given a trajectory, sample waypoints so that they are spaced, with respect
    to time, as the time_spacing variable outlines. So for example, if the 
    time_spacing is [1, 3, 4, 5], create a trajectory that has waypoints at
    the first, third fourth and fifth second, but is the same as the input
    trajectory.
    The essence of this function is that it should not change how a trajectory
    is percieved when it is implemented, only how it is represented in terms of
    waypoints. However, subtelties in calculating the new joint velocities might
    mean that it is not exactly the same as the input when implemented.
    Also, users should keep in mind that making a trajectory more coarse than
    it is in waypoint representation will definitely alter how it looks when
    implemented, as details will have been lost. The main use case of this
    function should be creating trajectories that have, in general, a higher
    resolution in waypoints than the input.
    @param trajectory the trajectory that is to be altered
    @param time_spacing the times at which waypoints should occur. Should be
        monotonically increasing, and the largest value should be less than
        the duration of the trajectory"
    """
    assert (x > y  for x, y in zip(time_spacing, time_spacing[1:])),\
        "time_spacing must be monotonically increasing"
    assert time_spacing[-1] <= trajectory.GetDuration(),\
        "The time_spacing should not sample beyond the trajectory's duration"

    configuration_specification = trajectory.GetConfigurationSpecification()
    new_trajectory = orp.RaveCreateTrajectory(env, "")
    new_trajectory.Init(configuration_specification)
    joint_values_offset =\
        configuration_specification.GetGroupFromName("joint_values").offset
    num_robot_dofs =\
        configuration_specification.GetGroupFromName("joint_values").dof
    joint_velocities_offset =\
        configuration_specification.GetGroupFromName("joint_velocities").offset
    delta_time_offset =\
        configuration_specification.GetGroupFromName("deltatime").offset
    iswaypoint_offset =\
        configuration_specification.GetGroupFromName("iswaypoint").offset
    for i in range(len(time_spacing)):
        if i == 0:
            delta_time=0
        else:
            delta_time = time_spacing[i] - time_spacing[i-1]
        time_to_sample = time_spacing[i]
        waypoint = np.zeros(configuration_specification.GetDOF())
        waypoint[joint_values_offset:joint_values_offset + num_robot_dofs]\
            = SampleSpecificGroup(trajectory, time_to_sample, "joint_values")
        waypoint[joint_velocities_offset:joint_velocities_offset +
            num_robot_dofs] =\
            SampleSpecificGroup(trajectory, time_to_sample, "joint_velocities")
        waypoint[delta_time_offset] = delta_time
        waypoint[iswaypoint_offset] = 1
        new_trajectory.Insert(new_trajectory.GetNumWaypoints(), waypoint)
    return new_trajectory

def ChangeWaypointsToBeEvenlyTimeSpaced(env, trajectory, num_waypoints):
    """
    Given a trajectory, sample waypoints so that they are evenly spaced
    with respect to time, and make these the waypoints of the returned
    trajectory.
    The essence of this function is that it should not change how a trajectory
    is percieved when it is implemented, only how it is represented in terms of
    waypoints. However, subtelties in calculating the new joint velocities might
    mean that it is not exactly the same as the input when implemented.
    Also, users should keep in mind that making a trajectory more coarse than
    it is in waypoint representation will definitely alter how it looks when
    implemented, as details will have been lost. The main use case of this
    function should be creating trajectories that have, in general, a higher
    resolution in waypoints than the input.
    @param trajectory the trajectory that is to be altered
    @param num_waypoints the number of waypoints that the new trajectory should
        have
    """
    duration = trajectory.GetDuration()*1.0
    time_spacing = np.arange(0, duration, duration / num_waypoints)
    return ChangeWaypointsToMatchTimeSpacing(env, trajectory, time_spacing)

def AddChangeToTrajectory(trajectory, change_index, change_dofs, env,
                          propagation_weights=None, keep_times=True):
    """
    Changes the trajectory so that the waypoint at change_index has the joint
    values change_dofs. Propagates the change out from the change_index
    according to propagation weights (if propagation_weights is None, the
    default is a 1 at change_index and zero everywhere else).
    @param trajectory the trajectory to be changed
    @param change_index which waypoint of the trajectory is changed
    @param change_dofs what the joint values will be changed to in the
        changed_index.
    @param propagation_weights how the change will be propagated out to the
        rest of the trajectory. An array of values in [0, 1], where
        propagation_weights[change_index] = 1 and the weights are montonically
        increasing up to change index and monotonically decreasing afterwards.
    @param keep_times if the delta times in the trajectory should be changed
        to accomodate the fact that the trajectory is changing. If True, the
        trajectory returned will take the same time but have very different
        joint velocities. If False, trajectory returned will have similar joint
        velocities to the input byt will have the same delta times.
    """
    assert trajectory.GetNumWaypoints() > 0,\
        "Trajectroy must have at least one waypoint"
    assert change_index < trajectory.GetNumWaypoints(),\
        "Must change an index inside the trajectory"
    if propagation_weights == None:
        # The default propagation is no propagation -- changing only the affected
        # waypoint.
        propagation_weights = np.zeros(trajectory.GetNumWaypoints())
        propagation_weights[change_index] = 1
    configuration_specification = trajectory.GetConfigurationSpecification()
    num_robot_dofs =\
        GroupLength(configuration_specification, "joint_values")
    assert len(change_dofs) == num_robot_dofs,\
        "The change must have the same number of joint values as the trajectory's\
        joint_values field"
    new_trajectory = orp.RaveCreateTrajectory(env, "")
    new_trajectory.Init(configuration_specification)
    for i in range(trajectory.GetNumWaypoints()):
        waypoint = trajectory.GetWaypoint(i)
        waypoint_joint_values = GetGroupValues(waypoint, "joint_values",
                                               configuration_specification)
        difference_to_change = change_dofs - waypoint_joint_values
        new_waypoint_joint_values = waypoint_joint_values +\
                                    propagation_weights[i] * difference_to_change
        waypoint = SetGroupValues(waypoint, new_waypoint_joint_values,
                                  "joint_values", configuration_specification)
        new_trajectory.Insert(new_trajectory.GetNumWaypoints(), waypoint)
    return new_trajectory
    if keep_times:
        new_trajectory = MakeLogicalKeepTimes(new_trajectory, env)
    else:
        new_trajectory = MakeLogicalKeepJointVelocities(new_trajectory, env)
    return new_trajectory

def AddPauseToTrajectory(trajectory, pause_index, pause_duration, env):
    """
    Create a pause at the index pause_index.
    @param the trajectory at which to pause. Not modified, a copy is made,
        modified and returned.
    @param pause_index the waypoint at which we want the robot to pause. So,
        for example, if pause_index=5, the robot will stay at the joint
        configuration specified by the fifth waypoint and pause.
    @param pause_duration how long (in seconds) the pause should be
    @param env the environment in which this trajectory will happen.
    """
    waypoint = trajectory.GetWaypoint(pause_index)
    new_trajectory = CopyTrajectory(trajectory, env)
    modified_waypoint = waypoint[:]
    modified_waypoint = SetGroupValues(
        modified_waypoint, 0, "joint_velocities",
        trajectory.GetConfigurationSpecification())
    new_trajectory.Insert(pause_index, modified_waypoint, overwrite=True)
    pause_waypoint = waypoint[:]
    pause__waypoint = SetGroupValues(
        pause_waypoint, 0, "joint_velocities",
        trajectory.GetConfigurationSpecification())
    pause_waypoint = SetGroupValues(
        pause_waypoint, pause_duration, "deltatime",
        trajectory.GetConfigurationSpecification())
    new_trajectory.Insert(pause_index + 1, pause_waypoint)
    return new_trajectory

def AddPausesToTrajectory(trajectory, pauses_list, env):
    """
    Create multiple pauses in the trajectory.
    @param trajectory the trajectory in which we insert the pauses.
    @param pauses_list a list of tuples of the form (pause_index, pause_duration)
    @param env the environment in which this trajectory will happen.
    """
    assert len(pauses_list) > 0, "Must insert at least one pause"
    new_trajectory = trajectory
    for index, duration in pauses_list:
        new_trajectory = AddPauseToTrajectory(
            new_trajectory, index, duration, env)
    return new_trajectory

# TODO
def MakeLogicalKeepTimes(trajectory, env):
    """
    Adjusts the velocities of a trajectory, so that they make sense given that the
    delta times in the trajectory stay the same. Does not check velocity limits.
    """
    configuration_spec = trajectory.GetConfigurationSpecification()
    joint_velocities_offset =\
        GroupOffset(trajectory.GetConfigurationSpecification(), "joint_velocities")
    delta_time_offset =\
        GroupOffset(trajectory.GetConfigurationSpecification(), "deltatime")
    num_robot_dofs =\
        GroupLength(trajectory.GetConfigurationSpecification(), "joint_values")
    new_trajectory = orp.RaveCreateTrajectory(env, "")
    new_trajectory.Init(configuration_spec)
    new_trajectory.Insert(0, trajectory.GetWaypoint(0))
    for i in range(1, trajectory.GetNumWaypoints() - 1):
        waypoint = trajectory.GetWaypoint(i)
        prev_waypoint = trajectory.GetWaypoint(i - 1)
        joint_values_difference = np.subtract(
            GetGroupValues(waypoint, "joint_values", configuration_spec),
            GetGroupValues(prev_waypoint, "joint_values", configuration_spec))
        delta_time = GetGroupValues(waypoint, "deltatime",
                                    configuration_spec)
        new_velocities = joint_values_difference / delta_time
        waypoint = SetGroupValues(waypoint, new_velocities, "joint_velocities",
                                  configuration_spec)
        new_trajectory.Insert(i, waypoint)
    new_trajectory.Insert(new_trajectory.GetNumWaypoints(),
        trajectory.GetWaypoint(trajectory.GetNumWaypoints() - 1))
    return new_trajectory

# TODO
def MakeLogicalKeepJointVelocities(trajectory):
    """
    Adjusts the delta times in the trajectory so that they make sense, given that
    the joint velocities stay the same.
    """
    pass

def GetLastWaypointBefore(trajectory, time):
    """
    Get the waypoint index that is the last waypoint before the time time in the
    trajectory.
    """
    assert time < trajectory.GetDuration(),\
        "The time we want cannot be larger than the trajectory's total duration"
    running_time = 0
    for i in range(trajectory.GetNumWaypoints()):
        running_time += GetGroupValues(
            trajectory.GetWaypoint(i), "deltatime",
            trajectory.GetConfigurationSpecification())
        if running_time > time:
            return i

#TODO
def RetimeBetweenWaypoints(trajectory, env, 
                           start_waypoint, end_waypoint, duration):
    """
    Change the trajectory so that the segment between start_waypoint and
    end_waypoint takes duration seconds to complete.
    Does not change the relative times between those waypoints, only the
    total duration of the segment.
    """
    new_trajectory = CopyTrajectory(trajectory, env)
    delta_times = []
    # the deltatime field specifies how long it takes to get to a waypoint
    # from the previous one. So, if we care about the time it takes to go
    # between start and end, we need to see what the delta times are from 
    # start + 1 to (including) end.
    for i in range(start_waypoint + 1, end_waypoint + 1):
        delta_times.append(GetGroupValuesAtWaypointIndex(
            trajectory, i, "deltatime"))
    delta_times = np.array(delta_times, dtype=float)
    normalized_delta_times = delta_times / np.sum(delta_times)
    new_delta_times = delta_times * duration
    for i in range(start_waypoint + 1, end_waypoint + 1):
        new_trajectory = SetGroupValuesAtWaypointIndex(
            new_trajectory, env, i, new_delta_times[i], "deltatime")
    return new_trajectory

# TODO
def DefaultArchieConfigurationSpec():
    """
    Returns the configuration specification for the Archie robot (jaco 7 dof
    with spherical wrist).
    """
    pass
