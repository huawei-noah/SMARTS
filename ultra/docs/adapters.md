# Adapters

An adapter is a function that receives a environment observation, environment reward,
and/or action from an agent, and then manipulates them (often by extracting or adding
relevant information) so that they can be processed by the agent or the environment.

## Action Adapters

An action adapter takes an action from an agent and adapts it so that it conforms to the
SMARTS simulator's action format. ULTRA has two default action adapters, one for
continuous action, and another for discrete action.

### [ultra.adapters.default_action_continuous_adapter](../ultra/adapters/default_action_continuous_adapter.py)

The default continuous action adapter requires the agent has a "continuous" action space
as defined by SMARTS. Therefore, when using this adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py)
of your agent needs its `action` parameter to be `ActionSpaceType.Continuous`. This
requirement is outlined in this module's `required_interface`.

SMARTS' `ActionSpaceType.Continuous` accepts actions in the form of a NumPy array with
shape (3,). That is, the action is a NumPy array of the form
`[throttle, brake, steering]` where throttle is a `float` in the range `[0, 1]`, brake 
is a `float` in the range `[0, 1]`, and steering is a `float` in the range `[-1, 1]`
(see [SMARTS action documentation](https://smarts.readthedocs.io/en/latest/sim/observations.html#actions)
for more on the behaviour of each part of the action). This action space is outlined in
this module's `gym_space`.

The behaviour of this action adapter is simply to return the action it was given,
without adapting it any further. It expects that the action outputted by the agent
already conforms to this NumPy array of shape (3,). The behaviour of this adapter is
fully defined in this module's `adapt` function.

### [ultra.adapters.default_action_discrete_adapter](../ultra/adapters/default_action_discrete_adapter.py)

The default discrete action adapter requires the agent has a "lane" action space as
defined by SMARTS. Therefore, when using this adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py)
of your agent needs its `action` parameter to be `ActionSpaceType.Lane`. This
requirement is outlined in this module's `required_interface`.

SMARTS' `ActionSpaceType.Lane` accepts actions in the form of a string. The string must
either be `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, or `"change_lane_right"`
where the behaviour of each action can be inferred from the action name (see
[SMARTS action documentation](https://smarts.readthedocs.io/en/latest/sim/observations.html#actions)
for more on the behaviour of each action). This action space is outlined in this
module's `gym_space`.

The behaviour of this action adapter is simply to return the action it was given,
without adapting it any further. It expects that the action outputted by the agent
already is one of the four available strings. The behaviour of this adapter is fully
defined in this module's `adapt` function.

## Info Adapters

An info adapter takes an [observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1), reward, and info dictionary from the environment and adapts them to include
more relevant information about the agent at each step. By default, the ULTRA
environment includes the ego vehicle's raw observation, and its score in the info
dictionary.
```python
info = {
    "score": ...,  # A float, the total distance travelled by the ego vehicle.
    "env_obs": ...,  # A smarts.core.sensors.Observation, the raw observation received by the ego vehicle.
}
```

ULTRA has a default info adapter that is used to include more data about the
agent that can be used to track the agent's learning progress and monitor the agent
during training and evaluation.

### [ultra.adapters.default_info_adapter](../ultra/adapters/default_info_adapter.py)

The default info adapter requires that the SMARTS environment include the next 20
waypoints in front of the ego vehicle, and all neighborhood (social) vehicles within a
radius of 200 meters around the ego vehicle. Therefore, when using this adapter, the
[AgentInterface](../../smarts/core/agent_interface.py) of your agent needs its
`waypoints` parameter to be `Waypoints(lookahead=20)` and its `neighborhood_vehicles`
parameter to be `NeighborhoodVehciles(radius=200.0)`. This requirement is outlined in
this module's `required_interface`.

The default info adapter modifies the given info dictionary passed to it by the
environment. Specifically, it adds another key, "logs", to the info dictionary. This
key's values is another dictionary that contains information about the agent:
```python
info = {
    "score": ...,  # A float, the total distance travelled by the ego vehicle.
    "env_obs": ...,  # A smarts.core.sensors.Observation, the raw observation received by the ego vehicle.
    "logs": {
        "position": ...,  # A np.ndarray with shape (3,), the x, y, z position of the ego vehicle.
        "speed": ...,  # A float, the speed of the ego vehicle in meters per second.
        "steering": ...,  # A float, the angle of the front wheels in radians.
        "heading": ...,  # A smarts.core.coordinates.Heading, the vehicle's heading in radians.
        "dist_center": ...,  # A float, the distance in meters from the center of the lane of the closest waypoint.
        "start": ...,  # A smarts.core.scenario.Start, the start of the ego evhicle's mission.
        "goal": ...,  # A smarts.core.scenario.PositionalGoal, the goal of the ego vehicle's mission.
        "closest_wp": ...,  # A smarts.core.waypoints.Waypoint, the closest waypoint to the ego vehicle.
        "events": ...,  # A smarts.core.events.Events, the events of the ego vehicle.
        "ego_num_violations": ...,  # An int, the number of violations committed by the ego vehicle (see ultra.utils.common.ego_social_safety).
        "social_num_violations": ...,  # An int, the number of violations committed by social vehicles (see ultra.utils.common.ego_social_safety).
        "goal_dist": ...,  # A float, the euclidean distance between the ego vehicle and its goal.
        "linear_jerk": ...,  # A float, the magnitude of the ego vehicle's linear jerk.
        "angular_jerk": ...,  # A float, the magnitude of the ego vehicle's angular jerk.
        "env_score": ...,  # A float, the ULTRA environment's reward obtained from the default reward adapter (see ultra.adapters.default_reward_adapter).
    }
}
```

This information contained in logs can ultimately be used by ULTRA's [Episode](../ultra/utils/episode.py)
object that is used to record this data to Tensorboard and also save this data to a
serializable object.

## Observation Adapters

An observation adapter takes an [observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1)
from the environment and adapts it so that it can be processed by the agent. ULTRA has
two default observation adapters, one that adapts the observation containing a top-down
RGB image into a gray-scale version of the same image, and another that adapts the
observation into a dictionary of vectors.

### [ultra.adapters.default_observation_image_adapter](../ultra/adapters/default_observation_image_adapter.py)

The default image observation adapter requires that the SMARTS environment include the
top-down RGB image in the agent's observation. Specifically, the image should be of
shape 64x64 with a resolution of 50 / 64. Therefore, when using this adapter, the
[`AgentInterface`](../../smarts/core/agent_interface.py) of your agent needs its `rgb`
parameter to be `RGB(width=64, height=64, resolution=(50 / 64))`. This requirement is
outlined in this module's `required_interface`.

This default image observation adapter produces a NumPy array of type `float32` and with
shape `(4, 64, 64)`. Each element of the array is normalized to be in the range
`[0, 1]`. This observation space is outlined in this module's `gym_space`.

This adapter receives an observation from the environment that contains a
`smarts.core.sensors.TopDownRGB` instance in the observation. The `data` attribute of
this class is a NumPy array of type `uint8` and shape `(4, 64, 64, 3)`. The adapter
converts this array to gray-scale by dotting it with `(0.1, 0.8, 0.1)`, resulting in the
value of each gray-scale pixel to be a linear combination of the red (R), green (G), and
blue (B) components of that pixel: `0.1 * R + 0.8 * G + 0.1 * B`. This gray-scale
weighting was chosen to accentuate the differences in gray values between the ego
vehicle, social vehicles, and the road. The gray-scale image is then normalized by
dividing the array by `255.0`. The output is a NumPy array of type `float32` and with
shape `(4, 64, 64)`. The most recent frame is at the highest index of this array. The
behaviour of this adapter is fully defined in this module's `adapt` function.

### [ultra.adapters.default_observation_vector_adapter](../ultra/adapters/default_observation_vector_adapter.py)

The default vector observation adapter requires that the SMARTS environment include the
next 20 waypoints in front of the ego vehicle, and all neighborhood (social) vehicles
within a radius of 200 meters around the ego vehicle. Therefore, when using this
adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py) of your agent
needs its `waypoints` parameter to be `Waypoints(lookahead=20)` and its
`neighborhood_vehicles` parameter to be `NeighborhoodVehicles(radius=200.0)`. This
requirement is outlined in this module's `required_interface`.

In addition to these aforementioned requirements, the observation, by default, contains
information about the ego vehicle's state. Provided that the observation has the
aforementioned requirements and the ego vehicle's state, this adapter adapts this
observation to a dictionary:
```python
{
    "low_dim_states": [
        ego_vehicle_speed / 30.0,
        distance_from_center / 1.0,
        steering / 3.14,
        angle_error / 3.14,
        relative_goal_position_x / 100.0,
        relative_goal_position_y / 100.0,
        relative_waypoint_position_x / 10.0,  # Nearest waypoint.
        relative_waypoint_position_y / 10.0,  # Nearest waypoint.
        relative_waypoint_position_x / 10.0,  # 2nd closest waypoint.
        relative_waypoint_position_y / 10.0,  # 2nd closest waypoint.
        ...
        relative_waypoint_position_x / 10.0,  # 20th closest waypoint.
        relative_waypoint_position_y / 10.0,  # 20th closest waypoint.
        road_speed / 30.0,
    ],
    "social_vehicles": [
        [
            relative_vehicle_position_x / 100.0,
            relative_vehicle_position_y / 100.0,
            heading_difference / 3.14,
            social_vehicle_speed / 30.0
        ],  # Closest social vehicle.
        [
            relative_vehicle_position_x / 100.0,
            relative_vehicle_position_y / 100.0,
            heading_difference / 3.14,
            social_vehicle_speed / 30.0
        ],  # 2nd closest social vehicle.
        ...
        [
            relative_vehicle_position_x / 100.0,
            relative_vehicle_position_y / 100.0,
            heading_difference / 3.14,
            social_vehicle_speed / 30.0
        ],  # 10th closest social vehicle.
    ]
}
```

Where:
- `ego_vehicle_speed` is the speed of the ego vehicle in meters per second.
- `distance_from_center` is the lateral distance between the center of the closest
waypoint's lane and the ego vehicle's position, divided by half of that lane's width.
- `steering` is the angle of ego vehicle's front wheels in radians.
- `angle_error` is the closest waypoint's heading minus the ego vehicle's heading.
- `relative_goal_position_x` is the x component of the vector obtained by calculating
the goal's `(x, y)` position minus the ego vehicle's `(x, y)` position, and rotating
that difference by the negative of the ego vehicle's heading. All in all, this keeps
this component completely relative from the ego vehicle's perspective.
- `relative_goal_position_y` is the y component of the vector obtained by calculating
the goal's `(x, y)` position minus the ego vehicle's `(x, y)` position, and rotating
that difference by the negative of the ego vehicle's heading. All in all, this keeps
this component completely relative from the ego vehicle's perspective.
- `relative_waypoint_position_x` is the x component of the vector obtained by
calculating the waypoint's `(x, y)` position minus the ego vehicle's `(x, y)` position,
and rotating that difference by the negative of the ego vehicle's heading. All in all,
this keeps the component completely relative from the ego vehicle's perspective.
- `relative_waypoint_position_y` is the y component of the vector obtained by
calculating the waypoint's `(x, y)` position minus the ego vehicle's `(x, y)` position,
and rotating that difference by the negative of the ego vehicle's heading. All in all,
this keeps the component completely relative from the ego vehicle's perspective.
- `road_speed` is the speed limit of the closest waypoint.
- `relative_vehicle_position_x` is the x component of vector obtained by calculating the
social vehicle's `(x, y)` position minus the ego vehicle's `(x, y)` position, and
rotating that difference by the negative of the ego vehicle's heading. All in all, this
keeps this component completely relative from the ego vehicle's perspective.
- `relative_vehicle_position_y` is the y component of vector obtained by calculating the
social vehicle's `(x, y)` position minus the ego vehicle's `(x, y)` position, and
rotating that difference by the negative of the ego vehicle's heading. All in all, this
keeps this component completely relative from the ego vehicle's perspective.
- `heading_difference` is the heading of the social vehicle minus the heading of the ego
vehicle.
- `social_vehicle_speed` is the speed of the social vehicle in meters per second.

Notice that the social vehicles are sorted by relative distance to the ego vehicle. This
was chosen under the assumption that the closest social vehicles are the ones that the
ego vehicle should pay attention to. While this is likely true in most situations, this
assumption may not be the most accurate in all cases. For example, if all the nearest
social vehicles are behind the ego vehicle, the ego vehicle will not observe any social
vehicles ahead of itself.

If the observation provided by the environment contains less than 10 social vehicles
(that is, there are less than 10 social vehicles in a 200 meter radius around the ego
vehicle), this adapter will pad the social vehicle adaptation with zero-vectors for the
remaining rows. For example, if there are no social vehicles in the observation from the
environment, the social vehicle adaptation would be a `(10, 4)` NumPy array with data:
`[[0., 0., 0., 0.], [0., 0., 0., 0.], ..., [0., 0., 0., 0.]]`.

If there are more than 10 social vehicles, this adapter will truncate the social vehicle
adaptation to only include 10 rows - the features of the 10 nearest social vehicles.

## Reward Adapters

A reward adapter takes an [observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1)
and the [environment reward](https://smarts.readthedocs.io/en/latest/sim/observations.html#rewards)
as arguments from the environment and adapts them, acting as a custom reward function.
ULTRA has one default reward adapter that uses elements from the agent's observation, as
well as the environment reward, to develop a custom reward.

### [ultra.adapters.default_reward_adapter](../ultra/adapters/default_reward_adapter.py)

The default reward adapter requires that the SMARTS environment include the next 20
waypoints in front of the ego vehicle in the ego vehicle's observation. Therefore, when
using this adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py) of your
agent needs its `waypoints` parameter to be `Waypoints(lookahead=20)`. This requirement
is outlined in this module's `required_interface`.

This default reward adapter combines elements of the agent's observation along with the
environment reward to create a custom reward. This custom reward consists of multiple
components:
```python
custom_reward = (
    ego_goal_reward +
    ego_collision_reward +
    ego_off_road_reward +
    ego_off_route_reward +
    ego_wrong_way_reward +
    ego_speed_reward +
    ego_distance_from_center_reward +
    ego_angle_error_reward +
    ego_reached_goal_reward +
    ego_step_reward +
    environment_reward
)
```

Where:
- `ego_goal_reward` is `0.0`
- `ego_collison_reward` is `-1.0` if the ego vehicle has collided, else `0.0`.
- `ego_off_road_reward` is `-1.0` if the ego vehicle is off the road, else `0.0`.
- `ego_off_route_reward` is `-1.0` if the ego vehicle is off its route, else `0.0`.
- `ego_wrong_way_reward` is `-0.02` if the ego vehicle is facing the wrong way, else
`0.0`.
- `ego_speed_reward` is `0.01 * (speed_limit - ego_vehicle_speed)` if the ego vehicle is
going faster than the speed limit, else `0.0`.
- `ego_distance_from_center_reward` is `-0.002 * min(1, abs(ego_distance_from_center))`.
- `ego_angle_error_reward` is `-0.0005 * max(0, cos(angle_error))`.
- `ego_reached_goal_reward` is `1.0` if the ego vehicle has reached its goal, else
`0.0`.
- `ego_step_reward` is
`0.02 * min(max(0, ego_vehicle_speed / speed_limit), 1) * cos(angle_error)`.
- `environment_reward` is `the_environment_reward / 100`.

And `speed_limit` is the speed limit of the nearest waypoint to the ego vehicle in
meters per second; the `ego_vehicle_speed` is the speed of the ego vehicle in meters per
second; the `angle_error` is the closest waypoint's heading minus the ego vehicle's
heading; the `ego_distance_from_center` is the lateral distance between the center
of the closest waypoint's lane and the ego vehicle's position, divided by half of that
lane's width; and `the_environment_reward` is the raw reward received from the SMARTS
simulator.
