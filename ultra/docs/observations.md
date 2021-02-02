# Observations

## Environment Observations

Raw environment observations can come in many different forms including images and physical metrics. The observations received by an agent can be configured with the agent's `AgentInterface`. To see more about SMARTS's environment observations, see the SMARTS documentation.

The baseline agents use an observation composed of pure physical metrics, that is, there are no images in the baseline agents observations. This is specified by specific parameters of the baseline agent's `AgentInterface` available in `ultra/baselines/agent_spec.py`:
```python
AgentInterface(
    waypoints=Waypoints(lookahead=20),  # Include the next 20 waypoints on the path of the current mission in the observation.
    neighborhood_vehicles=NeighborhoodVehicles(200),  # Include all social neighbouring vehicles in a radius of 200 in the observation.
    ...
    rgb=False,  # Do not include a top-down colour image in the observation.
    ...
    road_waypoints=False,  # Do not include a list of waypoints for all lanes in the observation.
    drivable_area_grid_map=False, # Do not include a sensor that says which areas of the map are drivable in the observation.
    ogm=False,  # Do not include a sensor that says which areas in space are occupied in the observation.
    lidar=False,  # Do not include the LIDAR point cloud sensor data in the observation.
    accelerometer=True,  # Include acceleration and jerk data in the observation.
)
```
As a result, a baseline agent's observation consists of:
- `events`, a `NamedTuple` describing the events that have occurred or are occurring. Examples include collisions, whether the ego vehicle is off the road, or whether the ego vehicle has reached its goal.
- `ego_vehicle_state`, a `VehicleObservation` `NamedTuple` for the ego vehicle describing physical properties of the ego vehicle such as position, heading, speed, and steering.
- `neighborhood_vehicle_states`, a list of `VehicleObservation` `NamedTuples`, each with aspects such as its position, heading, speed, acceleration, and jerk.
- `waypoint_paths`, a list of waypoints in front of the ego vehicle showing the potential routes ahead. Each waypoint contains information about things such as its own position, speed limit at that position, and lane width at that position.
- `distance_travelled`, the distance the ego vehicle has travelled.

## Observation Adapters

An observation adapter is a function that receives a raw environment observation and manipulates it (often by extracting only the relevant information) so that it can be processed by the agent. The observation adapter specifies how an agent will always observe the environment.

For example, the observation adapter used by the baseline agents (see `ultra/baselines/adapter.py`) selects specific information from the raw environment observation (`env_observation`), optionally manipulates this data, and returns this data as a dictionary. This new observation consists of properties pertaining to the `ego_state` (`env_observation.ego_vehicle_state`), and properties about the environment and neighbouring vehicles:
```python
state = dict(
    speed=ego_state.speed,
    relative_goal_position=relative_goal_position_rotated,
    distance_from_center=ego_dist_center,
    steering=ego_state.steering,
    angle_error=closest_wp.relative_heading(ego_state.heading),
    social_vehicles=env_observation.neighborhood_vehicle_states,
    road_speed=closest_wp.speed_limit,
    start=start.position,
    goal=goal.position,
    heading=ego_state.heading,
    goal_path=path,
    ego_position=ego_state.position,
    waypoint_paths=env_observation.waypoint_paths,
    events=env_observation.events,
)
```

Every observation passed to the baselines agents will be of the form specified by the observation adapter. For example, when a baseline agent's `act` method is called, it is passed this adapted observation from the environment. From there, the agent can continue to act directly from it, or continue to process this state.

For example, the DQN baseline's `act` method (`ultra/baselines/dqn/policy.py`) continues to process the given adapted observation with its `state_preprocessor`, further transforming the state into 3 categories:
- Images
- Low dimensional states (such as speed and steering angle), and
- Social vehicle information
