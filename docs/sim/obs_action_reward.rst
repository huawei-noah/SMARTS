.. _obs_action_reward:

Observation, Action, and Reward
===============================

We have introduced `AgentInterface` in :ref:`agent` which allows us to choose from the standard observation and action types for communication between an agent and a SMARTS environment.

============
Observations
============

Here we will introduce details of available observation types.
For `AgentType.Full`, which contains the most concrete observation details, the raw observation returned
is a Python `NamedTuple` with the following fields:

-----
Types
-----

* `GridMapMetadata` - Metadata for the observation maps with the following information,
    * `created_at` - time at which the map was loaded
    * `resolution` - map resolution in world-space-distance/cell
    * `width` - map width in # of cells
    * `height` - map height in # of cells
    * `camera_pos` - camera position when project onto the map
    * `camera_heading_in_degrees` - camera rotation angle along z-axis when project onto the map
* `Waypoint` - Metadata for the observation maps with the following information,
    * `id` - an integer identifier for this waypoint
    * `pos` - a numpy array (x, y) center point along the lane
    * `heading` - heading angle of lane at this point (radians)
    * `lane_width` - width of lane at this point (meters)
    * `speed_limit` - lane speed in m/s
    * `lane_id` - a globally unique identifier of lane under waypoint
    * `right_of_way` - `True` if this waypoint has right of way, `False` otherwise
    * `lane_index` - index of the lane under this waypoint, right most lane has index 0 and the index increments to the left
* `EgoVehicleObservation` - a `NamedTuple` describing the ego vehicle:
    * `id` - a string identifier for this vehicle
    * `position` - Coordinate of the center of the vehicle bounding box's bottom plane. shape=(3,). dtype=np.float64.
    * `bounding_box` - `Dimensions` data class for the `length`, `width`, `height` of the vehicle
    * `heading` - vehicle heading in radians
    * `speed` - agent speed in m/s
    * `steering` - angle of front wheels in radians
    * `yaw_rate` - rotational speed in radian per second
    * `road_id` - the identifier for the road nearest to this vehicle
    * `lane_id` - a globally unique identifier of the lane under this vehicle 
    * `lane_index` - index of the lane under this vehicle, right most lane has index 0 and the index increments to the left
    * `mission` - a field describing the vehicle plotted route
    * `linear_velocity` - Vehicle velocity along body coordinate axes. A numpy array of shape=(3,) and dtype=np.float64.
    * `angular_velocity` - Angular velocity vector. A numpy array of shape=(3,) and dtype=np.float64.
    * `linear_acceleration` (Optional) - Linear acceleration vector. A numpy array of shape=(3,). dtype=np.float64. Requires accelerometer sensor.
    * `angular_acceleration` (Optional) - Angular acceleration vector. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor. 
    * `linear_jerk` (Optional) - Linear jerk vector. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor.
    * `angular_jerk` (Optional) - Angular jerk vector. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor.
* `VehicleObservation` - a subset of `EgoVehicleObservation` describing a vehicle:
    * `position`, `bounding_box`, `heading`, `speed`, `lane_id`, `lane_index` - the same as with `EgoVehicleObservation`
* `ViaPoint` - 'Collectable' locations that can be placed within the simulation.
    * `position` - The `[x, y]` location of this `ViaPoint`.
    * `lane_index` - The lane index on the road this `ViaPoint` is associated with
    * `road_id` - The road id this `ViaPoint` is associated with
    * `required_speed` - The rough minimum speed required to collect this `ViaPoint`
* `Vias` - A listing of nearby `ViaPoint` and `ViaPoint` collected in the last step

--------
Features
--------

* `dt` - the amount of simulation time since the last step
* `step_count` - Number of steps taken by SMARTS thus far for the current scenario
* `elapsed_sim_time` - The amount of simulation time elapsed in SMARTS thus far for the current scenario
* `events` - a `NamedTuple` with the following fields:
    * `collisions` - collisions the vehicle has been involved with other vehicles (if any)
    * `off_road` - `True` if the vehicle is off the road
    * `wrong_way` - `True` if the vehicle is heading against the legal driving direction of the lane
    * `not_moving` - `True` if the vehicle has not moved for the configured amount of time
    * `reached_goal` - `True` if the vehicle has reached the ego agent's mission goal
    * `reached_max_episode_steps` - `True` if the vehicle has reached its max episode steps
    * `agents_alive_done` - `True` if all configured co-simulating agents are done (if any). This is useful for cases the ego has dependence on other agents
* `ego_vehicle_state` - A `EgoVehicleObservation` describing state information about the ego vehicle.
* `neighborhood_vehicle_states` - a list of `VehicleObservation` within the configured distance of the vehicle.
* `waypoint_paths` - A collection of `Waypoint` in front of the ego vehicle showing the potential routes ahead. Each item is a `Waypoint` instance.
* `distance_travelled` - The amount of distance that the ego vehicle has travelled along its mission route (or forward along road if no mission.)
* `lidar_point_cloud` (Optional) - The result of a simulated lidar array sourced from the ego vehicle's center.
  * Each item contains: a collection of hit points(or misses as an inf value), a collection of if the lidar point hit, and a collection of lines from emission point to hit(or inf).
* `drivable_area_grid_map` (Optional) - contains an observation image with its metadata
    * `metadata` - `GridMapMetadata`
    * `data` - A grid map (default 256x256) that shows the static drivable area around the ego vehicle
* `occupancy_grid_map` (Optional) - contains an observation image with its metadata
    * `metadata` - `GridMapMetadata`
    * `data` - An `OGM <https://en.wikipedia.org/wiki/Occupancy_grid_mapping>`_ (default 256x256) around the ego vehicle
* `top_down_rgb` (Optional) - contains an observation image with its metadata
    * `metadata` - `GridMapMetadata`
    * `data` - A RGB image (default 256x256) with the ego vehicle at the center
* `road_waypoints` (Optional) - A collection of `Waypoint` near the ego vehicle representing a `Waypoint` approximation of nearby lane centers.
    * `lanes` - The representation of each lane represented by `Waypoint`. Each item is list of `Waypoint`.
* `via_data` - A `Vias` describing collectable points the agent can visit.

See implementation in :class:`~smarts.core.sensors`


Then, you can choose the observations needed through :class:`~smarts.core.agent_interface.AgentInterface`.
Note: Some observations like `occupancy_grid_map`, `drivable_area_grid_map` and `top_down_rgb` requires the use of Panda3D package to render agent camera observations during simulations. So you need to install the required dependencies first using the command `pip install -e .[camera_obs]`

=======
Rewards
=======

The default reward from SMARTS environments is 

reward 

simply the distance travelled by the agent within the most recent single time step.
 given by a calculation within smarts; `env_reward` 
from smarts environments directly uses the reward from smarts. 

The given reward is 0 or `reward < -0.5` or `reward > 0.5` relating to distance travelled in meters 
on the step that a vehicle has gone at least 0.5 meters since the last given non-zero reward.

=======
Actions
=======

Prior to a simulation, an agent's action type and its policy to provide compliant actions, can be configured via its agent specification instance of :class:`~smarts.zoo.agent_spec.AgentSpec`. 
Refer to :ref:`agent` for details.

An agent can be configured to emit any one of the following action types from :class:`~smarts.core.controllers.ActionSpaceType`.

+ :attr:`~smarts.core.controllers.ActionSpaceType.Continuous`
+ :attr:`~smarts.core.controllers.ActionSpaceType.Lane`
+ :attr:`~smarts.core.controllers.ActionSpaceType.ActuatorDynamic`
+ :attr:`~smarts.core.controllers.ActionSpaceType.LaneWithContinuousSpeed`
+ :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
+ :attr:`~smarts.core.controllers.ActionSpaceType.Trajectory`
+ :attr:`~smarts.core.controllers.ActionSpaceType.MultiTargetPose`
+ :attr:`~smarts.core.controllers.ActionSpaceType.MPC`
+ :attr:`~smarts.core.controllers.ActionSpaceType.TrajectoryWithTime`
+ :attr:`~smarts.core.controllers.ActionSpaceType.Direct`
+ :attr:`~smarts.core.controllers.ActionSpaceType.Empty`

.. tip::

    Depending on the agent's policy, :attr:`~smarts.core.controllers.ActionSpaceType.ActuatorDynamic` action type might 
    allow the agent to learn faster than :attr:`~smarts.core.controllers.ActionSpaceType.Continous` action type because 
    learning to correct steering could be simpler than learning a mapping to all the absolute steering angle values. 
