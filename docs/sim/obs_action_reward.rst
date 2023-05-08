.. _obs_action_reward:

Observation, Action, and Reward
===============================

Observation
-----------

The complete set of possible :class:`~smarts.core.observations.Observation` returned by SMARTS environment is shown below.  

+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Observation                  | Type                                                                   | Remarks                                                                            |
+==============================+========================================================================+====================================================================================+
| dt                           | float                                                                  | Amount of simulation time the last step took.                                      |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| step_count                   | int                                                                    | Number of steps taken by SMARTS thus far in the current scenario.                  |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| steps_completed              | int                                                                    | Number of steps this agent has taken within SMARTS.                                |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| elapsed_sim_time             | float                                                                  | Amount of simulation time elapsed for the current scenario.                        |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| events                       | :class:`~smarts.core.events.Events`                                    | Classified observations that can trigger agent done status.                        |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ego_vehicle_state            | :class:`~smarts.core.observations.EgoVehicleObservation`               | Ego vehicle status.                                                                |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| under_this_agent_control     | bool                                                                   | Whether this agent currently has control of the vehicle.                           |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| neighborhood_vehicle_states  | Optional[List[:class:`~smarts.core.observations.VehicleObservation`]]  | List of neighborhood vehicle states.                                               |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| waypoint_paths               | Optional[List[List[:class:`~smarts.core.road_map.Waypoint`]]]          | Dynamic evenly-spaced points on the road ahead of the vehicle.                     |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| distance_travelled           | float                                                                  | Road distance driven by the vehicle.                                               |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| lidar_point_cloud            |                                                                        | Lidar point cloud consisting of [points, hits, (ray_origin, ray_vector)].          |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| drivable_area_grid_map       | Optional[:class:`~smarts.core.observations.DrivableAreaGridMap`]       | Drivable area map.                                                                 |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| occupancy_grid_map           | Optional[:class:`~smarts.core.observations.OccupancyGridMap`]          | Occupancy map.                                                                     |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| top_down_rgb                 | Optional[:class:`~smarts.core.observations.TopDownRGB`]                | RGB camera observation.                                                            |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| road_waypoints               | Optional[:class:`~smarts.core.observations.RoadWaypoints`]             | Per-road waypoints information.                                                    |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| via_data                     | :class:`~smarts.core.observations.Vias`                                | Listing of nearby collectable ViaPoints and ViaPoints collected in the last step.  |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| signals                      | Optional[List[:class:`~smarts.core.observations.SignalObservation`]]   | List of nearby traffic signal (light) states on this timestep.                     |
+------------------------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------+

.. note::

    The :attr:`~smarts.core.observations.Observation.occupancy_grid_map` is recommended when using scenarios with pedestrians. A higher resolution is preferable to ensure pedestrians are visible.

.. note::

    Some observations like :attr:`~smarts.core.observations.Observation.occupancy_grid_map`, :attr:`~smarts.core.observations.Observation.drivable_area_grid_map`,
    and :attr:`~smarts.core.observations.Observation.top_down_rgb`, require the installation of optional packages for image rendering, so install them via 
    ``pip install -e .[camera_obs]``.

Reward
------

The default reward from SMARTS environment is a function of the distance travelled.
Here, ``x`` is the distance travelled in meters from the last time step where a non-zero reward was given.

.. math::

    \begin{equation}
    reward(x)=
        \begin{cases}
            x, & \text{if $\|x\|>0.5$}\\
            0, & \text{otherwise}
        \end{cases}
    \end{equation}

Action
------

Prior to a simulation, an agent's action type and its policy to provide compliant actions, can be configured via its agent specification instance of :class:`~smarts.zoo.agent_spec.AgentSpec`. 
Refer to :ref:`agent` for details.

An agent can be configured to emit any one of the following action types from :class:`~smarts.core.controllers.ActionSpaceType`.

+ :attr:`~smarts.core.controllers.ActionSpaceType.Continuous`
+ :attr:`~smarts.core.controllers.ActionSpaceType.Lane`
+ :attr:`~smarts.core.controllers.ActionSpaceType.ActuatorDynamic`
+ :attr:`~smarts.core.controllers.ActionSpaceType.LaneWithContinuousSpeed`
+ :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
+ :attr:`~smarts.core.controllers.ActionSpaceType.RelativeTargetPose`
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
