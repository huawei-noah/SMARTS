.. _agent:

How to build an Agent
======================

SMARTS provides users the ability to customize their agents. :class:`smarts.zoo.agent_spec.AgentSpec` has the following fields:

.. code-block:: python

    class AgentSpec:
        interface: AgentInterface
        agent_builder: Callable[..., Agent] = None
        agent_params: Optional[Any] = None

An example of how to create an `Agent` instance is shown below.

.. code-block:: python

    AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )

    agent = agent_spec.build_agent()

We will further explain the fields of the `Agent` class later on this page. You can also read the source code at :class:`smarts.env.agent`.

==============
AgentInterface
==============

:class:`smarts.core.agent_interface.AgentInterface` regulates the flow of information between the agent and a SMARTS environment. It specifies the observations the agent expects to receive from the environment and the action the agent does to the environment. To create an agent interface, you can try

.. code-block:: python

    agent_interface = AgentInterface.from_type(
        interface = AgentType.Standard,
        max_episode_steps = 1000, 
        ...
    )

SMARTS provide some interface types, and the differences between them is shown in the table below. **T** means the `AgentType` will provide this option or information. 

.. code-block:: python

    |                        |       AgentType.Full       | AgentType.StandardWithAbsoluteSteering |       AgentType.Standard        |    AgentType.Laner     |          AgentType.LanerWithSpeed       |      AgentType.Tracker       |  AgentType.TrajectoryInterpolator  |   AgentType.MPCTracker  |          AgentType.Boid         |        AgentType.Loner       |       AgentType.Tagger       |     AgentType.Direct     |
    | :--------------------: | :------------------------: | :------------------------------------: | :-----------------------------: | :--------------------: | :-------------------------------------: | :--------------------------: | :--------------------------------: | :---------------------: | :-----------------------------: | :--------------------------: | :--------------------------: | :----------------------: |
    |         action         | ActionSpaceType.Continuous |       ActionSpaceType.Continuous       | ActionSpaceType.ActuatorDynamic |  ActionSpaceType.Lane  | ActionSpaceType.LaneWithContinuousSpeed |  ActionSpaceType.Trajectory  | ActionSpaceType.TrajectoryWithTime |   ActionSpaceType.MPC   | ActionSpaceType.MultiTargetPose |  ActionSpaceType.Continuous  |  ActionSpaceType.Continuous  |  ActionSpaceType.Direct  |
    |    max_episode_steps   |           **T**            |                 **T**                  |              **T**              |         **T**          |                   **T**                 |            **T**             |                **T**               |           **T**         |              **T**              |            **T**             |            **T**             |            **T**         |
    |  neighborhood_vehicles |           **T**            |                 **T**                  |              **T**              |                        |                                         |                              |                                    |                         |              **T**              |                              |            **T**             |            **T**         |
    |        waypoints       |           **T**            |                 **T**                  |              **T**              |         **T**          |                   **T**                 |            **T**             |                                    |           **T**         |              **T**              |            **T**             |            **T**             |                          |
    | drivable_area_grid_map |           **T**            |                                        |                                 |                        |                                         |                              |                                    |                         |                                 |                              |                              |                          |
    |           ogm          |           **T**            |                                        |                                 |                        |                                         |                              |                                    |                         |                                 |                              |                              |                          |
    |           rgb          |           **T**            |                                        |                                 |                        |                                         |                              |                                    |                         |                                 |                              |                              |                          |
    |          lidar         |           **T**            |                                        |                                 |                        |                                         |                              |                                    |                         |                                 |                              |                              |                          |
    |      accelerometer     |           **T**            |                 **T**                  |              **T**              |         **T**          |                   **T**                 |            **T**             |                **T**               |           **T**         |              **T**              |            **T**             |            **T**             |            **T**         |
    |         signals        |           **T**            |                                        |                                 |                        |                                         |                              |                                    |                         |                                 |                              |                              |            **T**         |
    |          debug         |           **T**            |                 **T**                  |              **T**              |         **T**          |                   **T**                 |            **T**             |                **T**               |           **T**         |              **T**              |            **T**             |            **T**             |            **T**         |

`max_episode_steps` controls the max running steps allowed for the agent in an episode. The default `None` setting means agents have no such limit.
You can move `max_episode_steps` control authority to RLlib with their config option `horizon`, but lose the ability to customize
different max_episode_len for each agent.

`action` controls the agent action type used. There are multiple `ActionSpaceType` options: `ActionSpaceType.Continuous`, `ActionSpaceType.Lane`, `ActionSpaceType.LaneWithContinuousSpeed` 
`ActionSpaceType.ActuatorDynamic`, `ActionSpaceType.Trajectory`, `ActionSpaceType.TrajectoryWithTime`, `ActionSpaceType.MPC`, `ActionSpaceType.TargetPose`, `ActionSpaceType.MultiTargetPose`, and `ActionSpaceType.Direct`.

- `ActionSpaceType.Continuous`: `(float, float, float)` continuous action space with throttle, brake, absolute steering angle. 
- `ActionSpaceType.ActuatorDynamic`: `(float, float float)` continuous action space with throttle, brake, steering rate. Steering rate means the amount of steering angle change *per second* (either positive or negative) to be applied to the current steering angle.
- `ActionSpaceType.Lane`: `str` discrete lane action space of strings including "keep_lane",  "slow_down", "change_lane_left", "change_lane_right".
- `ActionSpaceType.LaneWithContinuousSpeed`: `(int, float)` mixed action space of discrete lane change values `{-1,0,1}` corresponding to `{right_lane,current_lane,left_lane}`, and continuous target speed.
- `ActionSpaceType.Trajectory`: `(Sequence[float], Sequence[float], Sequence[float], Sequence[float])` continuous action space using trajectory as x coordinates, y coordinates, headings, and speeds to directly move a vehicle.
- `ActionSpaceType.TrajectoryWithTime`: `(Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[float])` continuous action space using trajectory as times, x coordinates, y coordinates, headings, and speeds to interpolate the vehicle along the trajectory.
- `ActionSpaceType.MPC`: `(Sequence[float], Sequence[float], Sequence[float], Sequence[float])` continuous action space using trajectory as x coordinates, y coordinates, headings, and speeds to adaptively perform controls on the vehicle model in an attempt to match the given trajectory. 
- `ActionSpaceType.TargetPose`: `Sequence[float, float, float, float]` continuous action space with a single vehicle x coordinate, y coordinate, heading, and time delta to reach the given pose.
- `ActionSpaceType.MultiTargetPose`: `Dict[str, (float, float, float, float)]` continuous action space that provides actions for multiple vehicles with each vehicle id mapped to pose as x coordinate, y coordinate, heading, and time delta to reach the given pose. 
- `ActionSpaceType.Direct`: `Union[float, (float,float)]` continuous action space where you can pass either (a) initial speed upon reset or (b) linear acceleration and angular velocity for other steps.


For other observation options, see :ref:`observations` for details.

We recommend you customize your `agent_interface`, like:

.. code-block:: python

    from smarts.core.agent_interface import AgentInterface
    from smarts.core.controllers import ActionSpaceType

    agent_interface = AgentInterface(
        max_episode_steps=1000,
        waypoints=True,
        neighborhood_vehicles=True,
        drivable_area_grid_map=True,
        ogm=True,
        rgb=True,
        lidar=False,
        action=ActionSpaceType.Continuous,
    )

For further customization, you can try:

.. code-block:: python

    from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, DrivableAreaGridMap, OGM, RGB, Waypoints
    from smarts.core.controllers import ActionSpaceType

    agent_interface = AgentInterface(
        max_episode_steps=1000,
        waypoints=Waypoints(lookahead=50), # lookahead 50 meters
        neighborhood_vehicles=NeighborhoodVehicles(radius=50), # only get neighborhood info with 50 meters.
        drivable_area_grid_map=True,
        ogm=True,
        rgb=True,
        lidar=False,
        action=ActionSpaceType.Continuous,
    )

Refer to :class:`smarts/core/agent_interface` for more details.


IMPORTANT: The generation of a DrivableAreaGridMap (`drivable_area_grid_map=True`), OGM (`ogm=True`) and/or RGB (`rgb=True`) images may significantly slow down the environment `step()`. If your model does not consume such observations, we recommend that you set them to `False`.

IMPORTANT: Depending on how your agent model is set up, `ActionSpaceType.ActuatorDynamic` might allow the agent to learn faster than `ActionSpaceType.Continuous` simply because learning to correct steering could be simpler than learning a mapping to all the absolute steering angle values. But, again, it also depends on the design of your agent model. 

======
Agent
======

An agent maps an observation to an action.

.. code-block:: python

    # A simple agent that ignores observations
    class IgnoreObservationsAgent(Agent):
        def act(self, obs):
            return [throttle, brake, steering_rate]

The observation passed in should be the observations that a given agent sees. In **continuous action space** the action is expected to produce values for `throttle` [0->1], `brake` [0->1], and `steering_rate` [-1->1].

Otherwise, only while using **lane action space**, the agent is expected to return a lane related command: `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, `"change_lane_right"`.

Another example:

.. literalinclude:: ../minimal_agent.py
   :language: python

===================
Spaces
===================

Spaces provide samples for variation. For reference, see https://gymnasium.farama.org/api/spaces/ .


.. code-block:: python

    # Observation space should match the observation. An example observation
    # space is as follows, if the observation only consists of speed and 
    # steering values.
    OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),        
    }

Observations can be customized. Some example customizations are provided in 
:class:`smarts.env.custom_observations`, with the following meanings:

- "distance_from_center": distance to lane center 
- "angle_error": ego heading relative to the closest waypoint
- "speed": ego speed
- "steering": ego steering
- "ego_ttc": time to collision in each lane
- "ego_lane_dist": closest cars' distance to ego in each lane



==================
Agent Observations
==================

Of all the information to work with it is useful to know a bit about the main agent observations in particular.

For that see the :ref:`observations` section for details.

