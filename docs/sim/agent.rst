.. _agent:

How to build an Agent
======================

SMARTS provides users the ability to customize their agents. :class:`smarts.core.agent.AgentSpec` has the following fields:

.. code-block:: python

    class AgentSpec:
        interface: AgentInterface
        agent_builder: Callable[..., Agent] = None
        agent_params: Optional[Any] = None
        observation_adapter: Callable = default_obs_adapter
        action_adapter: Callable = default_action_adapter
        reward_adapter: Callable = default_reward_adapter
        info_adapter: Callable = default_info_adapter

An example of how to create an `Agent` instance is shown below.

.. code-block:: python

    AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
        info_adapter=info_adapter,
    )

    agent = agent_spec.build_agent()

We will further explain the fields of the `Agent` class later on this page. You can also read the source code at :class:`smarts.env.agent`.

==============
AgentInterface
==============

:class:`smarts.core.agent_interface.AgentInterface` regulates the flow of informatoin between the an agent and a SMARTS environment. It specifies the observations an agent expects to receive from the environment and the action the agent does to the environment. To create an agent interface, you can try

.. code-block:: python

    agent_interface = AgentInterface.from_type(
        interface = AgentType.Standard,
        max_episode_steps = 1000, 
        ...
    )

SMARTS provide some interface types, and the differences between them is shown in the table below. **T** means the `AgentType` will provide this option or information. 

.. code-block:: python

    |                       |       AgentType.Full       | AgentType.StandardWithAbsoluteSteering |       AgentType.Standard        |   AgentType.Laner    |
    | :-------------------: | :------------------------: | :------------------------------------: | :-----------------------------: | :------------------: |
    |   max_episode_steps   |           **T**            |                 **T**                  |              **T**              |        **T**         |
    | neighborhood_vehicles |           **T**            |                 **T**                  |              **T**              |                      |
    |       waypoints       |           **T**            |                 **T**                  |              **T**              |        **T**         |
    |drivable_area_grid_map |           **T**            |                                        |                                 |                      |
    |          ogm          |           **T**            |                                        |                                 |                      |
    |          rgb          |           **T**            |                                        |                                 |                      |
    |         lidar         |           **T**            |                                        |                                 |                      |
    |        action         | ActionSpaceType.Continuous |       ActionSpaceType.Continuous       | ActionSpaceType.ActuatorDynamic | ActionSpaceType.Lane |
    |         debug         |           **T**            |                 **T**                  |              **T**              |        **T**         |

`max_episode_steps` controls the max running steps allowed for the agent in an episode. The default `None` setting means agents have no such limit.
You can move `max_episode_steps` control authority to RLlib with their config option `horizon`, but lose the ability to customize
different max_episode_len for each agent.

`action` controls the agent action type used. There are three `ActionSpaceType`: ActionSpaceType.Continuous, ActionSpaceType.Lane 
and ActionSpaceType.ActuatorDynamic.

- `ActionSpaceType.Continuous`: continuous action space with throttle, brake, absolute steering angle.
- `ActionSpaceType.ActuatorDynamic`: continuous action space with throttle, brake, steering rate. Steering rate means the amount of steering angle change *per second* (either positive or negative) to be applied to the current steering angle.
- `ActionSpaceType.Lane`: discrete lane action space of strings including "keep_lane",  "slow_down", "change_lane_left", "change_lane_right". (WARNING: This is the case in the current version 0.3.2b, but a newer version will soon be released. In this newer version, the action space will no longer consist of strings, but will be a tuple of an integer for `lane_change` and a float for `target_speed`.)

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

The observation passed in should be the observations that a given agent sees. In **contininuous action space** the action is expected to produce values for `throttle` [0->1], `brake` [0->1], and `steering_rate` [-1->1].

Otherwise, only while using **lane action space**, the agent is expected to return a lane related command: `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, `"change_lane_right"`.

Another example:

.. literalinclude:: ../minimal_agent.py
   :language: python

===================
Adapters and Spaces
===================

Adapters convert the data such as an agent's raw sensor observations to a more useful form. And spaces provide samples for variation.
 Adapters and spaces are particularly relevant to the :class:`rllib_example.agent` example. Also check out :class:`smarts.env.custom_observations` for some processing examples.

.. code-block:: python

    # Adapter
    def observation_adapter(env_observation):
        ego = env_observation.ego_vehicle_state

        return {
            "speed": [ego.speed],
            "steering": [ego.steering],
        }

    # Associated Space
    # You want to match the space to the adapter
    OBSERVATION_SPACE = gym.spaces.Dict(
    {
        ## see http://gym.openai.com/docs/#spaces
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),        
    }

You can customize your metrics and design your own observations like :class:`smarts.env.custom_observations`.
In :class:`smarts.env.custom_observations`, the custom observation's meaning is:

- "distance_from_center": distance to lane center 
- "angle_error": ego heading relative to the closest waypoint
- "speed": ego speed
- "steering": ego steering
- "ego_ttc": time to collision in each lane
- "ego_lane_dist": closest cars' distance to ego in each lane


Likewise with the action adapter

.. code-block:: python

    # this comes in from the output of the Agent
    def action_adapter(model_action):
        throttle, brake, steering = model_action
        return np.array([throttle, brake, steering])

    ACTION_SPACE = gym.spaces.Box(
        low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
    )

Because the reward is just a scalar value, no explicit specification of space is needed to go with the reward adapter. But the reward adapter is very important because it  allows further shaping of the reward to your liking:

.. code-block:: python

    def reward_adapter(env_obs, env_reward):
        return env_reward

Similarly, the info adapter allows further processing on the extra info, if you somehow need that.

.. code-block:: python

    def info_adapter(env_obs, env_reward, env_info):
        env_info[INFO_EXTRA_KEY] = "blah"
        return env_info

==================
Agent Observations
==================

Of all the information to work with it is useful to know a bit about the main agent observations in particular.

For that see the :ref:`observations` section for details.

