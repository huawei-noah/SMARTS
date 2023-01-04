.. _agent:

Building an Agent
=================

SMARTS provides users the ability to customize their agents. :class:`smarts.zoo.agent_spec.AgentSpec` has the following fields:

.. code-block:: python

    class AgentSpec:
        interface: AgentInterface
        agent_builder: Callable[..., Agent] = None
        agent_params: Optional[Any] = None

An example of how to create an ``Agent`` instance is shown below.

.. code-block:: python

    AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )

    agent = agent_spec.build_agent()

We will further explain the fields of the `Agent` class later on this page. You can also read the source code at :class:`smarts.env.agent`.


AgentInterface
--------------

:class:`smarts.core.agent_interface.AgentInterface` regulates the flow of information between the agent and a SMARTS environment. 
It specifies the observations the agent expects to receive from the environment and the action the agent provides to the environment.
To create an agent interface:

.. code-block:: python

    agent_interface = AgentInterface.from_type(
        interface = AgentType.Standard,
        max_episode_steps = 1000, 
        ...
    )

SMARTS provides several pre-designed agent interfaces. Their interface options are shown in the table below.

+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
|                        | :attr:`smarts.core.agent_interface.AgentType.Full`         | :attr:`smarts.core.agent_interface.AgentType.StandardWithAbsoluteSteering` | :attr:`smarts.core.agent_interface.AgentType.Standard`          | :attr:`smarts.core.agent_interface.AgentType.Laner`  | :attr:`smarts.core.agent_interface.AgentType.LanerWithSpeed`            | :attr:`smarts.core.agent_interface.AgentType.Tracker`      | :attr:`smarts.core.agent_interface.AgentType.TrajectoryInterpolator` | :attr:`smarts.core.agent_interface.AgentType.MPCTracker` | :attr:`smarts.core.agent_interface.AgentType.Boid`              | :attr:`smarts.core.agent_interface.AgentType.Loner`        | :attr:`smarts.core.agent_interface.AgentType.Tagger`       | :attr:`smarts.core.agent_interface.AgentType.Direct`   |
+========================+============================================================+============================================================================+=================================================================+======================================================+=========================================================================+============================================================+======================================================================+==========================================================+=================================================================+============================================================+============================================================+========================================================+
| action                 | :attr:`smarts.core.controllers.ActionSpaceType.Continuous` | :attr:`smarts.core.controllers.ActionSpaceType.Continuous`                 | :attr:`smarts.core.controllers.ActionSpaceType.ActuatorDynamic` | :attr:`smarts.core.controllers.ActionSpaceType.Lane` | :attr:`smarts.core.controllers.ActionSpaceType.LaneWithContinuousSpeed` | :attr:`smarts.core.controllers.ActionSpaceType.Trajectory` | :attr:`smarts.core.controllers.ActionSpaceType.TrajectoryWithTime`   | :attr:`smarts.core.controllers.ActionSpaceType.MPC`      | :attr:`smarts.core.controllers.ActionSpaceType.MultiTargetPose` | :attr:`smarts.core.controllers.ActionSpaceType.Continuous` | :attr:`smarts.core.controllers.ActionSpaceType.Continuous` | :attr:`smarts.core.controllers.ActionSpaceType.Direct` |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| max_episode_steps      | ✓                                                          | ✓                                                                          | ✓                                                               | ✓                                                    | ✓                                                                       | ✓                                                          | ✓                                                                    | ✓                                                        | ✓                                                               | ✓                                                          | ✓                                                          | ✓                                                      |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| neighborhood_vehicles  | ✓                                                          | ✓                                                                          | ✓                                                               |                                                      |                                                                         |                                                            |                                                                      |                                                          | ✓                                                               |                                                            | ✓                                                          | ✓                                                      |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| waypoints              | ✓                                                          | ✓                                                                          | ✓                                                               | ✓                                                    | ✓                                                                       | ✓                                                          |                                                                      | ✓                                                        | ✓                                                               | ✓                                                          | ✓                                                          |                                                        |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| drivable_area_grid_map | ✓                                                          |                                                                            |                                                                 |                                                      |                                                                         |                                                            |                                                                      |                                                          |                                                                 |                                                            |                                                            |                                                        |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| ogm                    | ✓                                                          |                                                                            |                                                                 |                                                      |                                                                         |                                                            |                                                                      |                                                          |                                                                 |                                                            |                                                            |                                                        |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| rgb                    | ✓                                                          |                                                                            |                                                                 |                                                      |                                                                         |                                                            |                                                                      |                                                          |                                                                 |                                                            |                                                            |                                                        |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| lidar                  | ✓                                                          |                                                                            |                                                                 |                                                      |                                                                         |                                                            |                                                                      |                                                          |                                                                 |                                                            |                                                            |                                                        |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| accelerometer          | ✓                                                          | ✓                                                                          | ✓                                                               | ✓                                                    | ✓                                                                       | ✓                                                          | ✓                                                                    | ✓                                                        | ✓                                                               | ✓                                                          | ✓                                                          | ✓                                                      |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| signals                | ✓                                                          |                                                                            |                                                                 |                                                      |                                                                         |                                                            |                                                                      |                                                          |                                                                 |                                                            |                                                            | ✓                                                      |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+
| debug                  | ✓                                                          | ✓                                                                          | ✓                                                               | ✓                                                    | ✓                                                                       | ✓                                                          | ✓                                                                    | ✓                                                        | ✓                                                               | ✓                                                          | ✓                                                          | ✓                                                      |
+------------------------+------------------------------------------------------------+----------------------------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------+-----------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------+--------------------------------------------------------+


``max_episode_steps`` controls the max running steps allowed for the agent in an episode. The default ``None`` setting means agents have no such limit.
You can move ``max_episode_steps`` control authority to RLlib with their config option ``horizon``, but lose the ability to customize
different max_episode_len for each agent.

``action`` controls the agent action type used. There are multiple ``ActionSpaceType`` options:


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

.. important::

    IMPORTANT: The generation of a DrivableAreaGridMap (``drivable_area_grid_map=True``), OGM (``ogm=True``) and/or RGB (``rgb=True``) images may significantly slow down the environment ``step()``. If your model does not consume such observations, we recommend that you set them to ``False``.

IMPORTANT: Depending on how your agent model is set up, ``ActionSpaceType.ActuatorDynamic`` might allow the agent to learn faster than ``ActionSpaceType.Continuous`` simply because learning to correct steering could be simpler than learning a mapping to all the absolute steering angle values. But, again, it also depends on the design of your agent model. 

Agent
-----

An agent maps an observation to an action.

.. code-block:: python

    # A simple agent that ignores observations
    class IgnoreObservationsAgent(Agent):
        def act(self, obs):
            return [throttle, brake, steering_rate]

The observation passed in should be the observations that a given agent sees. In **continuous action space** the action is expected to produce values for ``throttle`` [0,1], ``brake`` [0,1], and ``steering_rate`` [-1,1].

Otherwise, only while using **lane action space**, the agent is expected to return a lane related command: ``"keep_lane"``, ``"slow_down"``, ``"change_lane_left"``, ``"change_lane_right"``.

Another example:

.. literalinclude:: ../minimal_agent.py
   :language: python

Spaces
------

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
:class:`smarts.env.custom_observations`.