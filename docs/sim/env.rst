.. _environment:

Environment
===========

Base environments
-----------------

SMARTS environment module is defined in :mod:`~smarts.env` package. Currently SMARTS provides two kinds of training 
environments, namely:

+ ``HiWayEnv`` utilising ``gym.env`` style interface 
+ ``RLlibHiwayEnv`` customized for `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_ training

.. image:: ../_static/env.png

HiWayEnv
^^^^^^^^

``HiWayEnv`` inherits class ``gym.Env`` and supports gym APIs like ``reset``, ``step``, ``close``. An usage example is shown below.
Refer to :class:`~smarts.env.hiway_env.HiWayEnv` for more details.

.. code-block:: python

    # Make env
    env = gym.make(
            "smarts.env:hiway-v0", # Env entry name.
            scenarios=[scenario_path], # List of paths to scenario folders.
            agent_interfaces={AGENT_ID: agent_spec.interface}, # Dictionary mapping agents to agent interfaces.
            headless=False, # False to enable Envision visualization of the environment.
            seed=42, # RNG seed. Seeds are set at the start of simulation, and never automatically re-seeded.
        )

    # Reset env and build agent.
    observations = env.reset()
    agent = agent_spec.build_agent()

    # Step env.
    agent_obs = observations[AGENT_ID]
    agent_action = agent.act(agent_obs)
    observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})

    # Close env.
    env.close()

HiWayEnvV1
^^^^^^^^^^

``HiWayEnvV1`` inherits class ``gymnasium.Env`` and supports gym APIs like ``reset``, ``step``, ``close``. An usage example is shown below.
This version has two configurations of observation output: `ObservationOptions.full` which provides padded agents in the observations which
exactly matches the `env.observation_space`, and `ObservationOptions.multi_agent` which provides only agents as are currently active. Refer to
:class:`~smarts.env.gymnasium.hiway_env_v1.HiWayEnvV1` for more details.

.. code-block:: python

    # Make env
    env = gym.make(
            "smarts.env:hiway-v1", # Env entry name.
            scenarios=[scenario_path], # List of paths to scenario folders.
            agent_interfaces={AGENT_ID: agent_spec.interface}, # Dictionary mapping agents to agent interfaces.
            headless=False, # False to enable Envision visualization of the environment.
            seed=42, # RNG seed. Seeds are set at the start of simulation, and never automatically re-seeded.
            observation_options=ObservationOptions.multi_agent, # Configures between padded and un-padded agents in observations.
        )

    # Reset env and build agent.
    ## `reset(options) forwards "scenario" and "start_time" to `smarts.reset()`
    observations, infos = env.reset() 
    agent = agent_spec.build_agent()

    # Step env.
    agent_obs = observations[AGENT_ID]
    agent_action = agent.act(agent_obs)
    observations, rewards, terminated, truncated, infos = env.step({AGENT_ID: agent_action})

    # Close env.
    env.close()

To use this environment with certain frameworks you may want to convert the environment back into a 0.21 api version of gym.
This can be done with :class:`~smarts.env.gymnasium.wrappers.api_reversion.Api021Reversion`.

.. code-block:: python

    # Make env
    env = gym.make(
        "smarts.env:hiway-v1", # Env entry name.
        scenarios=[scenario_path], # List of paths to scenario folders.
    )
    env = Api021Reversion(env) # Turns the environment into roughly a 0.21 gym environment

RLlibHiwayEnv
^^^^^^^^^^^^^

``RLlibHiwayEnv`` inherits class ``MultiAgentEnv``, which is defined in RLlib. It also supports common env APIs like ``reset``, 
``step``, ``close``. An usage example is shown below. Refer to :class:`~smarts.env.rllib_hiway_env.RLlibHiWayEnv` for more details.

.. code-block:: python

    from smarts.env.rllib_hiway_env import RLlibHiWayEnv
    env = RLlibHiWayEnv(
        config={
            "scenarios": [scenario_path], # List of paths to scenario folders.
            "agent_specs": {AGENT_ID: agent_spec}, # Dictionary mapping agents to agent specs.
            "headless": False, # False to enable Envision visualization of the environment.
            "seed": 42, # RNG seed. Seeds are set at the start of simulation, and never automatically re-seeded.
        }
    )

    # Reset env and build agent.
    observations = env.reset()
    agent = agent_spec.build_agent()

    # Step env.
    agent_obs = observations[AGENT_ID]
    agent_action = agent.act(agent_obs)
    observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})

    # Close env.
    env.close()

Features
--------

Vehicle Diversity
^^^^^^^^^^^^^^^^^

SMARTS environments support the following types of vehicles:

+ **ego agents** - controlled by RL policy currently in training.
+ **social agents** - controlled by (pre-trained) policies from the Agent Zoo (see :mod:`~zoo.policies`). Like ego agents, social agents also use :class:`~smarts.zoo.agent_spec.AgentSpec` to register with the environment. They interact by watching the observation and returning action messages. Compared to ego agents, social agents are driven by trained models, hence they can provide behavioral characteristics we want.
+ **traffic vehicles** - controlled by an underlying traffic engine, like ``SUMO`` or ``SMARTS``.
+ **dataset vehicles** - controlled by replay of traffic history from naturalistic datasets such as Argoverse, NGSIM, and Waymo.

Refer to :ref:`scenario_studio` for designing scenarios, traffic vehicles, social agents, ego agents, and maps.

Multiagent Scenario
^^^^^^^^^^^^^^^^^^^

In multiagent scenarios, the agents may start and end at different time points
in the simulation. Consider the following multiagent scenario with 3 agents, 
namely ``Agent_0``, ``Agent_1``, and ``Agent_2``.

.. list-table::
   :header-rows: 1

   * - Time (s)
     - 0 - 10
     - 11 - 20
     - 21 - 30
     - 31 - 40
   * - Active agents
     - Agent_1, Agent_2
     - Agent_1
     - None
     - Agent_0
   * - Observation.keys()
     - Agent_1, Agent_2
     - Agent_1
     - None
     - Agent_0

Here, ``Agent_1`` and ``Agent_2`` start at time 0s, whereas ``Agent_0`` starts
at time 31s. ``Agent_2``, ``Agent_1``, and ``Agent_0`` become done at time 10s,
20s, and 40s, respectively. There could also be periods of time when there are
no active agents such as between time 21s and time 30s. For time 21s to 30s, 
SMARTS simply returns an empty dictionary as observation and the environment
has not finished yet because ``Agent_0`` is yet to become done.

Scenario Iterator
^^^^^^^^^^^^^^^^^

If (i) a list of scenarios, or (ii) a folder containing multiple scenarios, is passed through the environment arguments, then SMARTS automatically iterates over those scenarios. The next scenario is loaded after each ``env.reset()`` call. This feature is especially useful for training on multiple maps.

Moreover, if there are **n** routes file in ``scenario1/build/traffic`` dir, then each routes file will be combined with the map to form a scenario, leading to a total of **n** concrete scenarios (i.e., traffic-map combination) that SMARTS automatically iterates through for ``scenario1``. See :class:`~smarts.core.scenario.Scenario` for implementation details.

.. code-block:: python

    tune_config = {
        "env": RLlibHiwayEnv,
        "env_config": {
            "seed": tune.randint(1000),
            "scenarios": [scenario1, scenario2, ...],
            "headless": args.headless,
            "agent_specs": agent_specs,
        },
        ...
    }

In contrast to the above case, we can also use multiple maps for *different workers* in RLlib as follows.

.. code-block:: python

    tracks_dir = [scenario1, scenario2, ...]

    class MultiEnv(RLlibHiWayEnv):
        def __init__(self, env_config):
            env_config["sumo_scenarios"] = [tracks_dir[(env_config.worker_index - 1)]]
            super(MultiEnv, self).__init__(config=env_config)

    tune_config = {
        "env": MultiEnv,
        "env_config": {
            "seed": tune.randint(1000),
            "scenarios": tracks_dir,
            "headless": args.headless,
            "agent_specs": agent_specs,
        },
        ...
    }

.. note::

    The above two cases of scenario iteration are different. In the first case, samples are collected from different scenarios *across time*, but in the second case different workers collect samples from different scenarios *simultaneously* thanks to distributed computing of multiple workers.
    This means that in the first case, the agents get experiences from the same scenario, whereas in the second case, the agents get a mixture of experiences from different scenarios.

Determinism
^^^^^^^^^^^

SMARTS simulation is deterministic. Assuming all ego and social agents produce deterministic action, then the entire simulation will play back deterministically when repeated.
