.. _driving_smarts_2022:

Driving SMARTS 2022
===================

The Driving SMARTS 2022 is a benchmark used in the
`NeurIPS 2022 Driving SMARTS <https://smarts-project.github.io/archive/2022_nips_driving_smarts/competition/>`_ competition.

Objective
---------

Objective is to develop a single policy capable of controlling single-agent or multi-agent to complete different driving scenarios in the ``driving-smarts-v2022`` environment.
Refer to :func:`~smarts.env.gymnasium.driving_smarts_2022_env.driving_smarts_2022_env` for environment details.

In each driving scenario, the ego-agents must drive towards their respective mission goal locations. Each agent's mission goal is given in the observation returned by the environment at each time step. 
The mission goal could be accessed as ``observation.ego_vehicle_state.mission.goal.position`` which gives an ``(x, y, z)`` map coordinate of the goal location.
Any method such as reinforcement learning, offline reinforcement learning, behaviour cloning, generative models, predictive models, etc, may be used to develop the policy.

The scenario names and their missions are as follows. The desired task execution is illustrated by a trained baseline agent, which uses PPO algorithm from `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_ reinforcement learning library.

+ 1_to_2lane_left_turn_c
    A single ego agent must make a left turn at an uprotected cross-junction.

    .. image:: /_static/driving_smarts_2022/intersection-c.gif
+ 1_to_2lane_left_turn_t 
    A single ego agent must make a left turn at an uprotected T-junction.
  
    .. image:: /_static/driving_smarts_2022/intersection-t.gif
+ 3lane_merge_multi_agent
    One ego agent must merge onto a 3-lane highway from an on-ramp, while another agent must drive on the highway yielding appropriately to traffic from the on-ramp.
  
    .. image:: /_static/driving_smarts_2022/merge-multi.gif
+ 3lane_merge_single_agent
    One ego agent must merge onto a 3-lane highway from an on-ramp.
  
    .. image:: /_static/driving_smarts_2022/merge-single.gif
+ 3lane_cruise_multi_agent
    Three ego agents must cruise along a 3-lane highway with traffic.
  
    .. image:: /_static/driving_smarts_2022/cruise-multi.gif
+ 3lane_cruise_single_agent
    One ego agents must cruise along a 3-lane highway with traffic.
  
    .. image:: /_static/driving_smarts_2022/cruise-single.gif
+ 3lane_cut_in
    One ego agent must navigate (either slow down or change lane) when its path is cut-in by another traffic vehicle.
  
    .. image:: /_static/driving_smarts_2022/cut-in.gif
+ 3lane_overtake
    One ego agent, while driving along a 3-lane highway, must overtake a column of slow moving traffic vehicles and return to the same lane which it started at.

Observation space
-----------------

The underlying environment returns raw observations of type :class:`~smarts.core.observations.Observation` at each time point for each ego agent.

Action space
------------

This benchmark allows ego agents to use any one of the following action spaces.

+ :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
+ :attr:`~smarts.core.controllers.ActionSpaceType.RelativeTargetPose`

Zoo agents
----------

See the list of :ref:`available zoo agents <available_zoo_agents>` which are compatible with this benchmark. 
A compatible zoo agent can be evaluated in this benchmark as follows.

.. code-block:: bash

    $ cd <path>/SMARTS
    $ scl zoo install <agent path>
    # e.g., scl zoo install zoo/policies/interaction_aware_motion_prediction
    $ scl benchmark run driving_smarts_2022==0.0 <agent_locator> --auto_install
    # e.g., scl benchmark run driving_smarts_2022==0.0 zoo.policies:interaction-aware-motion-prediction-agent-v0 --auto-install