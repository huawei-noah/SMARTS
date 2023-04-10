.. _driving_smarts_2023_3:

Driving SMARTS 2023.3
=====================

Objective
---------

Objective is to develop a single-ego policy capable of controlling a single ego to perform a platooning task in the 
``platoon-v0`` environment. Refer to :func:`~smarts.env.gymnasium.platoon_env.platoon_env` for environment details. 

.. important::

    In a scenario with multiple egos, a single-ego policy is copied and pasted into every ego. Each ego is stepped 
    independently by calling their respective :attr:`~smarts.core.agent.Agent.act` function. In short, multiple
    egos are executed in a distributed manner. The single-ego policy should be capable of accounting for and 
    interacting with other egos, if any are present.

Each ego is supposed to track and follow its specified leader (i.e., lead vehicle) in a single file or in a 
platoon fashion. Name of the lead vehicle to be followed is given to the ego through its 
:attr:`~smarts.core.agent_interface.ActorsAliveDoneCriteria.actors_of_interest` attribute.

The episode ends for an ego when its assigned leader reaches the leader's destination. Egos do not have prior 
knowledge of the leader's destination.

Any method such as reinforcement learning, offline reinforcement learning, behaviour cloning, generative models,
predictive models, etc, may be used to develop the policy.

Several scenarios are provided for training. Their names and tasks are as follows. 
The desired task execution is illustrated by a trained baseline agent, which uses PPO algorithm from 
`Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_ reinforcement learning library.

+ straight_2lane_agents_1
    A single ego must follow a specified leader, with no background traffic.

    .. image:: /_static/driving_smarts_2023/platoon_straight_2lane_agents_1.gif

Observation space
-----------------

The underlying environment returns formatted :class:`~smarts.core.observations.Observation` using 
:attr:`~smarts.env.utils.observation_conversion.ObservationOptions.multi_agent`
option as observation at each time point. See 
:class:`~smarts.env.utils.observation_conversion.ObservationSpacesFormatter` for
a sample formatted observation data structure.

Action space
------------

Action space for each ego agent is :attr:`~smarts.core.controllers.ActionSpaceType.Continuous`.

Code structure
--------------

Users are free to use any training method and any folder structure for training the policy.

Only the inference code is required for evaluation, and therefore must follow the following 
folder structure, naming system, and contents.

.. code-block:: text

    inference                    # Main folder.
    ├── contrib_policy           # Contains code to train a model offline.
    │   ├── __init__.py          # Primary training script for training a new model.
    │   ├── policy.py            # Other necessary training files.
    |   .
    |   .
    |   .
    ├── submission                       
    |    ├── policy.py            # A policy with an act method, wrapping the saved model.
    |    ├── requirements.txt     # Dependencies needed to run the model.
    |    ├── explanation.md       # Brief explanation of the key techniques used in developing the submitted model.
    |    ├── ...                  # Other necessary files for inference.
    |    .
    |    .
    ├── __init__.py               # Dockerfile to build and run the training code.
    ├── MANIFEST.in               # Dockerfile to build and run the training code.
    ├── setup.cfg                 # Brief explanation of the key techniques used in developing the submitted model.
    └── setup.py                  # Other necessary files for inference.

policy.py
    Policy(Agent)

__init__.py
    Agent interface:
    Using the input argument agent_interface, users may configure all the fields of 
    :class:`~smarts.core.agent_interface.AgentInterface` except for the (i) accelerometer, 
    (ii) done_criteria, (iii) max_episode_steps, (iv) neighborhood_vehicle_states, and 
    (v) waypoint_paths. 

Example
-------

See the list of :ref:`available zoo agents <available_zoo_agents>` which are compatible with this benchmark. A compatible zoo agent can be run as follows.

.. code-block:: bash

    $ cd <path>/SMARTS
    $ scl zoo install <agent path>
    # e.g., scl zoo install zoo/policies/interaction_aware_motion_prediction
    $ scl benchmark run driving_smarts_2023.3==0.0 <agent_locator> --auto_install
    # e.g., scl benchmark run driving_smarts_2023.3==0.0 zoo.policies:interaction-aware-motion-prediction-agent-v0 --auto-install
