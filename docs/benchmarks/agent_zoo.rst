.. _agent_zoo:

Agent Zoo
=========

Numerous prebuilt and trained agents are hosted at `zoo/policies <https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies>`_ . 
These agents should be installed prior to being used in SMARTS environments or benchmarks.

.. code-block:: bash

    $ cd <path>/SMARTS
    # Install a zoo agent.
    $ scl zoo install <agent path>
    # e.g., scl zoo install zoo/policies/interaction_aware_motion_prediction

.. note::

    To build a wheel, execute:

    .. code-block:: bash

        $ scl zoo build <agent path>
        # e.g., scl zoo build zoo/policies/interaction_aware_motion_prediction

Contribute agents
-----------------

.. note::
    This section is only for contributing to the agent zoo.

First, add the new agent to `zoo/policies <https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies>`_. It should contain 

* package setup file,
* inference code with prebuilt model, and
* algorithm explanation.

Then, register the newly added zoo agent in `zoo/policies/__init__.py <https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies/__init__.py>`_.

Available zoo agents
--------------------

.. _available_zoo_agents:
.. list-table::
   :header-rows: 1

   * - Agent locator and path
     - Benchmark or Env
     - Action space
     - Source
     - Remarks
   * - | zoo.policies:interaction-aware-motion-prediction-agent-v0
       | zoo/policies/interaction_aware_motion_prediction
     - driving_smarts==0.0
     - :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
     - `code <https://github.com/smarts-project/smarts-project.rl/tree/master/interaction_aware_motion_prediction>`__
     - Contributed as part of `NeurIPS 2022 Driving SMARTS <https://smarts-project.github.io/archive/2022_nips_driving_smarts/>`__ competition.
   * - | zoo.policies:control-and-supervised-learning-agent-v0
       | zoo/policies/control_and_supervised_learning
     - driving_smarts==0.0
     - :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
     - `code <https://github.com/smarts-project/smarts-project.rl/tree/master/control_and_supervised_learning>`__
     - Contributed as part of `NeurIPS 2022 Driving SMARTS <https://smarts-project.github.io/archive/2022_nips_driving_smarts/>`__ competition.
   * - | zoo.policies:discrete-soft-actor-critic-agent-v0
       | zoo/policies/discrete_soft_actor_critic
     - driving_smarts==0.0
     - :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
     - `code <https://github.com/smarts-project/smarts-project.rl/tree/master/discrete_soft_actor_critic>`__
     - Contributed as part of `NeurIPS 2022 Driving SMARTS <https://smarts-project.github.io/archive/2022_nips_driving_smarts/>`__ competition.
