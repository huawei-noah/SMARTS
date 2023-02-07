.. _agent_zoo:

Agent Zoo
=========

Contributed agents are hosted at `Agent Zoo <https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies>`_ .

These agents should be installed prior to being used in SMARTS environments.

.. code-block:: bash

    $ cd <path>/SMARTS
    # Install a zoo agent.
    $ scl zoo install <agent>
    # e.g. scl zoo install zoo/policies/interaction_aware_motion_prediction

.. note::

    To build a wheel and install a zoo agent, execute:

    .. code-block:: bash

        $ scl zoo build <agent>
        # e.g. scl zoo build zoo/policies/interaction_aware_motion_prediction

Agents
------

#. Interaction-aware motion prediction agent 
    * This agent was contributed as part of NeurIPS2022 Driving SMARTS competition.
    * Agent `code <https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies/interaction_aware_motion_prediction>`_.
    * Run as follows:
    
        .. code-block:: bash

            $ cd <path>/SMARTS
            # Install a zoo agent.
            $ scl zoo install zoo/policies/interaction_aware_motion_prediction
            $ scl benchmark run driving_smarts==0.0 "./baselines/driving_smarts/v0/interaction_aware_motion_prediction.yaml" --auto-install 

Contribute agents
-----------------

.. note::
    This section is only for contributers to the agent zoo.

First, add the new agent to `Agent Zoo <https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies>`_. It should contain 

* package setup file,
* inference code with trained model, and
* algorithm explanation. 

Then, register the newly added zoo agent in ``SMARTS/zoo/policies/__init__.py``.