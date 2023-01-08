.. _quickstart:

Quickstart
==========

A typical workflow would look like this.

1. Design and build a scenario. 
   
   + Further reading: :ref:`scenario_studio` details the scenario design process. 
2. Build an agent by specifying its `interface` and `policy`.

   + Further reading: :ref:`agent` details the agent build process. 
3. Instantiate and run a SMARTS environment.

In this quickstart guide, we will run the ``chase_via_points.py`` example. Here,

1. a pre-designed scenario ``SMARTS/scenarios/sumo/loop`` is used.
2. a simple agent with `interface` == :attr:`~smarts.core.agent_interface.AgentType.LanerWithSpeed` and `policy` == `Chase Via Points` is demonstrated. The agent chases via points or follows nearby waypoints if a via point is unavailable.

Example
-------

File: ``SMARTS/examples/control/chase_via_points.py``

.. literalinclude:: ../examples/control/chase_via_points.py
    :language: python

Use the `scl` command to run SMARTS together with it's supporting processes. 

.. code-block:: bash

    $ cd <path>/SMARTS
    # Build the scenario `scenarios/sumo/loop`.
    $ scl scenario build scenarios/sumo/loop
    # Run SMARTS simulation with Envision display and `loop` scenario.
    $ scl run --envision examples/control/chase_via_points.py scenarios/sumo/loop 

Visit `http://localhost:8081/ <http://localhost:8081/>`_ to view the experiment.

The ``--envision`` flag runs the Envision server which displays the simulation. Refer to :ref:`visualization` for more information on Envision.

Other examples
^^^^^^^^^^^^^^

More examples on running an 
(i) egoless script, 
(ii) control  
are provided in the :ref:`examples <examples>` folder. Refer 
A handful of pre-built scenarios are available at :scenarios:`scenarios <>` folder. 

The `obs` argument to `ExampleAgent.act()` will contain the observations specified in the `AgentInterface` above, and it's expected that the return value of the `act` method matches the `ActionSpaceType` chosen as well.
