.. _visualization:

Visualization
=============

========
Envision
========

SMARTS includes a visualizer called *Envision* that runs on a process separate from the main simulation process. See [./envision/README.md](./envision/README.md) for more information on Envision, our front-end visualization tool. To manage these processes we use the `scl run` command line to run SMARTS together with it's supporting processes.
To run the default example simply build a scenario and run the following command,

.. code-block:: bash
    
    # Build scenarios/sumo/loop
    scl scenario build --clean scenarios/sumo/loop

    # Run an experiment
    scl run --envision examples/single_agent.py scenarios/sumo/loop

You need to add the `--envision` flag to run the Envision server where you can see the visualization of the experiment.

Feel free to change the above commands as necessary (the first is to specify location of the scenarios to use so that it can be build to be used by the experiment later, the latter to specify the example which will be used to run the experiment on the built scenario).

To see the front-end visualization visit `http://localhost:8081/` in your browser. Select the simulator instance in the top left dropdown. If you are using SMARTS on a remote machine you will need to port forward 8081.

======
Visdom
======

Use the `Visdom <https://github.com/facebookresearch/visdom>`_ integration to easily see the image-based observation outputs in real-time. 
Start the visdom server before running the scenario and open the server URL in your browser `http://localhost:8097 <http://localhost:8097>`_.

.. code-block:: bash

    # Install visdom
    $ pip install visdom
    # Start the server
    $ visdom

Enable Visdom in the SMARTS environment by setting `visdom=True`. For example:

.. code-block:: python

    env = gym.make(
        "smarts.env:hiway-v0", # env entry name
        ...
        visdom=True, # whether or not to enable visdom visualization (see Appendix).
        ...
    )

Below is a sample visualization of an agent's camera sensor observations.

.. figure:: ../_static/visdom.gif

    (Left) Drivable area grid map. (Center) Occupancy grid map. (Right) Top-down RGB image.