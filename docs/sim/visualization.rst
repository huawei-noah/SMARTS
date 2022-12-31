.. _visualization:

Visualization
=============

Envision
--------

SMARTS includes a front-end visualization tool called *Envision* that provides real-time view of the environment state.
Envision is built on web-technologies (including `React <https://reactjs.org/>`_, `WebGL <https://www.khronos.org/webgl/>`_, and `websockets <https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API`_) allowing it to run easily in the browser. 
It is composed of a few parts: a client which SMARTS uses directly; a server used for state broadcasting; and the web application where all the visualization and rendering happens.

Run
^^^

An example is shown below to run SMARTS with Envision.

.. code-block:: bash
    
    $ cd <path>/SMARTS
    # Build scenarios/sumo/loop
    $ scl scenario build --clean scenarios/sumo/loop
    # Run the single_agent.py example with the loop scenario
    $ scl run --envision examples/control/single_agent.py scenarios/sumo/loop

``--envision`` flag is added to ``scl run`` to enable the Envision server. Visit `http://localhost:8081/ <http://localhost:8081/>`_ in your browser to see the environment visualization. Select the simulator instance in the top left dropdown. If you are using SMARTS on a remote machine you will need to forward port 8081.

Data Recording and Replay
^^^^^^^^^^^^^^^^^^^^^^^^^

For recording simply add ``envision_record_data_replay_path`` to the `gym.make(...)` call,

.. code-block:: python

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agents={AGENT_ID: agent},
        headless=args.headless,
        visdom=False,
        fixed_timestep_sec=0.1,
        envision_record_data_replay_path="./data_replay",
    )

then run with `scl run --envision <examples/script_path> <scenarios/path>` (currently Envision server needs to be up for data recording to work).

For replay make sure you have Envision server running then use the following tool - passing in your replay files,

.. code-block:: bash

    $ scl scenario replay -d ./data_replay/1590892375a -t 0.1

    INFO:root:Replaying 1 record(s) at path=data_replay/1590892375a with timestep=0.1s

Development
^^^^^^^^^^^

To contribute to Envision it's easiest to start and control the processes manually. Start the Envision server by running,

.. code-block:: bash

    $ cd <path>/SMARTS
    # Runs on port 8081 by default
    $ python3 envision/server.py --debug


Then start the Envision web application. npm (version >= 6) and node (version >= 12) are required.

.. code-block:: bash

    $ cd envision/web
    # Install dependencies
    $ npm install
    # Build, run dev server, and watch code changes
    $ npm run watch

This development flow currently requires reloading the webpage after update.

Deployment
^^^^^^^^^^

If you've made changes to the Envision web application you'll want to save an updated distribution which users access directly (so they don't have to setup Envision's development dependencies). Simply run,

.. code-block:: bash

    # Saves to envision/web/dist
    $ npm run build

Visdom
------

Use the `Visdom <https://github.com/facebookresearch/visdom>`_ integration to easily see the image-based observation outputs in real-time. 
Start the visdom server before running the scenario and open the server URL in your browser `http://localhost:8097 <http://localhost:8097>`_.

.. code-block:: bash

    # Install visdom
    $ pip install visdom
    # Start the server
    $ visdom

Enable Visdom in the SMARTS environment by setting ``visdom=True``. For example:

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