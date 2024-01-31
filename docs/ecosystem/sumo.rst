.. _sumo:

SUMO
====

Learn SUMO through its user `documentation <https://sumo.dlr.de/docs/index.html>`_ . 

SMARTS currently directly installs SUMO version >=1.15.0 via `pip`. 

.. code-block:: bash

    $ pip install eclipse-sumo

Alternative installation methods, albeit more difficult, are described below.

Centralized TraCI management
----------------------------
.. _centralized_traci_management:

With the default behaviour each SMARTS instance will attempt to ask the operating system
    for a port to generate a ``TraCI`` server on which can result in cross-connection of SMARTS and ``TraCI`` server instances.

.. code-block:: bash

    ## console 1 (or in background OR on remote machine)
    # Run the centralized sumo port management server.
    # Use `export SMARTS_SUMO_CENTRAL_PORT=62232` or `--port=62232`
    $ python -m smarts.core.utils.centralized_traci_server

By setting ``SMARTS_SUMO_TRACI_SERVE_MODE`` to ``"central"`` SMARTS will use the ``TraCI`` management server.

.. code-block:: bash

    ## console 2
    ## Set environment variable to switch to the server.
    # This can also be set in the engine configuration.
    $ export SMARTS_SUMO_TRACI_SERVE_MODE=central
    ## Optional configuration
    # export SMARTS_SUMO_CENTRAL_HOST=localhost
    # export SMARTS_SUMO_CENTRAL_PORT=62232
    ## do run
    $ python experiment.py


Package managers
----------------

Instructions for installation from package managers are available at `https://sumo.dlr.de/docs/Installing/ <https://sumo.dlr.de/docs/Installing/>`_ .

Build from source
-----------------

If you wish to compile SUMO by yourself from source

+ Git repository can be found at `https://github.com/eclipse/sumo <https://github.com/eclipse/sumo>`_ .
+ Use SUMO version `1.7.0 <https://github.com/eclipse-sumo/sumo/commits/v1_7_0>`_ or higher.
+ Build instructions are available at `https://sumo.dlr.de/docs/Developer/index.html#build_instructions <https://sumo.dlr.de/docs/Developer/index.html#build_instructions>`_ . 
+ Please note that building SUMO might not install all other vital dependencies that SUMO requires to run.