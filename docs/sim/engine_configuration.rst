.. _engine_configuration:

Configuration
=============


SMARTS
------

You can change the behavior of the underlying SMARTS engine.

Configuration of the engine can come from several sources. These locations take precedence as noted:

1. Individual ``SMARTS_`` prefixed environment variables (e.g. ``SMARTS_SENSOR_WORKER_COUNT``)
2. Local directory engine configuration (``./smarts_engine.ini``)
3. Local user engine configuration, ``~/.smarts/engine.ini``, if local directory configuration is not found.
4. Global engine configuration, ``/etc/smarts/engine.ini``, if local configuration is not found.
5. Package default configuration, ``$PYTHON_PATH/smarts/engine.ini``, if global configuration is not found.

Note that configuration files resolve all settings at the first found configuration file (they do not layer.)


Options
-------

All settings demonstrated as environment variables are formatted to ``UPPERCASE`` and prefixed with ``SMARTS_`` 
 (e.g. ``[core] logging`` can be configured with ``SMARTS_CORE_LOGGING``).

Below is a comparison of valid approaches to changing an engine configuration value:

.. code:: ini

    ; Example configuration file
    ; For syntax see https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
    [assets]
    default_agent_vehicle = passenger


.. code:: bash

    $ # Another way to apply the value
    $ export SMARTS_ASSETS_DEFAULT_AGENT_VEHICLE=passenger


YAML resources
--------------
    
YAML files resolve as `PyYAML.safe_load() <https://pyyaml.org/wiki/PyYAMLDocumentation>` allows with a few extensions.

Dynamic module resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

The benchmark configuration directory can be dynamically found through
python using an evaluation syntax ``${{}}``. This is experimental and
open to change but the following resolves the python module location in
loaded configuration files:

.. code:: yaml

    somewhere_path: ${{module.to.resolve}}/file.txt # resolves to <path>/module/to/resolve/file.txt


This avoids loading the module into python but resolves to the first
path that matches the module.

Environment variable resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Resolving SMARTS engine configuration.

The engine configuration resolves by referencing the setting through
the evaluation syntax ``${}``. This is restricted to ``"SMARTS_"``
prefixed environment variables.

.. code:: yaml

    is_debug: ${SMARTS_CORE_DEBUG} # literal environment variable or engine setting `[core] debug`


Engine settings
---------------

The current list of engine settings are as follows:

.. list-table::
   :header-rows: 1

   * - **Setting**
     - ``SMARTS_ASSETS_PATH``
     - ``SMARTS_ASSETS_DEFAULT_AGENT_VEHICLE``
     - ``SMARTS_ASSETS_DEFAULT_AGENT_DEFINITIONS_LIST``
     - ``SMARTS_CORE_DEBUG``
     - ``SMARTS_CORE_MAX_CUSTOM_IMAGE_SENSORS``
     - ``SMARTS_CORE_OBSERVATION_WORKERS``
     - ``SMARTS_CORE_RESET_RETRIES``
     - ``SMARTS_CORE_SENSOR_PARALLELIZATION``
     - ``SMARTS_PHYSICS_MAX_PYBULLET_FREQ``
     - ``SMARTS_RAY_NUM_CPUS``
     - ``SMARTS_RAY_NUM_GPUS``
     - ``SMARTS_RAY_LOG_TO_DRIVER``
     - ``SMARTS_SUMO_SERVER_HOST``
     - ``SMARTS_SUMO_SERVE_MODE``
     - ``SMARTS_SUMO_SERVER_PORT``
     - ``SMARTS_VISDOM_ENABLED``
     - ``SMARTS_VISDOM_HOSTNAME``
     - ``SMARTS_VISDOM_PORT``
   * - **Section**
     - assets
     - assets
     - assets
     - core
     - core
     - core
     - core
     - core
     - physics
     - ray
     - ray
     - ray
     - sumo
     - sumo
     - sumo
     - visdom
     - visdom
     - visdom
   * - **Type**
     - string
     - string
     - string
     - boolean
     - integer
     - integer
     - integer
     - string
     - integer
     - integer|``None``
     - integer|``None``
     - boolean
     - string
     - string
     - integer
     - boolean
     - string
     - integer
   * - **Default**
     - ``"<SMARTS>/assets"``
     - ``"sedan"``
     - ``"<SMARTS>/assets/vehicles/vehicle_definitions_list.yaml"``
     - ``False``
     - 32
     - 0
     - 0
     - ``"mp"``
     - 240
     - ``None``
     - 0
     - ``False``
     - 8619
     - ``"localhost"``
     - ``"local"``
     - False
     - ``"http://localhost"``
     - 8097
   * - **Values**
     - Any existing path (not recommended to change)
     - Any defined vehicle name.
     - Any existing ``YAML`` file.
     - True|False
     - 0 or greater
     - 0 or greater (0 disables parallelization)
     - 0 or greater
     - [``"mp"`` ``"ray"``]
     - 1 or greater (240 highly recommended)
     - 0 or greater | None
     - 0 or greater | None
     - True|False
     - [``"localhost"``  ``"x.x.x.x"``  ``"https://..."``]
     - [``"local"``  ``"remote"``]
     - As dictated by OS.
     - True|False
     - [``localhost`` ``"x.x.x.x"`` ``"http://..."``]
     - As dictated by OS.
   * - **Description**
     - The path to SMARTS package assets.
     - This uses a vehicle from those defined in the ``SMARTS_ASSETS_DEFAULT_AGENT_DEFINITIONS_LIST`` file.
     - The path to a vehicle definition file. See :ref:`vehicle defaults <vehicle_defaults>` for more information.
     - Enables additional debugging information from SMARTS.
     - Reserves that number of custom image sensors for an individual vehicle.
     - Determines how many workers SMARTS will use when generating observations. 0 disables parallelization.
     - Increasing this value gives more attempts for SMARTS to reset to a valid initial state. This can be used to bypass edge case engine errors.
     - Selects the parallelization backing for SMARTS sensors and observation generation. ``"mp"`` uses python's inbuilt ``"multiprocessing"`` library and ``"ray"`` uses `ray <https://docs.ray.io>`.
     - **WARNING** change at peril. Configures pybullet's frequency.
     - Configures how many CPU's that ``ray`` will use.
     - Configures how many GPU's that ``ray`` will use.
     - Enables ``ray`` log debugging.
     - If ``SMARTS_SUMO_SERVE_MODE=remote``, the host name of the remote ``TraCI`` management server host.
     - If ``SMARTS_SUMO_SERVE_MODE=remote``, the port that the ``TraCI`` management server communicates on.
     - The ``TraCI`` server spin-up mode to use. ``"local"`` generates the ``TraCI`` server from the local process. ``"remote"`` uses an intermediary server to generate ``TraCI`` servers and prevent race conditions between process connections.
     - If to enable `visdom <https://github.com/fossasia/visdom>`_ visualization.
     - The host name for the ``visdom`` instance.
     - The port of the ``visdom`` instance.

