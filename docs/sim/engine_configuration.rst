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

All settings demonstrated as environment variables are formatted to ``UPPERCASE`` and prefixed with ``SMARTS_`` (e.g. ``[core] logging`` can be configured with ``SMARTS_CORE_LOGGING``)

These settings are as follows:

.. todo::

    List engine settings


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

