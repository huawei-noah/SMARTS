.. _configuration:

Configuration
=============

You can change the behavior of the underlying SMARTS engine.

Configuration of the engine can come from several sources. These locations take precedence as noted:

1. Individual ``SMARTS_`` prefixed environment variables (e.g. ``SMARTS_SENSOR_WORKER_COUNT``)
2. Local directory engine configuration (./smarts_engine.ini )
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