.. _vehicle_defaults:


Vehicle defaults
================

``SMARTS`` provides vehicle configuration for agent control.


Default agent vehicle details
-----------------------------

.. list-table::
   :header-rows: 1

   * - **Vehicle**
     - Sedan (Default)
     - Bus (class 4)
     - Pickup truck (class 2a)
     - Empty moving truck (class 5)
     - Loaded moving truck (class 5)
   * - **Resource**
     - "sedan" | "generic_sedan"
     - "bus" | "generic_bus"
     - "pickup" | "generic_pickup_truck"
     - "moving_truck_empty"
     - "moving_truck_loaded"
   * - **Dimensions** (LWH)
     - 3.68  1.47  1.30
     - 7.00  2.20  2.40
     - 5.00  1.91  1.89
     - 7.10  2.40  2.40
     - 7.10  2.40  2.40
   * - **Mass** (kg)
     - 2356.00
     - 6000.00
     - 2600.00
     - 6500.00
     - 8700.00


Note that the dimensions do not take into account elevation due to the wheels.

.. note::

    See also :assets:`vehicles/vehicle_definitions_list.yaml` and `truck classifications <https://en.wikipedia.org/wiki/Truck_classification>`.


Specifying vehicle definitions
------------------------------

Vehicles can be configured in a few different ways.


Configuration file
^^^^^^^^^^^^^^^^^^

.. code-block:: ini

    [assets]
    default_vehicle_definitions_list = path/to/file.yaml

.. note::

    See also :ref:`engine_configuration`.


Environment variable
^^^^^^^^^^^^^^^^^^^^

Setting ``SMARTS_ASSETS_DEFAULT_VEHICLE_DEFINITIONS_LIST`` will cause ``SMARTS`` to use the given vehicle definitions file as the default vehicle definitions.

.. note::

    See also :ref:`engine_configuration`.


Scenario
^^^^^^^^

Including a ``vehicle_definitions_list.yaml`` in your scenario will cause ``SMARTS`` to use those vehicle definitions for the duration of the scenario.

.. code-block:: bash

    $ tree scenarios/sumo/loop
    scenarios/sumo/loop
    ├── build
    │   └── ...
    ├── map.net.xml
    ├── rerouter.add.xml
    ├── scenario.py
    └── vehicle_definitions_list.yaml # <---


Usage
-----

Agent interface
^^^^^^^^^^^^^^^

.. code-block:: python

    from smarts.core.agent_interface import AgentInterface
    from smarts.core.controllers import ActionSpaceType

    agent_interface = AgentInterface(
        max_episode_steps=1000,
        waypoint_paths=True,
        vehicle_class="pickup",
        action=ActionSpaceType.Continuous,
    )

.. note::

    See also :ref:`agent`.


Syntax
------

A vehicle can be composed in the following way:


.. code-block:: yaml

    # vehicle_definitions_list.yaml
    f150: /usr/home/dev/vehicles/f150.yaml


.. code-block:: yaml

    # /usr/home/dev/vehicles/f150.yaml
    model: Ford F-150
    type: truck
    controller_params: ${SMARTS_ASSETS_PATH}/vehicles/controller_params/generic_pickup_truck.yaml
    chassis_params: ${SMARTS_ASSETS_PATH}/vehicles/chassis_params/generic_pickup_truck.yaml
    dynamics_model: /usr/home/dev/vehicles/f150_loaded.urdf
    visual_model: /usr/home/dev/vehicles/f150.glb
    tire_params: null # ${SMARTS_ASSETS_PATH}/vehicles/tire_params/base_tire_parameters.yaml


.. note::

    See :ref:`engine_configuration` for details about how YAML is resolved.


.. note::

    See :assets:`vehicles/controller_params/generic_pickup_truck.yaml` and :assets:`vehicles/chassis_params/generic_pickup_truck.yaml`.

