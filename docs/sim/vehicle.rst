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


YAML configurations
^^^^^^^^^^^^^^^^^^^

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


URDF hierarchy
^^^^^^^^^^^^^^


The vehicle urdf requires the following link configuration:


.. code-block:: text

    base_link 
    └── <base_link_connection (joint 0)> chassis 
        ├── <front_left_steer_joint (joint 1)> fl_axle 
        │   └── <front_left_wheel_joint (joint 2)> front_left_wheel
        ├── <front_right_steer_joint (joint 3)> fr_axle
        │   └── <front_right_wheel_joint (joint 4)> front_right_wheel 
        ├── <rear_left_wheel_joint (joint 5)> rear_left_wheel
        └── <rear_right_wheel_joint (joint 6)> rear_right_wheel


In XML this looks like:


.. code-block:: xml

    <!--vehicle.urdf-->
    <?xml version="1.0"?>
    <robot xmlns:xacro="http://ros.org/wiki/xacro" name="vehicle">
        <!--Link order is NOT important.-->
        <link name="base_link">...</link>
        <link name="chassis">...</link>
        <link name="fl_axle">...</link>
        <link name="fr_axle">...</link>
        <link name="front_left_wheel">...</link>
        <link name="front_right_wheel">...</link>
        <link name="rear_left_wheel">...</link>
        <link name="rear_right_wheel">...</link>

        <!--++++Joint order IS important.++++-->
        <joint name="base_link_connection" type="fixed">
            <parent link="base_link"/>
            <child link="chassis"/>
        </joint>
        <joint name="front_left_steer_joint" type="revolute">
            <parent link="chassis"/>
            <child link="fl_axle"/>
        </joint>
        <joint name="front_right_steer_joint" type="revolute">
            <parent link="chassis"/>
            <child link="fr_axle"/>
        </joint>
        <joint name="front_left_wheel_joint" type="continuous">
            <parent link="fl_axle"/>
            <child link="front_left_wheel"/>
        </joint>
        <joint name="front_right_wheel_joint" type="continuous">
            <parent link="fr_axle"/>
            <child link="front_right_wheel"/>
        </joint>
        <joint name="rear_left_wheel_joint" type="continuous">
            <parent link="chassis"/>
            <child link="rear_left_wheel"/>
        </joint>
        <joint name="rear_right_wheel_joint" type="continuous">
            <parent link="chassis"/>
            <child link="rear_right_wheel"/>
        </joint>
    </robot>


.. note::

    Joint name and order is critical. Joints and links in excess of the required joints will not cause problems.

