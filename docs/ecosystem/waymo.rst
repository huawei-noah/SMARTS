.. _waymo:

Waymo
=====

SMARTS supports loading and replaying scenarios from the **Waymo Motion Dataset**. See the Waymo `website <https://waymo.com/open/data/motion/>`_ for more info and download instructions.

Setup
-----

Download the dataset files from the `dataset website <https://waymo.com/open/download/>`_. It is recommended to download the dataset files from the ``uncompressed/scenario/training_20s`` folder as they have the full traffic capture for each scenario. Note: Waymo provides 2 different formats for the dataset files. SMARTS expects the ``Scenario protos`` format (not the ``tf.Example protos`` format). It is also recommended to use version 1.1 of the dataset, which includes enhanced map information.

Tools
-----

SMARTS provides some command-line tools to assist with visualizing and selecting Waymo scenarios.

.. code-block:: sh

    $ scl waymo --help
    Usage: scl waymo [OPTIONS] COMMAND [ARGS]...

    Utilities for using the Waymo Motion Dataset with SMARTS. See `scl waymo
    COMMAND --help` for further options.

    Options:
    --help  Show this message and exit.

    Commands:
    export    Export the Waymo scenario to a SMARTS scenario.
    overview  Display summary info for each scenario in the TFRecord file.
    preview   Plot the map and trajectories of the scenario.

To see all the scenario IDs in a tfrecord file:

.. code-block:: sh

    $ scl waymo overview ~/waymo/uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000
    Scenario ID         Timestamps    Vehicles    Pedestrians
    ----------------  ------------  ----------  -------------
    c84cde79e51b087c           199         151             37
    6cec26a9347e8574           199         165             13
    fe6141aeb4061824           198          74              6
    ...

To preview a scenario:

.. code-block:: sh

    $ scl waymo preview ~/waymo/uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000 ef903b7abf6fc0fa

Additionally, there are some other standalone tools in the `waymo module <https://github.com/huawei-noah/SMARTS/tree/master/smarts/waymo>`_

- ``gen_sumo_map.py``: a command-line program that converts the map from a Waymo scenario to a SUMO map

Example Scenario
----------------

An example SMARTS scenario is located `here <https://github.com/huawei-noah/SMARTS/tree/master/scenarios/waymo>`_. After downloading the dataset, modify the ``scenario_id`` and ``dataset_path`` variables to point to the desired Waymo scenario:

.. code-block:: python

   dataset_path = "/home/user/waymo/uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
   scenario_id = "ef903b7abf6fc0fa"

You can then run any of the examples with this scenario:

.. code-block:: sh

    $ scl run --envision examples/egoless.py scenarios/waymo

.. image:: /_static/waymo-replay.gif

Troubleshooting
---------------

SMARTS includes a module for loading Waymo scenarios (located in ``smarts/waymo/waymo_open_dataset``) that should work without any configuration. However, depending on the version of protobuf installed, some issues may come up which require the Python files to be regenerated. See this `README <https://github.com/huawei-noah/SMARTS/tree/master/smarts/waymo/waymo_open_dataset>`_ for instructions on how to do this.
