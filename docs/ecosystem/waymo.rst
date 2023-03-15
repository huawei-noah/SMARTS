.. _waymo:

Waymo
=====

SMARTS supports loading and replaying scenarios from the **Waymo Motion Dataset**. See the Waymo `website <https://waymo.com/open/data/motion/>`_ for more info and download instructions.

An example SMARTS scenario is located `here <https://github.com/huawei-noah/SMARTS/tree/master/scenarios/waymo>`_. After downloading the dataset, modify the ``scenario_id`` and ``dataset_path`` variables to point to the desired Waymo scenario:

.. code-block:: python

   dataset_path = "/home/user/waymo/uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
   scenario_id = "ef903b7abf6fc0fa"

You can then run any of the examples with this scenario:

.. code-block:: sh

    $ scl run --envision examples/egoless.py scenarios/waymo

.. image:: /_static/waymo-replay.gif
