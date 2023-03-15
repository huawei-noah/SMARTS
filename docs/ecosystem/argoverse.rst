.. _argoverse:

Argoverse 2
===========

SMARTS supports loading and replaying scenarios from the **Argoverse 2 Motion Forecasting Dataset**. See the Argoverse `website <https://www.argoverse.org/av2.html#forecasting-link>`_ for more info and download instructions.

An example SMARTS scenario is located `here <https://github.com/huawei-noah/SMARTS/tree/master/scenarios/argoverse>`_. After downloading the dataset, modify the ``scenario_id`` and ``scenario_path`` variables to point to the desired Argoverse scenario:

.. code-block:: python

    # scenario_path is a directory with the following structure:
    # /path/to/dataset/{scenario_id}
    # ├── log_map_archive_{scenario_id}.json
    # └── scenario_{scenario_id}.parquet

    scenario_id = "0000b6ab-e100-4f6b-aee8-b520b57c0530"
    scenario_path = Path("/home/user/argoverse/train/") / scenario_id

Make sure you've installed the required dependencies:

.. code-block:: sh

    $ pip install -e .[argoverse]

You can then run any of the examples with this scenario:

.. code-block:: sh

    $ scl run --envision examples/egoless.py scenarios/argoverse

.. image:: /_static/argoverse-replay.gif
