.. _ngsim:

NGSIM
=====

**NGSIM** is the Next Generation Simulation dataset, a free to use dataset created
by the US Department of Transportation (DOT) and described
`here <https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm>`__.


NGSIM traffic history scenarios
-------------------------------

These scenarios are located in the `smarts repository <https://github.com/huawei-noah/SMARTS/tree/master/scenarios/NGSIM>`_.

To use these scenarios, you must first download the dataset in an appropriate format.
SMARTS assumes that the data will be in tab-separated text files.
Its import scripts were tested on and work with the data downloaded from `here <http://bit.ly/PPUU-data>`__,
which is the same as was used by this `PPUU project <https://github.com/Atcold/pytorch-PPUU>`_.

After the dataset has been downloaded, you will need to update each ``scenario.py``
in the ngsim scenario directories (e.g., :scenarios:`NGSIM/i80/scenario.py`, :scenarios:`NGSIM/us101/scenario.py`, and/or :scenarios:`NGSIM/peachtree/scenario.py`) to point the `input_path` field
to the dataset location on your filesystem.

Once that is done, you should be able to build your scenarios in the normal way, for example: ``scl scenario build-all --clean scenarios/NGSIM``

For each traffic history dataset specified in your ``scenario.py``, 
this will create a corresponding ``.shf`` file that SMARTS will use
whenever this traffic will be added to a simulation.

Note that the SUMO maps created for NGSIM were made by hand by the open-source SMARTS team.
Although they attempt to align with the positions in the traffic dataset,
their level of exactness may not be enough for some model-training situations,
so you may want or need to refine them with SUMO's `netedit tool <https://sumo.dlr.de/docs/Netedit/index.html>`_.

An example of how traffic history might be saved as observations can be found in :mod:`~smarts.dataset.traffic_histories_to_observations`. 

To consume the generated observations you could use the following approach:

.. code-block:: python

    import os
    import pickle
    import re

    # a suggested approach
    import numpy as np
    from PIL import Image

    from smarts.core.observations import Observation, TopDownRGB

    scenarios = list()
    for scenario_name in os.listdir(input_path):
        scenarios.append(scenario_name)

    for scenario in scenarios:
        obs: List[Observation] = list()
        vehicle_ids = list()

        for filename in os.listdir(scenario_path):
            if filename.endswith(".pkl"):
                match = re.search("(.*).pkl", filename)
                assert match is not None
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        for id_ in vehicle_ids:
            with open(scenario_path / f"{id}.pkl", "rb") as f:
                vehicle_data = pickle.load(f)

            image_names = list()
            for filename in os.listdir(scenario_path):
                if filename.endswith(f"{id_}.png"):
                    image_names.append(filename)
            image_names.sort()

            for i in range(len(image_names)):
                with Image.open(scenario_path / image_names[i], "r") as image:
                    image.seek(0)
                    bev = np.moveaxis(np.asarray(image, dtype=np.uint8), -1, 0)
                
                sim_time = image_names[i].split("_")[0]
                current_obs: Observation = vehicle_data[float(sim_time)]
                obs.append((current_obs, bev))

        for o, rgb in obs:
            ...


Alternatively, an approach like :examples:`traffic_histories_vehicle_replacement.py` can be used to operate directly with the scenarios.


Samples
-------

Some specific dataset samples can be found at: https://github.com/smarts-project/smarts-project.offline-datasets