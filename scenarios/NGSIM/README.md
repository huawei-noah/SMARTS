# NGSIM traffic history scenarios

The scenarios in the sub-folders here are based on the Next Generation Simulation (NGSIM) 
dataset created by the US Department of Transportation (DOT) and described
[here](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).

To use these scenarios, you must first download the dataset in an appropriate format.
SMARTS assumes that the data will be in tab-separated files.
Its import scripts were tested on and work with the data downloaded from [here](http://bit.ly/PPUU-data),
which is the same as was used by this [PPUU project](https://github.com/Atcold/pytorch-PPUU).

After the dataset has been downloaded, you will need to update `scenario.py`
in the scenario folders to point the `input_path` field to its location on your filesystem.

Once that is done, you should be able to build your scenarios in the normal way, for example:
```bash
scl scenario build-all --clean scenarios/NGSIM
```

For each traffic history dataset specified in your `scenario.py`, 
this will creates a corresponding `.shf` file that SMARTS will use
whenever this traffic will be added to a simulation.

Note that the SUMO maps here were created by hand by the open-source SMARTS team.
Although they attempt to align with the positions in the traffic dataset,
their level of exactness may not be enough for some model-training situations,
so you may want or need to refine them with SUMO's [netedit tool](https://sumo.dlr.de/docs/Netedit/index.html).

An example of how traffic history might be replayed in SMARTS can be found in the 
[examples/observation_collection_for_imitation_learning.py](../../examples/observation_collection_for_imitation_learning.py)
script.
