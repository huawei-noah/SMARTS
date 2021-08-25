# Waymo Motion Dataset

SMARTS supports importing traffic histories from the Waymo motion dataset to replay in a SMARTS simulation.

More information about the dataset can be found at the Waymo [website](https://waymo.com/open/data/motion/).

## Setup

Follow the instructions for the setup of SMARTS in the main [README](https://github.com/huawei-noah/SMARTS/). Then install the `[waymo]` dependencies to install the `waymo-dataset` package:

```bash
pip install -e .[waymo]
```

Next, download the dataset files from [WAYMO Motion Database](https://waymo.com/open/download/#) to the folder `scenarios/waymo/waymo_dataset`. It is recommended to download the dataset files from the `uncompressed/scenario/training_20s` folder as they have the full traffic capture for each scenario.
The basic scenario map we developed is for a scenario from the `training_20s.tfrecord-00001-of-01000` dataset but for other scenarios and datasets, users must supply their own map. We provide a script to generate a simple SUMO map from the map data in the Waymo dataset as a starting point.

Example use of the map script:

```bash
python scenarios/waymo/gen_sumo_map.py scenarios/waymo/waymo_dataset/uncompressed_scenario_training_20s_training_20s.tfrecord-00001-of-01000 e211c9b4f68ff2c8
```

This will create a file called `map-{scenario_id}.net.xml` in `scenarios/waymo`. This map will have edges with a single lane for each lane in the map data. You can edit the map using SUMO's [netedit](https://sumo.dlr.de/docs/Netedit/index.html) tool to edit the map manually and make it more usable.

Some tips for editing the SUMO map manually in netedit:
* Clean up any edges/nodes that are detached from the main roads and/or have no traffic on them during the scenario
* There will likely be a lot of overlapping nodes. Select and join them into a single node using the `junctions.join` command under the `Processing` menu.
* The generated edges have very complicated shape data and should be deleted and replaced by drawing a new edge that approximates the original shape, preferably by connecting the start and end node of the original edge (and creating new nodes as needed for bends)
* Tweak the lane widths as needed -- they will all have a default width assigned
* Try to create new edges with multiple lanes directly overlapping the old ones
* You can move the entire polygons or change their lengths from one end by using the `move` tool

Once you have your map ready, save it as `map.net.xml` in the `scenarios/waymo` folder.

## Building the scenario

Edit `scenarios/waymo/waymo.yaml` so that `input_path` points to the TFRecord file containing the scenario you want to use, and use `scenario_id` to select the specific scenario in the file.

From the root of the SMARTS repo run the following command to build the scenario:

```bash
scl scenario build --clean scenarios/waymo
```

This will build the map and generate the SQLite database of trajectory data (with extension `.shf`).

## Using the traffic histories in SMARTS

There are currently 2 examples that use traffic histories. We explain their use below. First, start the envision server:

```bash
scl envision start &
```

Visit http://localhost:8081/ in your browser to visualize the replay examples.

### Example: observation collection

This example will create `.pkl` files storing the observations of each vehicle in the scenario that can be used by your own agent (see the history vehicle replacement example).

Run with:

```bash
python examples/observation_collection_for_imitation_learning.py scenarios/waymo
```

### Example: history vehicle replacement

This example queries the SQLite database for trajectory data to control the social vehicles. It creates one ego vehicle to hijack the autonomous vehicle for the scenario, and uses a simple agent (`ReplayCheckerAgent`) that just reads the observation data from the previous example to set its control inputs to match the recorded trajectory. You can subsitute your own custom agent for this instead.

Run with:

```bash
python examples/history_vehicles_replacement_for_imitation_learning.py --episodes=1 scenarios/waymo
```

