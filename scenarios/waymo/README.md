# Waymo Motion Dataset

SMARTS supports importing traffic histories from the Waymo motion dataset to replay in a SMARTS simulation.

More information about the dataset can be found at the Waymo [website](https://waymo.com/open/data/motion/).

## Setup

Follow the instructions for the setup of SMARTS in the main [README](https://github.com/huawei-noah/SMARTS/). Then install the `[waymo]` dependencies to install the `waymo-dataset` package:

```bash
pip install -e .[waymo]
```

Next, download the dataset files to the folder `scenarios/waymo/waymo_dataset`. It is recommended to download the dataset files from the `training_20s` folder as they have the full traffic capture for each scenario.

Currently, users must supply their own map. Place the `map.net.xml` file in the `scenarios/waymo` folder. Additionally, they can use the `scenarios/waymo/waymo_sumo_map_conversion.py` to create a naive sumo map of their waymo scenario and then edit it manually to create a cleaner sumo map.

## Generating the history database and creating the map
Inorder to create a naive sumo map of your waymo scenario, go to `scenarios/waymo` and run the following command,
```bash
python scenarios/waymo/waymo_sumo_map_conversion.py path/to/dataset scenario_id

# example
python scenarios/waymo/waymo_sumo_map_conversion.py scenarios/waymo/waymo_dataset/uncompressed_scenario_training_20s_training_20s.tfrecord-00007-of-01000 e211c9b4f68ff2c8
```
This will create the `net-{scenario_id}.net.xml` in `scenarios/waymo` which is the naive sumo map. This means the map will have lanes and edges accurate to their length and positions but will be full of holes and noise. There can be multiple junctions at the edge intersections and cracks in lanes.
So you can edit the map using SUMO's [netedit](https://sumo.dlr.de/docs/Netedit/index.html) tool to edit the map manually and make it more usable. 
Some Tips on how to edit the Sumo Map Manually:
    * Always use the current `net-{scenario_id}.net.xml` as skeleton and try to create new edges with multiple lanes directly overlapping the old ones by deleting the connections and then joining the nodes
    * You can remove all the overlapping junctions and create a single junction by removing the old junctions, selecting all the edge nodes that the junction will connect and using the `junctions.join` tool from taskbar.
    * You can move the entire polygons or change their lengths from one end by using the `move` tool from taskbar.
    * Make sure to remove any isolated edges or lanes and remove the old lanes, edges and junctions by deleting them manually.
    * Save your changes by in a new file called `map.net.xml` at the same level as `scenario.py` in `scenarios/waymo`.

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

