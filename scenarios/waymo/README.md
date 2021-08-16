# Waymo Motion Dataset

SMARTS supports importing traffic histories from the Waymo motion dataset to replay in a SMARTS simulation.

More information about the dataset can be found at the Waymo [website](https://waymo.com/open/data/motion/).

## Setup

Follow the instructions for the setup of SMARTS in the main [README](https://github.com/huawei-noah/SMARTS/). Then install the `[waymo]` dependencies to install the `waymo-dataset` package:

```bash
pip install -e .[waymo]
```

Next, download the dataset files to the folder `scenarios/waymo/waymo_dataset`. It is recommended to download the dataset files from the `training_20s` folder as they have the full traffic capture for each scenario.

Currently, users must supply their own map. Place the `map.net.xml` file in the `scenarios/waymo` folder.

## Generating the history database and creating the map

Edit `scenarios/waymo/waymo.yaml` so that `input_path` points to the TFRecord file containing the scenario you want to use, and use `scenario_index` to select the scenario within the list of scenarios in the file.

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

