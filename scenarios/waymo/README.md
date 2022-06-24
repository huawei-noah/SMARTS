# Waymo Motion Dataset

This scenario demonstrates using the Waymo Motion Dataset with SMARTS. Given a dataset file and scenario ID, SMARTS will load the map and replay the recorded trajectories of the vehicles from the scenario. More information about the dataset can be found at the [Waymo website](https://waymo.com/open/data/motion/).

## Dataset Tools (optional)

The SMARTS repository contains tools to assist with using the Waymo Motion Dataset. These can be found in `smarts/waymo/`. They include:
- `waymo_browser.py`: a text-based browser utility for exploring, visualizing, and tagging scenarios from the dataset
- `gen_sumo_map.py`: a command-line program that converts the map from a Waymo scenario to a SUMO map

## Setup

Follow the instructions for the setup of SMARTS in the main [README](https://github.com/huawei-noah/SMARTS/). Then install the `[waymo]` dependencies to install the `waymo-dataset` package:

```bash
pip install -e .[waymo]
```

Next, download the dataset files from the [Waymo Motion Dataset website](https://waymo.com/open/download/#). It is recommended to download the dataset files from the `uncompressed/scenario/training_20s` folder as they have the full traffic capture for each scenario. Note: Waymo provides 2 different formats for the dataset files. SMARTS expects the `Scenario protos` format (not the `tf.Example protos` format). It is also recommended to use version 1.1 of the dataset, which includes enhanced map information.

Edit `scenario.py` so that `input_path` points to the TFRecord file containing the scenario you want to use, and use `scenario_id` to select the specific scenario in the file.

From the root of the SMARTS repo run the following command to build the scenario:

```bash
scl scenario build --clean scenarios/waymo
```

This will build the map and generate the SQLite database of trajectory data (with extension `.shf`).

## Using the traffic histories in SMARTS

There are examples that use traffic histories. We explain their use below.

### Example: observation collection

This example will create `.pkl` files storing the observations of each vehicle in the scenario that can be used by your own agent (see the history vehicle replacement example).

Run with:

```bash
python examples/observation_collection_for_imitation_learning.py scenarios/waymo
```

### Example: history vehicle replacement

This example queries the SQLite database for trajectory data to control the social vehicles. It creates one ego vehicle to hijack the autonomous vehicle for the scenario, and uses a simple agent (`ReplayCheckerAgent`) that reads the observation data from the previous example to set its control inputs to match the recorded trajectory. You can subsitute your own custom agent for this instead.

Run with:

```bash
python examples/history_vehicles_replacement_for_imitation_learning.py --episodes=1 scenarios/waymo
```
