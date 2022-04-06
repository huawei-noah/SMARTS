# Text-Based Browser For Waymo Dataset
`scenarios/waymo_motion/waymo_utility.py` is a text based browser utility used to browse and explore Waymo tfRecord datasets and export them to a SMARTS scenarios if needed.
This also allows users to classify scenarios based on their interactive behavior and export and import their tags to a json file.
## Setup

The utility is independent of SMARTS and only has two dependencies.  You can install the `[waymo]` dependencies of SMARTS to install the `waymo-dataset` and `tabulate` packages using the following command:

```bash
pip install -e .[waymo]
```

Next, download the dataset files from [WAYMO Motion Database](https://waymo.com/open/download/#) to the folder `scenarios/waymo_motion/waymo_data` or your folder of choice. It is recommended to download the dataset files from the `uncompressed/scenario/training_20s` folder as they have the full traffic capture for each scenario.

## Running the Utility:
It is recommended to run this script from the root or source level directory of the repo. The script can be run using the following command:
```bash
python scenarios/waymo_motion/waymo_utility.py <path/to/waymo_dataset_directories> --target-base-path=<default/path/to/export/scenarios> --import-tags=<path/to/tag/containing/json/file/>
```
Ex:
```bash
python scenarios/waymo_motion/waymo_utility.py scenarios/waymo_motion/waymo_data
```

Or you can also use the click's scl command line at the source directory to launch the browser:
```bash
scl scenario browse-waymo <path/to/waymo_dataset_directories> -t=<default/path/to/export/scenarios> -i=<path/to/tag/containing/json/file/>
```
Ex:
```bash
scl scenario browse-waymo scenarios/waymo_motion/waymo_data
```
Some this:
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

