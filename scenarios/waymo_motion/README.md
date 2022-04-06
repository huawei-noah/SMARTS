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
Some things to keep in mind:
* You can pass in multiple paths to datasets which can be both directory or file. The paths need to be separated by space.
* `--target-base-path=<default/path/to/export/scenarios>` is an optional argument that can be passed to have a default target path to export scenarios or map images and trajectory animations of all scenarios of a tfRecord file. The path passed should be a valid directory path that needs to exist.
* `--import-tags=<path/to/tag/containing/json/file/>` is an optional argument that can be passed to pre-import tags for scenarios of a tfRecord file. The path should be .json file which contains a dictionary of structure `Dict["TfRecord Basename", Dict["Scenario ID", List["Tags"]]]`. Make sure the basename of the TfRecord file should not be modified when you download the dataset.

## TfRecord Browser
After running the program the first browser you will see is the `TfRecords Browser` which shows all the tfrecords you loaded in and the commands you can use to browse them further:
![tfrecord_browser.png](extra/tfrecord_browser.png)

Commands you can execute at this level:
1. `display all` &rarr; This displays the info of all the scenarios from every tfRecord file together. Displays can be filtered on the basis of tags which will be asked in subsequent option.
2. `display <indexes>` &rarr; This displays the info of tfRecord files at these indexes of the table. Displays can be filtered on the basis of tags which will be asked in subsequent option.
3. `explore <index>` &rarr; Explore the tfRecord file at this index of the table. This opens up another browser, `TfRecord Explorer`. The index passed should be an integer between 1 and the number of tfrecord files loaded in. You can see the total in the table printed above.
4. `import tags` &rarr; Import the tags of tfRecords from a previously saved .json file. Only tags of tfRecords which are displayed above will be imported. Ensure the name of tfRecord match with the ones displayed above. The structure of the json file should be same as the example shown below i.e.: `Dict["TfRecord Basename", Dict["Scenario ID", List["Tags"]]]`. If the filename of the tfRecord don't match the ones loaded in, they won't be displayed.
5. `export tags all/<indexes>` &rarr; Export the tags of the tfRecords at these indexes to a .json file. Optionally you can use all instead to export tags of all tfRecords. You will be asked to pass in the path to the .json file in a subsequent option where the path passed should be valid. An example of how the tags will be imported:
```json
{
  "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000": {
    "c84cde79e51b087c": [
      "2d"
    ],
    "6cec26a9347e8574": [
      "2d"
    ],
    "fe6141aeb4061824": [
      "2d"
    ],
    "cc6e41f0505f273": [
      "2d"
    ],
    "d9a14485bb4f49e8": [
      "2d"
    ]
  }
}
```
6. `exit` &rarr; To exit the program. You can also exit the program anytime by raising `EOF` like by pressing `Ctrl + D`.

## TfRecord Explorer
After selecting the tfRecord to explore further, the second browser you will see is the `TfRecord Explorer` which shows the scenario info of all the scenarios in this file and the commands you can use to explore them further:
![tfrecord_1](extra/tfrecord_1.png)
.\
.\
.\
![tfrecord_2.png](extra/tfrecord_2.png)

Commands you can execute at this level:
1. `display` &rarr; This displays the scenarios in this tfrecord filtered based on the tags chosen in a subsequent option.
2. `explore <index>` &rarr; Select and explore further the scenario at this index of the table. This opens up the third browser, `Scenario Explorer`. The index should be an integer between 1 and total number of scenarios displayed above.
3. `export all/<indexes>` &rarr; This command lets you export the scenarios at these indexes (or all the scenarios if used with `all`) to a target path. If you have run the script with `--target-base-path` option, the subsequent option will ask if you want to use custom path or use the default path passed. The indexes should be an integer between 1 and total number of scenarios displayed above separated by space. The exports can also be filtered based on the tags chosen in a subsequent option. 
    This will create a `<SCENARIO_ID>` directory at the path passed for every scenario and will consist of two files, `<SCENARIO_ID>/scenario.py` for scenario creation of `SMARTS`:
```python
from pathlib import Path
import yaml
import os
    
from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario, MapSpec

yaml_file = os.path.join(Path(__file__).parent, "waymo.yaml")
with open(yaml_file, "r") as yf:
    dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

dataset_path = dataset_spec["input_path"]
scenario_id = dataset_spec["scenario_id"]

gen_scenario(
    Scenario(
        map_spec=MapSpec(source=f"{dataset_path}#{scenario_id}", lanepoint_spacing=1.0),
        traffic_histories=["waymo.yaml"],
    ),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
```
   And `<SCENARIO_ID>/waymo.yaml` for generating history dataset and imitation learning aspects of `SMARTS`:,
```yaml
trajectory_dataset:
source: Waymo
input_path: ./waymo_dataset/uncompressed_scenario_training_20s_training_20s.tfrecord-00001-of-01000
scenario_id: <SCENARIO_ID>
```
Where the `input_path` and `scenario_id` will be modified accordingly.
4`preview all` &rarr; This will plot and dump the images of the map of all scenarios in this tf_record to a target path which you will be asked in a subsequent option. If you have run the script with `--target-base-path` option, the subsequent option will ask if you want to use custom path or use the default path passed.
5`preview <indexes>` &rarr; This plots and display the maps of these scenarios at these index of the table. Each map will be displayed in a separate canvas gui of `matplotlib` and you can only use other commands after closing all the plots. The indexes should be an integer between 1 and total number of scenarios displayed above and should be separated by space.
6`animate all` &rarr; This plot and dump the animations the trajectories of objects on map of all scenarios in this tf_record to a target path which you will be asked in a subsequent option. If you have run the script with `--target-base-path` option, the subsequent option will ask if you want to use custom path or use the default path passed.
7`animate <indexes>` &rarr; This plots and animate the trajectories of objects on map of scenario at this index of the table. Each animation will be displayed in a separate canvas gui of `matplotlib` and you can only use other commands after closing all the plots. The indexes should be an integer between 1 and total number of scenarios displayed above and should be separated by space.
8`tag all/<indexes>` or `tag imported all/<indexes>` &rarr; Tag the scenarios by adding the tags to their `Tags Added` list at these indexes of the table (or all the scenarios if used with `all`). Optionally if you call with `tag imported` then the tags for these scenarios will be added to `Imported Tags` list seen above. If indexes, then they need to be integers between 1 and total number of scenarios displayed above and should be separated by space.
 You will be asked to input the tags in a subsequent option, and they should be separated by space.
9. `untag all/<indexes>` or `untag imported all/<indexes>` &rarr; Untag the scenarios at these indexes of the table (or all the scenarios if used with `all`) by removing the tags from `Tags Added` list. Optionally if you call with `untag imported` then the tags for these scenarios will be removed from `Imported Tags` list seen above.  If indexes, then they need to be integers between 1 and total number of scenarios displayed above and should be separated by space.
 You will be asked to input the tags in a subsequent option, and they should be separated by space.
11. `back` &rarr; Go back to the `TfRecords Browser`.
12. `exit` &rarr; Exit the program. You can also exit the program anytime by raising `EOF` like by pressing `Ctrl + D`.

## Scenario Explorer
After selecting the scenario to explore further, the third browser you will see is the `Scenario Explorer` which shows the total number of different map features and their ids and total number of different track objects and their ids in this scenario:
![scenario_1.png](extra/scenario_1.png)
![scenario_2.png](extra/scenario_2.png)

Commands you can execute at this level:
1. `export` &rarr; Export the scenario to a target base path asked to input in a subsequent option. If you have run the script with `--target-base-path` option, the subsequent option will ask if you want to use custom path or use the default path passed.
2. `preview` or `preview <feature_ids>` &rarr; This plot and displays the map of the scenario with the feature ids highlighted in **Blue** if passed.  The feature ids need to be separated by space, be numbers from the map feature ids mentioned above and will not be highlighted if they don't exist.
3. `animate` or `animate <track_ids> &rarr; Animate the trajectories of track objects on the map of this scenario with the track ids highlighted in **Red** if passed. The ego vehicle will be higlighted in **Cyan** and objects of interests in **Green**.  The track ids need to be separated by space, be numbers from the track object ids mentioned above and will not be highlighted if they don't exist.
4. `tag` or `tag imported` &rarr; Tag the scenario by adding the tags to `Tags Added` list. Optionally if you call with `tag imported` then the tags  will be added to `Imported Tags` list seen above.  You will be asked to input the tags in a subsequent option, and they should be separated by space.
5. `untag` or `untag imported` &rarr; Untag the scenarios at these indexes of the table (or all the scenarios if used with `all`) by removing them from the `Tags Added` list. Optionally if you call with `tag imported` then the tags for these scenarios will be removed from `Imported Tags` list seen above. You will be asked to input the tags in a subsequent option, and they should be separated by space.
6. `back` &rarr; Go back to this scenario's tfrecord browser.
7. `exit` &rarr; Exit the program. You can also exit the program anytime by raising `EOF` like by pressing `Ctrl + D`.

## Caveats:
* All commands are case-sensitive but have specific rules to be matched with the user's input. 
* Space between words or parameters for commands can be variable but may lead to invalid command.
* When downloading dataset, make sure not to change the name of the TfRecord files as they are used for matching tfRecord names when importing tags.
* .json file having the tags for tfRecords scenarios need to have a very specific dictionary structure mentioned above.
* `animate <indexes>` command is relatively slow so it is recommended to animate only 5 to 6 scenarios together.
* Do not modify files in `scenarios/waymo_motion/templates` as it contains the templates for `scenario.py` and `waymo.py` that are exported during the `export` command.
```

