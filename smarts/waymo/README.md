# Text-Based Browser For Waymo Dataset
This is a text-based utility to browse, explore, and export  Waymo TFRecord datasets to SMARTS scenarios. Users are able to tag scenarios and export/import the tags to/from a JSON file.

## Setup
1. Install the extra dependencies.
    ```bash
    $ pip install waymo-open-dataset-tf-2-4-0 tabulate==0.8.9 pathos==0.2.8
    ```

2. Download the [Waymo Motion Dataset](https://waymo.com/open/download/) files to the folder `scenarios/waymo/waymo_data` or to your folder of choice. It is recommended to download `waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s` dataset as they have the full traffic capture for each scenario.

## Running the Utility
Execute the following command to run the script.
```bash
$ cd <path>/SMARTS
$ python smarts/waymo/waymo_browser.py <path/to/waymo_dataset> --target-base-path=<default/path/to/export/scenarios> --import-tags=<path/to/tag/containing/json/file/>
```
An example would be:
```bash
$ python smarts/waymo/waymo_browser.py scenarios/waymo/waymo_data
```
Note:
+ Multiple paths to either TFRecord files or directories, which are separated by spaces, can be passed in.
+ Optional argument `--target-base-path=<default/path/to/export/scenarios>` to set default target path to export scenarios, map images, and trajectory animations of all scenarios from a TFRecord file. Target-path directory must exist.
+ Optional argument `--import-tags=<path/to/tag/containing/json/file/>` to set pre-import tags for scenarios from a TFRecord file. The path should be a `.json` file containing a dictionary of structure `Dict["TFRecord Basename", Dict["Scenario ID", List["Tags"]]]`. Ensure the basename of the TFRecord files have not been modified.

## TFRecord Browser
After running the program the first browser you will see is the `TFRecord Browser` which shows all the TFRecords you loaded in and the commands you can use to browse them further:
```cmd
Waymo tfRecords:
  Index  TfRecords
-------  -------------------------------------------------------------------------------------------------------------------------------------
      1  /home/kyber/huawei/smarts_open/SMARTS/smarts/waymo/waymo_data/uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000
  
TfRecords Browser.
You can use the following commands to further explore these datasets:
1. `display all` --> Displays the info of all the scenarios from every tfRecord file together
                     Displays can be filtered on the basis of tags in a subsequent option.
2. `display <indexes>` --> Displays the info of tfRecord files at these indexes of the table.
                           The indexes should be an integer between 1 and 1 and space separated.
                            Displays can be filtered on the basis of tags.
3. `explore <index>` --> Explore the tfRecord file at this index of the table.
                         The index should be an integer between 1 and 1
4. `import tags` --> Import the tags of tfRecords from a previously saved .json file.
                     Only tags of tfRecords which are displayed above will be imported. Ensure the name of tfRecord match with the ones displayed above.
5. `export tags all/<indexes>` --> Export the tags of the tfRecords at these indexes to a .json file.
                                   Optionally you can use all instead to export tags of all tfRecords. The path to the .json file should be valid.
6. `exit` --> Exit the program


Command:
```
Commands you can execute at this level:
1. `display all` &rarr; This displays the info of all the scenarios from every TFRecord file together. Displays can be filtered on the basis of tags which will be asked in the subsequent prompt.
2. `display <indexes>` &rarr; This displays the info of TFRecord files at these indexes of the table. Displays can be filtered on the basis of tags which will be asked in the subsequent prompt.
3. `explore <index>` &rarr; Explore the TFRecord file at this index of the table. This opens up another browser, `TFRecord Explorer`. The index passed should be an integer between 1 and the number of TFRecord files loaded in. You can see the total in the table printed above.
4. `import tags` &rarr; Import the tags of TFRecords from a previously saved .json file. Only tags of TFRecords which are displayed above will be imported. Ensure the names of the TFRecords match with the ones displayed above. If the filenames of the TFRecords don't match the ones loaded in, they won't be displayed.
5. `export tags all/<indexes>` &rarr; Export the tags of the TFRecords at these indexes to a .json file. Optionally you can use all instead to export tags of all TFRecords. You will be asked to pass in the path to the .json file in a subsequent prompt where the path passed should be valid. An example of how the tags will be imported:
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
6. `exit` &rarr; To exit the program. You can also exit the program at any time by pressing `Ctrl + D`.

## TFRecord Explorer
After selecting the TFRecord to explore further, the second browser you will see is the `TFRecord Explorer` which shows the scenario info of all the scenarios in this file and the commands you can use to explore them further:
```cmd
TfRecord Explorer
-----------------------------------------------
61 scenarios in uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000:
  Index  Scenario ID         Timestamps    Track Objects    Traffic Light States    Objects of Interest  Tags Added    Tags Imported
-------  ----------------  ------------  ---------------  ----------------------  ---------------------  ------------  ---------------
      1  c84cde79e51b087c           199              188                     199                      2  []            []
      2  6cec26a9347e8574           199              178                     199                      2  []            []
      3  fe6141aeb4061824           198               80                     198                      2  []            []
      4  cc6e41f0505f273f           199               23                     199                      2  []            []
      5  d9a14485bb4f49e8           199               51                     199                      2  []            []
      6  e6cc567884b0e4f9           198              100                     198                      2  []            []
      7  ef903b7abf6fc0fa           199               23                     199                      2  []            []
      8  a7ea3da73ebb0ac7           199              152                     199                      2  []            []
      9  4f30f060069bbeb9           199               74                     199                      2  []            []
     10  20bf7bcc356ed3cd           198               30                     198                      2  []            []
     11  979c88d4c48e80a1           199              130                     199                      2  []            []
     12  570bd8a976d74b96           199               88                     199                      2  []            []
     13  4b6f47123bc2c8ac           199               70                     199                      2  []            []
     14  ea1fb0d50be9ae69           200              144                     200                      2  []            []
     15  27f5fc6e3f44bdde           199              124                     199                      2  []            []
     16  eaf07d60fa7bd546           199               47                     199                      2  []            []
     17  4c311861c0fffa9a           199              119                     199                      2  []            []
     18  1231c0c9a82e4f61           198              309                     198                      2  []            []
     19  69ce11e9b69203e0           198               51                     198                      2  []            []
     20  31e3acf12ee52d0c           198               77                     198                      2  []            []
```
.\
.\
.


```cmd
     58  b1b51cdb69de2d46           199              124                     199                      2  []            []
     59  7e019d5f96e54dbd           198              117                     198                      2  []            []
     60  7f625727984895a6           198              106                     198                      2  []            []
     61  d4d4be8abeb61d17           198               35                     198                      2  []            []
    

uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000 TfRecord Browser.
You can use the following commands to further explore these scenarios:
1. `display` --> Display the scenarios in this tfrecord filtered based on the tags chosen in a subsequent option.
2. `explore <index>` --> Select and explore further the scenario at this index of the table.
                        The index should be an integer between 1 and 61
3. `export all/<indexes>` --> Export the scenarios at these indexes or all of the table to a target path
                             The indexes should be an integer between 1 and 61 separated by space
                             The exports can be filtered based on the tags chosen in a subsequent option.
4. `preview all` --> Plot and dump the images of the map of all scenarios in this tf_record to a target path.
5. `preview` or `preview <indexes>` --> Plot and display the maps of these scenarios at these indexes of the table  (or all the scenarios if just `preview`) .
                                       The indexes should be an integer between 1 and 61 and should be separated by space.
6. `animate all` --> Plot and dump the animations the trajectories of objects on map of all scenarios in this tf_record to a target path.
7. `animate` or `animate <indexes>` --> Plot the map and animate the trajectories of objects of all scenarios if just `animate` or scenario at these indexes of the table.
                                        The indexes should be an integer between 1 and 61 and should be separated by space.
8. `tag all/<indexes>` or `tag imported all/<indexes>` --> Tag the scenarios at these indexes of the table or all with tags mentioned.
                                                           Optionally if you call with `tag imported` then the tags for these scenarios will be added to imported tag list.
                                                           If indexes, then they need to be integers between 1 and 61 and should be separated by space.
9. `untag all/<indexes>` or `untag imported all/<indexes>` --> Untag the scenarios at theses indexes of the table or all with tags mentioned.
                                                               Optionally if you call with `untag imported` then the tags for these scenarios will be removed from imported tag list.
                                                               If indexes, then they need to be integers between 1 and 61 and should be separated by space.
10. `back` --> Go back to the tfrecords browser
11. `exit` --> Exit the program


Command:
```

Commands you can execute at this level:
1. `display` &rarr; This displays the scenarios in this TFRecord filtered based on the tags chosen in a subsequent prompt.
2. `explore <index>` &rarr; Select and explore further the scenario at this index of the table. This opens up the third browser, `Scenario Explorer`. The index should be an integer between 1 and the total number of scenarios displayed above.
3. `export all/<indexes>` &rarr; This command lets you export the scenarios at these indexes (or all the scenarios if used with `all`) to a target path. If you have run the script with `--target-base-path` option, the subsequent prompt will ask if you want to use a custom path or use the default path passed. The indexes should be an integer between 1 and the total number of scenarios displayed above, separated by space. The exports can also be filtered based on the tags chosen in a subsequent prompt. This will create a `<SCENARIO_ID>` directory at the path passed for every scenario and will consist of two files, `<SCENARIO_ID>/scenario.py` for scenario creation in `SMARTS`:
```python
from pathlib import Path
import yaml
import os

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio import types as t


yaml_file = os.path.join(Path(__file__).parent, "waymo.yaml")
with open(yaml_file, "r") as yf:
    dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

dataset_path = dataset_spec["input_path"]
scenario_id = dataset_spec["scenario_id"]
traffic_history = t.TrafficHistoryDataset(
    name=f"waymo_{scenario_id}",
    source_type="Waymo",
    input_path=dataset_path,
    scenario_id=scenario_id,
)

gen_scenario(
    t.Scenario(
        map_spec=t.MapSpec(
            source=f"{dataset_path}#{scenario_id}", lanepoint_spacing=1.0
        ),
        traffic_histories=[traffic_history],
    ),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
```
   And `<SCENARIO_ID>/waymo.yaml` for generating history dataset and imitation learning aspects of `SMARTS`:
```yaml
trajectory_dataset:
  source: Waymo
  input_path: ./waymo_dataset/uncompressed_scenario_training_20s_training_20s.tfrecord-00001-of-01000
  scenario_id: <SCENARIO_ID>
```
Where the `input_path` and `scenario_id` will be modified accordingly.

4. `preview all` &rarr; Plot and dump the images of the map of all scenarios in this TFRecord to a target path which you will be asked in a subsequent prompt. If you have run the script with `--target-base-path` option, the subsequent prompt will ask if you want to use a custom path or use the default path passed.
5. `preview` or `preview <indexes>` &rarr; Plot and display the maps of these scenarios at these index of the table (or all the scenarios if just `preview`). Each map will be displayed in a separate GUI window of `matplotlib` and you can only use other commands after closing all the plots. The indexes should be an integer between 1 and the total number of scenarios displayed above and should be separated by space.
6. `animate all` &rarr; Plot and dump the animations of the trajectories of objects on the map of all scenarios in this TFRecord to a target path which you will be asked in a subsequent prompt. If you have run the script with `--target-base-path` option, the subsequent prompt will ask if you want to use custom path or use the default path passed.
7. `animate` or `animate <indexes>` &rarr; Plot and animate the trajectories of objects on the map of scenario at these indexes of the table (or all the scenarios if just `animate`). Each animation will be displayed in a separate GUI window of `matplotlib` and you can only use other commands after closing all the plots. The indexes should be an integer between 1 and the total number of scenarios displayed above and should be separated by space.
8. `tag all/<indexes>` or `tag imported all/<indexes>` &rarr; Tag the scenarios by adding the tags to their `Tags Added` list at these indexes of the table (or all the scenarios if used with `all`). Optionally if you call with `tag imported` then the tags for these scenarios will be added to `Imported Tags` list seen above. If indexes, then they need to be integers between 1 and the total number of scenarios displayed above and should be separated by space. You will be asked to input the tags in a subsequent prompt, and they should be separated by space.
9. `untag all/<indexes>` or `untag imported all/<indexes>` &rarr; Untag the scenarios at these indexes of the table (or all the scenarios if used with `all`) by removing the tags from `Tags Added` list. Optionally if you call with `untag imported` then the tags for these scenarios will be removed from `Imported Tags` list seen above. If indexes, then they need to be integers between 1 and the total number of scenarios displayed above and should be separated by space. You will be asked to input the tags in a subsequent prompt, and they should be separated by space.
10. `back` &rarr; Go back to the `TFRecords Browser`.
11. `exit` &rarr; Exit the program. You can also exit the program at any time by pressing `Ctrl + D`.

## Scenario Explorer
After selecting the scenario to explore further, the third browser you will see is the `Scenario Explorer` which shows the total number of different map features and their IDs, and the total number of different track objects and their IDs in the scenario:
```cmd
Scenario Explorer
-----------------------------------------------
Scenario c84cde79e51b087c Map Features:
Scenario ID         Timestamps    Track Objects    Traffic Light States    Objects of Interest  Tags Added    Tags Imported
----------------  ------------  ---------------  ----------------------  ---------------------  ------------  ---------------
c84cde79e51b087c           199              188                     199                      2  []            []

Scenario c84cde79e51b087c map data:
  Lanes    Road Lines    Road Edges    Stop Signs    Crosswalks    Speed Bumps
-------  ------------  ------------  ------------  ------------  -------------
    180             2            32             6            14              2

Lane Ids:  ['111', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '198', '202', '205', '238', '244', '245', '246', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '274', '291', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '436', '446', '451']

Road Line Ids:  ['26', '63']

Road Edge Ids:  ['3', '6', '14', '17', '25', '28', '29', '30', '38', '39', '42', '45', '46', '47', '48', '49', '50', '51', '52', '53', '55', '56', '57', '58', '59', '60', '61', '62', '64', '69', '77', '81']

Stop Sign Ids:  ['512', '513', '514', '515', '516', '517']

Crosswalk Ids:  ['496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509']

Speed Bumps Ids:  ['510', '511']

-----------------------------------------------
Trajectory Data
Scenario ID         Cars    Pedestrians    Cyclists    Others
----------------  ------  -------------  ----------  --------
c84cde79e51b087c     151             37           0         0

Track Object Ids:
Ego Id:  6335

Car Ids:  [3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3627, 3628, 3629, 3630, 3631, 3633, 3634, 3635, 3638, 3639, 3646, 3650, 3653, 3656, 3658, 3659, 3660, 3661, 3657, 3666, 3667, 3668, 3669, 3672, 3673, 3675, 3676, 3677, 3678, 3681, 3682, 3683, 3684, 3687, 3689, 3690, 3694, 3695, 3697, 3696, 3701, 3705, 3707, 3706, 3708, 3711, 3712, 3714, 3717, 3721, 3722, 3724, 3726, 3729, 3731, 3735, 3738, 3739, 3742, 3743, 3744, 3747, 3749, 3752, 3753, 3756, 3757, 3758, 3759, 3761, 3762, 3765, 3645, 3647, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583]

Pedestrian Ids:  [5178, 5179, 5180, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5188, 5189, 5191, 5210, 5211, 5220, 5212, 5213, 5231, 5233, 5238, 5242, 5246, 5195, 5252, 5254, 5255, 5222, 5223, 5225, 5234, 5235, 5241, 5201, 5248, 5202, 5207]

Cyclist Ids:  []

Other Ids:  []

Object of Interest Ids:  [3562, 5207]


Scenario c84cde79e51b087c.
You can use the following commands to further this scenario:
1. `display --> Display the scenario info which includes the map feature ids and track ids.
2. `export` --> Export the scenario to a target base path asked to input in a subsequent option.
3. `preview` or `preview <feature_ids>` --> Plot and display the map of the scenario with the feature ids highlighted in Blue if passed.
                                            The feature ids need to be separated by space, be numbers from the map feature ids mentioned above and will not be highlighted if they dont exist.
4. `animate` or `animate <track_ids> --> Animate the trajectories of track objects on the map of this scenario with the track ids highlighted in Red if passed.
                                        The track ids need to be separated by space, be numbers from the track object ids mentioned above and will not be highlighted if they dont exist.
5. `tag` or `tag imported` --> Tag the scenario with tags mentioned.
                                Optionally if you call with `tag imported` then the tags will be added to imported tag list.
6. `untag` or `untag imported` --> Untag the scenario with tags mentioned.
                                    Optionally if you call with `untag imported` then the tags will be removed to imported tag list.
7. `back` --> Go back to this scenario's tfrecord browser.
8. `exit` --> Exit the program


Command:
```

Commands you can execute at this level:
1. `display --> Display the scenario info which includes the map feature ids and track ids like the one shown above.
2. `export` &rarr; Export the scenario to a target base path asked to input in a subsequent prompt. If you have run the script with the `--target-base-path` option, the subsequent prompt will ask if you want to use custom path or use the default path passed.
3. `preview` or `preview <feature_ids>` &rarr; Plot and display the map of the scenario with the feature IDs highlighted in <span style="color:blue">**Blue**</span> if provided. The feature IDs need to be separated by space, be numbers from the map feature IDs mentioned above and will not be highlighted if they don't exist.
4. `animate` or `animate <track_ids>` &rarr; Animate the trajectories of track objects on the map of this scenario with the track IDs highlighted in <span style="color:red">**Red**</span> if provided. The ego vehicle will be highlighted in <span style="color:cyan">**Cyan**</span> and objects of interests in <span style="color:green">**Green**</span>. The track IDs need to be separated by space, be numbers from the track object IDs mentioned above and will not be highlighted if they don't exist.
5. `tag` or `tag imported` &rarr; Tag the scenario by adding the tags to `Tags Added` list. Optionally if you call with `tag imported` then the tags will be added to `Imported Tags` list seen above. You will be asked to input the tags in a subsequent prompt, and they should be separated by space.
6. `untag` or `untag imported` &rarr; Untag the scenarios at these indexes of the table (or all the scenarios if used with `all`) by removing them from the `Tags Added` list. Optionally if you call with `tag imported` then the tags for these scenarios will be removed from the `Imported Tags` list seen above. You will be asked to input the tags in a subsequent prompt, and they should be separated by space.
7. `back` &rarr; Go back to the TFRecord browser.
8. `exit` &rarr; Exit the program. You can also exit the program at any time by pressing `Ctrl + D`.

## Additional Notes:
* All commands are case-sensitive but have specific rules to be matched with the user's input. 
* Space between words or parameters for commands can be variable but may lead to invalid command.
* When downloading the dataset, make sure not to change the name of the TFRecord files as they are used for matching TFRecord names when importing tags.
* .json file having the tags for TFRecords scenarios need to have the specific dictionary structure mentioned above.
* `animate all` uses `ffmpeg` writer to save the animations which don't exist by default in linux and MacOS machines. 
   So you can install it using `sudo apt install ffmpeg` in linux or `brew install ffmpeg` in MacOS. You can read more about this issue [here](https://github.com/kkroening/ffmpeg-python/issues/251).
* `animate <indexes>` command is relatively slow, so it is recommended to animate only a small number of scenarios together.
```
