# Offline Learning

## Objective
The objective is to train a single **offline** learning policy capable of controlling single-agent or multi-agent to complete different tasks in various scenarios. In each scenario, the ego-agents must drive towards their respective goal locations. The challenge of this track is that participants have to develop a model that can be trained using offline dataset, without interactions with the simulator or any online adjustments. 

Important: we require participants to submit the code for us to train using offline dataset, and then we will use the trained model for evaluations. Do not submit a trained model. The contestants should also provide the training code as required. See example below.  

## Data and Model
1. For offline training, we provide information and tools for extracting training dataset from two Naturalistic Autonomous Driving datasets: [Waymo Open Motion dataset](https://waymo.com/open/data/motion/) and [Next Generation Simulation (NGSIM)](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).
1. Participants can download these two datasets, and use the provided tools to extract training data. 
1. It is required that the training dataset to be compatible with the SMARTS simulator. you can find the tools for visualize and extract the training data compatible with SMARTS here:
   1. Waymo utilities from https://github.com/huawei-noah/SMARTS/tree/saul/waymo-extraction/smarts/waymo, can be used to  
      + to browse Waymo dataset, and 
      + to extract offline training data from Waymo in form of SMARTS observation space.
   1. NGSIM utilities from ????, can used to 
      + to simulate NGSIM dataset in SMARTS, and
      + to extract offline training data from NGSIM in form of  SMARTS observation space.
1. We have provided a subset of each Waymo and NGSIM datasets that we found to include useful and interesting trajectories for training. You can use this subset to limit the size of your training set and exclude irrelevant scenes. The data extraction tools for each dataset are capable to extracting data from the provided subset. (these subsets are subject to be updated!)
1. Trained model should accept multi-agent observation of the format `Dict[agent_name: agent_observation]`. Observation space for each agent is `smarts.core.sensors.Observation`. For more details on the contents of `Observation` class, see https://github.com/huawei-noah/SMARTS/blob/comp-1/smarts/core/sensors.py#L186
1. Each agent's mission goal location is given in the observation returned at each time step. The mission goal could be accessed as `observation.ego_vehicle_state.mission.goal.position` which gives an `(x, y, z)` map coordinate of the goal location.
1. Trained model should output multi-agent action of the format `Dict[agent_name: agent_action]`. Action space for each agent is `smarts.core.controllers.ActionSpaceType.TargetPose` which is a sequence of [x-coordinate, y-coordinate, heading, and time-delta].

## Process Overview
1. Use `python3.8` to develop your model.
1. The submitted folder structure for Track-2 should be as follows. The folder and file names are to be maintained.
    ```text
    track2                       # Main folder.
    ├── train                    # Contains code to train an offline model.
    │   ├── train.py             # Primary training script for training a new model.
    │   ├── ...                  # Other necessary training files.
    |   .
    |   .
    |   .
    └── submission                       
        ├── policy.py            # A policy with an act method, wrapping the trained model.
        ├── requirements.txt     # Dependencies needed to run the model.
        ├── explanation.md       # Brief explanation of the key techniques used in developing the submitted model.
        ├── ...                  # Other necessary submission files.
        .
        .
        .
    ```
1. The `track2/train/train.py` code should be capable of reading in new offline data fed in by the competition organizers, train a new model with offline data from scratch, and save the newly trained model into a `Policy` class in `track2/submission/policy.py` file.
1. The command 
    ```bash
    $ python3.8 track2/train/train.py --input_dir=<path_to_offline_dataset>
    ```
    will be executed on the submitted code to train a new model using new offline data. Therefore, the training script should be named `track2/train/train.py` and take an argument `--input_dir` stating the path to the new offline data.
1. The `offline_data` directory contains a combination of selected Waymo and NGSIM datasets.
1. On completion of training, `track2/train/train.py` should save the trained model such that calling the `act(observation)` method of `submission/policy.py::Policy` returns an action.
1. The `track2/submission` folder will be read and evaluated by the same evaluation script as that of Track-1. See evaluation [README.md](../../evaluation/README.md).
1.  Finally, the offline training code in `track2/train` will be manually scrutinised. 


# Examples

## Offline RL

This example uses Convervative Q-learning (CQL) method from [d3rlpy](https://github.com/takuseno/d3rlpy) offline RL library.

**This example is only meant to demonstrate one potential method of developing an offline model using waymo dataset. The trained policy here does not fully solve the task environments.**

### Setup
+ Use `python3.8` to develop your model.
    ```bash
    $ cd <path>/SMARTS/competition/track2/train
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ pip install -e .
    ```
+ SMARTS is used as a dependent package.

### Preparing data
+ Observations: We use a 3-channel rgb birds eye view image plus an extended channel containing the location of the goal as the observation training. So the observation is of the form (4, 256, 256)

+ Actions: The action space (output of the policy) is using dx, dy and dh, which are the value change per step in x, y direction and heading for the ego vehicle in its birds eye view image coordinate. Since dx and dy can not be directly obtained from smarts observation, we have to get displacement change in global coordinate first and use a rotation matrix w.r.t the heading to get dx, dy. The bound for the action space is 
    + dx: [-0.1, 0.1]
    + dy: [0, 2]
    + dh: [-0.1, 0.1]

+ Rewards: The reward use the default reward in SMARTS which is the distance travelled per step plus an extra reward for reaching the goal. Since there is not a "goal" concept in the training set, we use the last point of each trajectory as the goal position for training. 



### Train
1. Train
    ```bash
    $ cd <path>/SMARTS/competition/track2/train
    $ python3.8 train.py --input_dir=<path_to_data> --ouput_dir=<path_to_saved_model>
    ```

    The default value for `input_dir` and `output_dir` are `/offline_dataset` and `/output`
 1. Since we can not load too many images in the training dataset at each time, we are training using data in one scenario at each time. After the end of each training iteration, we will save the model in `<path>/SMARTS/competition/track2/train/d3rlpy_logs/<scenario_index>`. The next trainig iteration will keep training on the latest trained model.  

## Imitation Learning

### Preparing data
The directory `/offline_dataset` looks like

```text
/offline_dataset                
    ├── <scenario_id>               # each scene in tfrecord
    |   ├── <time>_<vehicle_id>.png # ego-centric bird-eye view image
    |   |  .
    |   |  .
    |   |  <vehicle_id>.pkl # state space of the vehicle
    ├── <scenario_id>
    |   .
    |   .
   
```

### Data
+ Observations: We use a 3-channel rgb birds eye view image of the form (3, 256, 256) plus oridented and normalized dx & dy between the position of the ego vehicle and the goal location at each time step. dx & dy are calculated by first orienting both the current position and the goal location with respect to the current heading then substracting the oriented current position from the oriented goal location. dx & dy are then normalized using MinMaxScaler whose bound is (-0.1, 0.1).

+ Actions: The action space (output of the policy) is using dx, dy and dh, which are the value change per step in x, y direction and heading for the ego vehicle in its birds eye view image coordinate. dh is normalized by multiplying the values by 100. Since dx and dy can not be directly obtained from smarts observation, we have to get displacement change in global coordinate first and use a rotation matrix w.r.t the heading to get dx, dy. The bound for the action space is 
    + dh: [-1, 1]

    In evaluation, the values of predicted dh need to be divided by 100.

+ Rewards: The reward use the default reward in SMARTS which is the distance travelled per step plus an extra reward for reaching the goal. Since there is not a "goal" concept in the training set, we use the last point of each trajectory as the goal position for training. 


### Train
```bash
$ python train.py --dataset_path /offline_dataset \
                    --output_path ./output \
                    [--cache] False \
                    [--learning_rate] 0.001 \
                    [--save_steps] 10 \
                    [--batch_size] 32 \
                    [--num_epochs] 100 \
```
First time running `train.py`, please set `cache=False`, the processed data will be saved to `./output/dataset.npy`. For later use, set `cache=True` and it will use the cached dataset.