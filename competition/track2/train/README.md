# Offline Learning

## Objective
Objective is to train a single **offline** learning policy capable of controlling single-agent or multi-agent to complete different tasks in various scenarios. In each scenario the ego-agents must drive towards their respective goal locations. The challenge of this track is that participants have to develop a model that can be trained on offline dataset, without interactions with the simulator or any online adjustments. We require participants to submit the codes (not models) for us to train on the offline dataset, and then we used the trained model for evaluation. Examples of approaches for track 2 are Imitation Learning and Offline Reinforcement Learning. 

## Data and RL Model
1. For offline training, we provide information and tools for extracting training dataset from two Naturalistic Autonomous Driving datasets: [Waymo Open Motion dataset](https://waymo.com/open/data/motion/) and [Next Generation Simulation (NGSIM)](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).
1. Participants can download these 2 datasets, and use the provided tools to extract training data. 
1. It is required that the training dataset to be compatible with the SMARTS simulator. you can find the tools for visualize and extract the training data compatible with SMARTS here:
   1. Waymo utilities from https://github.com/huawei-noah/SMARTS/tree/saul/waymo-extraction/smarts/waymo, can be used to  
      + to browse Waymo dataset, and 
      + to extract offline training data from Waymo in form of SMARTS observation space.
   1. NGSIM utilities from ????, can used to 
      + to simulate NGSIM dataset in SMARTS, and
      + to extract offline training data from NGSIM in form of  SMARTS observation space.
1. We have provided a subset of each Waymo and NGSIM datasets that we found to include useful and interesting trajectories for training. You can use this subset to limit the size of your training set and exclude irrelevant scenes. The data extraction tools for each dataset are capable to extracting data from the provided subset. (these subsets are subject to be updated!)
1. Trained RL model should accept multi-agent observation of the format `Dict[agent_name: agent_observation]`. Observation space for each agent is `smarts.core.sensors.Observation`. For more details on the contents of `Observation` class, see https://github.com/huawei-noah/SMARTS/blob/comp-1/smarts/core/sensors.py#L186
1. Each agent's mission goal location is given in the observation returned at each time step. The mission goal could be accessed as `observation.ego_vehicle_state.mission.goal.position` which gives an `(x, y, z)` map coordinate of the goal location.
1. Trained RL model should output multi-agent action of the format `Dict[agent_name: agent_action]`. Action space for each agent is `smarts.core.controllers.ActionSpaceType.TargetPose` which is a sequence of [x-coordinate, y-coordinate, heading, and time-delta].

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
    $ python3.8 track2/train/train.py --input_dir=<path>/offline_data
    ```
    will be executed on the submitted code to train a new model using new offline data. Therefore, the training script should be named `track2/train/train.py` and take an argument `--input_dir` stating the path to the new offline data.
1. The `offline_data` directory contains a combination of selected Waymo and NGSIM datasets.
1. On completion of training, `track2/train/train.py` should save the trained model such that calling the `act(observation)` method of `submission/policy.py::Policy` returns an action.
1. The `track2/submission` folder will be read and evaluated by the same evaluation script as that of Track-1. See evaluation [README.md](../../evaluation/README.md).
1.  Finally, the offline training code in `track2/train` will be manually scrutinised. 
