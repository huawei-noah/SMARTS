# Offline Reinforcement Learning

## Objective
Objective is to train a single **offline** reinforcement learning (RL) policy capable of controlling single-agent or multi-agent to complete different tasks in various scenarios. In each scenario the ego-agents must drive towards their respective goal locations. 

## Data and RL Model
1. For offline RL training, use Waymo datasets.
1. Waymo utilities from https://github.com/huawei-noah/SMARTS/tree/saul/waymo-extraction/smarts/waymo, can be used to  
   + to browse Waymo dataset, and 
   + to extract Waymo data into SMARTS observations.
1. Trained RL model should accept multi-agent observation of the format `Dict[agent_name: agent_observation]`. Observation space for each agent is `smarts.core.sensors.Observation`. For more details on the contents of `Observation` class, see https://github.com/huawei-noah/SMARTS/blob/comp-1/smarts/core/sensors.py#L186
1. Each agent's mission goal is given in the observation returned at each time step. The mission goal could be accessed as `observation.ego_vehicle_state.mission.goal.position` which gives an `(x, y, z)` map coordinate of the goal location.
1. Trained RL model should output multi-agent action of the format `Dict[agent_name: agent_action]`. Action space for each agent is `smarts.core.controllers.ActionSpaceType.TargetPose` which is a sequence of [x-coordinate, y-coordinate, heading, and time-delta].

## Process Overview
1. Use `python3.8` to develop your model.
1. The submitted folder structure for Track-2 should be as follows. The folder and file names are to be maintained.
    ```text
    track2                       # Main folder.
    ├── train                    # Contains code to train an offline RL model.
    │   ├── train.py             # Primary training script for training a new model.
    │   ├── ...                  # Other necessary training files.
    |   .
    |   .
    |   .
    └── submission                       
        ├── policy.py            # A policy with an act method, wrapping the trained RL model.
        ├── requirements.txt     # Dependencies needed to run the RL model.
        ├── explanation.md       # Brief explanation of the key techniques used in developing the submitted model.
        ├── ...                  # Other necessary submission files.
        .
        .
        .
    ```
1. The `track2/train/train.py` code should be capable of reading in new offline data fed in by the competition organizers, train a new RL model offline from scratch, and save the newly trained model into a `Policy` class in `track2/submission/policy.py` file.
1. The command 
    ```bash
    $ python3.8 track2/train/train.py --input_dir=<path>/offline_data
    ```
    will be executed on the submitted code to train a new RL model offline using new offline data. Therefore, the training script should be named `track2/train/train.py` and take an argument `--input_dir` stating the path to the new offline data.
1. The `offline_data` directory contains selected Waymo datasets.
1. On completion of training, `track2/train/train.py` should save the trained model such that calling the `act(observation)` method of `submission/policy.py::Policy` returns an action.
1. The `track2/submission` folder will be read and evaluated by the same evaluation script as that of Track-1. See evaluation [README.md](../../evaluation/README.md).
1. Finally, the offline RL training code in `track2/train` will be manually scrutinised. 

## Submission
Once an RL model has been trained offline, place all necessary files to run the saved model for inference inside the folder named `submission`. 

The files named `policy.py`, `requirements.txt`, and `explanation.md`, must be included within this folder. Its contents are identical to that of Track-1 and they are explained at 
+ [Policy](../../track1/submission/README.md#Policy)
+ [Wrappers](../../track1/submission/README.md#Wrappers)
+ [Requirements](../../track1/submission/README.md#Requirements)
+ [Explanation](../../track1/submission/README.md#Explanation)

## Dockerfile
The submitted `Track2` folder


## Submit to Codalab
+ Zip the entire `track2` folder. 
    + If the `track2` folder is located at `<path>/SMARTS/competition/track2`, then run the following to easily create a zipped folder. 
        ```bash
        $ cd <path>/SMARTS/competition
        $ make track2_submission.zip 
        ```
+ Upload the `track2.zip` to CodaLab.
    + Go to the [CodaLab competition page](https://codalab.lisn.upsaclay.fr/).
    + Click `My Competitions -> Competitions I'm In`.
    + Select the SMARTS competition.
    + Click `Participate -> Submit/View Results -> Submit`
    + Upload the zipped folder.

