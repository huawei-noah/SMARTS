# Track-2: Offline Learning

## Objective
Objective is to develop a single **offline** learning policy capable of controlling single-agent or multi-agent to complete different driving scenarios. In each driving scenario the ego-agents must drive towards their respective goal locations. 

The challenge is to develop a model that can be trained using only offline datasets, without any interactions with any online environments.

Track-2 participants are required to submit their training code for us to train a new model from scratch using hidden offline dataset. The newly trained model will then be evaluated.

## Data and Model
1. For offline training, consider downloading and using the following two naturalistic autonomous driving datasets.
    + [Waymo Open Motion](https://waymo.com/open/data/motion/) 
    + [Next Generation Simulation (NGSIM)](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm). 
1. In order to browse and replay Waymo Dataset, use the `scl waymo overview` and `scl waymo preview` [commands](https://github.com/huawei-noah/SMARTS/blob/comp-1/cli/waymo.py).
1. In order to convert a Waymo dataset into an equivalent SMARTS scenario, use the `scl waymo export` [command](https://github.com/huawei-noah/SMARTS/blob/comp-1/cli/waymo.py).
1. In order to convert a Waymo dataset into an equivalent SMARTS dataset, do the following. 
   + First, convert the Waymo dataset into an equivalent SMARTS scenario and build the scenario.
   + Then, use the [traffic_histories_to_observations.py script](https://github.com/huawei-noah/SMARTS/blob/comp-1/examples/traffic_histories_to_observations.py) to generate equivalent SMARTS observations.
1. A subset of Waymo and NGSIM datasets which have useful and interesting trajectories are provided [here](https://github.com/smarts-project/smarts-project.offline-datasets). This subset may be used to focus the training. The provided data conversion tool can be used to convert these datasets into an equivalent offline SMARTS observation dataset.
1. The trained model should accept multi-agent observation of the format `Dict[agent_name: agent_observation]`. Observation space for each agent is `smarts.core.sensors.Observation`. For more details on the contents of `Observation` class, see [here](https://github.com/huawei-noah/SMARTS/blob/comp-1/smarts/core/sensors.py#L186).
1. Each agent's mission goal is given in the observation returned at each time step. The mission goal could be accessed as `observation.ego_vehicle_state.mission.goal.position` which gives an `(x, y, z)` map coordinate of the goal location.
1. Trained model should output multi-agent action of the format `Dict[agent_name: agent_action]`. Action space for each agent is `smarts.core.controllers.ActionSpaceType.TargetPose` which is a sequence of `[x-coordinate, y-coordinate, heading, and time-delta]`. Use `time-delta=0.1`.

## Process Overview
### Folder Structure
1. The structure of the zipped folder uploaded for Track-2 should be as follows. The folder and file names are to be maintained.
    ```text
    track2                       # Main folder.
    ├── train                    # Contains code to train a model offline.
    │   ├── train.py             # Primary training script for training a new model.
    │   ├── ...                  # Other necessary training files.
    |   .
    |   .
    |   .
    ├── submission                       
    |    ├── policy.py            # A policy with an act method, wrapping the saved model.
    |    ├── requirements.txt     # Dependencies needed to run the model.
    |    ├── explanation.md       # Brief explanation of the key techniques used in developing the submitted model.
    |    ├── ...                  # Other necessary files for inference.
    |    .
    |    .
    |    .
    └── Dockerfile                # Dockerfile to build and run the training code.
    ```
1. Do not include any pre-trained models within the submitted folder for Track-2.

### Train Folder
1. Use `python3.8` to develop your model.
1. The `track2/train/train.py` code should be capable of reading in new offline data fed in by the competition organizers, train a new model offline from scratch, and automatically save the newly trained model into the `track2/submission` folder.

### Submission Folder
1. On completion of training, the `track2/train/train.py` code should automatically save the trained model into the `track2/submission` folder. 
1. Place all necessary files to run the saved model for inference inside the `track2/submission` folder. 
1. The files named `policy.py`, `requirements.txt`, and `explanation.md`, must be included within this folder. Its contents are identical to that of Track-1 and they are explained at 
    + [Policy](../track1/submission/README.md#Policy)
    + [Wrappers](../track1/submission/README.md#Wrappers)
    + [Requirements](../track1/submission/README.md#Requirements)
    + [Explanation](../track1/submission/README.md#Explanation)

### Dockerfile, DockerHub, Training, and Evaluation
1. The submitted `track2` folder must contain a `track2/Dockerfile`. 
1. Build upon the base Dockerfile provided at `track2/Dockerfile`. Feel free to use any desired base image, install any additional packages, etc.
1. The Docker image should start training upon execution of `docker run` command, hence do not change the `ENTRYPOINT` command and do not change the `track2/entrypoint.sh` script.
1. Build the docker image and push it to [DockerHub](https://hub.docker.com/). 
    ```bash
    $ cd <path>/SMARTS/competition/track2
    $ docker build \
        --file=./Dockerfile \
        --network=host \ 
        --tag=<username/imagename:version>
        .
    ```
1. Provide the link to the DockerHub image in `track2/submission/explanation.md` file.
1. After uploading your Docker image, proceed to zip the entire `track2` folder and submit to Codalab Track-2.
1. In the server, the docker image will be pulled and executed as follows to train a new model. 
    ```bash
    $ docker run --rm -it \
        --volume=<path>/offline_dataset:/SMARTS/competition/offline_dataset
        --volume=<path>/output:/SMARTS/competition/output
        <username/imagename:version>
    ```
1. New offline dataset is made available to the container via a mapped volume at `/SMARTS/competition/offline_dataset` directory. The directory has the following structure.
    ```text
    offline_dataset                
        ├── <scenario_id>                      # One scene of variable time length
        |   ├── <time>_<vehicle_id>.png        # bird-eye view image of <vehicle_id> at <time>
        |   ├── <time>_<vehicle_id>.png        # bird-eye view image of <vehicle_id> at <time>         
        |   |  .
        |   |  .
        |   └── <vehicle_id>.pkl               # state space of <vehicle_id> over all time        
        ├── <scenario_id>                      # One scene of variable time length
        |   ├── <time>_<vehicle_id>.png        # bird-eye view image of <vehicle_id> at <time>
        |   ├── <time>_<vehicle_id>.png        # bird-eye view image of <vehicle_id> at <time>        
        |   |  .
        |   |  .
        |   ├── <vehicle_id>.pkl               # state space of <vehicle_id> over all time
        |   └── <vehicle_id>.pkl               # state space of <vehicle_id> over all time
        |   .
        |   .
    ```
1. The `/SMARTS/competition/offline_dataset` directory contains equivalent SMARTS observations, converted from selected Waymo and NGSIM datasets.
1. Inside the container, on completion of training, the trained model should be saved in `/SMARTS/competition/track2/submission` folder such that calling `/SMARTS/competition/track2/submission/policy.py::Policy.act(obs)` with a multi-agent SMARTS observation as input returns a multi-agent `TargetPose` action as output.
1. The `/SMARTS/competition/track2/submission` folder will be zipped, mapped out from the container, and evaluated by the same evaluation script as that of Track-1. See evaluation [README.md](../evaluation/README.md).
1. During development, it is strongly suggested to submit your zipped `track2/submission` folder to the Validation stage in Codalab, to verify that the evaluation works without errors.
1. Finally, the offline training code in `track2/train` directory will be manually scrutinised. 

### Submit to Codalab
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
    + Click `Participate -> Submit/View Results -> Track 2 -> Submit`
    + Upload the zipped folder.
