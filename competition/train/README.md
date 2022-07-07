# Multiple Scenarios
Objective is to train a single reinforcement learning (RL) policy capable of controlling single-agent or multi-agent to complete different tasks in various scenarios. In each scenario the ego-agents must drive towards their respective goal locations. The scenario names and their missions are as follows.

- 1_to_2lane_left_turn_c 
    + A single ego agent must make a left turn at an uprotected cross-junction.
- 1_to_2lane_left_turn_t 
    + A single ego agent must make a left turn at an uprotected T- junction.
- 3lane_merge_multi_agent
    + One ego agent must merge onto a 3-lane highway from an on-ramp, while another agent must drive on the highway yielding appropriately to traffic from the on-ramp.
- 3lane_merge_single_agent
    + One ego agent must merge onto a 3-lane highway from an on-ramp.
- 3lane_cruise_multi_agent
    + Three ego agents must cruise along a 3-lane highway with traffic.
- 3lane_cruise_single_agent
    + One ego agents must cruise along a 3-lane highway with traffic.
- 3lane_cut_in
    + One ego agent must navigate (either slow down or change lane) when its path is cut-in by another traffic vehicle.
- 3lane_overtake
    + One ego agent, while driving along a 3-lane highway, must overtake a column of slow moving traffic vehicles and return to the same lane which it started at.

# Example
This example uses PPO from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) reinforcement learning library.

The following setup, train, evaluate, and docker, instructions are meant for getting started and running the training example provided.

**This example is only meant to demonstrate one potential method of developing an RL model for the `multi-scenario-v0` environments. The trained RL policy here does not fully solve the task environments.**

## Setup
+ This example is packaged using its own `setup.py` file.
    ```bash
    $ git clone https://github.com/huawei-noah/SMARTS.git
    $ cd <path>/SMARTS/competition/train
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ pip install -e .
    ```
+ SMARTS is used as a dependent package.
+ Please use `python3.8`.

## Environment
+ Make separate environments for each scenario. 
+ Individual environments can be instantiated as follows, by inserting the name of the desired scenario.
    ```python
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario=<insert-name-of-scenario>,
        img_meters=img_meters, # Optional. Camera sensor ground size. Defaults to 64 meters.
        img_pixels=img_pixels, # Optional. Camera sensor pixel size. Defaults to 256 pixels.
        headless=True, # If False, enables Envision display. Defaults to True.
        sumo_headless=True. # If False, enables sumo-gui display. Defaults to True. 
    )
    ```
+ `img_meters` and `img_pixels` are optional parameters used to control the size in meters and in pixels, respectively, of the camera sensor output.
+ SMARTS is multiagent simulator which returns
    + `observations`: Dict[agent_name: agent_observation]
    + `rewards`: Dict[agent_name: agent_reward]
    + `dones`: Dict[agent_name: agent_done, \_\_all\_\_: all_done]
    Here `dones["__all__"]` becomes true when all agents are done, else false.
+ Default reward is distance travelled per step in meters.
+ Action space for each agent: `smarts.core.controllers.ActionSpaceType.Continuous`
    + ```python
      action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
      ```
    + Throttle: [0,1]
    + Brake: [0,1]
    + Steering: [-1,1]
+ Observation space for each agent: `smarts.core.sensors.Observation`
    + For more details on the contents of `Observation` class, see https://github.com/huawei-noah/SMARTS/blob/master/smarts/core/sensors.py#L179
+ In each environment, each agent's mission goal is given in the observation returned at each time step. The mission goal could be accessed as `observation.ego_vehicle_state.mission.goal.position` which gives an `(x, y, z)` map coordinate of the goal location.


## Visualize
+ Two methods are provided to visualize the simulation.
+ Envision:
    1. First, run the server in a separate terminal. 
        ```bash
        $ cd <path>/SMARTS/competition/train 
        $ scl envision start --scenarios ./.venv/lib/python3.8/site-packages/smarts/scenarios
        ```
    1. While instantiating the environment, set `headless` to `False`, i.e., 
        ```python 
        env = gym.make(
            "smarts.env:multi-scenario-v0",
            scenario=<insert-name-of-scenario>,
            headless=False,
        )
        ```
    1.  Go to `localhost:8081` to view the simulation.
+ SUMO GUI: 
    1. While instantiating the environment, set `sumo_headless` to `False`, i.e.,
        ```python 
        env = gym.make(
            "smarts.env:multi-scenario-v0",
            scenario=<insert-name-of-scenario>,
            sumo_headless=False,
        )
        ```
    1. A SUMO gui will automatically pop up during simulation.

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/competition/train
    $ python3.8 run.py
    ```
1. Trained model is saved into `<path>/SMARTS/competition/train/logs/<folder_name>` folder.

## Evaluate
1. To enable visualization, run the Envision server in a separate terminal.
    ```bash
    $ cd <path>/SMARTS/competition/train
    $ scl envision start --scenarios ./.venv/lib/python3.8/site-packages/smarts/scenarios
    ```
1. Start a new terminal and run the following to evaluate your model.
    ```bash
    $ cd <path>/SMARTS/competition/train
    $ python3.8 run.py --mode=evaluate --model="./logs/<folder_name>/<model>" --head
    ```
    + Add option `--head` to enable visualization in Envision.
1. If Envision was enabled, go to `localhost:8081` to view the simulation in Envision.

## Docker
1. Train a model inside docker
    ```bash
    $ cd <path>/SMARTS
    $ docker build --file=./competition/train/Dockerfile --network=host --tag=multi_scenario .
    $ docker run --rm -it --network=host --gpus=all multi_scenario
    (container) $ cd /src/competition/train
    (container) $ python3.7 run.py
    ```