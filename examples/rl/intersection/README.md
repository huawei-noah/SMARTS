# Intersection
This example illustrates the training of an ego agent to drive and make an uprotected left turn in an all-way-stop intersection, in traffic using the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) reinforcement-learning library. Here, we used the PPO algorithm.

The ego agent earns reward for the distance travelled per-step and is penalised for colliding with other vehicles, for going off-road, for going off-route, or for going on road-shoulder.

This example is only to demonstrate the use of SB3 library with SMARTS intersection-v0 task environments. The trained agents may not solve the task environment.

## Observation space
+ Topdown RGB image
    + size (width x height): 112 pixels x 112 pixels
    + resolution: 50 meter / 112 pixel
```
observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 112, 112), dtype=np.uint8)
```

## Action space
+ Throttle: [0,1]
+ Brake: [0,1]
+ Steering: [-1,1]
```
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/rl/intersection
$ python3.7 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
```

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/examples/rl/intersection
    $ python3.7 run.py 
    ```
1. Trained model is saved into `<path>/SMARTS/examples/rl/intersection/logs/<folder_name>` folder.
1. Monitor the RL agent during or after the training using tensorboard
    ```bash
    $ cd <path>/SMARTS/examples/rl/intersection
    $ tensorboard --logdir ./tensorboard/
    ```

## Evaluate your model
1. Run
    ```bash
    $ cd <path>/SMARTS/examples/rl/intersection
    $ scl envision start --scenarios ./.venv/lib/python3.7/site-packages/scenarios/intersections &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/rl/intersection/logs/<folder_name>" --head
    ```
1. Go to `localhost:8081` to view the simulation in Envision.


## Evaluate pre-trained model
1. Run
    ```bash
    $ cd <path>/SMARTS/examples/rl/intersection
    $ curl --o <path>/SMARTS/examples/rl/intersection/logs/pretrained/intersection.zip https://github.com/Adaickalavan/SMARTS-models/raw/main/intersection-v0/PPO_6200000_steps.zip
    $ scl envision start --scenarios ./.venv/lib/python3.7/site-packages/scenarios/intersections &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/rl/intersection/logs/pretrained/intersection.zip" --head
    ```
1. Go to `localhost:8081` to view the simulation in Envision.


## Docker
1. Train a model inside docker
    ```bash
    $ cd <path>/SMARTS
    $ docker build --file=<path>/SMARTS/examples/rl/intersection/Dockerfile --network=host --tag=intersection <path>/SMARTS
    $ docker run --rm -it --network=host --gpus=all intersection
    (container) $ cd /src/examples/rl/intersection
    (container) $ python3.7 run.py
    ```

