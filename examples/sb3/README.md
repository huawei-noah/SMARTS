# Stable Baselines3
This example illustrates the training of an ego agent to drive, as fast and as far as possible, in traffic using the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) reinforcement-learning library. Here, we used the PPO algorithm.

The ego agent earns reward for the distance travelled per-step and is penalised for colliding with other vehicles and for going off-road.

This example is only to demonstrate the use of SB3 library with SMARTS task environments. The trained agents may not solve the task environment.

## Observation space
+ Topdown RGB image
    + size (width x height): 64 pixels x 64 pixels
    + resolution: 1 meter/pixel
```
observation_space = gym.spaces.Dict({
    gym.spaces.Box(low=0, high=255, shape=(64,64,3), dtype=np.uint8),
})
```

## Action space:
+ Throttle: [0,1]
+ Brake: [0,1]
+ Steering: [-1,1]
```
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/sb3
$ python3.7 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
```

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/examples/sb3
    $ python3.7 run.py 
    ```
1. Trained model is saved into `<path>/SMARTS/examples/sb3/logs/<folder_name>` folder.
1. Monitor the RL agent during or after the training
    ```bash
    $ cd <path>/SMARTS/examples/sb3
    $ tensorboard --logdir ./tensorboard/
    ```

## Evaluate
1. Evaluate
    ```bash
    $ cd <path>/SMARTS/examples/sb3
    $ scl envision start -s ./scenarios &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/sb3/logs/<folder_name>" --head
    ```
1. Go to `localhost:8081` to view the simulation in Envision.


## Docker
1. Train a model inside docker
    ```bash
    $ cd <path>/SMARTS
    $ docker build --file=<path>/SMARTS/examples/sb3/Dockerfile --network=host --tag=sb3 <path>/SMARTS
    $ docker run --rm -it --network=host --gpus=all sb3
    (container) $ cd /src/examples/sb3
    (container) $ python3.7 run.py
    ```

