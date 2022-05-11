# Merge
This example illustrates the training of an ego agent to merge into a freeway (also called an expressway). The agent is tasked to drive along an entrance ramp, an acceleration lane, and then merge into the freeway. Finally, the ego agent should change lanes to the fastest lane or the rightmost lane of the freeway. Here, DQN algorithm from the [TF-Agents](https://www.tensorflow.org/agents) reinforcement-learning library is used.

The ego agent earns reward for the distance travelled per-step and is penalised for colliding with other vehicles, for going off-road, for going off-route, or for driving on road-shoulder.

This example is only meant to demonstrate the use of TF-Agents library with SMARTS task environments. The trained agent does not solve the task environment.

## Trained agent merging into a freeway
![](./docs/_static/merge.gif)

## Observation space
+ Topdown RGB image
    + size (width x height): 128 pixels x 128 pixels
    + resolution: 50 meter / 128 pixel
```
observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
```

## Action space
+ Throttle: [0,1]
+ Brake: [0,1]
+ Steering: [-1,1]
```
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

## Colab
1. Try it here: [![here](./docs/_static/colab-badge.svg)](https://colab.research.google.com/github/huawei-noah/SMARTS/blob/merge-v0/examples/rl/merge/merge.ipynb)

## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/rl/merge
$ python3.7 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
```

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/examples/rl/merge
    $ python3.7 run.py 
    ```
1. Trained model is saved into `<path>/SMARTS/examples/rl/merge/logs/<folder_name>` folder.
1. Monitor the RL agent during or after the training using tensorboard
    ```bash
    $ cd <path>/SMARTS/examples/rl/merge
    $ tensorboard --logdir ./logs/
    ```

## Evaluate
1. Start
    ```bash
    $ cd <path>/SMARTS/examples/rl/merge
    $ scl envision start --scenarios ./.venv/lib/python3.7/site-packages/scenarios/merge &
    ```
1. Run
    + Evaluate your own model 
        ```bash
        $ python3.7 run.py --mode=evaluate --model="./logs/<folder_name>/<model>" --head
        ```
    + Evaluate pre-trained agent
        ```bash
        $ curl -o ./logs/pretrained/merge.zip --create-dirs -L https://github.com/Adaickalavan/SMARTS-zoo/raw/main/merge-v0/DQN_5800000_steps.zip
        $ python3.7 run.py --mode=evaluate --model="./logs/pretrained/merge" --head
        ```
1. Go to `localhost:8081` to view the simulation in Envision.

## Docker
1. Train a model inside docker
    ```bash
    $ cd <path>/SMARTS
    $ docker build --file=./examples/rl/merge/Dockerfile --network=host --tag=merge .
    $ docker run --rm -it --network=host --gpus=all merge
    (container) $ cd /src/examples/rl/merge
    (container) $ python3.7 run.py
    ```