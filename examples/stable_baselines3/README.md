# SMARTS Stable Baselines3 Example
This example illustrates the training of an ego agent to drive, as fast and as far as possible, in traffic using the Stable Baselines3 (https://github.com/DLR-RM/stable-baselines3) reinforcement-learning algorithms.

Ego agent earns rewards based on the distance travelled and is penalised for colliding with other vehicles and for going off-road.

## Trained agent driving in traffic
![](./docs/_static/driving_in_traffic.gif)

## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/stable_baselines3
$ python3.7 -m venv ./venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
```

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/examples/stable_baselines3
    $ python3.7 run.py 
    ```
1. Trained model is saved into `<path>/SMARTS/examples/stable_baselines3/logs/<folder_name>` folder.

## Evaluate
1. Evaluate
    ```bash
    $ cd <path>/SMARTS/examples/stable_baselines3
    $ scl envision start -s ./scenarios &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/stable_baselines3/logs/<folder_name>" --head
    ```
1. Go to `localhost:8081` to view the simulation in Envision.

## Docker
1. Build and train inside docker container
    ```bash
    $ cd <path>/SMARTS
    $ docker build --file=<path>/SMARTS/examples/stable_baselines3/Dockerfile --network=host --tag=sb3 <path>/SMARTS
    $ docker run --rm -it --network=host --gpus=all sb3
    (container) $ cd /src/examples/stable_baselines3
    (container) $ python3.7 run.py
    ```