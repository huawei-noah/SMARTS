# Driving in Traffic
This example illustrates the training of an ego agent to drive in traffic using DreamerV2 (https://github.com/danijar/dreamerv2) reinforcement-learning algorithm.

Ego agent earns rewards based on the distance travelled and is penalised for colliding with other vehicles and for going off-road.

## Trained agent driving in traffic
![](./docs/_static/driving_in_traffic.gif)

## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/driving_in_traffic
$ python3.7 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
$ scl scenario build-all --clean ./scenarios/loop
```

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/examples/driving_in_traffic
    $ python3.7 run.py 
    ```
1. Trained model is saved into `<path>/SMARTS/examples/driving_in_traffic/logs/<folder_name>` folder.

## Evaluate
1. Evaluate
    ```bash
    $ cd <path>/SMARTS/examples/driving_in_traffic
    $ scl envision start -s ./scenarios &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/driving_in_traffic/logs/<folder_name>" --head
    ```
1. Go to `localhost:8081` to view the simulation in Envision.
