# Traffic Navigation
This example illustrates the training of an ego agent to drive in traffic using DreamerV2 (https://github.com/danijar/dreamerv2) reinforcement-learning algorithm.

Ego agent earns rewards based on the distance travelled and is penalised for colliding with other vehicles or for going off-road.

## Trained agent driving in traffic


## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/driving_in_traffic
$ python3.7 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
```

## Train
1. Train and save the trained model into `<path>/SMARTS/examples/driving_in_traffic/logs/<folder_name>` folder.
    ```bash
    $ cd <path>/SMARTS/examples/driving_in_traffic
    $ python3.7 run.py 
    ```

## Evaluate
1. Then execute,
    ```bash
    $ cd <path>/SMARTS/examples/driving_in_traffic
    $ scl envision start -s ../../scenarios &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/dreamer/logs/<folder_name>"
    ```
1. Go to `localhost:8081` to view the simulation in Envision.


## Docker
1. Build and train inside docker container
```bash
$ cd <path>/SMARTS
$ docker build -t driving_in_traffic --network=host -f <path>/SMARTS/examples/driving_in_traffic/Dockerfile <path>/SMARTS
$ docker run --rm -it --network=host driving_in_traffic 
(container) $ python3.7 run.py
```