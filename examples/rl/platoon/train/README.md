## Instructions

## Train
+ Setup
    ```bash
    # In terminal-A
    $ cd <path>/SMARTS/examples/rl/platoon
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip wheel
    $ pip install -e ./../../../.[camera_obs,argoverse]
    $ pip install -e ./inference/
    ```
+ Train without visualization
    ```bash
    # In terminal-A
    $ python3.8 train/run.py
    ```
+ Train with visualization
    ```bash
    # In terminal-A
    $ python3.8 train/run.py --head
    ```
    ```bash
    # In a different terminal-B
    $ scl envision start
    # Open http://localhost:8081/
    ```

## Evaluate
```bash
$ cd <path>/SMARTS
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -e .[camera_obs,argoverse]
$ scl zoo install examples/rl/platoon/inference
$ scl benchmark run driving_smarts_2023 examples.rl.platoon.inference:contrib-agent-v0 --auto-install
```