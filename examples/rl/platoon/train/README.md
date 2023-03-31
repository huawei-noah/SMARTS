## Instructions

# To train
```bash
$ cd <path>/SMARTS/examples/rl/platoon
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -e ./../../../.[camera_obs,argoverse]
$ pip install -e ./inference/
$ python3.8 train/run.py --head
```

On a different terminal
```bash
$ scl envision start
```

# To evaluate
```bash
$ cd <path>/SMARTS
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -e .[camera_obs,argoverse]
$ scl zoo install examples/rl/platoon/inference
$ scl benchmark run driving_smarts_2023 examples.rl.platoon.inference:contrib-agent-v0 --auto-install
```