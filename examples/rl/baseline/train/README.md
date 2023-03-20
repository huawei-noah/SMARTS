## Instructions

```bash
$ cd <path>/SMARTS/examples/rl/baseline
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -e ./../../../.[camera_obs,dev,doc,argoverse]
$ pip install -e ./inference/
$ python3.8 train/run.py --head
```

On a different terminal
```bash
$ scl envision start
```