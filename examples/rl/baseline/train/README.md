## Instructions

```bash
$ cd <path>/SMARTS/examples/rl/baseline
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --force-reinstall --no-cache-dir pip wheel setuptools==65.5.0
$ pip install --force-reinstall --no-cache-dir -e ./../../../.[camera_obs,dev,doc,argoverse]
$ pip install --force-reinstall --no-cache-dir -e ./inference/
$ python3.8 train/run.py --head
```

On a different terminal
```bash
$ scl envision start
```