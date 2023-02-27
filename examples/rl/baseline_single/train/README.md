$ cd <path>/SMARTS/examples/rl/baseline
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel setuptools
$ pip install -e ./../../../.[camera_obs,dev,doc]
$ pip install -e ./inference/
$ python3.8 train/run1.py