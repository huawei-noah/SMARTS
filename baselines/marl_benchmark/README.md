# Multi-Agent Benchmarks

This directory contains the scenarios, training environment, and agents used in the CoRL20 paper: [SMARTS: Scalable Multi-Agent ReinforcementLearning Training School for Autonomous Driving](...).

**Contents,**
- `agents/`: YAML files and some RLlib-based policy implementations
- `metrics/`: Class definition of metrics (default by a basic Metric class)
- `networks/`: Custom network implementations
  - `communicate.py`: Used for Networked agent learning
- `scenarios/`: Contains three types of scenarios tested in the paper
- `wrappers/`: Environment wrappers
- `evaluate.py`: The evaluation program
- `run.py`: Executes multi-agent training

## Setup
```bash
# git clone ...
cd <projec/baseline/marl_benchmark>

# setup virtual environment; presently at least Python 3.7 and higher is officially supported
python3.7 -m venv .venv

# enter virtual environment to install all dependencies
source .venv/bin/activate

# upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
pip install --upgrade pip

# install the current version of python package with the rllib dependencies
pip install -e .

## Running

If you have not already, it is suggested you checkout the benchmark branch.
```bash
$ git checkout marl_benchmark
```

To run the training procedure,

```bash
# from baselines/marl_benchmark/
$ python3.7 run.py <scenario> -f <config_file>
# E.x. python3.7 run.py scenarios/intersections/4lane -f agents/ppo/baseline-lane-control.yaml
```

To run the evaluation procedure for multiple algorithms,

```bash
# from baselines/marl_benchmark/
$ python evaluate.py <scenario> -f <config_files>
# E.x. python3.7  evaluate.py scenarios/intersections/4lane \
#          -f agents/ppo/baseline-lane-control.yaml \
#          --checkpoint ./log/results/run/4lane-4/PPO_Simple_977c1_00000_0_2020-10-14_00-06-10
```
