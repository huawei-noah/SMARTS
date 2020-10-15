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

## Running

To run the training procedure,

```bash
# from benchmarks/
$ python run.py <scenario> -f <config_file>
# E.x. python run.py scenarios/intersections/4lane -f agents/ppo/baseline-continuous-control.yaml
```

To run the evaluation procedure,

```bash
# from benchmarks/
$ python evaluate.py <scenario> -f <config_file> --checkpoint <checkpoint_path>
# E.x. python evaluate.py scenarios/intersections/4lane \
#          -f agents/ppo/baseline-continuous-control.yaml \
#          --checkpoint ./log/results/run/4lane-4/PPO_Simple_977c1_00000_0_2020-10-14_00-06-10
```
