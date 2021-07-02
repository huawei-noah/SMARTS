# Hyperparameter Tuning in ULTRA

## Summary

ULTRA supports hyperparameter tuning of single agents.

## Setup

ULTRA's tune is based off of [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

1. Generate the maps used by the ULTRA scenarios (if this has not been done already):
   ```sh
   $ scl scenario build-all ultra/scenarios/pool/experiment_pool/
   ```
2. Generate the scenarios to tune on (e.g. Task 1's easy level):
   ```sh
   $ python ultra/scenarios/interface.py generate --task 1 --level easy
   ```
   > Scenario folders should appear under `ultra/scenarios/task1/`.
3. A config module will need to be created in order to use Tune. See the PPO baseline's [tune_params.py](../ultra/baselines/ppo/ppo/tune_params.py) example and the [Ray Tune Search Space API documentation](https://docs.ray.io/en/latest/tune/api_docs/search_space.html) to create these configs.
4. (Optional) Start Envision to see the experiments in your browser:
   ```sh
   $ ./ultra/env/envision_base.sh
   ```

## Tuning

Tune your agent (e.g. the PPO baseline) on the train scenarios of your task (e.g. Task 1's easy level):
```sh
$ python ultra/tune.py --task 1 --level easy --policy ppo --config-module ultra.baselines.ppo.ppo.tune_params --metric episode_return --mode max
```
> This will tune the PPO baseline agent on Task 1's easy scenarios by sampling configurations from the PPO's tune hyperparameters. It will output the best performing combination of hyperparameters in YAML format determined by which hyperparameter combination maximizes episode return the most. Additionally, evaluation will be performed on the saved models trained with these best hyperparameters.

Available `ultra/tune.py` arguments:
- `--task`: The task to tune on (default is 1).
- `--level`: The level of the task to tune on (default is easy).
- `--policy`: The name of the policy class to tune (default is ppo).
- `--episodes`: The number of training episodes to perform for each sampled config (default is 10000).
- `--max-episode-steps`: The maximum number of steps the agent is allowed to take per episode (default is 10000).
- `--timestep`: The environment timestep in seconds (default is 0.1).
- `--headless`: Whether to run tuning without Envision (default is True).
- `--eval-episodes`: The number of evaluation episodes to perform on the best-performing config after tuning is complete (default is 200).
- `--save-rate`: How often to save the agent, measured in episodes (default is 1000).
- `--seed`: The environment seed (default is 2).
- `--log-dir`: The directory to put the tune experiment's data (default is tune_logs/).
- `--config-module`: The module containing a dictionary variable, 'config', that defines the config to tune (default is ultra.baselines.ppo.ppo.tune_params).
- `--metric`: The metric to optimize for; either episode_length, episode_return, or env_score (default is episode_return).
- `--mode`: How to optimize the metric; either max or min (default is max).
- `--scope`: How to compare amongst the metrics; either all, last, avg, last-5-avg, or last-10-avg (default is last). More information can be found [here](https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis).
- `--grace-period`: Used by the [ASHA Scheduler](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#asha-tune-schedulers-ashascheduler), will not terminate trials that are less than this old (default is episodes / 10).
- `--reduction-factor`: Used by the [ASHA Scheduler](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#asha-tune-schedulers-ashascheduler), sets the halving rate and amount (default is 2).
- `--brackets`: Used by the [ASHA Scheduler](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#asha-tune-schedulers-ashascheduler), sets the number of brackets (default is 1).
- `--num-samples`: The number of sample configs to draw from the provided config (default is 100). More information can be found [here](https://docs.ray.io/en/latest/tune/api_docs/search_space.html).
