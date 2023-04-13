

Hi @atanas-kom,

I see that you would like to develop a reinforcement learning model and evaluate it on the `driving_smarts_2022` benchmark.

We are in the midst of developing a new benchmark, named `driving_smarts_2023.3`. The mechanics of using a SMARTS environment, training models, and evaluating using benchmarks, are similar across different benchmarks, with the only difference being the underlying tasks, maps, and scoring formula.

Consider reading the [docs](https://smarts.readthedocs.io/en/latest/benchmarks/driving_smarts_2023_3.html) on `driving_smarts_2023.3` benchmark. It explains the task, observation space, action space, and required code structure. Additionally, an [example](https://github.com/huawei-noah/SMARTS/blob/master/examples/rl/platoon) for training & saving a model, registering an user's agent, and evaluating the model, is provided with instructions. Note that the `scl benchmark run driving_smarts_2023_3 examples.rl.platoon.inference:contrib-agent-v0 --auto-install` command, and scoring, would only be functional after PR #1952 is merged. In the meantime, do the following instead to evaluate the given example after training.

```bash
# On terminal-A
$ cd <path>/SMARTS/examples/rl/platoon
$ source ./.venv/bin/activate
$ scl envision start
# Open http://localhost:8081/

# On a different terminal-B
$ cd <path>/SMARTS/examples/rl/platoon
$ source ./.venv/bin/activate
$ python3.8 train/run.py --mode=evaluate --model=<path to desired saved model.zip> --head
```

On the question of "relation between a specific interface and action space", the user is able to specify their desired action space through the agent interface as shown [here](). SMARTS provides several action spaces, see [here](https://smarts.readthedocs.io/en/latest/api/smarts.core.controllers.html#smarts.core.controllers.ActionSpaceType), but only selected action spaces are allowed in the benchmarks. For example, the `driving_smarts_2023.3` benchmark currently only allows `ActionSpaceType.Continuous`.

I hope that walking through the example given for `driving_smarts_2023.3` benchmark helps to provide a better understanding on how to use SMARTS and its benchmarks. Wishing you the best to achieve an excellent driving model on the `driving_smarts_2022` benchmark.
