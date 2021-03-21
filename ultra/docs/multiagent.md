# Multi-agent Experiments in ULTRA

## Summary

ULTRA supports multi-agent experiments for independently learning agents.

## Setup

The following steps will show how to create a multi-agent task with 4 agents.
1. Generate the maps used by the ULTRA scenarios (if this has not been done already):
   ```sh
   $ scl scenario build-all ultra/scenarios/pool/
   ```
2. Create a multi-agent task:
   ```sh
   $ touch ultra/scenarios/task0-4agents/config.yaml
   ```
   Edit this `config.yaml` file to include `ego_missions` and `levels`.
   ```yaml
   levels:
     no-traffic:  # This is the level name.
       train:
         total: 10  # There will be 10 training scenarios.
         ego_missions:  # The number of missions determines the number of agents.
         - start: south-SN  # Turn left going from South to West.
           end:   west-EW
         - start: west-WE   # Turn left going from West to North.
           end:   north-SN
         - start: north-NS  # Turn left going from North to East.
           end:   east-WE
         - start: east-EW   # Turn left going from East to South.
           end:   south-NS
         intersection_types:
           2lane_c:
             percent: 1.0  # 100% of these scenarios will be 2 lane, c-intersections.
             specs: [[50kmh,no-traffic,0.34],[70kmh,no-traffic,0.33],[100kmh,no-traffic,0.33]]
       test:
         total: 2  # There will be 2 testing scenarios.
         ego_missions:  # Testing scenarios currenly only support one agent.
         - start: south-SN  # Turn left going from South to West.
           end:   west-EW
         intersection_types:
           2lane_c:
             percent: 1.0  # 100% of these scenarios will be 2 lane, c-intersections.
             specs: [[50kmh,low-density,0.34],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
   ```
   > This will create a task consisting of 12 scenarios (10 training, 2 testing). Each scenario supports 4 agents, and missions are shuffled so that agents do different missions in different training scenarios. Each agent will be tested separately, attempting to perform a left-turn from the South lane to the West lane in low-density traffic of various speeds.
3. Add your task to `ultra/config.yaml`:
   ```yaml
     task0-4agents:  # The name of the task.
       no-traffic:  # The level name.
         train: "ultra/scenarios/task0-4agents/train_no-traffic*"  # The relative path to the training scenarios.
         test:  "ultra/scenarios/task0-4agents/test_no-traffic*"  # The relative path to the testing scenarios.
   ```
4. Generate your task's scenarios:
   ```sh
   $ python ultra/scenarios/interface.py generate --task 0-4agents --level no-traffic
   ```
   > Scenario folders should appear under `ultra/scenarios/task0-4agents/`.
5. (Optional) Start Envision to see the experiments in your browser:
   ```sh
   $ ./ultra/env/envision_base.sh
   ```

## Training Agents

Train baseline agents on the train scenarios of your task:
```sh
$ python ultra/train.py --task 0-4agents --level no-traffic --policy bdqn,bdqn,bdqn,bdqn
```
> This will train 4 BDQN baseline agents in the task and create an experiment directory. Feel free to try other combinations of agents.

## Evaluating Agents

Once training is complete (or deemed sufficient), evaluate your trained agents on the test scenarios of your task.
```sh
$ python ultra/evaluate.py --task 0-4agents --level no-traffic --experiment-dir logs/<your-experiment-directory>/ --models logs/<your-experiment-directory>/models/000/
```
> Running ULTRA's evaluation will not work unless an `agent_metadata.pkl` file is available in the experiment directory. Experiments from older versions of ULTRA (< 0.2) do not have this file.
