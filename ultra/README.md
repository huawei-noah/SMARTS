[![coverage report](https://gitlab.smartsai.xyz/smarts/ULTRA/badges/master/coverage.svg)](https://gitlab.smartsai.xyz/smarts/ULTRA/-/commits/master)
[![pipeline status](https://gitlab.smartsai.xyz/smarts/ULTRA/badges/master/pipeline.svg)](https://gitlab.smartsai.xyz/smarts/ULTRA/-/commits/master)

# ULTRA

Unprotected Left Turn using Reinforcement-learning Agents
---
Ultra provides a gym-based environment using SMARTS for tackling intersection navigation and more specifically unprotected left turn.

Here is the summary of key features:
 - creating customized scenarios with different levels of difficulty
 - defining/analyzing traffic designs including low-mid-high densities
 - defining/analyzing social-vehicle behaviors
 - configurable train/test parameters
 - train/test custom RL algorithms against benchmark results


ignore this step if you already have smarts environment
### Setup
  ```sh
  python3.7 -m venv .smarts
  # 1-activate virtual environment to install all dependencies
  source .smarts/bin/activate
  # 2-install black for formatting (if you wish to contribute)
  pip install black
  # 3-install dependencies
  pip install -e .[train]
  ```

### Generate scenarios

Supported tasks are listed under `ultra/scenarios/`.

- For first time generating the maps:
  ```sh
  $ scl scenario build-all ultra/scenarios/pool
  ```
  > Maps only need to be compiled once and you don't need to run them again unless they are modifed
- Generate scenarios:
  ``` sh
  # task 1 generate 1000 sscenarios (800 train, 200 test) and supports 2 levels of difficulties for more info refer to our documentaion
  $ python ultra/scenarios/interface.py generate --task 1 --level easy
  ```
  > After running the command above, train and test scenarios are added under the task directory

### Run envision in the background
if you are running the train and evalaution in headless mode, ignore this step
```sh
$ ./ultra/env/envision_base.sh
```
envision runs as a background process, you can view the visualization on `localhost:8081/`

### Train and Evalutaion
We currently support PPO/SAC/TD3/DQN policies

- Train
  Training runs the policy with default configuration:
  `
  {train episodes: 1000000, maxsteps: 1200, timestep: 0.1 sec, evaluation rate: 10000, evaluation episodes: 200}
  `
  - For each experiment a folder with timestamp is added automatically under `logs/` and it saves tensorboad log, models and pkls
  - For every 10000 observation, evaluation is called automatically and policy is saved under `logs/your-expierment-name/models/`
  ```sh
  $ python ultra/train.py --task 1 --level easy
  # other arguments example: --episodes 20000 --eval-rate 500 --eval-episodes 200 --timestep 1 --headless
  ```
### Run Evaluation Separately
After training your agent, your models should be saved under `logs/your-experiment-name/models/` and you can re-run the evaluation:
  ```sh
  $ python ultra/evaluate.py --task 1 --level easy --models logs/your-expierment-name/models
  # other arguments --episodes 20000 --timestep 1 --headless
  ```

### Docker
- SMARTS is pre-installed and ULTRA source files are copied automatically into Docker image.
- Build a docker image alone
  ```sh
  $ cd path/to/repository/SMARTS/ultra
  $ docker build -t <container name> --network=host .
  # or run make
  ```
