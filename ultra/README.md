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

### Benchmark

Read the [documentation](https://gitlab.smartsai.xyz/smarts/ULTRA/-/wikis/Benchmark)


ignore this step if you already have smarts environment
### Setup
  ```sh
  python3.7 -m venv .smarts
  # 1-activate virtual environment to install all dependencies
  source .ultra/bin/activate
  # 2-install black for formatting (if you wish to contribute)
  pip install black
  # 3-install dependencies
  pip install -e .
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
We currently support PPO/SAC/TD3/DQN policies, for adding you own policy please follow the guidelines in [documentation](https://gitlab.smartsai.xyz/smarts/ULTRA/-/wikis/Benchmark)
- Train
  Training runs the policy with default configuration:
  `
  {train episodes: 10000, maxsteps:600, timestep:0.1 sec, evaluation rate: 1000, evaluation episodes:100}
  `
  - For each experiment a folder with timestamp is added automatically under `logs/` and it saves tensorboad log, models and pkls
  - For every 1000 observation, evaluation is called automatically and policy is saved under `logs/your-expierment-name/models/`
  ```sh
  $ python ultra/src/train.py --task 1 --level easy 
  # other arguments example: --episodes 20000 --eval-rate 500 --eval-episodes 200 --timestep 1 --headless
  ```
### Run Evaluation Separately
After training your agent, your models should be saved under `logs/your-experiment-name/models/` and you can re-run the evaluation:
  ```sh
  $ python ultra/src/evaluate.py --task 1 --level easy --models logs/your-expierment-name/models
  # other arguments --episodes 20000 --timestep 1 --headless
  ```

### Server
- Server name, address, and destination path `${DST}` for storage

  |Server|Address|${DST}|
  |:----|:----|:----|
  |CX3|10.193.241.233|/data/research|
  |CX4|10.193.241.234|/data/research|
  |Compute-4|10.193.192.17|/data/$USER|
  |Compute-11|10.193.192.113|/data/$USER|
  |GX3|10.193.241.239|/data/research|  

### Docker
- SMARTS is pre-installed and ULTRA source files are copied automatically into Docker image.
- Build a docker image alone
  ```sh
  $ cd path/to/repository/ULTRA/
  $ docker build -t <container name> --network=host .
  ```
- Build, push, load, and run interactive docker cotainer in detached mode in remote server
  ```sh
  $ sudo apt-get install sshpass
  $ cd path/to/repository/ULTRA/
  $ source ./ultra/docker/remote.sh <username> <password> <server name> <container name> .
  ```
  - \<username>       : username for server   
  - \<password>       : password for server   
  - \<server name>    : CX3, CX4, Compute-4, Compute-11, or GX3   
  - \<container name> : string with all lower case, e.g., ultratest
- Docker container has a default memory limit of 100GB. To change, edit the line `--memory=100g` in `ULTRA/ultra/docker/remote.sh`.
- By default, `ULTRA/logs` folder is mapped from Docker container to local storage at `${DST}/logs`.
- To map additional Docker container volumes such as `ULTRA/ultra/scenarios/task1` to local storage, edit `ULTRA/ultra/docker/remote.sh` to include `--volume=${DST}/ultra/scenarios/task1/:/ULTRA/ultra/scenarios/task1/`.
- Some regular commands
  ```sh
  # Copy folder into remote server
  $ sshpass -p <password> scp -r <folder to copy> <username>@<server address>:<destination path>
  $ sshpass -p abcd1234 scp -r ${PWD}/ultra/scenarios/task3 z84216771@10.193.241.239:/data/research/ultra/scenarios/

  # Login into remote server
  $ sshpass -p <password> ssh <username>@<server address>
  $ sshpass -p abcd1234 ssh z84216771@10.193.241.239

  # Enter the interactive docker container
  $ docker exec -ti <container name> bash
  $ docker exec -ti ultratest bash

  # Exit the interactive docker container
  $ ctrl-p ctrl-q
  ```
