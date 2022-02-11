# SMARTS
[![SMARTS CI Base Tests Linux](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml?query=branch%3Amaster) 
[![SMARTS CI Format](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg) 
[![Downloads](https://img.shields.io/pypi/dm/smarts)](https://pypi.org/project/smarts/)
[![GitHub contributors](https://img.shields.io/github/contributors/huawei-noah/smarts)](https://github.com/huawei-noah/smarts/graphs/contributors)

SMARTS (Scalable Multi-Agent RL Training School) is a simulation platform for reinforcement learning (RL) and multi-agent research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776) for background on some of the project goals.

![](docs/_static/smarts_envision.gif)

# Multi-Agent experiment as simple as ...
```python
import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent

class SimpleAgent(Agent):
    def act(self, obs):
        return "keep_lane"

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    agent_builder=SimpleAgent,
)

agent_specs = {
    "Agent-007": agent_spec,
    "Agent-008": agent_spec,
}

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/loop"],
    agent_specs=agent_specs,
)

agents = {
    agent_id: agent_spec.build_agent()
    for agent_id, agent_spec in agent_specs.items()
}
observations = env.reset()

for _ in range(1000):
    agent_actions = {
        agent_id: agents[agent_id].act(agent_obs)
        for agent_id, agent_obs in observations.items()
    }
    observations, _, _, _ = env.step(agent_actions)
```

# Contents
1. [Documentation](#Documentation)
1. [Setup](#Setup)
    + [Installation](#Installation)
    + [Running](#Running)
1. [Examples](#Examples)
    + [Usage](#Usage)
    + [Reinforcement Learning](#Reinforcement-Learning)
1. [Standard Environments](#Standard-Environment)
1. [Containers](#Containers)
    + [Docker](#Docker)
    + [Singularity](#Singularity)
1. [Troubleshooting](#Troubleshooting)
    + [General](#General)
    + [SUMO](#SUMO)
1. [Bug Reports](#Bug-Reports)
1. [Contributing](#Contributing)
1. [Citing](#Citing)

# Documentation
Documentation is available at [smarts.readthedocs.io](https://smarts.readthedocs.io/en/latest).

# Setup
### Installation
```bash
git clone https://github.com/huawei-noah/SMARTS.git
cd <path/to/SMARTS>

# For Mac OS X users, ensure XQuartz is pre-installed.
# Install the system requirements, then follow the instructions for setting up the `SUMO_HOME` environment variable. You may use the `-y` option to enable automatic assumption of "yes" to all prompts to avoid timeout from waiting for user input. 
bash utils/setup/install_deps.sh

# Verify sumo is >= 1.5.0.
# If you have issues see ./docs/SUMO_TROUBLESHOOTING.md .
sumo

# Setup virtual environment. Presently at least Python 3.7 and higher is officially supported.
python3.7 -m venv .venv

# Enter virtual environment to install dependencies.
source .venv/bin/activate

# Upgrade pip.
pip install --upgrade pip

# Install smarts with extras as needed. [train] version of python package with the rllib dependencies
pip install -e '.[train]'

# Install [camera-obs] version of python package with the panda3D dependencies if you want to run sanity tests or render camera sensor observations in your simulations
# make sure to install [test] version of python package with the rllib dependencies so that you can run sanity-test (and verify they are passing)
pip install -e '.[camera-obs]'  

pip install -e '.[test]'

# Make sure you install the [camera-obs] dependencies first and then can run sanity-test (and verify they are passing)
# if tests fail, check './sanity_test_result.xml' for test report. 
make sanity-test

# Then you can run a scenario, see following section for more details
```

### Running
Use the `scl` command line to run SMARTS together with it's supporting processes. To run the default example simply build a scenario and run the following command:

```bash
# Build `scenarios/loop`.
scl scenario build --clean scenarios/loop

# Run an experiment with Envision display and `loop` scenario. 
scl run --envision examples/single_agent.py scenarios/loop 
```

Add the `--envision` flag to run the Envision server where you can see the visualization of the experiment. See [./envision/README.md](./envision/README.md) for more information on Envision, our front-end visualization tool.

After executing the above command, visit http://localhost:8081/ in your browser to view your experiment.

Several example scripts are provided under [`SMARTS/examples`](./examples), as well as a handful of scenarios under [`SMARTS/scenarios`](./scenarios). You can create your own scenarios using the [Scenario Studio](./smarts/sstudio). Below is the generic command to run and visualize one of the example scripts with a scenario.

```bash
scl run --envision <examples/script_path> <scenarios/path> 
```

Pass in the agent example path and scenarios folder path above to run an experiment like the one mentioned above.

# Examples 
### Usage
Illustration of various ways to use SMARTS. 
1. [Single agent](examples/single_agent.py) example.
1. [Multi agent](examples/multi_agent.py) example.
1. [Parallel environments](examples/parallel_environment.py) to run multiple SMARTS environments in parallel.

### Reinforcement Learning
1. [MARL benchmark](baselines/marl_benchmark)
1. [Stable Baselines 3](examples/sb3) using PPO.
1. [Driving in traffic](examples/driving_in_traffic) using world model based RL.

# Standard Environments
1. `intersection-v0`: In this task, the ego-vehicle needs to drives towards an intersection with an all-way-stop traffic, and make a left turn without any collisions. Further task description is available at [/smarts/env/intersection_env.py](smarts/env/intersection_env.py).  
    ```python
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=True, # If False, enables Envision display.
        sumo_headless=False, # If True, enables sumo-gui display.
    )
    ```

## CLI tool

SMARTS provides a command-line tool to interact with scenario studio and Envision.

Usage
```
scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...
```

Commands:
* scenario
* envision
* zoo
* run

Subcommands of scenario:
* build-all: Generate all scenarios under the given directories
* build: Generate a single scenario
* clean: Clean generated artifacts

Subcommands of envision:
* start: start envision server

Subcommands of zoo:
* zoo: Build an agent, used for submitting to the agent-zoo

Subcommands of run:
No subcommands of `run`. You can directly use `run` to simulate an experiment as mentioned in the example above.

### Examples:

```
# Start envision, serve scenario assets out of ./scenarios
scl envision start --scenarios ./scenarios

# Build all scenario under given directories
scl scenario build-all ./scenarios ./eval_scenarios

# Rebuild a single scenario, replacing any existing generated assets
scl scenario build --clean scenarios/loop

# Clean generated scenario artifacts
scl scenario clean scenarios/loop
```


### Building Docs Locally

Assuming you have run `pip install .[dev]`.

```bash
make docs

python -m http.server -d docs/_build/html
# Open http://localhost:8000 in your browser
```

## Extras

### Visualizing Agent Observations
If you want to easily visualize observations you can use our [Visdom](https://github.com/facebookresearch/visdom) integration. Start the visdom server before running your scenario,

```bash
visdom
# Open the printed URL in your browser
```

And in your experiment, start your environment with `visdom=True`

```python
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/loop"],
    agent_specs=agent_specs,
    visdom=True,
)
```

### Interfacing w/ PyMARL and malib

[PyMARL](https://github.com/oxwhirl/pymarl) and [malib](https://github.com/ying-wen/malib) have been open-sourced. You can run them via,

```bash
git clone git@github.com:ying-wen/pymarl.git

ln -s your-project/scenarios ./pymarl/scenarios

cd pymarl

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py --config=qmix --env-config=smarts
```

```bash
git clone git@github.com:ying-wen/malib.git

ln -s your-project/scenarios ./malib/scenarios

cd malib

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python examples/run_smarts.py --algo SAC --scenario ./scenarios/loop --n_agents 5
```

# Containers
### Docker
SMARTS docker images are hosted at [dockerhub](https://hub.docker.com/orgs/huaweinoah).

```bash
$ cd </path/to/SMARTS>
$ docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:<version>
# E.g. docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:v0.5.1

# If visualization is needed, run Envision server in the background.
$ scl envision start -s ./scenarios -p 8081 &

# Build the scenario. 
# This step is required on the first time, and whenever the scenario is modified.
$ scl scenario build scenarios/loop --clean

# Run an example. 
# Add --headless if visualisation is not needed.
$ python examples/single_agent.py scenarios/loop

# Visit http://localhost:8081 in the host machine to see the running simulation in Envision.
```

### Singularity
```bash
$ cd </path/to/SMARTS>

# Build container from definition file.
$ sudo singularity build ./utils/singularity/smarts.sif ./utils/singularity/smarts.def

# Use the container to build the required scenarios.
$ singularity shell --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif
# Inside the container
Singularity> scl scenario build /src/scenarios/loop/
Singularity> exit

# Then, run the container using one of the following methods.

# 1. Run container in interactive mode.
$ singularity shell --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif
# Inside the container
Singularity> python3.7 /src/examples/single_agent.py /src/scenarios/loop/ --headless

# 2. Run commands within the container from the host system.
$ singularity exec --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif python3.7 /src/examples/single_agent.py /src/scenarios/loop/ --headless

# 3. Run container instance in the background.
$ singularity instance start --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif smarts_train /src/examples/single_agent.py /src/scenarios/loop/ --headless
```

# Troubleshooting
### General
In most cases SMARTS debug logs are located at `~/.smarts`. These can be helpful to diagnose problems.

### SUMO
SUMO can have some problems in setup. Please look through the following for support for SUMO:
* If you are having issues see: **[SETUP](docs/setup.rst)** and **[SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)**
* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)**
* If you wish to compile from source see: **[SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions)**. 
    * **Please note that building SUMO may not install other vital dependencies that SUMO requires to run.**
    * If you build from the git repository we recommend you use: **[SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0)** or higher

# Bug Reports
Please read [how to create a bug report](https://github.com/huawei-noah/SMARTS/wiki/How-To-Make-a-Bug-Report) and then open an issue [here](https://github.com/huawei-noah/SMARTS/issues).

# Contributing
Please read [contributing](CONTRIBUTING.md).

# Citing
If you use SMARTS in your research, please cite the [paper](https://arxiv.org/abs/2010.09776). In BibTeX format:

```bibtex
@misc{zhou2020smarts,
      title={SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving},
      author={Ming Zhou and Jun Luo and Julian Villella and Yaodong Yang and David Rusu and Jiayu Miao and Weinan Zhang and Montgomery Alban and Iman Fadakar and Zheng Chen and Aurora Chongxi Huang and Ying Wen and Kimia Hassanzadeh and Daniel Graves and Dong Chen and Zhengbang Zhu and Nhat Nguyen and Mohamed Elsayed and Kun Shao and Sanjeevan Ahilan and Baokuan Zhang and Jiannan Wu and Zhengang Fu and Kasra Rezaee and Peyman Yadmellat and Mohsen Rohani and Nicolas Perez Nieves and Yihan Ni and Seyedershad Banijamali and Alexander Cowen Rivers and Zheng Tian and Daniel Palenicek and Haitham bou Ammar and Hongbo Zhang and Wulong Liu and Jianye Hao and Jun Wang},
      url={https://arxiv.org/abs/2010.09776},
      primaryClass={cs.MA},
      booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
      year={2020},
      month={11}
 }
```
