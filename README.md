# Welcome to the 2022 NeurIPS Driving SMARTS Competition
Thank you for your interest in the 2022 NeurIPS Driving SMARTS competition. Please take a look at the [competition site](https://codalab.lisn.upsaclay.fr/competitions/6618).

## Registration
In order to be elligible for any rewards, either 

+ Fill out the following form: [Registration Form](https://docs.google.com/forms/d/1bIXTQL420q-cB65j1df2vhbh-79NMlm0M2G0uQtwvds)

+ Or, send a response to smarts4ad@gmail.com including the following information:
    ```text
    Public group name [required]:      
    Group members [required]:      
    Declared affiliations (orgs or any relation to organisers) [required]:      
    Primary contact email [required]:
    ```

## Competition Tracks
Validation Stage: This stage is to validate that your submission will work without errors during Track1 and Track2 evaluation.

There are two competition tracks.
+ Track 1: The participants may use any method to develop their solutions.
+ Track 2: The participants are only allowed to train their methods on the offline datasets.

## Prizes
Top participants in each track will receive the following prizes:

* Gold US$6000
* Silver US$4000
* Bronze US$2000

Additional prizes:

* US$1000 for the most innovative approach out of top-6 entries in both tracks
* US$1000 given to one of the valid submissions not in top-3 positions in either track

Winners in each track will receive cash prizes and will get a chance to present their innovative solutions during a virtual ceremony.

## First Steps
Code and instructions related to the competition may be found in the [competition directory](./competition/) where it is recommended you read the READMEs of each section.
- [Track 1 training](./competition/track1/train/README.md)
- [Track 1 submission](./competition/track1/submission/README.md)
- [Track 2](./competition/track2/README.md)
- [Evaluation](./competition/evaluation/README.md)

## Starting Kits
Starting code may be found for each track in the following locations.
- [Track 1](./competition/track1/)
- [Track 2](./competition/track2/)

## Submission
Deliverables may be submitted to the following site: https://codalab.lisn.upsaclay.fr/competitions/6618#participate-submit_results

Track1 and Track2 deliverables can only be submitted a limited number of times. Therefore, the Validation Stage may be used to ensure your model does not encounter errors during Track1 and Track2 submission.

#

# SMARTS
[![SMARTS CI Base Tests Linux](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml?query=branch%3Amaster) 
[![SMARTS CI Format](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml?query=branch%3Amaster)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg) 

SMARTS (Scalable Multi-Agent RL Training School) is a simulation platform for reinforcement learning (RL) and multi-agent research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776) for background on some of the project goals.

![](docs/_static/smarts_envision.gif)

# Multi-agent experiment as simple as ...
```python
import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec

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
    scenarios=["scenarios/sumo/loop"],
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
1. [CLI Tool](#CLI-Tool)  
    + [CLI Usage](#CLI-Usage)
    + [CLI Examples](#CLI-Examples)
1. [Visualizing Observations](#Visualizing-Observations)
1. [PyMARL and MALib](#PyMARL-and-MALib)
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
# Install the system requirements. You may use the `-y` option to enable automatic assumption of "yes" to all prompts to avoid timeout from waiting for user input. 
bash utils/setup/install_deps.sh

# Setup virtual environment. Presently at least Python 3.7 and higher is officially supported.
python3.7 -m venv .venv

# Enter virtual environment to install dependencies.
source .venv/bin/activate

# Upgrade pip.
pip install --upgrade pip

# Install smarts with extras as needed. Extras include the following: 
# `camera-obs` - needed for rendering camera sensor observations, and for testing.
# `test` - needed for testing.
# `train` - needed for RL training and testing.
pip install -e '.[camera-obs,test,train]'

# Run sanity-test and verify they are passing.
# If tests fail, check './sanity_test_result.xml' for test report. 
make sanity-test
```

### Running
Use the `scl` command to run SMARTS together with it's supporting processes. 

To run the default example, firstly build the scenario `scenarios/sumo/loop`.
```bash
scl scenario build --clean scenarios/sumo/loop
```

Then, run a single-agent SMARTS simulation with Envision display and `loop` scenario.
```bash 
scl run --envision examples/single_agent.py scenarios/sumo/loop 
```

The `--envision` flag runs the Envision server which displays the simulation visualization. See [./envision/README.md](./envision/README.md) for more information on Envision, SMARTS's front-end visualization tool.

After executing the above command, visit http://localhost:8081/ to view the experiment.

Several example scripts are provided in [examples](./examples) folder, as well as a handful of scenarios in [scenarios](./scenarios) folder. You can create your own scenarios using the [Scenario Studio](./smarts/sstudio). Below is the generic command to run and visualize one of the example scripts with a scenario.

```bash
scl run --envision <examples/path> <scenarios/path> 
```

# Examples 
### Usage
Illustration of various ways to use SMARTS. 
1. [Single agent](examples/single_agent.py) example.
1. [Multi agent](examples/multi_agent.py) example.
1. [Parallel environments](examples/parallel_environment.py) to run multiple SMARTS environments in parallel.

### Reinforcement Learning
1. [MARL benchmark](baselines/marl_benchmark)
1. [Driving in traffic](examples/driving_in_traffic) using world model based RL.

# CLI Tool
SMARTS provides a command-line tool to interact with scenario studio and Envision.

### CLI Usage
```bash
scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...
```

Commands:
* scenario
* envision
* zoo
* run

Subcommands of `scenario`:
* build: Generate a single scenario.
* build-all: Generate all scenarios under the given directories.
* clean: Clean generated artifacts.

Subcommands of `envision`:
* start: Start Envision server.

Subcommands of `zoo`:
* build: Build a policy, to submit to the agent zoo.
* install: Attempt to install the specified agents from the given paths/url.
* manager: Start the manager process which instantiates workers.

Subcommands of `run`:
* No subcommands. Use `run` directly to simulate as shown [above](#Running).

### CLI Examples
```bash
# Start envision and serve scenario assets out of ./scenarios
scl envision start --scenarios ./scenarios

# Build all scenarios under given directories
scl scenario build-all ./scenarios ./eval_scenarios

# Rebuild a single scenario, replacing any existing generated assets
scl scenario build --clean scenarios/sumo/loop

# Clean generated scenario artifacts
scl scenario clean scenarios/sumo/loop
```

# Visualizing Observations
Use the [Visdom](https://github.com/facebookresearch/visdom) integration to easily visualize the observations. 

Firstly, start the Visdom server in a terminal.
```bash
visdom
# Open the printed URL in a browser.
```

Secondly, in a separate terminal, run SMARTS simulation. Enable Visdom in the environment by setting `visdom=True`. For example:
```python
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/sumo/loop"],
    agent_specs=agent_specs,
    visdom=True,
)
```
Below is a sample visualization of an agent's camera sensor observations.
<p align="center">
<img src="docs/_static/visdom.gif" ><br/>
(Left) Drivable area grid map. (Center) Occupancy grid map. (Right) Top-down RGB image.
</p>

# PyMARL and MALib
Run SMARTS with [PyMARL](https://github.com/oxwhirl/pymarl).
```bash
git clone git@github.com:ying-wen/pymarl.git

ln -s your-project/scenarios ./pymarl/scenarios

cd pymarl

# Setup virtual environment.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py --config=qmix --env-config=smarts
```

Run SMARTS with [MALib](https://github.com/ying-wen/malib). 
```bash
git clone git@github.com:ying-wen/malib.git

ln -s your-project/scenarios ./malib/scenarios

cd malib

# Setup virtual environment.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python examples/run_smarts.py --algo SAC --scenario ./scenarios/sumo/loop --n_agents 5
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
$ scl scenario build scenarios/sumo/loop --clean

# Run an example. 
# Add --headless if visualisation is not needed.
$ python examples/single_agent.py scenarios/sumo/loop

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
Singularity> scl scenario build /src/scenarios/sumo/loop/
Singularity> exit

# Then, run the container using one of the following methods.

# 1. Run container in interactive mode.
$ singularity shell --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif
# Inside the container
Singularity> python3.7 /src/examples/single_agent.py /src/scenarios/sumo/loop/ --headless

# 2. Run commands within the container from the host system.
$ singularity exec --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif python3.7 /src/examples/single_agent.py /src/scenarios/sumo/loop/ --headless

# 3. Run container instance in the background.
$ singularity instance start --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif smarts_train /src/examples/single_agent.py /src/scenarios/sumo/loop/ --headless
```

# Troubleshooting
### General
In most cases SMARTS debug logs are located at `~/.smarts`. These can be helpful to diagnose problems.

### SUMO
SUMO might encounter problems during setup. Please look through the following for support for SUMO:
* If you are having issues see: [Setup](docs/setup.rst) and [SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)
* If you wish to find binaries: [SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)
* If you wish to compile from source see: [SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions). 
    * Please note that building SUMO may not install other vital dependencies that SUMO requires to run.
    * If you build from the git repository we recommend to use [SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0) or higher

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
