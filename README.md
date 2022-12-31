# SMARTS
[![SMARTS CI Base Tests Linux](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml?query=branch%3Amaster) 
[![SMARTS CI Format](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/smarts/badge/?version=latest)](https://smarts.readthedocs.io/en/latest/?badge=latest)
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

# Documentation
:rotating_light: :bell: Read the docs :notebook_with_decorative_cover: at [smarts.readthedocs.io](https://smarts.readthedocs.io/en/latest) . :bell: :rotating_light:

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
# `camera_obs` - needed for rendering camera sensor observations, and for testing.
# `test` - needed for testing.
# `train` - needed for RL training and testing.
pip install -e '.[camera_obs,test,train]'

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
1. [Single agent](examples/control/single_agent.py) example.
1. [Multi agent](examples/control/multi_agent.py) example.
1. [Parallel environments](examples/control/parallel_environment.py) to run multiple SMARTS environments in parallel.

### RL Model
1. [MARL benchmark](baselines/marl_benchmark)
1. [Intersection](examples/rl/intersection) using PPO from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).
<img src="examples/rl/intersection/docs/_static/intersection.gif" height="450" width="600"/>
1. [Racing](examples/rl/racing) using world model based RL.
<img src="examples/rl/racing/docs/_static/racing.gif" height="350" width="600"/>

### RL Environment
1. [ULTRA](https://github.com/smarts-project/smarts-project.rl/blob/master/ultra) provides a gym-based environment built upon SMARTS to tackle intersection navigation, specifically the unprotected left turn.

# Issues, Bugs, Feature Requests 
1. First, read how to communicate issues, report bugs, and request features [here](./docs/resources/contributing.rst#communication).
1. Next, raise them using appropriate tags at [https://github.com/huawei-noah/SMARTS/issues](https://github.com/huawei-noah/SMARTS/issues).

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
