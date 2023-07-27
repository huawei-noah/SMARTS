# SMARTS
[![SMARTS CI Base Tests Linux](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml?query=branch%3Amaster) 
[![SMARTS CI Format](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/smarts/badge/?version=latest)](https://smarts.readthedocs.io/en/latest/?badge=latest)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg) 
[![Pyversion](https://img.shields.io/pypi/pyversions/smarts.svg)](https://badge.fury.io/py/smarts)
[![PyPI version](https://badge.fury.io/py/smarts.svg)](https://badge.fury.io/py/smarts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SMARTS (Scalable Multi-Agent Reinforcement Learning Training School) is a simulation platform for multi-agent reinforcement learning (RL) and research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776).

![](docs/_static/smarts_envision.gif)

# Documentation
:rotating_light: :bell: Read the docs :notebook_with_decorative_cover: at [smarts.readthedocs.io](https://smarts.readthedocs.io/) . :bell: :rotating_light:

# Examples 
### Primitive
1. [Egoless](examples/1_egoless.py) example.
   + Run a SMARTS simulation without any ego agents, but with only background traffic.
1. [Single-Agent](examples/2_single_agent.py) example.
   + Run a SMARTS simulation with a single ego agent.
1. [Multi-Agent](examples/3_multi_agent.py) example.
   + Run a SMARTS simulation with multiple ego agents.
1. [Environment Config](examples/4_environment_config.py) example.
   + Demonstrate the main observation/action configuration of the environment.
1. [Agent Zoo](examples/5_agent_zoo.py) example.
   + Demonstrate how the agent zoo works.
1. [Agent interface example](examples/6_agent_interface.py)
   + TODO demonstrate how the agent interface works.

### Integration examples
A few more complex integrations are demonstrated.

1. Configurable example
   + script: [control/7_experiment_base.py](examples/control/7_experiment_base.py)
   + Configurable agent number.
   + Configurable agent type.
   + Configurable environment.
1. Parallel environments
   + script: [control/8_parallel_environment.py](examples/control/8_parallel_environment.py)
   + Multiple SMARTS environments in parallel
   + ActionSpaceType: LaneWithContinuousSpeed

### RL Examples
1. [Drive](examples/10_drive). See [Driving SMARTS 2023.1 & 2023.2](https://smarts.readthedocs.io/en/latest/benchmarks/driving_smarts_2023_1.html) for more info.
1. [VehicleFollowing](examples/11_platoon). See [Driving SMARTS 2023.3](https://smarts.readthedocs.io/en/latest/benchmarks/driving_smarts_2023_3.html) for more info.
1. [PG](examples/12_rllib/pg_example.py). See [RLlib](https://smarts.readthedocs.io/en/latest/ecosystem/rllib.html) for more info.
1. [PG Population Based Training](examples/12_rllib/pg_pbt_example.py). See [RLlib](https://smarts.readthedocs.io/en/latest/ecosystem/rllib.html) for more info.

### RL Environment
1. [ULTRA](https://github.com/smarts-project/smarts-project.rl/blob/master/ultra) provides a gym-based environment built upon SMARTS to tackle intersection navigation, specifically the unprotected left turn.

# Issues, Bugs, Feature Requests 
1. First, read how to communicate issues, report bugs, and request features [here](./docs/resources/contributing.rst#communication).
1. Next, raise them using appropriate tags at [https://github.com/huawei-noah/SMARTS/issues](https://github.com/huawei-noah/SMARTS/issues).

# Cite this work
If you use SMARTS in your research, please cite the [paper](https://arxiv.org/abs/2010.09776). In BibTeX format:

```bibtex
@misc{SMARTS,
    title={SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving},
    author={Ming Zhou and Jun Luo and Julian Villella and Yaodong Yang and David Rusu and Jiayu Miao and Weinan Zhang and Montgomery Alban and Iman Fadakar and Zheng Chen and Aurora Chongxi Huang and Ying Wen and Kimia Hassanzadeh and Daniel Graves and Dong Chen and Zhengbang Zhu and Nhat Nguyen and Mohamed Elsayed and Kun Shao and Sanjeevan Ahilan and Baokuan Zhang and Jiannan Wu and Zhengang Fu and Kasra Rezaee and Peyman Yadmellat and Mohsen Rohani and Nicolas Perez Nieves and Yihan Ni and Seyedershad Banijamali and Alexander Cowen Rivers and Zheng Tian and Daniel Palenicek and Haitham bou Ammar and Hongbo Zhang and Wulong Liu and Jianye Hao and Jun Wang},
    url={https://arxiv.org/abs/2010.09776},
    primaryClass={cs.MA},
    booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
    year={2020},
    month={11}
}
```
