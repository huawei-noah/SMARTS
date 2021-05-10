# SMARTS
![SMARTS CI](https://github.com/junluo-huawei/SMARTS/workflows/SMARTS%20CI/badge.svg?branch=master) ![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

SMARTS (Scalable Multi-Agent RL Training School) is a simulation platform for reinforcement learning and multi-agent research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776) for background on some of the project goals.

![](docs/_static/smarts_envision.gif)

## Installation of SMARTS package
You can install SMARTS package from [PyPI](https://pypi.org/project/smarts/)
Requires Python >= 3.7. 
```bash
# If you dont have python 3.7 or higher, make sure to install or update python first

# For windows user 
py -m pip install smarts
 
# For Unix/MACOSX user
python3 -m pip install smarts

# Follow the instructions given by prompt for setting up the SUMO_HOME environment variable
./install_deps.sh

# verify sumo is >= 1.5.0
# if you have issues see ./doc/SUMO_TROUBLESHOOTING.md
sumo

# setup virtual environment; presently only Python 3.7.x is officially supported
python3.7 -m venv .venv

# enter virtual environment to install all dependencies
source .venv/bin/activate

# upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
pip install --upgrade pip

# install [train] version of python package with the rllib dependencies
pip install -e .[train]

# make sure you can run sanity-test (and verify they are passing)
# if tests fail, check './sanity_test_result.xml' for test report. 
pip install -e .[test]
make sanity-test

# then you can run a scenario, see following section for more details
```
