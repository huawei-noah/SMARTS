# SMARTS
![Pipeline status](TODO/pipeline.svg) ![Code coverage](TODO/coverage.svg?style=flat) ![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

This is the HiWay Edition of Scalable Multi-Agent Training School (SMARTS). It consists of a suite of autonomous driving tasks and support for population based evaluation and training.

![](media/envision_video.gif)

## Multi-Agent environment as simple as...

```python
import gym

from smarts.core.utils.episodes import episodes
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, AgentPolicy

class Policy(AgentPolicy):
    def act(self, obs):
        return "keep_lane"

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    policy_builder=Policy,
)

agent_specs = {"Agent-007": agent_spec, "Agent-008": agent_spec}

env = gym.make(
    "smarts.env:hiway-v0", scenarios=["scenarios/loop"], agent_specs=agent_specs,
)

for episode in range(100):
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }
    observations = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        agent_actions = {
            a_id: agents[a_id].act(agent_obs)
            for a_id, agent_obs in observations.items()
        }
        observations, _reward, dones, _infos = env.step(agent_actions)
```

## Setup

```bash
# For Mac OS X users, make sure XQuartz is pre-installed as SUMO's dependency

# git clone ...
cd <project>

# Follow the instructions given by prompt for setting up the SUMO_HOME environment variable
./install_deps.sh

# verify sumo is >= 1.5.0
# if you have issues see ./doc/SUMO_TROUBLESHOOTING.md
sumo

# test sumo installation, make sure you see a window displaying a curvy loop road network.
make sumo-gui scenario=./scenarios/loop

# setup virtual environment; presently only Python 3.7.x is officially supported
python3.7 -m venv .venv

# enter virtual environment to install all dependencies
source .venv/bin/activate

# upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
pip install --upgrade pip

# install [train] version of python package with the rllib dependencies
pip install -e .[train]

# make sure you can run tests (and verify they are passing)
make test

# if you wish to contribute to SMARTS, also install the [dev] dependencies.
pip install .[dev]

# then you can run a scenario, see following section for more details
```

## Running

We use [supervisord](http://supervisord.org/introduction.html) to run SMARTS together with it's supporting processes. To run the default example simply build a scenario and start supervisord:

```bash
# build scenarios/loop
scl scenario build --clean scenarios/loop

# start supervisord
supervisord
```

With `supervisord` running, visit http://localhost:8081/ in your browser to view your experiment.

See [./envision/README.md](./envision/README.md) for more information on Envision, our front-end visualization tool.

Several example scripts are provided under [`SMARTS/examples`](./examples), as well as a handful of scenarios under [`SMARTS/scenarios`](./scenarios). You can create your own scenarios using the [Scenario Studio](./smarts/sstudio). Here's how you can use one of the example scripts with a scenario.

```bash
# Update the command=... in ./supervisord.conf
#
# [program:smarts]
# command=python examples/single_agent.py scenarios/loop
# ...
```

## CLI tool
SMARTS provides a command-line tool to interact with scenario studio and Envision.

Usage
```
scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...
```

Commands:
* envision
* scenario
* zoo

Subcommands of scenario:
* build-all: Generate all scenarios under the given directories
* build: Generate a single scenario
* clean: Clean generated artifacts

Subcommands of envision:
* start: start envision server

Subcommands of zoo:
* zoo: Build an agent, used for submitting to the agent-zoo

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

## Interfacing with Gym

See the provided ready-to-go scripts under the [examples/](./examples) directory.

## Contributing

Please read [Contributing](doc/CONTRIBUTING.md)

## Extras

### Visualizing Agent Observations
If you want to easily visualize image-based observations you can use our [Visdom](https://github.com/facebookresearch/visdom) integration. Start the visdom server before running your scenario,

```bash
visdom
# Open the printed URL in your browser
```

### Building Docs Locally

```bash
make docs

python -m http.server -d docs/_build/html
# Open http://localhost:8000 in your browser
```

### Generating Flame Graphs (Profiling)

Things inevitably become slow, when this happens, Flame Graphs are a great tool to for finding hot spots in your code.

```bash
# You will need python-flamegraph to generate flamegraphs
pip install git+https://github.com/asokoloski/python-flamegraph.git

make flamegraph scenario=scenarios/loop script=examples/single_agent.py
```


### Interfacing w/ PyMARL and malib

[PyMARL](https://github.com/oxwhirl/pymarl) and [malib](https://github.com/ying-wen/malib) presently live under the contrib package. You can run them via,

```bash
# somewhere on your machine, outside the HiWay directory
# TODO: Update this to our fork
git clone git@gitlab.smartsai.xyz:smarts/pymarl.git
cd pymarl

# or wherever you have placed your pymarl repo
ln -s $(PWD)/scenarios ../pymarl/scenarios

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py --config=qmix --env-config=smarts
```

```bash
# somewhere on your machine, outside the HiWay directory
git clone git@github.com:ying-wen/malib.git
cd malib

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# or wherever you have placed your malib repo
ln -s $(PWD)/scenarios ../malib/scenarios

python examples/run_smarts.py --algo SAC --scenario ./scenarios/loop --n_agents 5
```

### Using Docker


If you're comfortable using docker or are on a platform without suitable support to easily run SMARTS (e.g. an older version of Ubuntu) you can run the following,

```bash
docker build -t smarts .
docker run --rm -it -v $(PWD):/src smarts
cd /src
pip install -e .[train]

$ python examples/single_agent.py scenarios/loop
```

# TODO: once we've moved our docker images to github, update this:

If you want to push new images to our **public** GitLab container registry (e.g. to update the environment that our GitLab CI tests use) run,

```bash
docker login gitlab.smartsai.xyz:5050
docker build -t gitlab.smartsai.xyz:5050/smarts/smarts-dockerfiles/smarts-base .
docker push gitlab.smartsai.xyz:5050/smarts/smarts-dockerfiles/smarts-base
```

### SUMO Troubleshooting

* If you are having issues see: **[SETUP](docs/setup.rst)** and **[SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)**
* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)**
* If you wish to compile from source see: **[SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions)**
* If you build from the git repository use: **[SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0)** or higher
