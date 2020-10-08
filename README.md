# SMARTS
![Pipeline status](https://gitlab.smartsai.xyz/smarts/smarts/badges/master/pipeline.svg) ![Code coverage](https://gitlab.smartsai.xyz/smarts/smarts/badges/master/coverage.svg?style=flat) ![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

This is the HiWay Edition of Scalable Multi-Agent Training School (SMARTS). It consists of a suite of autonomous driving tasks and support for population based evaluation and training.

![](media/envision_video.gif)

[[_TOC_]]

## Multi-Agent environment as simple as...

```python
import gym
import random

from smarts.core.mission_planner import Mission

agent_configs = {
    "AGENT-007": { "mission": Mission.random() },
    "AGENT-008": { "mission": Mission.random() }
}

env = gym.make("smarts.env:hiway-v0",
               sumo_scenario="./scenarios/loop",
               agent_configs=agent_configs)

env.reset()
for _ in range(1000):
    env.step({
        "AGENT-007": random.choice(["keep_lane", "turn_left", "turn_right"]),
        "AGENT-008": random.choice(["keep_lane", "turn_left", "turn_right"])
    })

env.close()
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

# test sumo installation
make sumo-gui scenario=./scenarios/loop

# setup virtual environment; presently only Python 3.7.x is officially supported
python3.7 -m venv .venv

# enter virtual environment to install all dependencies
source .venv/bin/activate

# upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
pip install --upgrade pip

# install black for formatting (if you wish to contribute)
pip install black

# install dev version of python package with the rllib dependencies
pip install -e .[train]

# make sure you can run tests (and verify they are passing)
make test

# if you wish to view documentation
pip install .[dev]

# then you can run a (built-in) scenario, see following section
```

## Running

We use [supervisord](http://supervisord.org/introduction.html) to run SMARTS together with it's surrounding processes. To run the default example simply execute:

```bash
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

If you want to easily visualize image-based observations you can use our [Visdom](https://github.com/facebookresearch/visdom) integration. Start the visdom server before running your scenario,

```bash
visdom
# Open the printed URL in your browser
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

Subcommands of scenario:
* build-all: Generate all scenarios under the given directories
* build: Generate a single scenario
* clean: Clean generated artifacts

Subcommands of envision:
* start: start envision server

### Examples:

Build all scenario

```
scl scenario build-all
```

Or specify one or more scenarios directories
```
scl scenario build-all ./scenarios
```

Generate a single scenario
```
scl scenario build --clean scenarios/loop
```

Clean generated artifacts
```
scl scenario clean scenarios/loop
```

## Interfacing with Gym

See the provided ready-to-go scripts under the [examples/](./examples) directory.

## Setting up an additional Scenario directory

In order to load a registered social agent it needs to be reachable from a directory contained in the PYTHONPATH.

Consider that `local_scenarios/scenario.py` has this assumption:

```python
social_agent = t.SocialAgentActor(
    name="zoo-car",
    agent_locator="local_scenarios.4lane_t.agent_prefabs:za-v0",
)
```

This is customizable and not all solutions will be exactly the same.

One option is to reorder the directory like so since the directory of the calling script, i.e. `single_agent.py`, is guarenteed to be in PYTHONPATH:

```bash
.
├── examples
│   ├── local_scenarios
│   │   └── 4lane_t
│   │       ├── agent_prefabs.py <- local_scenarios.4lane_t.agent_prefabs
│   │       ├── bubbles.pkl
│   │       ├── map.egg
│   │       ├── map.glb
│   │       ├── map.net.xml
│   │       ├── scenario.py <- scenario script
│   │       └── traffic
│   │           └── basic.rou.xml
│   └── single_agent.py <- calling script
└── requirements.txt
```

Another option is for the calling script, i.e. `single_agent.py`, to update the PYTHONPATH path with `from sys import path; path.append(...)`. In this specific case `path.append('.')` would suffice.

## Building the Package (for Distribution)

These instructions are to build a pip wheel for distribution (e.x. to users directly or through a package manager).

```bash
cd <project>

# virtual env was created during setup instructions
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
make wheel
```

This will create the following directories,
- `build/`
- `dist/`
- `*.egg-info/`

The output wheel will be under `dist/`.

## Contributing

Before pushing python code always run formatting.

```bash
make format
```

Please also read [Contributing](doc/CONTRIBUTING.md)

## Extras

### Building Docs Locally

```bash
make docs

python -m http.server -d docs/_build/html
# Open http://localhost:8000 in your browser
```

### Generating Flame Graphs (Profiling)

Things inevitably become slow, when this happens, Flame Graphs are a great tool to have for finding hot spots in your code.

```bash
# You will need python-flamegraph to generate flamegraphs
pip install git+https://github.com/asokoloski/python-flamegraph.git

make scenario=scenarios/loop script=examples/single_agent.py flamegraph
```

### Visualizing URDFs

Collision models like the ground plane, and vehicles are represented as [Unified Robot Description Format](http://wiki.ros.org/urdf) files. You can easily visualize them with the [urdf-viz](https://github.com/OTL/urdf-viz) tool ([binaries](https://github.com/OTL/urdf-viz/releases)). Once downloaded, simply run,

```bash
./urdf_viz hiway/models/vehicle/vehicle.urdf
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

If you want to push new images to our **public** GitLab container registry (e.g. to update the environment that our GitLab CI tests use) run,

```bash
docker login gitlab.smartsai.xyz:5050
docker build -t gitlab.smartsai.xyz:5050/smarts/smarts-dockerfiles/smarts-base .
docker push gitlab.smartsai.xyz:5050/smarts/smarts-dockerfiles/smarts-base
```

**`smarts-dockerfiles` is PUBLIC and none of those images contain source code and shouldn't!**

### SUMO Troubleshooting

* If you are having issues see: **[SETUP](docs/setup.rst)** and **[SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)**
* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)**
* If you wish to compile from source see: **[SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions)**
* If you build from the git repository use: **[most current version of 1.7](https://github.com/eclipse/sumo/commits/v1_7_0)** or higher
