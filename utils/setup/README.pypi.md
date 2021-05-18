# SMARTS
![SMARTS CI](https://github.com/junluo-huawei/SMARTS/workflows/SMARTS%20CI/badge.svg?branch=master) ![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

SMARTS (Scalable Multi-Agent RL Training School) is a simulation platform for reinforcement learning and multi-agent research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776) for background on some of the project goals.

![](docs/_static/smarts_envision.gif)

## Installation of the SMARTS package
You can install the SMARTS package from [PyPI](https://pypi.org/project/smarts/).

Package installation requires Python >= 3.7.
If you dont have python 3.7 or higher, make sure to install or update python first.

```bash
# For windows user 
py -m pip install smarts
 
# For Unix/MACOSX user
python3 -m pip install smarts
```
## Installation of SUMO
[SUMO or "Simulation of Urban Mobility"](https://sumo.dlr.de/docs/index.html) is an open source, highly portable, microscopic and continuous traffic simulation package which SMARTS currently uses as a tool to create scenarios. You need to install SUMO to use the different modules and packages installed through the command above.
You can find a general location for sources and binaries for all platforms here: 
- https://sumo.dlr.de/docs/Downloads.php

If you wish to compile SUMO yourself, the repository is located here: 
 - https://github.com/eclipse/sumo.
 - If you do so make sure to check out the [most current version of 1.7](https://github.com/eclipse/sumo/commits/v1_7_0) or higher.

and the build instructions:  
 - https://sumo.dlr.de/docs/Developer/Main.html#build_instructions

### Linux

SUMO primarily targets Ubuntu versions >= 16.04. So you may not be able to download pre-built binaries for SUMO 1.7 from a package manager if you're running another OS.

If you try through a package manager make sure to command-line call `sumo` to make sure that you have the right version of SUMO.

We would recommend using the prebuilt binaries but if you are using Ubuntu 16 (Xenial), there is a bash script in `extras/sumo/ubuntu_build` that you can use to automate the compilation of SUMO version 1.5.0.

### MacOS

MacOS installation of SUMO is straight-forward. See https://sumo.dlr.de/docs/Downloads.php#macos_binaries for details.

* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)**
    * **Please note that building SUMO may not install other vital dependencies that SUMO requires to run.**
    * If you build from the git repository we recommend you use: **[SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0)** or higher
  
## Modules installed
Through the SMARTS package, you have access to ['core'](https://github.com/huawei-noah/SMARTS/tree/master/smarts/core), ['env'](https://github.com/huawei-noah/SMARTS/tree/master/smarts/env), ['sstudio'](https://github.com/huawei-noah/SMARTS/tree/master/smarts/sstudio) and ['zoo'](https://github.com/huawei-noah/SMARTS/tree/master/smarts/zoo) as sub-modules under `smarts` module and 
then ['envision'](https://github.com/huawei-noah/SMARTS/tree/master/envision) as a separate package.
The `scl` module is also provided to support the command line tool.

## How To Use
SMARTS provides users the ability to customize their agents. The agent is defined in terms of the interface it expects from the environment, and the responses an agent produces.

You can learn more about how to build agents [here](https://smarts.readthedocs.io/en/latest/sim/agent.html).

Here is a simple example of how a single agent experiment can be built:

```python
import gym
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent

class SimpleAgent(Agent):
    def act(self, obs):
        return "keep_lane"

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner),
    agent_builder=SimpleAgent,
)

agent_specs = {
    "Agent-007": agent_spec,
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
To enrich your training datasets, you can edit your own map through [SUMO’s NETEDIT](https://sumo.dlr.de/docs/NETEDIT.html) and export it in a `map.net.xml` format. 

This example `map.net.xml` code creates a simple straight path:
```xml
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <sumo-net-file value="/path/to/scenarios/map.net.xml"/>
    </input>

    <output>
        <output-file value="/path/to/scenarios/map.net.xml"/>
    </output>

    <processing>
        <geometry.min-radius.fix.railways value="false"/>
        <geometry.max-grade.fix value="false"/>
        <offset.disable-normalization value="true"/>
        <lefthand value="false"/>
    </processing>

    <junctions>
        <no-internal-links value="true"/>
        <no-turnarounds value="true"/>
        <junctions.corner-detail value="5"/>
        <junctions.limit-turn-speed value="5.5"/>
        <rectangular-lane-cut value="false"/>
    </junctions>

    <pedestrian>
        <walkingareas value="false"/>
    </pedestrian>

    <report>
        <aggregate-warnings value="5"/>
    </report>

</configuration>
```
The `sstudio` package supports flexible and expressive scenario specification which you can use to generate traffic with different traffic vehicle numbers and routes, and agent missions.

You can create a simple `scenario.py` script to generate the scenario like:

```python
from pathlib import Path
from typing import Any, Tuple

import smarts.sstudio.types as types
from smarts.sstudio import gen_missions, gen_traffic

scenario = str(Path(__file__).parent)

patient_car = types.TrafficActor(
    name="car",
)

shared_route = types.Route(
    begin=("edge-east", 0, 20),
    end=("edge-west", 0, 0),
)

traffic = types.Traffic(
    flows=[
        types.Flow(
            route=shared_route,
            rate=1,
            actors={patient_car: 1},
        )
    ]
)

gen_missions(
    scenario,
    missions=[
        types.Mission(shared_route),
    ],
)
gen_traffic(scenario, traffic, "traffic")
```
That creates a social agent going from east to west on a straight one-way road.

You can read more about the Scenario Studio [here](https://smarts.readthedocs.io/en/latest/sim/scenario_studio.html).
## CLI tool

SMARTS provides a command-line tool to interact with scenario studio and Envision using the command `scl`.
The `scl` command is an abbreviation for "smarts command line".

Usage:
```bash
scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...
```

Commands:
* `envision`
* `scenario`
* `zoo`

Subcommands of `scenario`:
* `build-all`: Generate all scenarios under the given directories
* `build`: Generate a single scenario
* `clean`: Clean generated artifacts

Subcommands of `envision`:
* `start`: start an Envision server

Subcommands of `zoo`:
* `zoo`: Build an agent, used for submitting to the agent-zoo

### Examples:
If you make a directory where you keep all your scenarios, you can:

```bash
# Start envision, serve scenarios assets out of ./scenarios
scl envision start --scenarios ./scenarios

# Build all scenario under given directories
scl scenario build-all ./scenarios ./eval_scenarios

# Rebuild a single scenario, replacing any existing generated assets
scl scenario build --clean scenarios/loop

# Clean generated scenario artifacts
scl scenario clean scenarios/loop
```

## Documentation
Documentation is available at [smarts.readthedocs.io](https://smarts.readthedocs.io/en/latest).
