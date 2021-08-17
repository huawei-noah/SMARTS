# Policy Zoo

## Packaging Policies

**Build policies** (and make available in zoo/policies)

```bash
# cd <project root>
scl zoo build <policy>
# e.g. scl zoo build zoo/policies/open-agent
```

**Installing policies** (and make available in zoo/policies)

```bash
# cd <project root>
scl zoo install <policy>
# e.g. scl zoo install zoo/policies/open-agent
```

**Use** (in scenarios)
Under your scenario directory, create a requirements.txt
```
# we'll be serving packages from a local pip index
--extra-index-url http://localhost:8080
open-agent==0.1.187
```

Then in your scenario.py:
```python
# */scenario.py (example)
# ...
t.SocialAgentActor(
    name="my-open-agent",
    agent_locator="open_agent:open_agent-v0"
)
# ...
```

## Evaluating Policies

### Introduction

The evaluation module contains a collection of scripts that are used to assess the behaviour of agents.

### Metrics

Agents can be tested in the evaluation model for useful traits. These evaluations include:

* Diversity
* Collision
* Off-road
* Kinematics

### Scenarios

The given agents are run through a number of different scenarios and the the per-step state of each agent is recorded for evaluation.

### Usage

Before running the evaluation, ensure the config is correct. You can specify the agents and the scenarios that they will be evaluated on in `./batch_list.yaml`.

The base path to the referenced scenario directories can be configured through the `scenarios_root` key value, the `agent_list` values can be used to configure the grouping of agents to compare, and the `scenario_list` values are used to specify the trial scenarios and additional per-scenario configuration.

An agent is specified by the locator, you can also give it a name and specify the parameters of the agent.

```
agent_list:
  group_1:
    - locator: smarts.zoo.policies:keep-lane-agent-v0
      name: keep_lane_agent_v0
    - locator: smarts.zoo.policies.open_agent.open_agent:open_agent-v0
      name: open_agent_v0
      params:
        Q_u_accel: 0
```

To run the evaluation:

```bash
cd <project root>/zoo/evaluation
make evaluation
```

Evaluation results are written to `./results` into time-stamped named directories.

### Results

The the evaluation results will be stored in `diversity_evaluation_result.csv`, `rank.csv` and `report.csv` and the driven path and speed will be drawn to diagrams as a visual aid.

## Extras

### Running Policy Tests

If policies come with tests, you can run them all via,

```bash
cd <project root>/zoo
make test
```
