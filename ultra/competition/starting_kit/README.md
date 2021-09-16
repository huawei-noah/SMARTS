# Starting Kit

Welcome to the ULTRA competition! This README provides instructions on how to get started by installing the dependencies and running a quick example. Additionally, it will walk through the basics of creating your own agent, creating your own scenarios, and submitting your agent for evaluation.

For more comprehensive documentation on SMARTS and ULTRA, please visit these links:
- SMARTS : https://smarts.readthedocs.io/
- ULTRA : https://github.com/huawei-noah/SMARTS/tree/ultra-develop/ultra/docs

## Setup
If not available, install Docker from https://docs.docker.com/get-docker/.

Build and run Docker container:
```bash
$ cd /path/to/SMARTS/ultra/competition/starting_kit
$ docker build --no-cache --network=host -f Dockerfile -t starting_kit:latest .
$ docker run --rm -it --network=host -v ${PWD}:/starting_kit starting_kit:latest

# Inside the Docker container
$ cd /starting_kit
```

### Run a quick example
Try running a quick example.

```bash
# Generate the maps (this only has to be done once).
$ scl scenario build-all scenarios/pool/

# Build the example scenarios (this only has to be done again when the config.yaml changes). See the "Creating your own scenarios" section for more information.
$ python scenarios/build_scenarios.py --task example_scenarios --level example --save-dir scenarios/example_scenarios/ --root-dir scenarios/ --pool-dir scenarios/pool/

# Optional: Start Envision to allow viewing of experiments.
$ scl envision start -s ./scenarios -p 8081 &

# Run the random agent baseline in one of the scenarios you just created.
$ python agents/random_baseline_agent/run.py scenarios/example_scenarios/test_example_2lane_c_50kmh_low-density-flow-10
```
> If you have started Envision, open `localhost:8081` in your browser to view the example.

## Creating your own Agent

### Agent
A bare agent looks like:

```python
from smarts.core.agent import Agent


class MyAgent(Agent):
    def __init__(self):
        pass

    def act(self, observation):
        """Returns an action in the form of a numpy array with shape (3,), where this action
        represents the [throttle, brake, steering] of the agent. Range of throttle, brake, 
        and steering are [0, 1], [0, 1], and [-1, 1], respectively.

        Args:
        observation: The observation provided to the agent from the environment. This
            observation will be the adapted observation, that is, it will be what is
            returned from the agent's observation adapter.

        Returns:
        Union[Sequence[float], np.ndarray]: Either a Sequence of three float values, or a 
            numpy array with shape (3,). The action is the [throttle, brake, steering] of 
            the agent's vehicle.
        """
        pass
```

The agent must contain an act method that returns an action given an observation.

### AgentSpec
In addition to the agent class, a `smarts.core.agent.AgentSpec` will be needed. The `AgentSpec` defines how this agent class will be instantiated and how it will interact with the environment.

The `AgentSpec` takes a few specific parameters.

- `interface`:

  The `AgentInterface` provides the SMARTS simulator with information about what sensors the vehicle needs and also what action the agent outputs. Recall that we restrict the agent's observation to either be of "vector" or "image" type, and that the agent should use a continuous action space. So depending on the type of observation you would like to use, your `AgentInterface` will look like:

  ```python
  from smarts.core.controllers import ActionSpaceType
  from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, RGB, Waypoints

  # If using the image observation
  AgentInterface(
    rgb=RGB(width=64, height=64, resolution=(50 / 64)),
    action=ActionSpaceType.Continuous,
  )

  # If using the vector observation
  AgentInterface(
    neighborhood_vehicles=NeighborhoodVehicles(radius=200.0),
    waypoints=Waypoints(lookahead=20),
    action=ActionSpaceType.Continuous,
  )
  ```

  The image observation adapter requires the car to have an RGB sensor that produces an image with width of 64 pixels and height of 64 pixels. The resolution of the image is recommended to be 50 / 64. That is, the sensor will produce a 64x64 image that spans 50 meters by 50 meters. The values of 64, 64, and 50 / 64 for the `width`, `height`, and `resolution`, respectively, are what will be used for the evaluation on CodaLab, so it is recommended that these values be used. If your agent does not have these values for the `width`, `height`, and `resolution` when submitted for evaluation on CodaLab, the evaluation will fail.

  The vector observation adapter requires the car to have a NeighborhoodVehicles sensor that senses all nearby vehicles within a certain `radius`, and a Waypoints sensor that produces waypoints up to a certain `lookahead` distance along the vehicle's mission path. The values of 200.0 and 20 for the `radius` and `lookahead`, respectively, can be changed. However, these are the values that will be used for the evaluation on CodaLab, so it is recommended that these values be used. If your agent does not have these values for the `radius` and `lookahead` when submitted for evaluation on CodaLab, the evaluation will fail.

  The `AgentInterface` also has a `done_criteria` argument that can be used to specify a `smarts.core.agent_interface.DoneCriteria` object that tells the simulation under which circumstances the agent becomes done in the episode:

  ```python
  from smarts.core.agent_interface import AgentInterface, DoneCriteria

  AgentInterface(
    done_criteria=DoneCriteria(
      collision=...,  # End the episode if the agent collides.
      off_road=...,  # End the episode if the agent drives off the road.
      off_route=...,  # End the episode if the agent drives off the specified mission route.
      on_shoulder=...,  # End the episode if the agent drives on the shoulder.
      wrong_way=...,  # End the episode if the agent drives in a lane with oncoming traffic.
      not_moving=...,  # End the episode if the agent is not moving for 60 seconds or more.
    ),
    ...
  )
  ```

  For evaluation, your agent will be equipped with a `DoneCriteria` that terminates the episode if the agent collides, goes off road, goes off route, or goes the wrong way.

  To see more about what the `AgentInterface` can configure, please see its documentation [here](https://smarts.readthedocs.io/en/latest/sim/agent.html?highlight=agent%20interface#agentinterface).

- `agent_builder`:

  This is usually a class that inherits from `smarts.core.agent.Agent`, e.g. `MyAgent`.

- `agent_params`:

  These are the parameters that the `agent_builder` will take to instantiate an instance. It is usually in the form of a dictionary.

- `observation_adapter`:

  An observation adapter is a `Callable` object that takes a [raw environment observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1) (a `smarts.core.sensors.Observation`) and preprocesses it into something the agent can use. For this competition, we restrict agents to only use an "image" observation or a "vector" observation defined by observation adapters in ULTRA:

  ```python
  import ultra.adapters as adapters
  from smarts.core.agent import AgentSpec

  # If using the image observation
  AgentSpec(
    ...
    observation_adapter=adapters.default_observation_image_adapter.adapt,
    ...
  )

  # If using the vector observation
  AgentSpec(
    ...
    observation_adapter=adapters.default_observation_vector_adapter.adapt,
    ...
  )
  ```

  See the [ULTRA adapters documentation](https://github.com/huawei-noah/SMARTS/blob/master/ultra/docs/adapters.md) to see what these adapters produce. The output of these observation adapters is passed into your agent's `act` method.

  > NOTE: If your agent does not use one of these ULTRA adapters, it will fail the
  evaluation.

- `action_adapter`:

  The action adapter is a `Callable` object that takes the action returned from your agent's `act` method and attempts to do some post-processing on the action so that it can be interpreted by the SMARTS environment. Of course, if you wanted, you could simply return a valid action from your agent's `act` method and you would then have no need for an action adapter.

- `reward_adapter`:

  The reward adapter is a `Callable` object that takes the raw environment observation (a `smarts.core.sensors.Observation`) and the [raw environment reward](https://smarts.readthedocs.io/en/latest/sim/observations.html?highlight=reward#rewards) (a `float`) in order to produce a processed reward. This allows you to create a custom reward function for your agent.

- `info_adapter`:

  Conforming to OpenAI's Gym interface, we allow the agent to specify what information they would like to receive from the environment. An info adapter is a `Callable` object that takes a raw environment observation (a `smarts.core.sensors.Observation`), the raw environment reward (a `float`), and environment information from the ULTRA environment (a `Dict`). The info adapter is meant to return further processed information, if needed.

### Example
See below for a full-fledged example of what an `agent.py` file could look like.

```python
from typing import Dict

import numpy as np
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, Waypoints
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
import ultra.adapters as adapters


class MyAgent(Agent):
    def __init__(self, my_param1, my_param2):
        """Make an agent that will attempt to go as fast as it can."""
        self._my_param1 = my_param1
        self._my_param2 = my_param2

    def act(self, observation) -> np.ndarray:
        """Return the action that corresponds to fully pressing down the gas pedal. Also
        note that this `observation` that is passed to the `act` method is NOT a
        `smarts.core.sensors.Observation`. It is actually the observation that is returned
        by our observation adapter."""
        return np.ndarray([1.0, 0.0, 0.0])


def my_action_adapter(self, action: np.ndarray):
    """Don't change the action. The `action` we receive is the action that our agent's
    `act` method returns. We assume the action that our agent's `act` method returns is
    is already valid."""
    return action


def my_reward_adapter(self, observation: Observation, reward: float):
    """Add the vehicle's speed to the reward. This is the reward that the environment will
    return for our agent upon each step."""
    return reward + observation.ego_vehicle_state.speed


def my_info_adapter(self, observation: Observation, reward: float, info: Dict):
    """Add the reward and raw environment observation to the info. This is the info that
    the environment will return for our agent after each step."""
    info["reward"] = reward
    info["observation"] = observation
    return info


agent_spec = AgentSpec(
    interface=AgentInterface(
        max_episode_steps=100,
        neighborhood_vehicles=NeighborhoodVehicles(radius=200.0),
        waypoints=Waypoints(lookahead=20.0),
        action=ActionSpaceType.Continuous,  # Must use continuous action space.
    ),
    agent_builder=MyAgent,
    agent_params={"my_param1": 1.0, "my_param2": True},
    observation_adapter=adapters.default_observation_vector_adapter.adapt,  # Must use one of the allowed observations.
    action_adapter=my_action_adapter,  # Use the custom action adapter.
    reward_adapter=my_reward_adapter,  # Use the custom reward adapter.
    info_adapter=my_info_adapter,  # Use the custom info adapter.
)
```

## Creating your own scenarios

### Specify scenarios
Scenarios can be defined through a `config.yaml` file.

The `config.yaml` has a specific format that is used to define multiple aspects about the scenarios it creates.

```yaml
levels:
  <level-name>:
    train:
      total: <total>
      ego_missions:
      - start: <start-road>
        end: <end-road>
      intersection_types:
        <intersection-type>:
          percent: <percentage>
          specs: <specs>
          stops: null
          bubbles: null
        ...
    test:
      total: <total>
      ego_missions:
      - start: <start-road>
        end: <end-road>
      intersection_types:
        <intersection-type>:
          percent: <percentage>
          specs: <specs>
          stops: null
          bubbles: null
        ...
  <level-name>:
    ...
  ...
```

The `config.yaml` consists of multiple levels. Each level is given a name, `<level-name>`, and has training and testing scenarios.

Both training and testing scenarios require a `<total>` number of scenarios to be generated.

Additionally, you can specify a mission for your ego agent. This is done by setting a `<start-lane>` and an `<end-lane>` under `ego_missions`. The intersections that are provided allow for multiple values for these two attributes. For t-intersections, the available values for these attributes are `south-SN`, `south-NS`, `west-EW`, `west-WE`, `east-EW`, and `east-WE`. The same values exist for cross-intersections in addition to `north-NS` and `north-SN`. These names, although at first confusing, can be easily interpreted. For example, `south-SN` means "start on the South (`south`) road, and move from South to North (`SN`)". Some combinations of `<start-lane>` and `<end-lane>` are unsupported. However, the combination of `south-SN` for the `<start-lane>` and `<west-EW>` will always work on all intersections. For evaluation, your agent will only be tested on South to West left-turns and East to South left-turns.

Finally, the actual intersections can be specified. Values for the `<intersection-type>` attribute include all directory names under the `scenarios/pool/` directory. For example, `2lane_c` and `4lane_t`.

For each intersection type, the percentage of the scenarios with that intersection type can be specified with the `<percentage>` attribute. This number is an element of the interval [0.0, 1.0] and specifies the proportion of scenarios that will have the given intersection type.

The `<specs>` attribute is a list of lists that describe the road speed, traffic density, and proportion of scenarios with this intersection type that have this speciifc road speed and traffic density. For example, a valid value for `<specs>` would be `[[50kmh,no-traffic,0.33],[70kmh,mid-density,0.33],[100kmh,high-density,0.34]]`. This indicates that 33% of the scenarios with this intersection type, are on 50 km/h roads with no traffic, 33% are on 70 km/h roads with medium-density traffic, and 34% of the scenarios with this intersection type are on 100 km/h roads with high-density traffic. The available values for the road speed are `50kmh`, `70kmh`, and `100kmh`. And the available values for the traffic density are `no-traffic`, `low-density`, `mid-density`, `high-density`, `low-interaction`, `mid-interaction`, and `high-interaction`. The "`-interaction`" distributions are meant for the agent to experience interaction in the intersection, and only in the intersection. They use a more limited number of social vehicles, and their are no social vehicles in the default ego mission route.

The `stop` and `bubbles` are higher-level features that allow you to specify stopped vehicles and utilize zoo agents in your scenario, respectively. We imagine that these will not necessarily be needed for the competition, however, ways to implement them can be found in the main ULTRA code in the [SMARTS repository](https://github.com/huawei-noah/SMARTS).

A level can have multiple intersection types, and a `config.yaml` can have multiple levels. You can find an example of a task's `config.yaml` under `scenarios/example_scenarios/`.

### Building Scenarios
First, ensure that the maps used to create the scenarios are built. This step only has to be done once, and only again if the maps change.

```bash
$ scl scenario build-all scenarios/pool/
```

Once your task's `config.yaml` is created, you can build your scenarios with the provided `scenarios/build_scenarios.py`. Using this script allows you to build scenarios quickly and easily.

Say, for example, we have a `config.yaml` describing scenarios for a level called "`my_level`". We can create a new directory to hold our (to-be-created) scenarios and our task's `config.yaml`:

```bash
$ mkdir scenarios/my_task/
$ mv config.yaml scenarios/my_task/
```

Next, we can run the `build_scenarios.py` script which will read our `config.yaml` and create the `my_level` scenarios:

```bash
$ python scenarios/build_scenarios.py --task my_task --level my_level --save-dir scenarios/my_task/ --root-dir scenarios/ --pool-dir scenarios/pool/
```

After this command is run, you should see scenarios for your `my_level` under `scenarios/my_task/`.

The arguments for `build_scenarios.py` are as follows:
- `--task`: The name of the task to build. It should match a directory name under the specified `--root-dir` argument.
- `--level`: The level to build from the task's `config.yaml`.
- `--save-dir`: The directory in which to save the created scenarios.
- `--root-dir`: The directory containing the task's folder.
- `--pool-dir`: Where the maps used to build the scenarios can be found.

## Submitting an Agent for Evaluation
1. Ensure you have an `agent.py` file and an `agent_spec` variable in that file

   Your agent submission should contain at least one file called `agent.py`. In this file should be a variable called `agent_spec` that references a `smarts.core.agent.AgentSpec` instance. The `agent_spec` defines how your agent will be built.

2. If it is one of the baselines (or similar), ensure that the agent has access to your desired checkpoint.

   If you are using a reinforcement learning baseline agent (or an agent which takes a checkpoint directory as an argument to load from), ensure that this checkpoint directory is specified in the `AgentSpec`. For example, the SAC baseline `Agent` takes a `checkpoint_dir` argument that can be specified through its `agent_spec`'s `policy_params`. Ensure that this `policy_params` specifies the `checkpoint_dir` argument to be the directory in which your desired neural network weights are saved.

3. Ensure your agent works with the provided evaluation script

   This starting kit comes provided with an `evaluation` directory containing an `evaluate.py` script that can be used to verify that your agent will work with the evaluation that CodaLab will do on your submission.

   The `evaluate.py` script takes the following arguments:
   - `--submission-dir`: The directory of your `agent.py` file and other files needed for your submission.
   - `--evaluation-scenarios-dir`: The directory containing the scenarios you would like to use to evaluate your agent.
   - `--scores-dir`: The directory in which a `scores.txt` file will be saved. This file contains metrics about the evaluation. The metrics in `scores.txt` are the same metrics that CodaLab will use to evaluate your submission.
   - `--verbose`: Print extra information regarding the evaluation.

   For example:

   ```bash
   $ python evaluation/evaluate.py local --submission-dir agents/my_agent/ --evaluation-scenarios-dir scenarios/example_scenarios/ --scores-dir ./my_scores/
   ```
   > This will evalute the agent in the `my_agent` directory on the scenarios in the `scenarios/example_scenarios/` directory, and output evaluation metrics in a `scores.txt` file that will by saved in the `my_scores/` directory.

4. Zip up the `agent.py` and all other files your agent needs in order to run

   ```bash
   $ cd agents/my_agent/
   $ zip my_agent.zip *
   ```
   > You can verify that the submission was zipped properly by unzipping your compressed submission. You should see all the files and directories you zipped when you uncompress `my_agent.zip`, that is, make sure extra directories are not created within the zip.

5. Go to the competition page on CodaLab and upload your zipped agent submission
