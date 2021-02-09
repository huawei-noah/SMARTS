# Adding a Custom Agent

## Create Your Agent
Similar to the baseline agents available in `ultra/baselines/` a custom agent conforming to ULTRA's specification is defined by a directory containing: `<your_agent_name>/setup.py`,  
`<your_agent_name>/<your_agent_name>/__init__.py`,  
`<your_agent_name>/<your_agent_name>/policy.py`, and  `<your_agent_name>/<your_agent_name>/params.yaml`.
- The creation of `setup.py` and `__init__.py` can be done by following the implementations of these files available in the baseline agents.
- The `params.yaml` file contains parameters for your policy class to use.
- The `policy.py` file contains your agent. Minimally, it must have a class that inherits from `smarts.core.agent.Agent`, and this class must define at least the following three methods:
  - ```python
    # Given your params.yaml as a dictionary in the form of policy_params, and
    # a checkpoint directory, checkpoint_dir, initialize an instance of your
    # agent.
    def __init__(self, policy_params, checkpoint_dir):
        pass
    ```
  - ```python
    # Given a state (and an option to explore), return an action according to
    # your agent's "action space type" (see step 2).
    def act(self, state, explore: bool):
        pass
    ```
  - ```python
    # Step your agent (e.g. add the experience to the replay buffer and have the
    # neural network learn from experiences).
    def step(self, state, action, reward, next_state, done, others):
        pass
    ```
  See the baseline agents for examples of these method implementations.
## Allow for Agent Registration
Once your agent is configured as above, we must ensure that it can be registered. In the package containing your agent, add the following  to the package's `__init__.py`
```python
from smarts.zoo.registry import register
from smarts.core.controllers import ActionSpaceType
from ultra.baselines.agent_spec import BaselineAgentSpec

from <your.agents.package.name>.<your_agent_name>.<your_agent_name>.policy import <YourAgentsPolicyClass>

register(
    locator="<your_agent_name>-v<your_agent_version_number>",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.<YourActionSpaceType>,
        policy_class=<YourAgentsPolicyClass>,
        **kwargs
    ),
)
```
For example, the package containing the baseline agents, `ultra.baselines`, has `ultra/baselines/__init__.py` that registers each baseline agent with its ULTRA Agent Specification.
Additionally, see `smarts/core/controllers/__init__.py` to view available action space types and how they are performed.

In `ultra/agent_pool.json`, add your agent's information:
```json
  "<your_agent_name>" : {
    "path" : "<your.agents.package.name>",
    "name" : "<your_agent_name>",
    "locator : "<your_agent_name>-v<your_agent_version_number>",
  }
```

## Use Your Agent
When running `train.py` or `evaluate.py`, ensure the `--policy` flag is set as `<your_agent_name>`. For example:
```sh
$ python ultra/train.py --task 1 --level easy --policy <your_agent_name>
```
