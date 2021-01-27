# ULTRA Agents

An ULTRA agent specification inherits from SMARTS's `AgentSpec`. The baseline agents of ULTRA use the `BaslineAgentSpec` available in `ultra/baselines/agent_spec.py`:
```python
# The baseline agent specification in ultra/baselines/agent_spec.py.
spec = AgentSpec(
    interface=AgentInterface(
        waypoints=Waypoints(lookahead=20),
        neighborhood_vehicles=NeighborhoodVehicles(200),
        action=action_type,
        rgb=False,
        max_episode_steps=max_episode_steps,
        debug=True,
    ),
    agent_params=dict(
        policy_params=policy_params, checkpoint_dir=checkpoint_dir
    ),
    agent_builder=policy_class,
    observation_adapter=adapter.observation_adapter,
    reward_adapter=adapter.reward_adapter,
    # action_adapter=... (not specified for baseline agents),
    # info_adapter=... (not specified for baseline agents),
)
```

- `interface` is a SMARTS `AgentInterface` that specifies how the agent will interface with the environment. For example, the types of observations (whether they include images, neighbouring social vehicles, accelerometer data, etc.), vehicle type, and action type can be specified with the agent interface.
- `agent_params` are the parameters that are passed to the agent builder class.
- `agent_builder` is a class inheriting from SMARTS's `Agent`. This class must define `act` and `step` methods.
- `observation_adapter` is a function that takes a raw environment observation and returns a modified observation readable by the agent. This is an optional parameter for any AgentSpec, but is chosen to be assigned for the baseline agents. See more about observation adapters [here](observations.md).
- `reward_adapter` is a function that takes a raw environment observation and a raw environment reward and returns a number corresponding to the agent's reward. This is an optional parameter for any AgentSpec, but is chosen to be assigned for the baseline agents. See more about reward adapters [here](rewards.md).
- `action_adapter` is a function that takes an action of some form and returns an action. This is an optional parameter for any AgentSpec and is chosen to be left as the default implementation for the baseline agents. The default implementation simply returns the action passed to it. See `smarts/core/agent.py` for more details.
- `info_adapter` is a function that takes a raw environment observation, a raw environment reward, and raw environment info, returning a dictionary of information about the agent. This is an optional parameter for any AgentSpec and is chosen to be left as the default implementation for the baseline agents. The default implementation simply returns the raw environment info. See `smarts/core/agent.py` for more details.
